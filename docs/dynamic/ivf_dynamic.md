# Dynamic IVF: adding & deleting after build

This document describes what it takes to make `rabitqlib::ivf::IVF`
(`include/rabitqlib/index/ivf/ivf.hpp`) support **add** and **delete** after the
initial `construct()`, turning the current build-once index into a mutable one.

## 1. How the index works today (and why it's static)

`construct(data, centroids, cluster_ids, …)` (`ivf.hpp:170`) takes the *entire*
dataset plus externally computed centroids and per-vector cluster assignments,
then:

1. Counts per-cluster sizes and builds `id_lists` (`ivf.hpp:178-188`).
2. `allocate_memory(counts)` (`ivf.hpp:216`) makes **three single contiguous
   allocations**, each sized exactly for the whole dataset:
   - `batch_data_` — 1-bit codes + factors, packed in FastScan **batches of
     `kBatchSize = 32`** (`fastscan.hpp:19`);
   - `ex_data_` — extra-bit codes, per vector;
   - `ids_` — `PID`s, ordered by cluster.
3. `init_clusters(counts)` (`ivf.hpp:236`) slices those buffers: each `Cluster`
   just stores **raw pointers** (`batch_data`, `ex_data`, `ids`) + `num` into the
   shared buffers at offsets computed from the prefix sum of cluster sizes
   (`cluster.hpp`).
4. `quantize_cluster` (`ivf.hpp:260`) rotates each vector, then calls
   `quant::quantize_split_batch` (`quantization/rabitq.hpp:224`) on chunks of 32
   vectors, writing transposed (bit-sliced) FastScan blocks.

### The blockers for in-place mutation

| Blocker | Where | Why it blocks add/delete |
|---|---|---|
| **Single contiguous, exactly-sized buffers** | `allocate_memory` `ivf.hpp:216-231` | Cluster *i*'s region is wedged between *i-1* and *i+1*. Growing one cluster needs shifting every later cluster — no room to append. |
| **Clusters hold raw pointers, not storage** | `cluster.hpp` | A cluster can't grow itself; it has no ownership or capacity. |
| **FastScan 1-bit block is transposed & nibble-interleaved** | `pack_codes` `fastscan.hpp:69-106` | A vector's 1-bit code is **not** contiguous in the packed block — see [§1.1](#11-what-can-and-cant-be-swapped). Moving one vector needs a one-batch repack, not a memcpy. (Only `batch_data` is affected; `ids`/`ex_data` are plain swaps.) |
| **No reverse id map** | `ids_` is slot→`PID` only | Delete-by-id needs `PID → (cluster, offset)`; today that requires an O(N) scan. |
| **`num_` and offsets are fixed** | `ivf.hpp`, `init_clusters` | Totals and per-cluster offsets are computed once at build. |
| **Centroids are external & fixed** | passed into `construct` | Fine for routing (assign new vectors to nearest existing centroid), but no rebalancing — see [balance](#7-cluster-balance--known-limitations). |

### 1.1 What can and can't be swapped

Each vector occupies the same *number* of bits, but "same size" ≠ "contiguous".
The three buffers behave differently:

| Buffer | Per-vector contiguous? | Move/swap cost |
|---|---|---|
| `ids_` (`PID`) | ✅ yes | trivial assignment |
| `ex_data` (extra bits) | ✅ yes — `ExDataMap::data_bytes * n`, laid out per vector | simple `memcpy` |
| `batch_data` (1-bit FastScan block) | ❌ no | one-batch repack (or nibble RMW) |

The 1-bit block is **transposed by byte-column and nibble-interleaved** by
`pack_codes` (`fastscan.hpp:91-102`): within a 32-vector batch, lane `v` (0–15) and
lane `v+16` are packed into the **low/high nibble of the same bytes**, at a
permuted position (`kPerm0`, `fastscan.hpp:40-42`), one nibble per byte-column. So
a single vector's bits are scattered across the block *and share bytes with another
lane* — you cannot move it with a `memcpy` or whole-byte swap; it requires a
nibble-level read-modify-write.

**Practical shortcut — no re-quantization needed.** The codes already exist; the
simplest correct mutation is to re-run `pack_codes` on just the **affected ≤32-vector
batch** from its unpacked codes (keep the tail batch's unpacked bytes around, or
unpack on demand). That's O(32 · `padded_dim/8`) for one batch and avoids manual
nibble surgery.

## 2. Recommended approach

Two changes, combined:

- **(A) Per-cluster growable storage** — give each `Cluster` ownership of its own
  growable buffers (mirrors FAISS `ArrayInvertedLists`). Growth becomes *local* to
  one cluster (amortized doubling) instead of a global re-layout.
- **(B) Tombstone delete + lazy compaction** — mark deletions in a bitset and skip
  them at scan time; reclaim space on an explicit `compact()`/rebuild. This avoids
  expensive FastScan repacking on every delete.

Keep the existing contiguous layout **only as the on-disk / load format** (and as
the result of `compact()`), so the hot search path and file format stay intact.

### Alternative (lower effort, lower quality)
Append-only **overflow region** per cluster + tombstones, periodic full rebuild:
leave the static packed arrays immutable, send new vectors to a small secondary
per-cluster buffer that search also scans, and rebuild to merge. Lowest risk to
the FastScan fast path, but adds a second scan path and bookkeeping. Good as a
first milestone; (A)+(B) is the cleaner end state.

## 3. Data-structure changes

- **`Cluster`** (`cluster.hpp`): own its data instead of pointing into shared
  buffers. Add `capacity_` and growable storage, e.g.
  ```cpp
  std::vector<char> batch_data_;   // FastScan blocks; size = n_batches * BatchDataMap::data_bytes(padded_dim)
  std::vector<char> ex_data_;      // per-vector ex codes
  std::vector<PID>  ids_;
  size_t num_;
  // helpers: append_one(...), remove_at(offset), n_batches(), last_batch_fill()
  ```
  (Keep alignment: allocate with the existing `memory::align_allocate<64>` or an
  aligned allocator so SIMD loads stay valid.)
- **`IVF`** (`ivf.hpp`): drop the single `batch_data_/ex_data_/ids_` pointers as the
  primary store (or keep them only for `save`/`load`/`compact`). Add:
  - `std::unordered_map<PID, std::pair<uint32_t,uint32_t>> id_to_loc_;` — reverse
    map `PID → (cluster_id, offset)` for delete/update (FAISS `DirectMap`
    analogue).
  - `std::vector<boost/std bitset or std::vector<uint8_t>> deleted_;` per cluster
    (or a global `dynamic_bitset`) for tombstones.
  - a `dirty_`/`num_deleted_` counter to trigger compaction.
  - optional per-cluster mutex for thread-safe concurrent add/delete.

## 4. Add — algorithm

New public API (and bindings): `void add(const float* vec, PID id);`
(or batched `add(const float* vecs, const PID* ids, size_t n)`).

1. **Rotate**: `rotator_->rotate(vec, rotated)` (same as search, `ivf.hpp:421-422`).
2. **Assign cluster**: nearest existing centroid via the initializer with
   `nprobe = 1`: `initer_->centroids_distances(rotated, 1, cand)` (`ivf.hpp:426`);
   `cid = cand[0].id`. Centroids are fixed, so this just routes.
3. **Quantize**: encode the single vector against that cluster's **rotated**
   centroid `initer_->centroid(cid)` (`initializer.hpp:82`) using
   `quant::quantize_split_batch(..., n = 1, …)` (`rabitq.hpp:224`).
4. **Append to the cluster** (the FastScan-aware part — see [§1.1](#11-what-can-and-cant-be-swapped)):
   - `ex_data` / `ids`: plain append (`vector::push_back` / memcpy) — easy.
   - `batch_data`: only the **tail block** is touched. If the last block is partial
     (`num % 32 != 0`) the new vector joins it; if full, start a new 32-wide block.
     Either way, **re-run `pack_codes` on that one tail batch** from its unpacked
     codes (no re-quantization — the new vector's code was already produced in step
     3, and the existing tail vectors' codes are recoverable/kept unpacked).
5. **Update bookkeeping**: `id_to_loc_[id] = {cid, offset}`, `cluster.num_++`,
   `num_++`.

Notes: reject/replace duplicate `id` (check `id_to_loc_`). For batched adds, group
by assigned cluster and repack each tail block once.

## 5. Delete — algorithm

New public API: `void remove(PID id);` (batched `remove(const PID* ids, size_t n)`).

Recommended **tombstone** path (cheap, no repacking):
1. Look up `(cid, offset)` in `id_to_loc_` (O(1)); if absent, no-op.
2. Set `deleted_[cid][offset] = 1`; `num_deleted_++`; erase from `id_to_loc_`.
3. Search skips tombstoned slots (see §6).
4. When `num_deleted_` exceeds a threshold, `compact()` rebuilds the affected
   clusters (or the whole index) dropping tombstones and repacking FastScan blocks.

Eager alternative (swap-with-last, like FAISS): move the cluster's last vector into
the deleted slot and shrink. `ids`/`ex_data` move with a simple `memcpy`, but the
1-bit block move spans **two** FastScan batches (source tail + destination), each
needing a one-batch repack (or nibble RMW), plus an `id_to_loc_` update for the
moved vector — more code, more per-delete cost. Prefer tombstones + batch
compaction (which repacks each batch once for many deletions).

## 6. Search changes

`scan_one_batch` (`ivf.hpp:494`) must skip tombstoned ids before inserting into the
KNN buffer — add a check next to each `knns.insert(id, …)` (`ivf.hpp:524`, `539`):

```cpp
if (!is_deleted(cid, base_offset + i)) knns.insert(id, ex_dist);
```

This needs the cluster id / base offset passed into `search_cluster` /
`scan_one_batch` (currently they only get raw pointers). Tombstoned vectors still
cost distance computation until compaction — that's the tombstone trade-off.

## 7. Cluster balance & known limitations

- **Fixed centroids ⇒ drift.** New vectors route to the nearest *existing*
  centroid; deletes shrink lists. Over time clusters become uneven (one large, some
  empty), degrading the time/accuracy trade-off. There is no online re-clustering —
  refresh by rebuilding from new centroids. (Same property as FAISS IVF.)
- **No memory reclaimed without `compact()`** — tombstones keep slots until rebuild
  (and `std::vector` shrink keeps capacity anyway).
- **FastScan repacking** is the main correctness risk; isolate it in
  `Cluster::append_one` / `compact()` and unit-test against a from-scratch
  `construct()` of the same final set.
- **Thread safety**: `construct` is OpenMP-parallel; concurrent `add`/`remove` need
  per-cluster locking (or document single-writer).

## 8. Persistence & bindings

- **save/load** (`ivf.hpp:309`, `353`): if clusters become growable, serialize each
  cluster's `num` (already done) and write contiguous compacted blocks — i.e.
  `save()` implicitly compacts. On `load`, sizing from cluster sizes still works.
  Bump a format version if the on-disk layout changes; otherwise keep it identical
  by compacting on save.
- **bindings** (`python_bindings/ivf_bindings.cpp`): add `.def("add", …)` and
  `.def("remove", …)` (+ optional `.def("compact", …)`), plus the Python wrapper in
  `python/ivf.py`. Mirror the existing `build`/`search` argument-handling style
  (`ivf_bindings.cpp:155-174`).

## 9. Implementation checklist

- [ ] `cluster.hpp`: `Cluster` owns growable, 64B-aligned `batch_data`/`ex_data`/
      `ids` + `capacity`; add `append_one`, `remove_at`/tombstone, `n_batches`.
- [ ] `ivf.hpp`: add `id_to_loc_` reverse map, per-cluster `deleted_` bitsets,
      `num_deleted_`.
- [ ] `ivf.hpp`: `add(vec, id)` — rotate → assign (nprobe=1) → quantize n=1 →
      append (repack tail block) → update maps.
- [ ] `ivf.hpp`: `remove(id)` — tombstone via `id_to_loc_`.
- [ ] `ivf.hpp`: `compact()` — rebuild packed blocks dropping tombstones.
- [ ] `ivf.hpp`: thread `cid`/offset into `search_cluster`/`scan_one_batch`; skip
      tombstones at `knns.insert`.
- [ ] `ivf.hpp`: `save`/`load` updated (compact-on-save keeps the format stable).
- [ ] `python_bindings/ivf_bindings.cpp` + `python/ivf.py`: expose `add`,
      `remove`, `compact`.
- [ ] tests under `tests/`: add/remove then compare recall & results vs a fresh
      `construct()` of the same final vector set; FastScan tail-block repack
      correctness; delete-then-search excludes tombstoned ids.

## Source references (current code)

| What | Location |
|---|---|
| `IVF` class & contiguous buffers | `include/rabitqlib/index/ivf/ivf.hpp:29-34` |
| `construct` | `include/rabitqlib/index/ivf/ivf.hpp:170-214` |
| `allocate_memory` (single allocations) | `include/rabitqlib/index/ivf/ivf.hpp:216-231` |
| `init_clusters` (offset packing) | `include/rabitqlib/index/ivf/ivf.hpp:236-258` |
| `quantize_cluster` (FastScan blocks) | `include/rabitqlib/index/ivf/ivf.hpp:260-307` |
| `search` / `search_cluster` / `scan_one_batch` | `include/rabitqlib/index/ivf/ivf.hpp:402-544` |
| `Cluster` (raw pointers) | `include/rabitqlib/index/ivf/cluster.hpp` |
| `Initializer` (`centroid`, `centroids_distances`) | `include/rabitqlib/index/ivf/initializer.hpp:74-88` |
| `kBatchSize = 32` | `include/rabitqlib/fastscan/fastscan.hpp:19` |
| `pack_codes` (transpose + nibble-interleave) | `include/rabitqlib/fastscan/fastscan.hpp:69-106` |
| `kPerm0` (lane permutation) | `include/rabitqlib/fastscan/fastscan.hpp:40-42` |
| `quantize_split_batch` | `include/rabitqlib/quantization/rabitq.hpp:224` |
| save / load | `include/rabitqlib/index/ivf/ivf.hpp:309-400` |
| bindings | `python_bindings/ivf_bindings.cpp:59-174` |
