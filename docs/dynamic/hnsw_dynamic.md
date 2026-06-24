# Dynamic HNSW: adding & deleting after build

This document describes what it takes to make `rabitqlib::hnsw::HierarchicalNSW`
(`include/rabitqlib/index/hnsw/hnsw.hpp`) support **add** and **delete** after the
initial `construct()`. Unlike IVF, the HNSW storage is already
per-element/contiguous and an internal `add_point` already exists — but two
HNSW-specific issues (exact-distance graph building and positional graph ids) shape
the design. For **delete** we use **tombstones** (FAISS HNSW has no delete; a
tombstone + keep-in-graph approach, as in hnswlib, is the standard way).

## 1. How the index works today (and why it's static)

`construct(cluster_num, centroids, data_num, data, cluster_ids, …)`
(`hnsw.hpp:587`) sets `rawDataPtr_ = data` (`hnsw.hpp:616`) and builds the graph by
calling `add_point(idx, cluster_id, config)` for every vector in parallel
(`hnsw.hpp:618-623`). So insertion logic already exists.

Storage (`hnsw.hpp:115-160`):
- `data_level0_memory_` — one contiguous array of `max_elements_` fixed-size slots
  (`size_data_per_element_`). Per element the layout is
  **`[#edges + edges][cluster_id][external_id][BinData (1-bit)][ExData]`**
  (`hnsw.hpp:145`). Crucially, each vector's 1-bit code here is **per-element and
  contiguous** (single-query estimation, not the IVF FastScan transpose).
- `linkLists_` — per-element upper-level edges (levels > 0).
- `element_levels_`, `enterpoint_node_`, `maxlevel_`.
- `label_lookup_` — `external label → internal id` (`hnsw.hpp:157`); internal id is
  the slot index = `cur_element_count_++` at insert (`hnsw.hpp:646-647`).

### The blockers

| Blocker | Where | Why it blocks add/delete |
|---|---|---|
| **Fixed capacity, no resize** | `add_point` throws at `cur_element_count_ >= max_elements_` | `hnsw.hpp:642-644` | Can't grow past the size chosen at init. Need a `resizeIndex`. |
| **Graph build uses *exact* distances via a borrowed raw pointer** | `get_data_dist` / `search_base_layer` read `rawDataPtr_ + label*dim_` | `hnsw.hpp:322`, `782-804`; set only in `construct` (`:616`) | Raw vectors are **not stored** in the index (only quantized codes are). After build/load `rawDataPtr_` is dangling/null, so edge selection for a *new* node has no exact distances to work with. See [§1.1](#11-the-rawdataptr-problem). |
| **No delete / tombstone** | — | grep: no `deleted`/`markDelete` | Removal unsupported entirely. |
| **Internal ids are positional in the graph** | edges store internal ids; no reverse (incoming-edge) index | Compacting/relocating a node would require rewriting every *incoming* edge — see [§1.2](#12-why-tombstones-not-compaction). |
| **`add_point` keys off `rawDataPtr_` + a label index, not a vector** | `hnsw.hpp:626-686` | It quantizes `rawDataPtr_ + label*dim_` and takes a caller-supplied `cluster_id`; there is no "add this vector" entry point. |

### 1.1 The `rawDataPtr_` problem

The build deliberately selects neighbors with **non-quantized** distances
(`construct`: *"Build edges with non-quantized vectors"*, `hnsw.hpp:617`) via
`get_data_dist` (`hnsw.hpp:322`) and `search_base_layer` (`hnsw.hpp:782-804`), all
reading `rawDataPtr_`. But the index only persists **quantized** codes — query-time
`searchBaseLayerST_AdaptiveRerankOpt` (`hnsw.hpp:306`) uses the RaBitQ estimator and
never touches `rawDataPtr_`. So a dynamic `add` after build/load has no raw vectors
for exact edge selection.

Two ways out:
- **(recommended) Build edges for new nodes with the quantized estimator** — reuse
  the same estimation path the query search already uses, so no raw vectors are
  needed. Slightly lower graph quality than the exact-distance bulk build, but
  self-contained.
- **(costly) Retain raw vectors** in the index to keep exact-distance insertion —
  defeats the compression goal; only worth it if insertion-time recall is critical.

### 1.2 Why tombstones, not compaction

Each element slot is fixed-size and contiguous, so *moving* a node's bytes is a
simple `memcpy` (unlike IVF's FastScan block). The hard part is the **graph**:
edges everywhere store the node's **internal id**, and there is no reverse
(incoming-edge) index. Relocating or removing a node would require scanning the
whole graph to rewrite incoming edges, plus possibly repairing levels and the
enterpoint. That's exactly the work FAISS declines to do. **Tombstones avoid it**:
mark the node deleted, keep it in the graph as a *routing* node, and just exclude it
from results.

## 2. Recommended approach

- **Add**: expose insertion (an `add_point` overload that takes a vector), assign it
  to the nearest centroid for quantization, select neighbors with the **estimator**
  (§1.1), and **grow capacity** via a `resizeIndex` when full.
- **Delete**: **tombstone** — a `deleted_` bitset, filtered at search, with nodes
  kept in the graph for navigation. Optional slot **reuse** on later adds, and a
  periodic **rebuild** to shed tombstones and refresh graph quality.

## 3. Data-structure changes

- **Capacity / resize** (`hnsw.hpp`): add `resizeIndex(new_max)` (the hnswlib
  operation) — `realloc` `data_level0_memory_` to `new_max * size_data_per_element_`,
  and resize `element_levels_`, `link_list_locks_`, `linkLists_`, and the
  `visited_list_pool_`. Per-element fixed size makes this a straightforward
  contiguous realloc + copy.
- **Tombstones**: `std::vector<uint8_t> deleted_;` (indexed by internal id) +
  `std::atomic<size_t> num_deleted_;`. (hnswlib instead steals a reserved bit in the
  level-0 link count; a separate vector is simpler.)
- **Free list (optional)**: `std::vector<PID> deleted_slots_;` to reuse tombstoned
  internal ids on add (`allow_replace_deleted` in hnswlib).
- **Insertion entry point**: an `add_point` overload taking `const float* vec`
  (rotate + quantize the passed vector) instead of indexing `rawDataPtr_`.
- **Centroid assignment**: nearest of the `num_cluster_` rotated centroids in
  `centroids_memory_` (a small scan, or a helper) to pick the RaBitQ residual
  reference for quantization.

## 4. Add — algorithm

New public API (and bindings): `void add(const float* vec, PID label);`

1. **Capacity**: if `cur_element_count_ == max_elements_`, `resizeIndex(growth)`
   (or pop a `deleted_slots_` entry to reuse).
2. **Assign centroid**: nearest of `centroids_memory_` (rotated) → `cluster_id`
   (used only as the RaBitQ residual reference).
3. **Quantize**: rotate `vec`, `quant::quantize_split_single(rotated, centroid,
   …, bindata, exdata, …)` into the new slot — same call `add_point` already uses
   (`hnsw.hpp:677`), but fed the passed vector instead of `rawDataPtr_`.
4. **Pick level / link**: assign a random level and connect neighbors using the
   **estimator** (§1.1) rather than exact `rawDataPtr_` distances; update the
   enterpoint if the new level is higher (existing logic in `add_point`,
   `hnsw.hpp:651-735`).
5. **Bookkeeping**: `label_lookup_[label] = cur_c`; `deleted_[cur_c] = 0`;
   `cur_element_count_++`. Duplicate labels already throw (`hnsw.hpp:635-640`) —
   decide replace vs reject.

Concurrency: `add_point` already takes label/link/global locks
(`hnsw.hpp:629,651,658`); `resizeIndex` must hold `global_` and block concurrent
inserts.

## 5. Delete — algorithm (tombstone)

New public API: `void mark_delete(PID label);` (and `unmark_delete`).

1. `internal_id = label_lookup_[label]` (under `label_lookup_lock_`).
2. `deleted_[internal_id] = 1`; `num_deleted_++`; optionally push to
   `deleted_slots_`.
3. **Do not unlink** — the node stays in the graph as a routing hub so the small
   world stays navigable.
4. Search excludes it (see §6). If the **enterpoint** is deleted it still routes
   fine; only drop it as enterpoint on a rebuild.
5. When `num_deleted_` is large, **rebuild** (or reuse slots on add) to reclaim
   space and restore graph quality.

**Slot reuse is not a plain overwrite.** If a later `add` reuses a tombstoned
slot, the node must be **re-linked into the graph**, not just have its bytes
replaced — otherwise it keeps the deleted node's old (now-wrong) edges. hnswlib does
this in `updatePoint` (re-runs neighbor selection for the reused id); see the
[reference appendix](#appendix-how-hnswlib-implements-this).

## 6. Search changes

Query search is `search_knn` → `searchBaseLayerST_AdaptiveRerankOpt`
(`hnsw.hpp:304-306`). Add a tombstone check **only at result collection**, not at
traversal: keep visiting a deleted node's neighbors (routing), but don't admit the
deleted node itself into the top-k heap. Concretely, guard the
`emplace`/`insert`-into-results sites with `if (!deleted_[candidate_id]) …`.
Tombstoned nodes still cost distance computations until rebuild — the tombstone
trade-off. (Optionally fetch a few extra candidates to offset deleted hits.)

## 7. Quality & known limitations

- **Graph drift**: incremental adds (especially with estimator-based neighbor
  selection) and accumulated tombstones gradually lower recall and bloat traversal.
  Rebuild periodically.
- **Quantization-centroid drift**: new vectors quantize against the nearest
  *existing* centroid (residual reference). Vectors far from all centroids get
  poorer codes over time — centroids are fixed until rebuild. (Routing is via the
  graph, not centroids, so this affects accuracy, not list balance.)
- **Tombstones don't free memory**: slots persist until reused or rebuilt.
- **No graph repair on delete**: deleted nodes remain as routing hubs; correctness
  is preserved, efficiency degrades with the deleted fraction.
- **Thread-safety**: reuse the existing per-label/link/global locks; guard
  `deleted_` and `resizeIndex`.

## 8. Persistence & bindings

- **save/load** (`hnsw.hpp:431`, `486`): both already serialize
  `cur_element_count_ * size_data_per_element_` (`:470`, `:534`) and per-element
  link lists (`:473`, `:552`). Add the `deleted_` bitset (and free list) to the
  format — or **compact on save** (drop tombstones, rebuild) to keep the format
  unchanged. Bump a version if the layout changes. **Tip:** if you store the
  tombstone as a bit inside the level-0 link header (the hnswlib trick, see
  appendix), it persists **for free** with the existing `data_level0_memory_`
  blob — no format change needed.
- **bindings** (`python_bindings/hnsw_bindings.cpp`): add `.def("add", …)`,
  `.def("mark_delete", …)`, optionally `.def("resize", …)` / `.def("rebuild", …)`,
  mirroring the existing `build`/`search`/`save`/`load` style (`hnsw_bindings.cpp:172-192`).

## 9. Implementation checklist

- [ ] `hnsw.hpp`: `resizeIndex(new_max)` — realloc `data_level0_memory_`, resize
      `element_levels_`/`link_list_locks_`/`linkLists_`/`visited_list_pool_`.
- [ ] `hnsw.hpp`: `add_point` overload taking `const float* vec`; nearest-centroid
      assignment; **estimator-based** neighbor selection (no `rawDataPtr_`).
- [ ] `hnsw.hpp`: public `add(vec, label)` (capacity check + slot reuse).
- [ ] `hnsw.hpp`: `deleted_` bitset + `num_deleted_`; `mark_delete`/`unmark_delete`.
- [ ] `hnsw.hpp`: tombstone filter at result-collection in
      `searchBaseLayerST_AdaptiveRerankOpt`.
- [ ] `hnsw.hpp`: optional `rebuild()`/`compact()` and slot free list.
- [ ] `hnsw.hpp`: `save`/`load` persist tombstones (or compact-on-save).
- [ ] `python_bindings/hnsw_bindings.cpp` + python wrapper: expose `add`,
      `mark_delete`, `resize`/`rebuild`.
- [ ] tests under `tests/`: add-after-build recall vs fresh `construct`; deleted ids
      never returned but still route; capacity growth; save/load round-trip with
      tombstones.

## Source references (current code)

| What | Location |
|---|---|
| `HierarchicalNSW` class | `include/rabitqlib/index/hnsw/hnsw.hpp:38` |
| element layout comment | `include/rabitqlib/index/hnsw/hnsw.hpp:145` |
| storage members (`data_level0_memory_`, `linkLists_`, `label_lookup_`) | `hnsw.hpp:146-157` |
| `construct` (parallel `add_point`, sets `rawDataPtr_`) | `hnsw.hpp:587-624` |
| `add_point` (capacity throw, dup throw, quantize) | `hnsw.hpp:626-686` |
| capacity limit throw | `hnsw.hpp:642-644` |
| exact-distance build (`get_data_dist`) | `hnsw.hpp:322` |
| `search_base_layer` uses `rawDataPtr_` | `hnsw.hpp:782-804` |
| query search (estimator; no `rawDataPtr_`) | `hnsw.hpp:304-306` |
| `quantize_split_single` (per-vector code) | `hnsw.hpp:677` |
| free per-element link lists | `hnsw.hpp:185-201` |
| save / load | `hnsw.hpp:431-585` |
| bindings | `python_bindings/hnsw_bindings.cpp:72-192` |

## Appendix: how hnswlib implements this

khorgosh's `HierarchicalNSW` is derived from
[hnswlib](https://github.com/nmslib/hnswlib), which already supports
add / delete / resize. The mechanics below (from a local checkout,
`hnswlib/hnswlib/hnswalg.h`) map almost 1:1 onto the changes proposed above.

### Tombstone stored *in* the level-0 link header (no separate bitset)
- `static const unsigned char DELETE_MARK = 0x01;` (`hnswalg.h:21`),
  `std::atomic<size_t> num_deleted_` (`hnswalg.h:27`).
- The mark lives in a byte of the **level-0 link-list header**:
  `markDeletedInternal` does `ll_cur = (unsigned char*)get_linklist0(id) + 2;
  *ll_cur |= DELETE_MARK` (`hnswalg.h:872-885`); `unmarkDeletedInternal`
  (`:914-927`) clears it; `isMarkedDeleted` reads it (`:933-936`).
  Cost: `maxM0_` is limited to the lower 16 bits — fine in practice. Bonus: the
  flag is serialized **for free** inside `data_level0_memory_`.
- `markDelete(label)` / `unmarkDelete(label)` resolve via `label_lookup_` then call
  the internal versions (`hnswalg.h:852-907`).

### Search traverses deleted nodes but excludes them from results
- The entry point is skipped as a *result* if deleted but still used to route
  (`hnswalg.h:235`, `326`); candidates are admitted only if
  `!isMarkedDeleted(candidate_id)` (and any user filter) — `hnswalg.h:291`, `407` —
  while traversal continues through them. This is exactly the "filter at result
  collection, keep traversing" rule in §6.

### Resize
- `resizeIndex(new_max)` (`hnswalg.h:633-656`): `realloc` `data_level0_memory_` and
  `linkLists_`, `resize` `element_levels_` and `link_list_locks_`, and reset the
  `visited_list_pool_`. Matches §3.

### Slot reuse on add re-links the graph (`updatePoint`)
- `addPoint(data, label, replace_deleted=true)` (`hnswalg.h:953-991`): if
  `deleted_elements` has a vacant slot, it reassigns the label, `unmarkDeletedInternal`,
  and calls `updatePoint` — which **overwrites the vector AND re-runs neighbor
  selection** to reconnect the reused id (`hnswalg.h:994+`). So reuse ≠ overwrite;
  the graph is repaired. Otherwise it falls back to a normal append.
- `allow_replace_deleted_` + the `deleted_elements` set gate this (`hnswalg.h:68-71`).

### Note
hnswlib never truly removes or compacts a node from the graph — deleted nodes
remain as routing hubs until a slot is reused or the index is rebuilt. That's the
same trade-off this design adopts.

| hnswlib reference | Location |
|---|---|
| `DELETE_MARK`, `num_deleted_`, `deleted_elements` | `hnswlib/hnswlib/hnswalg.h:21-71` |
| `markDelete` / `unmarkDelete` | `hnswlib/hnswlib/hnswalg.h:852-907` |
| `markDeletedInternal` / `isMarkedDeleted` (bit in link0 header) | `hnswlib/hnswlib/hnswalg.h:872-936` |
| search excludes deleted from results | `hnswlib/hnswlib/hnswalg.h:235,291,326,407` |
| `resizeIndex` | `hnswlib/hnswlib/hnswalg.h:633-656` |
| `addPoint(replace_deleted)` + `updatePoint` | `hnswlib/hnswlib/hnswalg.h:953-994` |
