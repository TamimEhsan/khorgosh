# khorgosh HNSW Search — Optimization Opportunities (vs SVS Vamana)

Focused on the **search path** of `rabitqlib::hnsw::HierarchicalNSW`. Reference is SVS's `greedy_search` over Vamana — SVS doesn't ship HNSW, but Vamana's inner loop is structurally identical to HNSW's base-layer search (greedy beam-search over a graph with prefetching and a visited filter), so the techniques transfer directly.

(IVF and QG comparisons are out of scope here — to be revisited later.)

## Search call graph (where we'll focus)

User entry: `HierarchicalNSW::search()` (`include/rabitqlib/index/hnsw/hnsw.hpp:1065`)

```
search(queries, query_num, TOPK, efSearch, thread_num)         // line 1065
  ├── parallel_for over queries                                // line 1070
  └── per-query lambda
        ├── rotator_->rotate(...)                              // line 1076
        └── search_knn(rotated_query, TOPK)                    // line 1077
              ├── q_to_centroids[]   (allocated per query)     // line 1101
              ├── greedy walk on upper layers (level > 0)      // line 1133
              └── searchBaseLayerST_AdaptiveRerankOpt(...)     // line 1162
                    └── tight inner loop over neighbors        // line 1220
```

## Findings, ranked by expected impact

### 1. Per-query heap allocations inside the parallel loop  ⭐⭐⭐

#### The problem

`search()` and `search_knn()` allocate four heap structures on **every single query**:

```cpp
// hnsw.hpp:1075-1076 (inside the parallel_for lambda — runs once per query)
std::vector<float> rotated_query(padded_dim_);
this->rotator_->rotate(queries + (idx * dim_), rotated_query.data());

// hnsw.hpp:1101 (search_knn — runs once per query)
std::vector<float> q_to_centroids(num_cluster_);

// hnsw.hpp:1195-1206 (searchBaseLayerST_AdaptiveRerankOpt — runs once per query)
buffer::SearchBuffer<float> candidate_set(ef);   // allocates AlignedAllocator vector of (ef+1) entries

// hnsw.hpp:1192 — pulled from a pool but the pool itself is mutex-guarded (see #7)
HashBasedBooleanSet* vl = visited_list_pool_->get_free_vislist();
```

Sizing for a representative workload (d=768, padded_dim=768, num_cluster=1024, ef=100):

| Structure | Per-query bytes | Notes |
| --- | --- | --- |
| `rotated_query` | 768 × 4 = **3,072 B** | `std::vector<float>` default allocator |
| `q_to_centroids` | 1024 × 4 = **4,096 B** (L2) or 8,192 B (IP, line 1112 doubles it) | `std::vector<float>` default allocator |
| `SearchBuffer.data_` | (ef+1) × sizeof(`AnnCandidate<float>`) ≈ **1,212 B** | uses `memory::AlignedAllocator` (good — but the call still hits the allocator) |
| `BoundedKNN.queue_` | grows up to TOPK × sizeof(Candidate) ≈ **160 B** for top-10 | `std::vector` reallocates as it grows |
| Visited list `clear()` | O(table_size) = **128 KB** of writes for a 1M-element index | not an allocation, but a hot memory write per query |

That's **~9 KB of fresh allocation per query**, plus 128 KB of zero-fill on the visited table. Under the `ivf::parallel_for` work-stealing pattern (`include/rabitqlib/index/ivf/initializer.hpp:36-60`) every worker thread races on the global allocator — the classic glibc malloc arena contention pattern. At thousands of QPS this becomes measurable.

The lambda already receives a `threadId` parameter and silently throws it away:

```cpp
// hnsw.hpp:1074
[&](size_t idx, size_t /*threadId*/) {  // tid available, unused
```

#### What SVS does

SVS centralizes per-thread state in a `SearchScratchspace<Buffer, Scratch>` struct (`include/svs/index/vamana/index.h:196-232`):

```cpp
template <typename Buffer, typename Scratch> struct SearchScratchspace {
    Buffer buffer;                                 // the search buffer (beam queue + visited filter)
    Scratch scratch;                               // dataset-specific scratch (e.g. dequant buffer for LVQ)
    GreedySearchPrefetchParameters prefetch_parameters;

    SearchScratchspace& apply(const VamanaSearchParameters& p) {
        buffer.change_maxsize(p.buffer_config_);             // resize in place — no realloc if capacity OK
        buffer.configure_visited_set(p.search_buffer_visited_set_);
        prefetch_parameters = {p.prefetch_lookahead_, p.prefetch_step_};
        return *this;
    }
};
```

`VamanaIndex::search()` (`include/svs/index/vamana/index.h:556-603`) constructs the scratchspace **once per thread per call**, *inside* the `parallel_for` lambda — but everything inside is reused across the query batch that thread owns:

```cpp
// SVS index.h:563-602 (abridged)
threads::parallel_for(
    threadpool_,
    threads::StaticPartition{queries.size()},
    [&](const auto is, uint64_t /*tid*/) {
        auto search_buffer = search_buffer_type{...};                    // once per thread
        auto prefetch_parameters = GreedySearchPrefetchParameters{...};  // once per thread
        auto scratch = extensions::per_thread_batch_search_setup(data_, distance_);  // once per thread

        extensions::per_thread_batch_search(
            data_, search_buffer, scratch,
            queries, result, threads::UnitRange{is},
            greedy_search_closure(prefetch_parameters, cancel),
            *this, cancel
        );
    }
);
```

The two important moves SVS makes:

1. **`StaticPartition`** (`include/svs/lib/threads/types.h:341`) gives each thread a contiguous range of queries up front. So the lambda body runs **once per thread**, not once per query. All scratch is created once and the inner `per_thread_batch_search` loops over the assigned range, mutating that scratch in place.
2. The **scratchspace can be pre-allocated by the caller** and passed into a lower-level `search(query, scratch, cancel)` overload (`include/svs/index/vamana/index.h:507-516`). For external-threading scenarios (thread-per-request servers, async runtimes) the scratch lifetime is owned by the caller. There's even a public `scratchspace()` factory (`index.h:447-462`) that builds one with the index's current default parameters.

For dataset-specific scratch (e.g., LVQ needs a small dequantization buffer per thread), the size is decided by the dataset implementation through `extensions::per_thread_batch_search_setup` (`index.h:587`). That keeps the threading layer agnostic to the kernel's memory needs.

SVS also uses a `CacheAlignedAllocator` (`include/svs/lib/threads/threadlocal.h:62-91`) for per-thread containers so adjacent threads' scratch never shares a cache line — kills false sharing on the hot writes inside the search buffer.

#### Concrete implementation for khorgosh

Two changes, one architectural and one mechanical.

**Step A — define a scratch struct with everything the per-query path needs:**

```cpp
// new file: include/rabitqlib/index/hnsw/search_scratch.hpp
#pragma once
#include <vector>
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/hashset.hpp"   // (or the new direct-mapped filter from finding #2)
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib::hnsw {

// Cache-line aligned so adjacent threads' scratch can't share a line.
struct alignas(64) SearchScratch {
    std::vector<float, memory::AlignedAllocator<float>> rotated_query;
    std::vector<float, memory::AlignedAllocator<float>> q_to_centroids;
    buffer::SearchBuffer<float>                         candidate_set;
    HashBasedBooleanSet                                 visited;     // owned, not pooled
    std::vector<HierarchicalNSW::Candidate>             knn_buf;     // for BoundedKNN

    void prepare(size_t padded_dim,
                 size_t num_cluster_floats,    // num_cluster_  * (metric==IP ? 2 : 1)
                 size_t ef,
                 size_t max_elements,
                 size_t topk) {
        if (rotated_query.size()   != padded_dim)         rotated_query.assign(padded_dim, 0.f);
        if (q_to_centroids.size()  != num_cluster_floats) q_to_centroids.assign(num_cluster_floats, 0.f);
        if (candidate_set.capacity_unsafe() != ef)        candidate_set.resize(ef);
        if (visited.capacity() != /*matching size derivation*/) visited.initialize(/*…*/);
        knn_buf.reserve(topk);
        visited.clear();          // O(table) zero-fill — unavoidable, but no realloc
        candidate_set.clear();
    }
};

}  // namespace rabitqlib::hnsw
```

**Step B — give `HierarchicalNSW` a vector of scratches sized to `thread_num` and reuse it across the parallel batch:**

```cpp
// include/rabitqlib/index/hnsw/hnsw.hpp — add member
std::vector<SearchScratch> search_scratches_;

// rewritten search(): one scratch per thread, prepared once, reused for the thread's slice of queries
inline std::vector<std::vector<std::pair<float, PID>>> HierarchicalNSW::search(
    const float* queries, size_t query_num, size_t TOPK, size_t efSearch, size_t thread_num
) {
    if (thread_num == 0) thread_num = std::thread::hardware_concurrency();
    if (search_scratches_.size() < thread_num) search_scratches_.resize(thread_num);

    std::vector<std::vector<std::pair<float, PID>>> results(query_num);
    const size_t qcent_size = (metric_type_ == METRIC_IP ? 2 : 1) * num_cluster_;

    rabitqlib::ivf::parallel_for(
        0, query_num, thread_num,
        [&](size_t idx, size_t tid) {
            auto& s = search_scratches_[tid];
            s.prepare(padded_dim_, qcent_size, std::max(efSearch, TOPK), max_elements_, TOPK);

            rotator_->rotate(queries + (idx * dim_), s.rotated_query.data());
            search_knn_into(s.rotated_query.data(), TOPK, efSearch, s, results[idx]);
        }
    );
    return results;
}
```

`search_knn_into` is `search_knn` rewritten to take the scratch by reference — `q_to_centroids` reads from `s.q_to_centroids`, `candidate_set` is `s.candidate_set`, and `searchBaseLayerST_AdaptiveRerankOpt` uses `s.visited` instead of pulling from `visited_list_pool_`. The visited pool can be deleted entirely (closes finding #7).

Roughly **+30 lines added, ~10 lines deleted**, no algorithm changes, fully measurable with the existing `sample/hnsw_rabitq_querying.cpp` rig.

**Step C (optional but free) — expose the lower-level entry point:**

If you want to support callers doing their own threading (e.g. a serving loop with a thread per HTTP request), add a `search_one(query, scratch, out)` public method that takes a caller-owned `SearchScratch&` and skips the parallel_for entirely. This is exactly what SVS's `index.h:506-516` `search(query, scratch)` overload provides.

#### Why this is worth doing first

- Touches only the search path, no recall change, no new dependency.
- Removes contention on the global allocator and the `VisitedListPool` mutex in one shot.
- Sets up the data structure that findings #2 (direct-mapped visited filter) and #5 (lazy `q_to_centroids`) want to live in.
- Estimated win: **5–15% QPS** at moderate thread counts, bigger at high concurrency where allocator contention dominates.

### 2. Visited set is mostly the slow path  ⭐⭐⭐

`HashBasedBooleanSet` (`utils/hashset.hpp`) sizes its direct-mapped table at `2^((bit_size+4)/2 + 3)`. For 1M elements that's `bit_size=20`, table size = `2^15 = 32 768`. With ef≈100 and per-query ~5–20× expansion, the table sees thousands of inserts per query → **collisions are the common case**, and every collision falls back into `std::unordered_set` (line 86, 97):

```cpp
bool get(PID data_id) const {
    PID val = this->table_[hash1(data_id)];
    if (val == data_id) return true;
    return (val != kPidMax && stl_hash_.find(data_id) != stl_hash_.end());  // hash-table lookup
}
```

`std::unordered_set::find` is a node-based lookup (cache-unfriendly, allocator hits), and `clear()` walks both structures.

SVS's `VisitedFilter` (`include/svs/index/vamana/filter.h`) takes the opposite stance: a **fixed-size direct-mapped filter with no fallback**. Collisions just produce false negatives — the algorithm tolerates them because rediscovering a node costs one distance computation, which is cheap compared to dragging in a hash-table miss. With 16-bit values it stores only the upper bits of the ID, so a 128 KB filter (2^16 slots × 2 B) covers every 32-bit ID exactly. `reset()` is a single `std::fill` over a contiguous array.

**Fix:** add a small direct-mapped filter (say 2^17 = 128K entries × 4 B = 512 KB per thread, or 2^16 × 2 B = 128 KB) and drop the unordered_set fallback. Sized as a power of two so `index = id & mask` is one AND. Add a `prefetch(id)` method.

Estimated win: **15–30%** on the inner loop for million-scale graphs — visited-set ops dominate when distance is cheap (which is exactly the RaBitQ case).

### 3. Prefetch lookahead is too short  ⭐⭐⭐

The inner loop (`hnsw.hpp:1218-1260`) prefetches **one** neighbor ahead:

```cpp
rabitqlib::memory::mem_prefetch_l1(get_bindata_by_internalid(*(data + 1)), 2);
for (size_t j = 1; j <= size; j++) {
    int candidate_id = *(data + j);
    rabitqlib::memory::mem_prefetch_l1(get_bindata_by_internalid(*(data + j + 1)), 2);
    ...
}
```

DRAM latency is ~80–100 ns; a single distance computation on RaBitQ-binary codes is much faster than that, so the line prefetched for `j+1` won't have arrived by the time we touch it. SVS uses **lookahead=4 with step=2** (`greedy_search.h:44-50`) and skips prefetches for already-visited neighbors:

```cpp
auto prefetcher = lib::make_prefetcher(
    lib::PrefetchParameters{lookahead=4, step=2},
    num_neighbors,
    [&](size_t i) { accessor.prefetch(dataset, neighbors[i]); },
    [&](size_t i) {
        if (search_buffer.visited_set_enabled()) {
            if (i + 1 < num_neighbors) search_buffer.unsafe_prefetch_visited(neighbors[i + 1]);
            return !search_buffer.unsafe_is_visited(neighbors[i]);  // skip prefetch
        }
        return true;
    }
);
```

Two changes worth porting:
- Bump `lookahead` to 4 (parameterize so it can be tuned per machine).
- Predicate-skip: don't prefetch nodes the visited filter says are already seen — half your prefetches are wasted otherwise.
- **Prefetch the visited-filter slot for the next neighbor**, not just its data. This is `search_buffer.unsafe_prefetch_visited(neighbors[i+1])` in SVS. With the current `HashBasedBooleanSet` you'd prefetch `&table_[hash1(next_id)]` — and once #2 above is done, the slot lookup is the inner-loop hot path.

Estimated win: **10–25%** on the base-layer search depending on graph degree and DRAM bandwidth.

### 4. Upper-layer walk has no prefetching at all  ⭐⭐

The upper-layer descent (`hnsw.hpp:1133-1159`) iterates neighbors one at a time and computes a `bin_est` per neighbor with no prefetch:

```cpp
for (int i = 0; i < size; i++) {
    PID cand = datal[i];
    EstimateRecord candest;
    get_bin_est(q_to_centroids, query_wrapper, cand, candest);
    ...
}
```

Upper layers are tiny (log scale) so the absolute time is small — but on multi-million-vector indexes with maxlevel ≥ 4 it adds up across queries. Cheap fix: same prefetch pattern as the base layer (smaller lookahead is fine, 2 is plenty).

Estimated win: **1–3%** end-to-end.

### 5. `q_to_centroids` is fully populated even when only a fraction is needed  ⭐

`search_knn` precomputes the query's distance/IP to **every** centroid (`hnsw.hpp:1103-1126`). For 1024 clusters × 768 dims that's a non-trivial chunk of work per query before the graph walk even starts. This isn't HNSW's fault — it comes from RaBitQ needing the centroid distance for any node it visits. But a search only visits a small subset of nodes (and therefore a small subset of clusters).

Two options:
- **Lazy compute**: a `compute_or_get(cluster_id)` helper backed by an array marked with sentinel "not yet computed". First touch fills it. Saves time when `clusters_visited << num_clusters`. Benchmark first — if visit coverage is high (small graphs, large ef), the bookkeeping cost wins out.
- **Precompute only the K nearest centroids** (e.g. via a smaller IVF probe) and treat the rest as ineligible / fall back. Bigger algorithmic change; only worth it if `num_cluster_` is in the thousands.

Estimated win: situational, **0–10%** depending on `num_cluster_` and `ef`.

### 6. `search()` always calls `set_ef(efSearch)` on the shared instance  ⭐

`hnsw.hpp:1068`: `set_ef(efSearch);` mutates a member variable. With per-query parallelism and concurrent calls to `search()` from different threads (e.g. multiple query batches), this is a data race on `ef_`. In the current sample (`sample/hnsw_rabitq_querying.cpp:76`) it's called once per ef sweep so it doesn't bite, but it's a footgun.

Fix: pass `ef` through the call chain instead of mutating state, or make `ef_` per-call. The fix is mechanical.

### 7. VisitedListPool mutex on every search  ⭐

`utils/visited_pool.hpp:21-39`: every `get_free_vislist()` and `release_vis_list()` takes a global `std::mutex`. With N threads doing K queries each, that's `2 × N × K` lock acquisitions on a single mutex. With #2's per-thread scratch redesign this becomes a non-issue (each thread owns its filter, no pool needed).

If you keep the pool, switch to a lock-free stack (atomic head + Treiber stack) or — simpler — a thread-local cache with the global pool as fallback.

---

---

## Threading-layer improvements (orthogonal to algorithmic wins above)

The findings above are about what runs *inside* the parallel region. The items below are about *how the parallel region itself is set up* — the threading infrastructure. Most of these don't affect a single-query latency benchmark much, but they show up at high QPS, on multi-socket machines, or in long-running serving processes.

### T1. Stop spawning fresh threads per call ⭐⭐

`include/rabitqlib/index/ivf/initializer.hpp:17-69` is the `parallel_for` HNSW search uses:

```cpp
std::vector<std::thread> threads;
threads.reserve(numThreads);
for (size_t thread_id = 0; thread_id < numThreads; ++thread_id) {
    threads.push_back(std::thread([&, thread_id] { ... while (true) { fetch_add; fn(); } ... }));
}
for (auto& thread : threads) thread.join();
```

**Every call to `HierarchicalNSW::search()` spawns N `std::thread`s, runs the work, then joins and destroys them.** On Linux that's an `clone()` syscall + kernel stack + scheduler handshake per thread, per call. For a serving loop dispatching `search()` per RPC, this is pure overhead (single-digit milliseconds at high QPS, or worse if the kernel decides to migrate threads).

**SVS reference**: `include/svs/lib/threads/threadpool.h:147-230` (`NativeThreadPoolBase`) and `include/svs/lib/threads/thread.h:355-766` (`ThreadControlBlock`). The pool spawns workers once at construction and they live until the pool is destroyed. Workers spin on an atomic state for ~1M cycles, then sleep on a `std::condition_variable` if no work arrives. New `parallel_for` calls flip the atomic — no syscall, no thread creation.

**Fix:** introduce a small `ThreadPool` that owns N persistent workers and reuses them across calls. You don't need SVS's full templated abstraction — a simple class with `parallel_for(size_t n, F&& fn)` is enough. The interface mirrors what `initializer.hpp` already exposes, so HNSW (and IVF/QG later) only need a one-line swap.

Sketch:
```cpp
class ThreadPool {
    std::vector<std::thread> workers_;
    std::atomic<size_t> cursor_{0};
    std::atomic<size_t> end_{0};
    std::function<void(size_t, size_t)> fn_;       // shared
    std::atomic<int>  generation_{0};              // workers spin on changes
    std::atomic<int>  done_count_{0};
    // … condition_variable for sleep after spin budget exhausted
};
```

Estimated win: **~2–5% at high QPS**, more if your workload has many short search calls.

### T2. Thread pinning / CPU affinity ⭐

khorgosh sets none — threads can be migrated by the kernel scheduler at any time, which trashes warm L1/L2 caches (visited table, search buffer, dataset rows it just read).

OpenMP gives you `OMP_PROC_BIND=close` / `OMP_PLACES=cores` as environment knobs that work for any OpenMP-parallelized region, but the custom `std::thread` `parallel_for` ignores all of that.

**SVS reference:** affinity is bundled into the NUMA layer (`include/svs/lib/numa.h:313-336`):

```cpp
class NodeBind {
public:
    explicit NodeBind(size_t node) : affinity_{} {
        affinity_.set(node, true);
        numa_bind(affinity_.ptr());
        tls::assigned_node = node;
    }
};
```

Workers in `InterNUMABuilder` (`threadpool.h:129-144`) construct a `NodeBind` inside their thread function so binding happens once at startup and persists.

**Fix for khorgosh** (assuming you adopt T1):
- After `pthread_create` in your worker, call `pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset)` with a single-CPU mask. Round-robin assign worker `i` to CPU `i % nproc` by default.
- Make this a parameter on the pool: `ThreadPool(num_threads, bool pin = false, std::span<int> cpu_ids = {})`. Off by default — enabling affinity in shared environments (containers with CPU quotas, multi-tenant boxes) can hurt.
- macOS has no `pthread_setaffinity_np`; gate this on `#ifdef __linux__`.

Estimated win: **5–15% on bandwidth-bound workloads** (RaBitQ distance is L1-friendly so the win here is mostly visited-set + search-buffer warmth). Non-trivial only on machines with > 1 NUMA node or aggressive scheduling.

### T3. NUMA awareness for billion-scale indexes ⭐⭐ (situational)

Currently the entire `data_level0_memory_` block is one `malloc` (`hnsw.hpp:393`) — it lives on whichever NUMA node first-touched it (almost certainly node 0). A worker thread on node 1 reads every dataset row across the inter-socket link.

For datasets that fit in a single socket's RAM this is invisible. For 100M+ vector indexes on a dual-socket box with separate memory channels, it's the difference between scaling and not scaling.

**SVS reference:**
- `include/svs/lib/numa.h:230-307` — `NumaLocal<T>`: holds one copy per node, accessed via `tls::assigned_node`.
- `include/svs/lib/threads/threadpool.h:129-144` — `InterNUMABuilder` spawns a worker per NUMA node, each binding to its node before constructing thread-local objects.
- `include/svs/lib/numa.h:199-219` — `numa_num_nodes()`, `numa_cpus_on_node()` for sizing.

**Fix for khorgosh** is invasive but well-scoped, gate it behind a build flag:
```cmake
option(KHORGOSH_ENABLE_NUMA "Enable NUMA-aware allocation and thread binding" OFF)
```

When enabled:
1. Allocate `data_level0_memory_` per NUMA node (replicate the dataset N times — costs N× memory but each node touches local memory only). For very large datasets this is impractical → instead use `numa_alloc_interleaved` to round-robin pages across nodes (degrades cleanly without replication).
2. Pin worker `i` to a CPU on node `i % num_nodes`.
3. Use `numa_alloc_local` for per-thread `SearchScratch` allocation.

Only build if you're actually deploying on > 1 NUMA node. Most khorgosh users (research / single-box) won't need it.

### T4. False sharing on per-thread state ⭐

When you build the per-thread scratch vector from finding #1, make sure adjacent `SearchScratch` instances don't share a cache line. `std::vector<SearchScratch>` packs them densely; if `sizeof(SearchScratch)` is, say, 80 bytes, two threads will write to the same 64-byte line and ping-pong it.

Two ways to fix:
- Mark the struct `alignas(64)` (already shown in finding #1's sketch).
- Or use SVS's `CacheAlignedAllocator` (`include/svs/lib/threads/threadlocal.h:62-91`) for the outer vector.

Trivial change, real ~1–3% win at high thread counts.

### T5. The work-stealing pattern is the wrong default for HNSW search ⭐

`initializer.hpp` uses an atomic counter — every worker calls `current.fetch_add(1)` to pick up the next query. That's good when query cost varies wildly (some queries terminate early, some explore the whole graph). For HNSW search at fixed `ef`, the variance per query is low, so static partitioning wins on two counts:
1. No atomic on the hot path.
2. Each thread's slice is contiguous → if you ever want to prefetch the *next query's* `q_to_centroids` source data while finishing the current, contiguous beats random.

**SVS reference:** `include/svs/lib/threads/types.h:341-380` (`StaticPartition`) + `types.h:311-329` (`balance(n, nthreads, tid)`). Each thread computes `[start, stop)` independently with one division.

**Fix:** make the partitioning strategy a parameter on the pool, default to static for HNSW search. Keep dynamic available for build-time work where iteration cost varies. Tiny code change once T1's pool exists.

### T6. Spin-then-sleep workers (only worth it if you do T1)

If you adopt the persistent pool from T1, the simplest correct implementation has workers `cv.wait()` for new work — but that means a syscall per dispatch, defeating much of T1's purpose for serving loops with sub-millisecond search latency.

**SVS reference:** `include/svs/lib/threads/thread.h:130-182` (`detail::spin_while_true`, `spin_until`) + the `ThreadControlBlock` state machine at `thread.h:355-766`.

The pattern: workers spin on an atomic state for ~1M cycles (~300 µs on modern x86) before falling back to `condition_variable::wait`. For HNSW search where `parallel_for` returns in microseconds, the spin window catches every dispatch and the CV is only used during true idle gaps. Tunable spin budget so you don't burn cycles when idle for long stretches.

Worth implementing only if benchmarks show CV wakeup latency dominating. Adds ~50 LOC to the pool.

### Summary table

| Item | Where SVS does it | Estimated khorgosh win | Effort |
| --- | --- | --- | --- |
| T1 — Persistent thread pool | `threadpool.h:147-230` | 2–5% high QPS | Medium (1 new file ~150 LOC) |
| T2 — CPU pinning | `numa.h:313-336` (`NodeBind`) | 5–15% on multi-socket | Small after T1 |
| T3 — NUMA-aware data placement | `numa.h:199-307` (`NumaLocal`, `numa_alloc_*`) | 1.5–2× on bandwidth-bound, dual-socket | Large; gate behind cmake flag |
| T4 — `alignas(64)` on per-thread scratch | `threadlocal.h:62-91` (`CacheAlignedAllocator`) | 1–3% | Trivial; do as part of finding #1 |
| T5 — Static partition for query batches | `types.h:311-380` | Marginal alone; sets up cleaner code | Trivial after T1 |
| T6 — Spin-then-sleep workers | `thread.h:355-766` | Recovers most of T1's overhead in serving loops | Medium |

The natural sequencing: do finding #1 first (it's algorithmic and gives you the per-thread struct), then T1 + T4 + T5 together (one pool refactor). T2 follows naturally on Linux. T3 and T6 are deeper investments — only do them if your deployment shape demands it.

---

## Things khorgosh already does right (don't regress these)

- Single-thread per query at the algorithm level — graph search has tight data dependencies that don't parallelize within a query without significant complexity (intra-query parallelism is rarely a net win below ef ≈ 1000).
- **Bucketed label locks**: `label_op_locks_(kMaxLabelOperationLock = 65536)` (line 337). This is exactly the bucketed-mutex pattern that's missing in the QG builder — HNSW build already does this. Good.
- Aligned allocations in `SearchBuffer` and `HashBasedBooleanSet` via `memory::AlignedAllocator`.
- `searchBaseLayerST_AdaptiveRerankOpt` skips the `link_list_locks_` taken in build-time `search_base_layer` — search-time path is correctly lock-free.
- The two-tier estimate (`get_bin_est` → `get_full_est` only when `low_dist < distk`) at lines 1230–1246 is a well-tuned reranking pattern; don't disturb it.

---

## Suggested implementation order

Algorithmic findings (#) and threading-layer items (T) are interleaved here in a risk-minimizing order:

1. **#1 — Per-thread `SearchScratch`** for `rotated_query`, `q_to_centroids`, `SearchBuffer`, visited set. Includes **T4** (`alignas(64)`). Mechanical, no algorithm change, immediate gain, sets up later items. ~1–2 days.
2. **#3 — Prefetch lookahead = 4** with a parameter knob. One-line change to test, then add the predicate-skip variant. ~1 day.
3. **#6 — Fix the `set_ef` race** by passing `ef` through the call chain. ~1 hour.
4. **T1 + T5 — Persistent thread pool with static partitioning**, swap into `HierarchicalNSW::search`. Single new file (~150 LOC) replacing `ivf::parallel_for` for the search path. ~2 days.
5. **T2 — Optional CPU pinning** on the new pool, gated on `__linux__` and a constructor flag. ~half day after T1.
6. **#2 — Replace `HashBasedBooleanSet`** with a fixed-size direct-mapped filter. Bigger change, needs recall benchmarking — false negatives should be invisible but verify. ~2–3 days including bench.
7. Add **prefetch on the visited filter slot** for next neighbor (depends on #2). ~half day.
8. **#4 — Upper-layer prefetch** and **#7 — `VisitedListPool` removal** (#7 disappears for free if #1 is done; remove the dead code). Cleanup pass.
9. **#5 — Lazy `q_to_centroids`** — only if benchmarks show it's worthwhile.
10. **T6 — Spin-then-sleep** worker pattern, **T3 — NUMA awareness** — only if profiling justifies it. T3 needs a deployment story (multi-socket box).

Each step can be measured independently with `sample/hnsw_rabitq_querying.cpp` (already sweeps `ef`, reports QPS and recall — good rig for A/B). Recommend running it under `perf stat -e cache-misses,cache-references,context-switches,cpu-migrations,LLC-load-misses` for the threading items, since their wins won't show up in QPS averages alone.

---

## File / line reference

khorgosh:
- `include/rabitqlib/index/hnsw/hnsw.hpp:1065-1086` — `search()` entry, parallel_for over queries
- `include/rabitqlib/index/hnsw/hnsw.hpp:1088-1175` — `search_knn()` (per-query orchestration)
- `include/rabitqlib/index/hnsw/hnsw.hpp:1133-1159` — upper-layer greedy walk
- `include/rabitqlib/index/hnsw/hnsw.hpp:1183-1267` — `searchBaseLayerST_AdaptiveRerankOpt` (base-layer hot loop)
- `include/rabitqlib/utils/hashset.hpp:34-100` — `HashBasedBooleanSet` (direct map + unordered_set fallback)
- `include/rabitqlib/utils/visited_pool.hpp` — `VisitedListPool` with global mutex
- `include/rabitqlib/utils/buffer.hpp:16-104` — `SearchBuffer` (the beam queue; already aligned, looks good)
- `include/rabitqlib/index/ivf/initializer.hpp:17-69` — `parallel_for` used by HNSW search

SVS reference (Vamana, structurally analogous to HNSW base-layer):
- `include/svs/index/vamana/greedy_search.h:124-203` — main loop with prefetcher
- `include/svs/index/vamana/greedy_search.h:44-50` — `GreedySearchPrefetchParameters` (lookahead=4, step=2)
- `include/svs/index/vamana/filter.h:46-130` — `VisitedFilter<uint32_t, N>` direct-mapped filter
