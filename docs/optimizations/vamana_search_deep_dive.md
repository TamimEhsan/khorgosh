# SVS Vamana Search: Why It's 5-10x Faster Than HNSW (Same Node Expansions)

If both SVS Vamana and HNSW expand the same number of nodes, the entire speedup comes from **per-node expansion cost**. This document traces exactly what happens inside each node expansion and identifies every optimization.

---

## 1. The Hot Loop: What Happens Per Node Expansion

**File:** `include/svs/index/vamana/greedy_search.h:104-149`

Each node expansion does these steps:
```
1. Pop best unvisited candidate from search buffer     [O(1)]
2. Fetch adjacency list from graph                      [1 cache miss, but pre-prefetched]
3. For each neighbor in adjacency list:
   a. Check visited filter                              [1 lookup, pre-prefetched]
   b. If not visited: prefetch NEXT neighbor's data     [async, non-blocking]
   c. Compute distance(query, neighbor)                 [SIMD, data already in cache]
   d. Insert into search buffer                         [O(1) skip or O(log k) insert]
```

The key insight: **every memory access is prefetched before it's needed.** The pipeline is:

```
Iteration i:   compute distance(neighbor_i)
               while CPU does SIMD math on neighbor_i's data...
               prefetch neighbor_{i+lookahead}'s data from DRAM → L1
               prefetch visited_filter slot for neighbor_{i+1}
```

This converts what would be serial memory stalls into overlapped computation + memory access.

---

## 2. Optimization #1: The Prefetcher (The Biggest Win)

**File:** `include/svs/lib/prefetch.h:79-163`

This is likely the single largest contributor to the speedup. In a typical HNSW implementation, each neighbor expansion goes:

```
// Typical HNSW (simplified):
for each neighbor_id in adjacency_list:
    if visited[neighbor_id]: continue        // CACHE MISS #1: visited set lookup
    vec = data[neighbor_id]                  // CACHE MISS #2: load vector (cold)
    dist = compute_distance(query, vec)      // STALL: waiting for #2
    update_candidate_list(dist)
```

Each vector load from main memory costs ~100-200 CPU cycles on modern hardware. During those cycles, the CPU is **completely stalled** waiting for data. With 128-dim float32 vectors (512 bytes), that's 8 cache lines per vector.

SVS's prefetcher eliminates these stalls:

```cpp
// SVS (greedy_search.h:114-148):
auto prefetcher = lib::make_prefetcher(
    {lookahead=4, step=2},  // prefetch 4 neighbors ahead
    num_neighbors,
    [&](size_t i) { accessor.prefetch(dataset, neighbors[i]); },  // prefetch data
    [&](size_t i) {
        // Also prefetch the visited filter slot for the NEXT neighbor
        search_buffer.unsafe_prefetch_visited(neighbors[i + 1]);
        // Only prefetch data if not already visited (saves bandwidth)
        return !search_buffer.unsafe_is_visited(neighbors[i]);
    }
);

prefetcher();  // Initial burst: prefetch first `lookahead` items
for (auto id : neighbors) {
    if (search_buffer.emplace_visited(id)) continue;
    prefetcher();  // Steady state: prefetch one more item
    auto dist = distance::compute(...);  // Data is already in L1!
    search_buffer.insert(...);
}
```

### How the Prefetcher Works Internally

The `Prefetcher` class (`prefetch.h:79-163`) has two phases:

**Startup phase (slack > 0):** Rapidly prefetch `lookahead` items to fill the pipeline. Each call prefetches `min(slack+1, step)` items, decrementing slack.

**Steady state (slack == 0):** Each call prefetches exactly 1 valid item (skipping already-visited nodes). This maintains the lookahead distance.

The predicate callback is critical: it checks the visited filter BEFORE issuing a prefetch. If a neighbor was already visited, don't waste prefetch bandwidth on it - advance to the next unvisited neighbor instead.

### What Gets Prefetched

**Vector data** (`prefetch.h:37-57`):
```cpp
void prefetch_l0(std::span<T, Extent> span) {
    const std::byte* base = byte_span.data();
    size_t num_prefetches = div_round_up(bytes, 64);  // 64-byte cache lines
    for (size_t i = 0; i < num_prefetches; ++i)
        _mm_prefetch(base + 64 * i, _MM_HINT_T0);    // into L1
}
```
For a 128-dim float32 vector (512 bytes), this issues 8 prefetch instructions, pulling the entire vector into L1 cache before it's needed.

**Graph adjacency list** (`graph.h:98-108`):
```cpp
const_reference get_node(Idx i) const {
    auto raw_data = data_.get_datum(i);
    size_t bytes = (1 + num_neighbors) * sizeof(Idx);
    if (bytes > 64)  // if > 1 cache line
        lib::prefetch(raw_data.subspan(64));  // prefetch remaining lines
    return raw_data.subspan(1, num_neighbors);
}
```
The first cache line of the adjacency list is already fetched (the access itself triggers it). SVS prefetches the *remaining* cache lines of large adjacency lists.

**Visited filter slot** (`filter.h:101`):
```cpp
void prefetch(integer_type key) const {
    lib::prefetch_l0(&values_[hash(key)]);
}
```
The visited filter lookup for the NEXT neighbor is prefetched while processing the CURRENT neighbor.

### Estimated Impact: 2-4x

In memory-bound workloads (which ANN search is), hiding memory latency is the single most impactful optimization. With ~100-200 cycle memory latency and ~10-30 cycle SIMD distance computation (for cached data), eliminating stalls can give 3-10x speedup on the memory access alone. The actual improvement depends on dimensionality and dataset size relative to LLC.

---

## 3. Optimization #2: AVX-512 SIMD Distance Computation

**File:** `include/svs/core/distance/euclidean.h:198-217` and `simd_utils.h:196-244`

### The Generic SIMD Framework

SVS uses a 4x-unrolled, multi-accumulator SIMD framework (`simd_utils.h:196-244`):

```cpp
// Process 64 floats per iteration (4 × 16-wide AVX-512)
auto s0 = op.init();  // accumulator 0
auto s1 = op.init();  // accumulator 1
auto s2 = op.init();  // accumulator 2
auto s3 = op.init();  // accumulator 3

for (; i + 64 <= count; i += 64) {
    // Load 4 chunks of query
    auto a0 = _mm512_loadu_ps(a + i);
    auto a1 = _mm512_loadu_ps(a + i + 16);
    auto a2 = _mm512_loadu_ps(a + i + 32);
    auto a3 = _mm512_loadu_ps(a + i + 48);
    
    // Load 4 chunks of data vector
    auto b0 = _mm512_loadu_ps(b + i);
    auto b1 = _mm512_loadu_ps(b + i + 16);
    auto b2 = _mm512_loadu_ps(b + i + 32);
    auto b3 = _mm512_loadu_ps(b + i + 48);
    
    // Accumulate into 4 INDEPENDENT accumulators
    s0 = _mm512_fmadd_ps(_mm512_sub_ps(a0,b0), _mm512_sub_ps(a0,b0), s0);
    s1 = _mm512_fmadd_ps(_mm512_sub_ps(a1,b1), _mm512_sub_ps(a1,b1), s1);
    s2 = _mm512_fmadd_ps(_mm512_sub_ps(a2,b2), _mm512_sub_ps(a2,b2), s2);
    s3 = _mm512_fmadd_ps(_mm512_sub_ps(a3,b3), _mm512_sub_ps(a3,b3), s3);
}
// Combine at the end
s0 = (s0 + s1) + (s2 + s3);
```

### Why 4 Accumulators Matter

The comments in `simd_utils.h:128-146` explain this directly:

> "Floating point arithmetic is not associative, so generally the compiler must strictly obey program semantics when optimizing. This means that if a single accumulator register is used, we introduce a long chain dependency in the instruction stream."

> "Cascadelake and greater servers have two execution ports that offer the bulk of AVX-512 functionality. When fully utilized, SIMD instructions can obtain a throughput of 2 ops per cycle."

With **1 accumulator**: FMA has 4-cycle latency → throughput limited to 1 FMA every 4 cycles.
With **4 accumulators**: 4 independent chains → 1 FMA per cycle (latency hidden by parallelism), and on dual-port CPUs, up to 2 FMA per cycle.

This is a **4-8x improvement** over single-accumulator SIMD.

### L2 Distance: 2 Instructions Per 16 Elements

The L2 kernel (`euclidean.h:205-207`):
```cpp
static __m512 accumulate(__m512 accumulator, __m512 a, __m512 b) {
    auto c = _mm512_sub_ps(a, b);           // 16 subtractions
    return _mm512_fmadd_ps(c, c, accumulator); // 16 multiply-adds (fused)
}
```

`_mm512_fmadd_ps(c, c, acc)` computes `c*c + acc` in a single instruction. This is the **fused multiply-add** - one instruction does two operations. So for 16 elements: 1 SUB + 1 FMA = 2 instructions.

### Masked Tail Handling

For vectors where `dims % 16 != 0` (`simd_utils.h:239-242`):
```cpp
if (i < count.size()) {
    auto mask = create_mask<simd_width>(count);  // e.g., 0b0000001111111111 for 10 remaining
    s0 = op.accumulate(mask, s0, op.load_a(mask, a + i), op.load_b(mask, b + i));
}
```

Uses **masked SIMD** (`_mm512_maskz_loadu_ps`) to handle the tail without scalar fallback. This avoids a branch and scalar loop for the remainder.

### VNNI for Integer Types

For int8/uint8 vectors (`euclidean.h:223-248`), SVS uses AVX-512 VNNI:
```cpp
static reg_t accumulate(reg_t accumulator, reg_t a, reg_t b) {
    auto c = _mm512_sub_epi16(a, b);
    return _mm512_dpwssd_epi32(accumulator, c, c);  // 32 squared differences in one instruction
}
```

`_mm512_dpwssd_epi32` performs a **dot product of word-sized signed differences** - it takes 32 int16 values, multiplies pairs, and accumulates into 16 int32 accumulators. This processes **32 dimensions per instruction**.

### Estimated Impact: 1.5-3x over typical HNSW

Most HNSW libraries use SSE4 or AVX2 (8-wide), often without multi-accumulator unrolling. SVS uses AVX-512 (16-wide) with 4x unrolling and FMA. The raw SIMD throughput difference is 2-4x, but since distance computation is often not the bottleneck (memory access is), the practical impact is smaller for large datasets. For datasets that fit in LLC, this becomes more dominant.

---

## 4. Optimization #3: Visited Filter (Direct-Mapped Cache)

**File:** `include/svs/index/vamana/filter.h:44-125`

This is NOT a hash set. It's a **direct-mapped cache** with 2^16 = 65,536 slots, using only 128KB of memory (fits entirely in L2 cache).

### Design
```cpp
// Hash: just mask the lower 16 bits (zero-cost)
static size_t hash(uint32_t key) { return key & 0xFFFF; }

// Store: only the UPPER 16 bits (saves 50% memory)
static value_type value(uint32_t key) { return key >> 16; }

// Check: compare stored upper bits against key's upper bits
static bool check(uint32_t key, uint16_t value) { return (key >> 16) == value; }

// Emplace: check + store in one operation
bool emplace(uint32_t key) {
    auto i = hash(key);
    auto& v = values_[i];
    bool was_present = check(key, v);
    v = value(key);  // unconditionally overwrite
    return was_present;
}
```

### Why This Is Fast

1. **No hash function overhead.** The "hash" is a single AND instruction. Compare to std::unordered_set which computes a real hash.

2. **Constant memory.** 65K × 2 bytes = 128KB. Fits in L2 cache. A hash set for a million-node graph would be many megabytes and cause cache thrashing.

3. **No collision handling.** Direct-mapped = O(1) guaranteed. Hash sets have worst-case O(n) probe sequences.

4. **False negatives are cheap.** If two IDs collide (same lower 16 bits, different upper 16), the filter says "not visited" even if one was. Result: an extra distance computation. This is wasted work but doesn't affect correctness, and happens rarely (~0.015% per lookup for typical search visiting ~100 nodes).

5. **No memory allocation.** The filter is allocated once per thread and reused across queries via `reset()` (memset).

6. **Prefetchable.** The single `_mm_prefetch(&values_[hash(key)])` in the search loop ensures the filter lookup never stalls.

### Estimated Impact: 1.1-1.3x

The filter eliminates redundant distance computations. In dense graphs (high max_degree), many neighbors of different nodes overlap. Without the filter, you'd recompute distances for already-visited nodes. The savings are proportional to the ratio of redundant-to-unique neighbor encounters.

---

## 5. Optimization #4: Search Buffer Design

**File:** `include/svs/index/vamana/search_buffer.h`

### O(1) Early Rejection
```cpp
size_t insert(value_type neighbor) {
    if (can_skip(neighbor.distance())) return size();  // O(1)
    return insert_inner(neighbor);
}
```

`can_skip` checks if the buffer is full AND the new distance is worse than the worst candidate. This rejects the majority of candidates in O(1) without touching the buffer at all. In a typical search, after the first few expansions, most candidates are too far and get rejected here.

### Binary Search Insertion
When a candidate passes the O(1) check:
```cpp
auto it = std::lower_bound(begin, end, neighbor, comparator);  // O(log k)
std::copy_backward(it, end - 1, end);  // shift elements right
*it = neighbor;  // insert
```

This maintains a sorted array, not a heap. Why a sorted array over a heap?
- `next()` (get best unvisited) is O(1) - just return the element at `best_unvisited_` index
- `done()` is O(1) - compare `best_unvisited_` against `search_window_size`
- Most insertions are rejected by `can_skip`, so the O(log k) insertion is rarely invoked

### Cache-Aligned Allocation
```cpp
using vector_type = std::vector<value_type, threads::CacheAlignedAllocator<value_type>>;
```

The buffer is aligned to 64-byte cache line boundaries, preventing false sharing when multiple threads have adjacent buffers.

### Estimated Impact: 1.05-1.1x

The O(1) rejection is the main win. The sorted array vs. heap difference is minor for typical k values (10-100).

---

## 6. Optimization #5: LVQ Fused Decompress+Distance (Quantized Only)

**File:** `include/svs/quantization/lvq/vectors.h:431-536`

For the quantized version, this is where the massive additional speedup comes from. Instead of:
1. Decompress 4-bit vector → float32 buffer (512 bytes for 128-dim)
2. Compute L2 distance on float32 vectors

SVS fuses these into a single pass:

```cpp
// vectors.h:459-478 (Sequential strategy)
for (size_t i = 0; i < unrolled_iterations; ++i) {
    // Load 16 floats from query
    auto lhs = accumulator_t{&x[simd_width * j]};
    
    // Unpack 16 compressed integers from data (4-bit → int32)
    auto unpacked = unpack_as(v, j, eve::as<int_wide_t>(), helper, eve::ignore_none);
    
    // Fused: scale, add bias, subtract query, square, accumulate
    // apply_step does: converted = scale * int_to_float(unpacked) + bias
    //                  temp = query - converted
    //                  accum += temp * temp
    a0 = apply_step(distance, a0, lhs, unpacked, aux, eve::ignore_none);
}
```

The `apply_step` for L2 (`vectors.h:275-290`):
```cpp
auto converted = eve::add[pred](aux.scale * eve::convert(y, eve::as<float>()), aux.bias);
auto temp = x - converted;
return accum + temp * temp;
```

### Why Fusion Matters

Without fusion (naive approach):
1. Load 64 bytes of compressed data → decompress to 512 bytes of float32 → write to temp buffer
2. Read 512 bytes from temp buffer + 512 bytes from query → compute distance

With fusion:
1. Load 64 bytes of compressed data → decompress IN REGISTER → compute distance immediately

Fusion eliminates:
- **512 bytes of writes** to the decompression buffer
- **512 bytes of reads** back from the decompression buffer
- **A temporary allocation** for the buffer

Plus: the compressed data is 8x smaller (64 bytes vs 512 for 128-dim, 4-bit), so the vector load from memory is 8x faster.

### The Same 4x Unrolling

LVQ distance also uses 4 accumulators (`vectors.h:454-478`):
```cpp
auto a0 = accumulator_t(0);
auto a1 = accumulator_t(0);
auto a2 = accumulator_t(0);
auto a3 = accumulator_t(0);
for (size_t i = 0; i < unrolled_iterations; ++i) {
    // 4 parallel decompress+distance chains
    a0 = apply_step(..., unpack(j),   ...);
    a1 = apply_step(..., unpack(j+1), ...);
    a2 = apply_step(..., unpack(j+2), ...);
    a3 = apply_step(..., unpack(j+3), ...);
}
a0 = (a0 + a1) + (a2 + a3);
```

### FastIP: Algebraic Optimization for Inner Product

For inner product with LVQ, SVS exploits the linearity of the dot product (`vectors.h:257-259, 307-318, 336-340`):

```
<query, scale * compressed + bias>  =  scale * <query, compressed> + bias * sum(query)
```

`sum(query)` is precomputed once per query via `eve::algo::reduce(query, 0.0f)` (`vectors.h:693`). Then the inner loop only computes `<query, compressed>` (no scale/bias per element), and the scalar multiply + add happens once at the end (`vectors.h:337-339`):

```cpp
float finish_step(DistanceFastIP distance, wide_<float, N> accum, ScaleBias aux) {
    return aux.scale * eve::reduce(accum, eve::plus) + aux.bias * distance.query_sum;
}
```

This saves 2 operations per element (scale multiply + bias add) in the hot loop.

### Turbo Packing Strategy

There's also a `Turbo` packing strategy (`vectors.h:506-536`) that rearranges the bit layout of compressed vectors for more efficient SIMD unpacking. The `for_each_slice` primitive processes the compressed data in SIMD-friendly chunks that align with register widths.

### Estimated Impact: 2-4x (for quantized variant)

The combination of 8x smaller data (less memory bandwidth) and fused decompress+compute (no intermediate buffer) gives a very large speedup. The memory bandwidth savings alone can be 3-5x since ANN search is memory-bound.

---

## 7. Optimization #6: Query Preprocessing for Biased Distance

**File:** `include/svs/quantization/lvq/vectors.h:552-636`

For LVQ, each compressed vector has a centroid offset. Instead of adding the centroid to each compressed vector during distance computation, SVS **subtracts the centroid from the query once**:

```
L2(query, centroid + compressed) = L2(query - centroid, compressed)
```

The `EuclideanBiased` class (`vectors.h:552-636`) does this preprocessing:
```cpp
void fix_argument(const std::span<const float>& query) {
    for (size_t i = 0; i < num_centroids; ++i) {
        for (size_t j = 0; j < dims; ++j)
            processed_query_[i][j] = query[j] - centroid[i][j];
    }
}
```

This is called once per query. Then every distance computation against vectors in centroid group `k` uses `processed_query_[k]` directly, avoiding per-vector centroid addition.

---

## 8. Optimization #7: Two-Phase Search with Reranking (Quantized, Two-Level)

**File:** `include/svs/extensions/vamana/lvq.h:159-181`

For two-level LVQ datasets:

```cpp
// Phase 1: Graph search using PRIMARY (4-bit) vectors only
{
    auto accessor = PrimaryAccessor();
    search(query, accessor, distance, search_buffer);
}

// Phase 2: Rerank using full precision (primary + residual)
for (size_t j = 0; j < search_buffer.size(); ++j) {
    auto id = search_buffer[j].id();
    auto new_distance = distance::compute(distance, query, dataset.get_datum(id));
    search_buffer[j].set_distance(new_distance);
}
search_buffer.sort();
```

Phase 1 is fast because it uses 4-bit vectors. Phase 2 is expensive but only runs on the ~`search_window_size` candidates (typically 10-200), not the entire graph traversal. This gives graph traversal speed close to brute-force 4-bit while maintaining accuracy close to full precision.

---

## 9. Summary: Where the 5-10x Comes From

Assuming SVS vs a typical HNSW library (e.g., hnswlib) with the same number of node expansions:

| Optimization | Mechanism | Estimated Impact | Applies To |
|---|---|---|---|
| **Prefetching** | Overlap memory latency with computation | **2-4x** | Both normal & quantized |
| **AVX-512 SIMD** | 16-wide FMA, 4x unrolled, multi-accumulator | **1.5-3x** | Both |
| **Visited filter** | 128KB direct-mapped cache, zero-cost hash | **1.1-1.3x** | Both |
| **Search buffer** | O(1) early rejection, sorted array | **1.05-1.1x** | Both |
| **LVQ fusion** | Fused decompress+distance, 8x less bandwidth | **2-4x** | Quantized only |
| **Query preprocessing** | Centroid subtracted from query once | **1.1-1.2x** | Quantized only |
| **Two-phase search** | Fast 4-bit traversal + precise reranking | **1.2-1.5x** | Quantized two-level |

### For Normal (float32) Vamana:
The gains compound multiplicatively on the bottleneck:
- Prefetching removes the memory stall bottleneck → **2-4x**
- AVX-512 speeds up computation (matters more now that memory isn't stalling) → **1.5-2x**
- Visited filter + buffer → **~1.2x**
- **Combined: ~4-10x** over a naive HNSW that doesn't prefetch and uses SSE/AVX2

### For Quantized (LVQ) Vamana:
All of the above, PLUS:
- 4-8x less memory bandwidth per vector → more fits in cache → fewer cache misses
- Fused decompress+distance → no intermediate buffer
- **Combined: ~8-15x** over float32 HNSW without prefetching

### The Dominant Factor

**Prefetching is the #1 factor.** In a memory-bound workload like graph-based ANN search, the CPU spends most of its time waiting for data from DRAM. A 128-dim float32 vector spans 8 cache lines. Without prefetching, each distance computation stalls for ~100-200ns. With SVS's prefetcher running 4 iterations ahead, by the time you need the data, it's already in L1 cache. This alone can explain a 3-5x speedup.

The second factor is AVX-512 with proper unrolling and multi-accumulator design, which makes the computation itself 2-3x faster than typical SSE/AVX2 implementations.

Everything else (visited filter, search buffer, query preprocessing) provides incremental 10-30% improvements that compound on top.
