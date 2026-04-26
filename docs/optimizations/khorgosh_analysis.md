# Khorgosh (RaBitQ) - Complete Analysis: Architecture, Bottlenecks & Improvements

## 1. Architecture Overview

Khorgosh is a header-only C++ ANNS library implementing **RaBitQ** (Random Bit Quantization) on top of HNSW, IVF, and SymQG graph structures. The `transpose-query` branch optimizes query layout for sequential memory access during search.

### File Map (non-third-party)

| File | Purpose |
|------|---------|
| `defines.hpp` | Types (PID, AnnCandidate, MetricType) |
| `quantization/rabitq_impl.hpp` | Core RaBitQ: 1-bit + extra-bits quantization |
| `quantization/data_layout.hpp` | Memory layout for batch/single quantized data |
| `quantization/pack_excode.hpp` | Packing extra-bit codes |
| `fastscan/fastscan.hpp` | PSHUFB-based batch distance (32 vectors at once) |
| `fastscan/highacc_fastscan.hpp` | Higher-accuracy FastScan (int32 accumulation) |
| `index/hnsw/hnsw.hpp` | HNSW graph build + search (**main search loop**) |
| `index/ivf/ivf.hpp` | IVF index |
| `index/symqg/qg.hpp` | SymQG index |
| `index/query.hpp` | Query preprocessing (quantize query, transpose) |
| `index/estimator.hpp` | Distance estimation from quantized codes |
| `index/lut.hpp` | Lookup table for FastScan |
| `utils/warmup_space.hpp` | **Hot path:** transposed binary inner product (AVX-512/AVX2) |
| `utils/space.hpp` | SIMD utilities, excode IP, rotation helpers |
| `utils/buffer.hpp` | SearchBuffer (sorted linear buffer) |
| `utils/memory.hpp` | Aligned allocation, prefetch helpers |
| `utils/rotator.hpp` | FHT-Kac random rotation |
| `utils/hashset.hpp` | Hash-based visited set |
| `utils/visited_pool.hpp` | Thread pool for visited sets |

---

## 2. RaBitQ Quantization Scheme

### How It Works

Each vector is quantized relative to its cluster centroid:

1. **Rotation:** Apply Fast Hadamard Transform (FHT) + Kac walk to the vector. This makes components near-Gaussian, improving quantization quality. Dimension padded to multiple of 64.

2. **Residual:** `r = rotated_vector - rotated_centroid`

3. **1-bit code:** `b[i] = (r[i] > 0) ? 1 : 0` — one bit per dimension, packed into uint64 blocks.

4. **Factors:** Three floats computed per vector:
   - `f_add`: additive bias for distance estimate
   - `f_rescale`: multiplicative scaling factor
   - `f_error`: error bound (controls pruning aggressiveness)

5. **Extra-bits code (optional):** Additional N bits per dimension for higher accuracy.
   - Uses greedy optimization to find the best rescale factor
   - Stored separately from 1-bit codes

### Distance Estimation Formula

For L2 distance:
```
est_dist = f_add + ||q - centroid||^2 + f_rescale * (IP(q_quantized, b) + offset)
low_dist = est_dist - f_error * ||q - centroid||
```

The key operation is `IP(q_quantized, b)` — inner product between 4-bit quantized query and 1-bit data codes. This reduces to **popcount of bitwise AND**, which is extremely fast.

---

## 3. Search Workflow (HNSW)

### Entry: `search()` → `search_knn()` → `searchBaseLayerST_AdaptiveRerankOpt()`

**Step 1: Query preparation** (`query.hpp:139-171`)
- Rotate query via FHT-Kac
- Compute sum of query components
- Quantize query to 4-bit codes (uint16 per dimension)
- Pack into uint64 binary representation
- **Transpose** the binary representation for cache-friendly access

**Step 2: Centroid distances** (`hnsw.hpp:1100-1126`)
- Compute `||q - centroid_i||` for ALL centroids (full float precision)
- These are used as `g_add` and `g_error` for each vector's distance estimate

**Step 3: Upper-layer greedy descent** (`hnsw.hpp:1133-1158`)
- Use 1-bit distance estimates to greedily descend through HNSW layers

**Step 4: Base layer search** (`hnsw.hpp:1183-1267`)
The hot loop. For each candidate node:
1. Get adjacency list from link list
2. Prefetch bin data of first neighbor (2 cache lines)
3. For each neighbor:
   a. Prefetch NEXT neighbor's bin data (2 cache lines)
   b. Check visited set
   c. Compute 1-bit distance estimate via `get_bin_est()`
   d. If promising (`low_dist < distk`), compute full estimate via `get_full_est()`
   e. Insert into BoundedKNN and SearchBuffer
   f. Prefetch link list of current best candidate

---

## 4. Hot Path: Per-Node Distance Computation

### 4.1 `warmup_ip_x0_q` — The Core Kernel (`warmup_space.hpp:37-166`)

This computes the weighted popcount inner product between 1-bit data and 4-bit query:

```
result = delta * SUM_i( SUM_j( popcount(data[i] & query[j][i]) << j ) ) + vl * SUM_i( popcount(data[i]) )
```

**AVX-512 version (lines 47-102):**
- Processes 8 × 64-bit blocks per iteration (512 data bits)
- Uses `_mm512_popcnt_epi64()` (VPOPCNTDQ instruction)
- Inner loop over `b_query=4` query bits: AND + popcount + shift-left + accumulate
- Final horizontal reduction via `_mm512_reduce_add_epi64()`

**AVX2 version (lines 103-160):**
- Processes 4 × 64-bit blocks per iteration
- Uses `popcount_avx2()` — Mula's nibble-lookup method via PSHUFB
- Same structure but half the throughput

### 4.2 Distance Estimate Assembly (`estimator.hpp:164-189`)

```cpp
ip_x0_qr = warmup_ip_x0_q<4>(bin_code, transposed_query, delta, vl, padded_dim, 4);
est_dist = f_add + g_add + f_rescale * (ip_x0_qr + k1xsumq);
low_dist = est_dist - f_error * g_error;
```

Three multiplies, three adds after the kernel. Negligible compared to the popcount loop.

### 4.3 Full Estimate (with extra bits, `estimator.hpp:133-159`)

When `ex_bits > 0` and the 1-bit estimate is promising:
```cpp
ip_x0_qr = mask_ip_x0_q(rotated_query, bin_code, padded_dim);  // float×bit inner product
ex_ip = ip_func_(rotated_query, ex_code, padded_dim);           // float×Nbit inner product
est_dist = f_add_ex + g_add + f_rescale_ex * (scale * ip_x0_qr + ex_ip + kbxsumq);
```

The `ip_func_` is selected at load time based on `ex_bits` and dispatches to optimized SIMD inner product routines in `space.hpp`.

---

## 5. Bottlenecks Identified

### BOTTLENECK 1: Visited Set — Hash Collision Fallback to `std::unordered_set` (CRITICAL)

**File:** `hashset.hpp:80-99`

```cpp
bool get(PID data_id) const {
    PID val = this->table_[hash1(data_id)];
    if (val == data_id) return true;
    return (val != kPidMax && stl_hash_.find(data_id) != stl_hash_.end());  // <-- SLOW
}

void set(PID data_id) {
    PID& val = table_[hash1(data_id)];
    if (val == kPidMax) { val = data_id; }
    else { stl_hash_.emplace(data_id); }  // <-- HEAP ALLOCATION
}
```

**Problem:** The visited set is a direct-mapped table with overflow to `std::unordered_set`. On collision (which happens frequently — table size is only `sqrt(max_elements)` scaled), it falls back to `stl_hash_.find()` which:
- Has O(1) average but involves pointer chasing (hash bucket → linked list node)
- Each `emplace` into the unordered_set does a **heap allocation**
- `clear()` calls `stl_hash_.clear()` which frees all nodes — **O(n) deallocation per query**
- The `stl_hash_` is cold in cache and destroys L1/L2 locality

**How SVS handles this:** A fixed-size direct-mapped cache (128KB, 65K slots) with NO overflow. False negatives (collisions) simply result in recomputation — much cheaper than heap allocation.

**Impact:** This is likely the #1 bottleneck. Every node expansion touches the visited set at least once per neighbor (typically 20-64 times per expansion). Any heap allocation or pointer chase in this path is devastating.

### BOTTLENECK 2: Prefetching Is Minimal and Poorly Placed (HIGH)

**File:** `hnsw.hpp:1218-1261`

Current prefetching:
```cpp
// Before the loop: prefetch first neighbor's bindata (2 cache lines)
mem_prefetch_l1(get_bindata_by_internalid(*(data + 1)), 2);

// Inside loop: prefetch NEXT neighbor's bindata (2 cache lines)
mem_prefetch_l1(get_bindata_by_internalid(*(data + j + 1)), 2);

// After insertion: prefetch link list of best candidate
mem_prefetch_l2((char*)get_linklist0(candidate_set.next_id()), 2);
```

**Problems:**
1. **Only prefetches 2 cache lines (128 bytes) of bin data.** For 768-dim vectors, `padded_dim=768`, the bin data is `768/8 + 12 = 108 bytes` (fits in 2 lines). But for higher dims (1024+), this is insufficient.

2. **Lookahead of only 1.** Only prefetches the immediately next neighbor. With the popcount kernel + estimate computation taking ~10-30 cycles, and DRAM latency at ~100-200 cycles, prefetching just 1 ahead is not enough. SVS prefetches **4 items ahead**.

3. **No prefetch of the visited set.** SVS prefetches the visited filter slot for the next neighbor while processing the current one. Khorgosh's `HashBasedBooleanSet::get()` involves a table lookup that may cache-miss, but is never prefetched.

4. **No prefetch of adjacency list.** When the search decides to explore a candidate, the link list of that candidate is only prefetched to L2 (not L1) and only 2 cache lines.

5. **Potential out-of-bounds read:** `*(data + j + 1)` when `j == size` reads one past the valid neighbor list. This is technically undefined behavior (the `+ 1` element after the last neighbor in the link list could be anything).

6. **No predicate-based skip.** SVS skips prefetching for already-visited nodes. Khorgosh prefetches unconditionally, wasting bandwidth.

### BOTTLENECK 3: BoundedKNN Uses `std::vector::insert` (MODERATE)

**File:** `hnsw.hpp:77-92`

```cpp
void insert(const Candidate& cand) {
    auto it = std::upper_bound(queue_.begin(), queue_.end(), cand, ...);
    queue_.insert(it, cand);           // <-- O(n) shift via memmove
    if (queue_.size() > capacity_)
        queue_.pop_back();             // <-- O(1) but redundant allocation check
}
```

**Problems:**
- `std::vector::insert` in the middle causes a `memmove` of all trailing elements
- No O(1) early rejection before doing the binary search. SVS's SearchBuffer checks `can_skip(dist)` first — if the buffer is full and this distance is worse than the worst, skip immediately without touching the buffer.
- The `Candidate` struct contains a `ResultRecord` (2 floats) + PID (4 bytes) = 12 bytes per element. For ef=200, that's 2.4KB of data being shifted on each insertion.

### BOTTLENECK 4: Centroid Distance Precomputation Is O(num_clusters) (MODERATE)

**File:** `hnsw.hpp:1100-1126`

```cpp
for (size_t i = 0; i < num_cluster_; i++) {
    q_to_centroids[i] = std::sqrt(raw_dist_func_(...));  // full L2 + sqrt for each cluster
}
```

For `num_cluster_ = 1` (typical for HNSW), this is fine. But if many clusters are used, this is a linear scan with full float distance + sqrt per cluster, done for EVERY query. There's no caching or SIMD batch computation.

### BOTTLENECK 5: Memory Layout Causes Cache Conflicts (MODERATE)

**File:** `hnsw.hpp:137-138`

```
Layout per element:
| Link list (maxM0*4 + 4 bytes) | cluster_id (4) | label (4) | BinData (~108) | ExData (~var) |
```

The link list and quantized data are in the same contiguous block. When you fetch the link list to get neighbors, you also pull the quantized data of the CURRENT node into cache — but you don't need it (you need neighbors' data, not your own). This wastes cache lines.

**How SVS handles this:** Graph data and vector data are stored in **separate arrays**. Fetching the adjacency list doesn't pollute the cache with vector data.

### BOTTLENECK 6: `data_level0_memory_` Is `malloc`'d Without Alignment (LOW-MODERATE)

**File:** `hnsw.hpp:392-393`

```cpp
data_level0_memory_ = reinterpret_cast<char*>(malloc(max_elements_ * size_data_per_element_));
```

Plain `malloc` — not `aligned_alloc`. While individual element access is computed via `internal_id * size_data_per_element_`, if `size_data_per_element_` is not a multiple of 64, then elements will NOT be cache-line aligned. This causes:
- Split cache line loads when accessing elements
- Worse prefetch efficiency (prefetching one cache line may miss part of the data)

### BOTTLENECK 7: Per-Query Vector Allocations (LOW-MODERATE)

**File:** `hnsw.hpp:1075, 1101`

```cpp
// Per query:
std::vector<float> rotated_query(padded_dim_);           // allocation
std::vector<float> q_to_centroids(num_cluster_);         // allocation
// Inside SplitSingleQuery constructor:
std::vector<uint16_t> quant_query(padded_dim);           // allocation
std::vector<uint64_t> QueryBin_(padded_dim * 4 / 64);   // allocation
std::vector<uint64_t> TransposedQueryBin_(...);          // allocation
```

Five heap allocations per query. For high-QPS scenarios, this causes allocator contention and cache pollution. SVS pre-allocates per-thread scratch space.

### BOTTLENECK 8: No Multi-Accumulator Unrolling in Popcount Kernel (LOW)

**File:** `warmup_space.hpp:62-87`

The AVX-512 loop processes 8 blocks per iteration with a SINGLE pair of accumulators (`ip_vec`, `ppc_vec`). There's no 2x or 4x unrolling with independent accumulators. While popcount + add have low latency (~3 cycles), the inner `b_query` loop (4 iterations) creates a dependency chain:

```
for j in 0..3:
    q_vec = load(...)
    and_vec = and(x_vec, q_vec)
    popcnt = popcnt(and_vec)
    weighted = slli(popcnt, j)
    ip_vec = add(ip_vec, weighted)   // <-- dependency chain on ip_vec
```

Each iteration of `j` depends on the previous `ip_vec` value. With 4 dependent adds, this serializes to ~12 cycles per 8-block chunk. Unrolling the outer `i` loop with independent accumulators would allow the CPU to execute multiple chunks in parallel.

---

## 6. Lackings

### LACKING 1: No Bloom/Direct-Mapped Visited Filter

The `HashBasedBooleanSet` is a custom hash table with `std::unordered_set` overflow. A simple direct-mapped filter (like SVS's `VisitedFilter`) would be:
- Fixed 128KB memory (fits in L2)
- Zero allocation
- No pointer chasing
- `clear()` is just `memset`
- False negatives are acceptable (just recompute distance)

### LACKING 2: No Adaptive Prefetch Strategy

SVS's `Prefetcher` class dynamically adjusts how many items to prefetch ahead, skips already-visited items, and prefetches both data and metadata (visited set). Khorgosh has a simple "prefetch next one" approach with no adaptation.

### LACKING 3: No Compile-Time Dimension Specialization

SVS uses template parameters for dimensions (`MaybeStatic<N>`), allowing the compiler to generate specialized code for common dimensions (128, 256, 768). Khorgosh passes dimensions as runtime `size_t` everywhere, preventing loop unrolling optimizations.

### LACKING 4: No Reranking with Original Vectors

SVS's two-level LVQ does graph traversal with compressed vectors, then reranks the final candidates with full-precision reconstruction. Khorgosh does 1-bit search → optional ex-bits refinement, but never reranks with the original float vectors. This caps accuracy.

### LACKING 5: No Graph Construction Optimizations

The HNSW construction (`hnsw.hpp:618-737`) uses raw float distances. It's a straightforward port of hnswlib with:
- Per-node mutexes (fine-grained locking, but high contention)
- No batched construction (each node processed individually)
- `std::unordered_map` for label lookup (slow)

SVS uses batched construction with coarse-grained bucket locking, two-pass alpha refinement, and backedge pruning.

### LACKING 6: No Thread-Local Scratch Reuse

Each query allocates its own `SplitSingleQuery`, vectors, etc. In SVS, these are pre-allocated per thread and reused across queries.

### LACKING 7: ExCode Inner Product Uses Function Pointer Dispatch

**File:** `hnsw.hpp:159`
```cpp
float (*ip_func_)(const float*, const uint8_t*, size_t);
```

The excode inner product function is dispatched via a function pointer, preventing inlining. SVS uses template dispatch which is resolved at compile time.

---

## 7. Improvement Recommendations

### Priority 1: Replace HashBasedBooleanSet with Direct-Mapped Filter (Est. 1.5-2.5x)

Replace the current visited set with a fixed-size direct-mapped cache:
```cpp
class VisitedFilter {
    std::vector<uint16_t> slots_;  // 65536 slots, 128KB
    static uint32_t hash(PID id) { return id & 0xFFFF; }
    static uint16_t tag(PID id) { return id >> 16; }
    bool check_and_set(PID id) {
        auto h = hash(id);
        bool present = (slots_[h] == tag(id));
        slots_[h] = tag(id);
        return present;
    }
    void clear() { memset(slots_.data(), 0xFF, slots_.size() * 2); }
};
```

Benefits: No heap allocation, no pointer chasing, O(1) guaranteed, fits in L2, memset clear.

**Note on table sizing:** The current `HashBasedBooleanSet` uses `~32 * sqrt(max_elements / 10)` slots (e.g., ~8K for 1M elements). Even without the `/10`, the table would be ~32K — comparable to SVS's 64K. The table size is NOT the main problem. The problem is what happens on collision: SVS silently overwrites (O(1), no allocation), while Khorgosh overflows into `std::unordered_set` (heap allocation + pointer chasing per collision).

**Handling duplicates from false negatives:** A direct-mapped filter will occasionally report a previously-visited node as unvisited (false negative). This causes a duplicate distance computation and insertion attempt. Impact on each buffer:

- **`candidate_set` (SearchBuffer):** Duplicate X gets inserted with the same distance. When `pop()` returns X the second time, all its neighbors are already marked visited in the filter, so the second expansion is a no-op scan. Cost: one wasted pop (~20 cache-line loads for the adjacency list, zero distance computations). **Harmless.**

- **`boundedKNN`:** Duplicate X could appear twice in the final TOPK result. **This needs a fix.** Since the duplicate has the same `est_dist`, the binary search (`upper_bound`) places the new entry right after the existing one. A simple O(1) check at the insertion boundary handles it:

```cpp
void insert(const Candidate& cand) {
    if (queue_.size() >= capacity_ && cand.record.est_dist >= queue_.back().record.est_dist)
        return;  // O(1) early reject (Priority 3)

    auto it = std::upper_bound(queue_.begin(), queue_.end(), cand, by_est_dist);

    // O(1) duplicate check: same node → same distance → adjacent position
    if (it != queue_.begin() && std::prev(it)->id == cand.id) return;
    if (it != queue_.end() && it->id == cand.id) return;

    queue_.insert(it, cand);
    if (queue_.size() > capacity_) queue_.pop_back();
}
```

Two integer comparisons on data already in cache from the binary search. Zero overhead in practice. The `candidate_set` does NOT need this fix — duplicates there are harmless (just wasted pops).

### Priority 2: Add Proper Prefetching Pipeline (Est. 1.5-3x)

Implement a lookahead prefetcher similar to SVS:
```cpp
// Before loop: prefetch first N neighbors' bindata
for (size_t p = 0; p < min(4, size); p++)
    mem_prefetch_l1(get_bindata_by_internalid(*(data + 1 + p)), needed_lines);

// Inside loop:
for (size_t j = 1; j <= size; j++) {
    // Prefetch neighbor j+4's data (if exists and not visited)
    if (j + 4 <= size) {
        PID future = *(data + j + 4);
        // Prefetch visited set slot
        prefetch_l1(&visited_table[future & mask]);
        // Only prefetch data if not visited
        if (!visited_fast_check(future))
            mem_prefetch_l1(get_bindata_by_internalid(future), needed_lines);
    }
    // ... process neighbor j (data should be in L1 by now)
}
```

Also prefetch the correct number of cache lines based on actual `size_bin_data_` and `size_ex_data_`.

### Priority 3: Add O(1) Early Rejection to BoundedKNN (Est. 1.1-1.2x)

```cpp
void insert(const Candidate& cand) {
    if (queue_.size() >= capacity_ && cand.record.est_dist >= queue_.back().record.est_dist)
        return;  // O(1) reject
    // ... existing binary search + insert
}
```

### Priority 4: Pre-Allocate Per-Thread Scratch (Est. 1.1-1.3x)

Move allocations outside the per-query loop:
```cpp
struct ThreadScratch {
    std::vector<float> rotated_query;
    std::vector<float> q_to_centroids;
    SplitSingleQuery<float> query_wrapper;  // needs redesign to be reusable
    // ...
};
// Pre-allocate one per thread
```

### Priority 5: Unroll the Popcount Kernel (Est. 1.1-1.3x)

Process 2 or 4 data blocks per outer iteration with independent accumulators:
```cpp
__m512i ip_vec0 = _mm512_setzero_si512();
__m512i ip_vec1 = _mm512_setzero_si512();
__m512i ppc_vec0 = _mm512_setzero_si512();
__m512i ppc_vec1 = _mm512_setzero_si512();

for (size_t i = 0; i < vec_end; i += 16) {  // 2x unroll
    __m512i x0 = _mm512_loadu_si512(&data[i]);
    __m512i x1 = _mm512_loadu_si512(&data[i + 8]);
    ppc_vec0 = _mm512_add_epi64(ppc_vec0, _mm512_popcnt_epi64(x0));
    ppc_vec1 = _mm512_add_epi64(ppc_vec1, _mm512_popcnt_epi64(x1));
    for (uint32_t j = 0; j < b_query; j++) {
        // ... independent accumulation into ip_vec0 and ip_vec1
    }
}
ip_vec0 = _mm512_add_epi64(ip_vec0, ip_vec1);
```

### Priority 6: Separate Graph and Data Storage (Est. 1.05-1.15x)

Store link lists and quantized data in separate arrays:
```cpp
char* link_lists_memory_;   // Only link lists, tightly packed
char* quant_data_memory_;   // Only quantized data, tightly packed
```

This prevents adjacency list fetches from evicting useful quantized data from cache.

### Priority 7: Use `aligned_alloc` for `data_level0_memory_` (Est. 1.05x)

```cpp
data_level0_memory_ = static_cast<char*>(
    std::aligned_alloc(64, max_elements_ * size_data_per_element_)
);
```

And ensure `size_data_per_element_` is rounded up to a multiple of 64.

### Priority 8: Template the Dimension for Common Cases (Est. 1.05-1.1x)

```cpp
template <size_t PaddedDim = 0>  // 0 = runtime
float warmup_ip_x0_q(...) {
    constexpr size_t num_blk = (PaddedDim > 0) ? PaddedDim / 64 : 0;
    // Compiler can now unroll for known dimensions
}
// Dispatch:
if (padded_dim == 128) return warmup_ip_x0_q<128>(...);
if (padded_dim == 768) return warmup_ip_x0_q<768>(...);
return warmup_ip_x0_q<0>(...);  // fallback
```

### Priority 9: Fix Potential Out-of-Bounds Read in Prefetch (BUG)

**File:** `hnsw.hpp:1223-1225`
```cpp
mem_prefetch_l1(get_bindata_by_internalid(*(data + j + 1)), 2);
```
When `j == size`, `*(data + j + 1)` reads past the neighbor list. Fix:
```cpp
if (j < size) {
    mem_prefetch_l1(get_bindata_by_internalid(*(data + j + 1)), 2);
}
```

---

## 8. SIMD Optimization Opportunities (Detailed)

### SIMD Issue 1: Single Accumulator in All ExCode IP Functions (HIGH)

**Files:** `space.hpp:324-761` — all 7 functions (`ip16_fxu1..7_avx`)

Every excode inner product function uses a **single accumulator** for FMA:
```cpp
// ip64_fxu2_avx (representative), space.hpp:367-402
__m512 sum = _mm512_setzero_ps();
for (size_t i = 0; i < dim; i += 64) {
    sum = _mm512_fmadd_ps(q, cf, sum);   // 4-cycle latency, depends on prev sum
    sum = _mm512_fmadd_ps(q, cf, sum);   // stalls until above completes
    sum = _mm512_fmadd_ps(q, cf, sum);   // stalls
    sum = _mm512_fmadd_ps(q, cf, sum);   // stalls
}
```

FMA has **4-cycle latency** on Intel but **0.5-cycle throughput** (2 FMA ports). With 4 dependent FMAs per iteration, that's 16 cycles minimum. With 2 independent accumulators and 2x unrolling, you could reach 2 FMA/cycle — an **up to 8x throughput improvement** on the compute-bound portion.

**Fix:** Use 2-4 independent accumulators with outer-loop unrolling:
```cpp
__m512 sum0 = _mm512_setzero_ps();
__m512 sum1 = _mm512_setzero_ps();
for (size_t i = 0; i < dim; i += 128) {
    // First 64 dims → sum0
    sum0 = _mm512_fmadd_ps(q0, cf0, sum0);
    sum0 = _mm512_fmadd_ps(q1, cf1, sum0);
    sum0 = _mm512_fmadd_ps(q2, cf2, sum0);
    sum0 = _mm512_fmadd_ps(q3, cf3, sum0);
    // Second 64 dims → sum1 (independent chain)
    sum1 = _mm512_fmadd_ps(q4, cf4, sum1);
    sum1 = _mm512_fmadd_ps(q5, cf5, sum1);
    sum1 = _mm512_fmadd_ps(q6, cf6, sum1);
    sum1 = _mm512_fmadd_ps(q7, cf7, sum1);
}
sum0 = _mm512_add_ps(sum0, sum1);
```

**Estimated impact:** 1.5-3x on excode IP (which is the hot path for `get_full_est()` when `ex_bits > 0`).

### SIMD Issue 2: Single Accumulator in `mask_ip_x0_q` with Commented-Out Fix (HIGH)

**File:** `space.hpp:941-979`

The code for the full-bits float×binary inner product has 4 masked loads per block, all added into one `sum`. Lines 948-951 show the **fix was already written but commented out**:

```cpp
//    __m512 sum0 = _mm512_setzero_ps();   // <-- COMMENTED OUT
//    __m512 sum1 = _mm512_setzero_ps();
//    __m512 sum2 = _mm512_setzero_ps();
//    __m512 sum3 = _mm512_setzero_ps();

__m512 sum = _mm512_setzero_ps();          // <-- single accumulator used instead
for (size_t i = 0; i < num_blk; ++i) {
    sum = _mm512_add_ps(sum, masked0);     // dependency chain
    sum = _mm512_add_ps(sum, masked1);
    sum = _mm512_add_ps(sum, masked2);
    sum = _mm512_add_ps(sum, masked3);
}
//    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
```

**Fix:** Uncomment the 4-accumulator version. Each `masked0..3` is independent and should go into separate accumulators, combined only at the end.

**Estimated impact:** 1.3-2x on `split_single_fulldist` path.

### SIMD Issue 3: No Outer-Loop Unrolling in `warmup_ip_x0_q` (MODERATE)

**File:** `warmup_space.hpp:62-87`

The AVX-512 loop processes 8 blocks per iteration with a single `ip_vec`/`ppc_vec` pair. The inner `j` loop (4 iterations for `b_query=4`) creates a dependency chain on `ip_vec`:

```cpp
for (size_t i = 0; i < vec_end; i += 8) {
    __m512i x_vec = _mm512_loadu_si512(&data[i]);
    ppc_vec = _mm512_add_epi64(ppc_vec, _mm512_popcnt_epi64(x_vec));
    for (uint32_t j = 0; j < b_query; j++) {
        __m512i q_vec = _mm512_loadu_si512(&query[j * num_blk + i]);
        __m512i weighted = _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x_vec, q_vec)), j);
        ip_vec = _mm512_add_epi64(ip_vec, weighted);  // dependency chain
    }
}
```

**Fix:** Process 16 or 24 blocks per iteration with 2-3 independent accumulator pairs:
```cpp
for (size_t i = 0; i < vec_end; i += 16) {
    __m512i x0 = _mm512_loadu_si512(&data[i]);
    __m512i x1 = _mm512_loadu_si512(&data[i + 8]);
    ppc_vec0 = _mm512_add_epi64(ppc_vec0, _mm512_popcnt_epi64(x0));
    ppc_vec1 = _mm512_add_epi64(ppc_vec1, _mm512_popcnt_epi64(x1));
    for (uint32_t j = 0; j < b_query; j++) {
        // x0's chain → ip_vec0, x1's chain → ip_vec1 (independent)
    }
}
```

**Estimated impact:** 1.1-1.3x on 1-bit distance estimate.

### SIMD Issue 4: Scalar `reverse_bits_u64` in Hot Loop (MODERATE)

**File:** `space.hpp:817-825, 955`

`mask_ip_x0_q` calls `reverse_bits_u64()` **once per 64-dim block** in the inner loop:
```cpp
for (size_t i = 0; i < num_blk; ++i) {
    uint64_t bits = reverse_bits_u64(*it_data);  // 12 shift/mask/OR ops
    ...
}
```

This is pure scalar work (12 dependent operations) in a SIMD-heavy loop. It exists because `pack_binary` (line 180-188) packs bits MSB-first, but masked loads expect LSB-first.

**Fix options:**
- **Best:** Change `pack_binary` to pack LSB-first. Eliminates `reverse_bits_u64` entirely with zero runtime cost.
- **Alternative:** Use BMI2 intrinsics (`_pdep_u64`/`_pext_u64`) for faster bit rearrangement.

The same `reverse_bits` (32-bit version) is also called in `new_transpose_bin` (line 845-846, marked `[TODO: remove all reverse_bits]`).

**Estimated impact:** ~1.1x on full-estimate path per call.

### SIMD Issue 5: AVX2 Horizontal Reduction via Store-to-Memory (LOW-MODERATE)

**File:** `space.hpp:308-316`

```cpp
inline float mm256_reduce_add_ps(__m256 v) {
    std::array<float, 8> accumulator{};
    _mm256_storeu_ps(accumulator.data(), v);
    float result = 0.0F;
    for (const auto& i : accumulator) { result += i; }
    return result;
}
```

This stores to stack memory then loads back 8 individual floats — a store-forwarding stall (~10 cycles) plus 7 scalar additions. The standard in-register reduction is ~4 instructions:

```cpp
inline float mm256_reduce_add_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ss(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, shuf));
}
```

Every AVX2 excode IP function and `mask_ip_x0_q` calls this at the end.

**Estimated impact:** ~1.05x per AVX2 distance call.

### SIMD Issue 6: AVX2 `mask_ip_x0_q` Processes 8 Floats at a Time (LOW-MODERATE)

**File:** `space.hpp:980-1010`

The AVX2 path processes binary data **one byte at a time** (8 floats per step):
```cpp
for (int j = 0; j < 8; ++j) {  // 8 iterations per 64-bit block
    uint8_t current_byte = static_cast<uint8_t>(bits >> (j * 8));
    __m256i v_byte = _mm256_set1_epi32(current_byte);  // expensive broadcast
    // ... mask 8 floats
}
```

This means 8 loop iterations per block, each with a `_mm256_set1_epi32` broadcast (which involves a cross-lane shuffle). Processing 16 or 32 floats per iteration by extracting 2+ bytes at once would reduce iteration count and broadcast overhead.

**Estimated impact:** 1.3-1.5x on AVX2 full-estimate path.

### SIMD Issue 7: Commented-Out Prefetch Targeting Wrong Data (INFO)

**File:** `space.hpp:972`

```cpp
// _mm_prefetch(reinterpret_cast<const char*>(it_query + 128), _MM_HINT_T1);
```

This prefetches the **query** (which is hot in cache, reused across all expansions). What should be prefetched is the next node's **binary data** (cold, fetched from DRAM). This indicates prefetching was attempted but aimed at the wrong target.

### SIMD Issue 8: Eigen Delegation for Raw Distances (LOW)

**File:** `space.hpp:129-155`

```cpp
template <typename T>
inline T euclidean_sqr(const T* vec0, const T* vec1, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return (v0 - v1).dot(v0 - v1);  // computes (v0-v1) TWICE
}
```

Used during construction and centroid-distance precomputation. Eigen auto-vectorizes but:
- May compute `v0-v1` twice (depends on expression template optimization)
- No guaranteed multi-accumulator unrolling
- Runtime dimension prevents compile-time loop unrolling

Hand-written SIMD (like SVS's `L2FloatOp<16>` with 4 accumulators) would be reliably 1.2-1.5x faster.

### SIMD Issue 9: `ip_fxi` Uses Eigen for float×uint8 Inner Product (MODERATE for ex_bits=8)

**File:** `space.hpp:764-772`

```cpp
template <typename TF, typename TI>
inline TF ip_fxi(const TF* vec0, const TI* vec1, size_t dim) {
    ConstVectorMap<TF> v0(vec0, dim);
    ConstVectorMap<TI> v1(vec1, dim);
    return v0.dot(v1.template cast<TF>());
}
```

Used when `ex_bits == 8` (selected at `space.hpp:801`). Eigen's `cast<float>()` from `uint8_t` lacks a specialized SIMD conversion path — it falls back to scalar or suboptimal widening. The hand-written excode functions (`ip64_fxu2..7_avx`) all use explicit `_mm512_cvtepu8_epi32` + `_mm512_fmadd_ps`, which is 3 instructions per 16 elements vs Eigen's ~4 instructions per element.

**Replacement — `ip16_fxu8_avx` with multi-accumulator unrolling:**

```cpp
inline float ip16_fxu8_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ code, size_t dim
) {
    float result = 0.0F;
#if defined(__AVX512F__)
    const size_t unroll_stride = 64;  // 4 × 16 elements
    size_t i = 0;

    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    // Main loop: 4x unrolled, 4 independent accumulators
    for (; i + unroll_stride <= dim; i += unroll_stride) {
        __m128i c0 = _mm_loadu_si128((__m128i*)(code + i));
        __m128i c1 = _mm_loadu_si128((__m128i*)(code + i + 16));
        __m128i c2 = _mm_loadu_si128((__m128i*)(code + i + 32));
        __m128i c3 = _mm_loadu_si128((__m128i*)(code + i + 48));

        __m512 q0 = _mm512_loadu_ps(query + i);
        __m512 q1 = _mm512_loadu_ps(query + i + 16);
        __m512 q2 = _mm512_loadu_ps(query + i + 32);
        __m512 q3 = _mm512_loadu_ps(query + i + 48);

        sum0 = _mm512_fmadd_ps(q0, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c0)), sum0);
        sum1 = _mm512_fmadd_ps(q1, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c1)), sum1);
        sum2 = _mm512_fmadd_ps(q2, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c2)), sum2);
        sum3 = _mm512_fmadd_ps(q3, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c3)), sum3);
    }

    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));

    // Epilogue: handle remaining 16-element chunks
    for (; i + 16 <= dim; i += 16) {
        __m128i c = _mm_loadu_si128((__m128i*)(code + i));
        __m512 q = _mm512_loadu_ps(query + i);
        sum = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c)), sum);
    }

    result = _mm512_reduce_add_ps(sum);

    // Scalar tail (dim not multiple of 16)
    for (; i < dim; ++i) {
        result += query[i] * static_cast<float>(code[i]);
    }

#elif defined(__AVX2__)
    const size_t unroll_stride = 32;  // 4 × 8 elements
    size_t i = 0;

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    // Main loop: 4x unrolled
    for (; i + unroll_stride <= dim; i += unroll_stride) {
        // Load 8 uint8 values each, zero-extend to int32, convert to float
        __m128i raw0 = _mm_loadl_epi64((__m128i*)(code + i));
        __m128i raw1 = _mm_loadl_epi64((__m128i*)(code + i + 8));
        __m128i raw2 = _mm_loadl_epi64((__m128i*)(code + i + 16));
        __m128i raw3 = _mm_loadl_epi64((__m128i*)(code + i + 24));

        __m256 cf0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw0));
        __m256 cf1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw1));
        __m256 cf2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw2));
        __m256 cf3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw3));

        __m256 q0 = _mm256_loadu_ps(query + i);
        __m256 q1 = _mm256_loadu_ps(query + i + 8);
        __m256 q2 = _mm256_loadu_ps(query + i + 16);
        __m256 q3 = _mm256_loadu_ps(query + i + 24);

        sum0 = _mm256_fmadd_ps(q0, cf0, sum0);
        sum1 = _mm256_fmadd_ps(q1, cf1, sum1);
        sum2 = _mm256_fmadd_ps(q2, cf2, sum2);
        sum3 = _mm256_fmadd_ps(q3, cf3, sum3);
    }

    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    // Epilogue: handle remaining 8-element chunks
    for (; i + 8 <= dim; i += 8) {
        __m128i raw = _mm_loadl_epi64((__m128i*)(code + i));
        __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw));
        __m256 q = _mm256_loadu_ps(query + i);
        sum = _mm256_fmadd_ps(q, cf, sum);
    }

    // In-register horizontal reduction (no store-to-memory)
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ss(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    result = _mm_cvtss_f32(_mm_add_ss(lo, shuf));

    // Scalar tail
    for (; i < dim; ++i) {
        result += query[i] * static_cast<float>(code[i]);
    }

#else
    // Scalar fallback
    for (size_t i = 0; i < dim; ++i) {
        result += query[i] * static_cast<float>(code[i]);
    }
#endif
    return result;
}
```

Then update the dispatch in `select_excode_ipfunc`:
```cpp
if (ex_bits == 8) {
    return excode_ipimpl::ip16_fxu8_avx;  // was: ip_fxi
}
```

**Estimated impact:** 3-5x faster than Eigen's `ip_fxi` for `ex_bits=8`, due to:
- Explicit `_mm512_cvtepu8_epi32` / `_mm256_cvtepu8_epi32` (optimal uint8→float conversion)
- 4 independent accumulators (breaks FMA dependency chain)
- In-register horizontal reduction on AVX2 (no store-forwarding stall)
- No Eigen expression template overhead

### SIMD Summary Table

| Issue | Location | Fix | Estimated Impact |
|-------|----------|-----|-----------------|
| Single accumulator in excode IP | `space.hpp:324-761` (all 7 functions) | 2-4 independent accumulators + unroll | **1.5-3x** on excode IP |
| Single accumulator in `mask_ip_x0_q` | `space.hpp:953` (fix commented out!) | Uncomment 4-accumulator code | **1.3-2x** on full estimate |
| No outer unrolling in `warmup_ip_x0_q` | `warmup_space.hpp:62` | Process 16+ blocks with 2 acc pairs | **1.1-1.3x** on 1-bit estimate |
| Scalar `reverse_bits_u64` per block | `space.hpp:955` | Fix bit packing order or use BMI2 | **~1.1x** on full estimate |
| Store-to-array horizontal reduction | `space.hpp:308-316` | In-register SIMD reduction | **~1.05x** per AVX2 call |
| AVX2 byte-at-a-time in mask IP | `space.hpp:986-1003` | Process 16+ floats per step | **1.3-1.5x** on AVX2 full estimate |
| `reverse_bits` in transpose | `space.hpp:844` (marked TODO) | Change packing convention | **~1.05x** query init |
| Eigen for raw L2/IP | `space.hpp:129-155` | Hand-written SIMD with multi-acc | **1.2-1.5x** on construction |
| Eigen `ip_fxi` for ex_bits=8 | `space.hpp:764-772` | `ip16_fxu8_avx` with 4 accumulators | **3-5x** on ex_bits=8 IP |

---

## 9. Candidate Set / Search Buffer Comparison

Both libraries use essentially the **same data structure**: a sorted linear array with binary search insertion and memmove to shift elements. The differences are in the details.

### Side-by-Side Comparison

| Aspect | Khorgosh `SearchBuffer` | SVS `SearchBuffer` |
|--------|------------------------|---------------------|
| **File** | `utils/buffer.hpp` | `index/vamana/search_buffer.h` |
| **Storage** | `std::vector<AnnCandidate<T>>` with 64-byte aligned allocator | `std::vector<SearchNeighbor<Idx>>` with `CacheAlignedAllocator` |
| **Capacity** | `capacity + 1` (guard element) | `capacity + 1` (guard element) |
| **Insertion** | Binary search → `memmove` → write | Binary search → `copy_backward` → write |
| **Early reject** | `is_full(dist)`: checks `dist > top_dist()` | `can_skip(dist)`: checks `compare_(back(), dist) && full()` |
| **Pop / next** | `pop()`: returns ID, marks checked via top-bit flag, scans forward | `next()`: returns reference, marks visited, scans forward |
| **Done check** | `has_next()`: `cur_ < size_` | `done()`: `best_unvisited_ == min(size_, search_window_size_)` |
| **Visited tracking** | Top bit of PID used as "checked" flag | Separate `visited` bool in `SearchNeighbor` |
| **Visited filter** | None (separate `HashBasedBooleanSet`) | Optional built-in `VisitedFilter` (128KB direct-mapped) |
| **Duplicate detection** | None | Backward scan from insertion point to find same ID |
| **Sort** | Not needed (always sorted) | `sort()` method for post-rerank re-ordering |

### Key Differences

**1. SVS has a split window concept.** SVS's `SearchBuffer` has both `search_window_size_` and `capacity_`. The `done()` condition checks against `search_window_size_`, not `capacity_`. This means the buffer can hold MORE candidates than the search window — the extra overflow space stores candidates that might become relevant if better candidates appear. Khorgosh's buffer has a single `capacity_` that serves both purposes.

**2. SVS deduplicates on insert.** SVS scans backward from the insertion point to check if the same ID already exists at the same distance (lines 336-347). Khorgosh does not deduplicate — it can insert the same ID multiple times with different estimated distances. This is intentional for SymQG where multi-code quantization can give different estimates, but wastes buffer slots for HNSW.

**3. SVS integrates the visited filter.** The 128KB direct-mapped `VisitedFilter` is embedded inside the search buffer and can be prefetched/checked as a unit. Khorgosh uses a completely separate `HashBasedBooleanSet` with no integration.

**4. Both use the same O(1) early rejection.** Both check if the buffer is full and the new distance is worse than the worst element before doing any work. The logic is identical.

**5. Both use binary search + shift.** The insertion cost is O(log n) for the search + O(n) for the shift. For typical buffer sizes (ef=64-200), this is fast enough. Neither uses a heap — a sorted array gives O(1) access to the best unvisited and O(1) `done()` checks, which matter more in the hot loop.

### What Khorgosh Could Adopt from SVS

- **Split window/capacity**: Allow the buffer to hold more candidates than the search window. This can improve recall without increasing the number of node expansions.
- **Integrated visited filter**: Embedding a direct-mapped filter inside the buffer enables prefetching the filter slot alongside buffer operations.

---

## 10. Reranking Analysis

### How Reranking Works in Each Library

#### Khorgosh: Adaptive Multi-Stage Reranking

Khorgosh uses a **two-buffer architecture** with adaptive reranking during search:

- **`candidate_set`** (`SearchBuffer`, size = `ef`): Controls graph traversal. Holds candidates sorted by estimated distance, popped in order for neighbor expansion.
- **`boundedKNN`** (`BoundedKNN`, size = `TOPK`): Holds the final result. Only candidates that pass a lower-bound check get promoted into this.

The reranking flow per neighbor (from `searchBaseLayerST_AdaptiveRerankOpt`, hnsw.hpp:1229-1253):

```
1. Compute 1-bit estimate:    get_bin_est()        → est_dist, low_dist
2. Lower-bound check:         low_dist < distk?     (distk = worst in boundedKNN)
3. If promising, full estimate: get_full_est()      → refined est_dist using ex_bits
4. Insert into boundedKNN     (capacity = TOPK)
5. Always insert into candidate_set (capacity = ef) for graph traversal
```

**How many candidates get the full estimate?**
- ALL visited nodes get the 1-bit estimate (cheap: ~10-30 cycles via popcount)
- Only nodes where `low_dist < distk` OR `boundedKNN.size() < TOPK` get the full estimate (expensive: excode IP via FMA)
- As search progresses and `distk` tightens, fewer and fewer candidates pass the lower-bound check
- In practice, roughly **2-5x TOPK** candidates get the full estimate, though this varies by dataset and recall target

**Sizes:**
- `candidate_set` capacity = `ef` (user parameter, typically 64-200, always >= TOPK)
- `boundedKNN` capacity = `TOPK` (user parameter, typically 10-100)
- `ef` controls how many nodes the graph traversal explores
- `TOPK` controls how many final results are returned

**When `ex_bits == 0`:** No reranking happens. Every candidate goes directly into `boundedKNN` with only the 1-bit estimate. The lower-bound check is skipped entirely (line 1247-1253).

#### SVS (Two-Level LVQ): Post-Search Reranking

SVS uses a **single buffer** with post-search bulk reranking:

- **`search_buffer`** (`SearchBuffer`, size = `search_window_size`): Used for both graph traversal AND result collection.

The reranking flow (from `lvq.h:167-181`):

```
Phase 1: Full graph search using PRIMARY (4-bit compressed) vectors
          search_buffer fills up to search_window_size candidates

Phase 2: Rerank ALL candidates in the buffer
          for j = 0 to search_buffer.size():
              recompute distance using full reconstruction (primary + residual)
          sort buffer by new distances
```

**How many candidates get reranked?**
- Exactly `search_buffer.size()` candidates — which equals `min(num_visited, search_window_size)`
- In practice, this is almost always `search_window_size` (the buffer fills up quickly)
- `search_window_size` is typically 10-200, set by the user
- If `search_window_size < num_neighbors`, it's automatically increased to `num_neighbors`

**For one-level LVQ:** No reranking at all. The search uses the same (only) precision level throughout.

### Comparison

| Aspect | Khorgosh | SVS (Two-Level LVQ) |
|--------|----------|---------------------|
| **When reranking happens** | During search (adaptive, per-candidate) | After search (bulk, all at once) |
| **What gets reranked** | Only candidates passing `low_dist < distk` | ALL candidates in the buffer |
| **Number reranked** | ~2-5x TOPK (adaptive, depends on data) | = `search_window_size` (fixed) |
| **Rerank cost per vector** | ExCode IP (float × N-bit, FMA-based) | Full reconstruction (primary + residual → float, then L2) |
| **Cheap estimate cost** | Popcount (1-bit, ~10-30 cycles) | 4-bit compressed L2 (~30-100 cycles) |
| **Cheap/expensive ratio** | ~3-10x cheaper for cheap vs expensive | ~2-4x cheaper for cheap vs expensive |
| **Buffer architecture** | Two buffers (candidate_set + boundedKNN) | One buffer (dual-purpose) |
| **Result buffer size** | TOPK | search_window_size (>= num_neighbors) |
| **Traversal buffer size** | ef (>= TOPK) | search_window_size |

### Key Insight

Khorgosh's adaptive approach is theoretically better — it only pays the expensive reranking cost for candidates likely to be in the final result. SVS's bulk approach is simpler but wastes effort reranking candidates that won't make the final cut.

However, SVS's cheap estimate (4-bit LVQ) is more accurate than Khorgosh's 1-bit estimate, so SVS's search buffer is better-ordered before reranking. This means SVS's graph traversal is more efficient (fewer wasted hops), even though its reranking is less selective.

The practical trade-off: Khorgosh does less total reranking work, but SVS's per-node infrastructure (prefetching, visited filter, cache layout) is faster, often more than compensating.

---

## 11. Comparison with SVS at the Per-Node Level (Updated)

| Aspect | Khorgosh | SVS (LVQ) | Gap |
|--------|----------|-----------|-----|
| **Visited set** | Hash table + unordered_set overflow | 128KB direct-mapped filter | **2-3x** |
| **Prefetch lookahead** | 1 neighbor ahead | 4 neighbors ahead | **1.5-2x** |
| **Prefetch coverage** | Bin data only (2 cache lines) | Data + visited set + graph | **1.2x** |
| **Data/graph separation** | Interleaved in same memory block | Separate arrays | **1.1x** |
| **Search buffer** | No early rejection | O(1) skip check | **1.1x** |
| **Per-query allocations** | 5 vector allocations | Pre-allocated thread-local | **1.1x** |
| **SIMD unrolling** | Single accumulator pair | 4x unrolled accumulators | **1.1x** |
| **Dimension dispatch** | Runtime only | Compile-time for known dims | **1.05x** |
| **Distance compute** | Popcount (1-bit, very fast) | LVQ decompress+FMA (fast) | ~**1x** (RaBitQ wins) |
| **Function dispatch** | Function pointer (no inline) | Template (zero-cost) | **1.05x** |

**Estimated total gap:** Khorgosh leaves ~2-5x performance on the table compared to a fully-optimized implementation at the same node expansion count.

**Note:** RaBitQ's distance computation (popcount-based) is inherently faster per operation than LVQ's decompress+FMA. The data is also smaller (1 bit vs 4 bits per dimension), meaning less memory bandwidth per vector. Khorgosh should theoretically be faster than SVS in the distance kernel — the gap comes entirely from the surrounding infrastructure (visited set, prefetching, memory layout, allocations).

---

## 12. The Transpose-Query Optimization

The `transpose-query` branch rearranges the quantized query binary representation:

**Before:** `QueryBin[block_i * b_query + bit_j]` — for each block, all 4 query bits are contiguous.

**After:** `TransposedQueryBin[bit_j * num_blk + block_i]` — for each bit position, all blocks are contiguous.

**Impact in `warmup_ip_x0_q`:**
- **Before:** Required gather instructions (`_mm512_i64gather_epi64`) to collect the j-th bit across blocks.
- **After:** Simple sequential load (`_mm512_loadu_si512(&query[j * num_blk + i])`).

Gather instructions have ~10-20 cycle latency on Intel CPUs and can only execute on one port. A simple load is 4-7 cycles and can use both load ports. This is a ~2x improvement on the inner loop's load bandwidth, which translates to ~1.3-1.5x overall for the warmup kernel.

---

## 13. Summary

Khorgosh's RaBitQ quantization is algorithmically elegant — 1-bit codes with popcount give very cheap distance estimates. But the infrastructure around the hot path (visited set, prefetching, memory management) and SIMD utilization have significant room for improvement. The biggest gains would come from:

**Infrastructure (estimated 3-5x combined):**
1. **Direct-mapped visited filter** (replace HashBasedBooleanSet): ~2x
2. **Proper prefetch pipeline** (lookahead 4, cover data + visited + graph): ~1.5-2x
3. **Eliminate per-query allocations** (thread-local scratch): ~1.2x
4. **O(1) early rejection** in BoundedKNN: ~1.1x

**SIMD (estimated 1.5-3x combined on distance compute):**
5. **Multi-accumulator excode IP** (all 7 functions use single accumulator): 1.5-3x on excode
6. **Uncomment 4-accumulator `mask_ip_x0_q`** (fix is already written!): 1.3-2x on full estimate
7. **Outer-loop unrolling in `warmup_ip_x0_q`**: 1.1-1.3x on 1-bit estimate
8. **Eliminate `reverse_bits_u64`** from inner loop: ~1.1x on full estimate
9. **Fix AVX2 horizontal reduction**: ~1.05x per AVX2 call

These compound to an estimated **4-8x total improvement** without changing the algorithm or quantization scheme.
