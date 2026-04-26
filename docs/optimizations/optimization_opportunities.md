# HNSW + RaBitQ Optimization Opportunities

**Rating Scale:**
- **Impact**: 1-10 (10 = highest performance gain)
- **Difficulty**: 1-10 (10 = most complex to implement)
- **Priority Score**: Impact / Difficulty (higher is better)

---

## Quick Wins (High Impact, Low Difficulty)

### 1. Enable Faster Quantization by Default
**Impact:** 8 | **Difficulty:** 1 | **Priority:** 8.0

**Location:** `sample/hnsw_rabitq_indexing.cpp:86`

**Description:**
Currently, faster quantization is optional. The `faster_quant` flag uses precomputed constant scaling factors instead of computing them per-vector, significantly reducing quantization time.

**Implementation:**
```cpp
// Change from:
bool faster_quant = false;
// To:
bool faster_quant = true;  // Default enabled
```

**Expected Gain:** 30-50% faster index construction time

---

### 2. Optimize mask_ip_x0_q Function
**Impact:** 7 | **Difficulty:** 2 | **Priority:** 3.5

**Location:** `include/rabitqlib/index/estimator.hpp:150`

**Description:**
The function has a TODO comment indicating it needs optimization. This function is called during every single-vector full distance estimation.

**Implementation:**
- Add SIMD vectorization for bit operations
- Use AVX2 `_mm256_movemask_epi8` for efficient bit extraction
- Unroll loops for small dimension cases

**Expected Gain:** 10-15% faster query latency

---

### 3. Batch Neighbor Distance Computations
**Impact:** 6 | **Difficulty:** 2 | **Priority:** 3.0

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:786-813`

**Description:**
During graph construction, neighbor distances are computed one-by-one. Batching these computations allows for better cache utilization and SIMD opportunities.

**Implementation:**
- Prefetch multiple neighbor data points
- Compute distances in batches of 4-8
- Use SIMD for parallel distance computation

**Expected Gain:** 15-20% faster index construction

---

### 4. Cache-Line Align Quantized Data
**Impact:** 5 | **Difficulty:** 1 | **Priority:** 5.0

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:392-393`

**Description:**
Ensure all quantized data structures start on cache-line boundaries (64 bytes) to avoid false sharing and improve prefetch efficiency.

**Implementation:**
```cpp
// Add alignment attribute
alignas(64) char* data_level0_memory_;

// Use aligned allocation
data_level0_memory_ = static_cast<char*>(
    aligned_alloc(64, max_elements_ * size_data_per_element_)
);
```

**Expected Gain:** 5-8% faster query latency

---

### 5. Optimize SearchBuffer Insertion
**Impact:** 6 | **Difficulty:** 2 | **Priority:** 3.0

**Location:** `include/rabitqlib/utils/buffer.hpp:46-56`

**Description:**
The current implementation uses `memmove` for every insertion, which is expensive for small buffers.

**Implementation:**
- For buffers < 16 elements, use insertion sort with SIMD
- For larger buffers, use a min-heap instead of sorted array
- Add batch insertion mode

**Expected Gain:** 8-12% faster query latency

---

## Medium Effort Optimizations

### 6. Multi-Stage Distance Estimation
**Impact:** 8 | **Difficulty:** 4 | **Priority:** 2.0

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:1183-1267`

**Description:**
Current adaptive reranking uses 2 stages: bin_est → full_est. Add intermediate ex_est stage for better filtering.

**Implementation:**
```cpp
// Stage 1: Cheap bin_est for all candidates
get_bin_est(q_to_centroids, query_wrapper, candidate_id, candest);

// Stage 2: Medium-cost ex_est for promising candidates
if (candest.low_dist < threshold_1) {
    get_ex_est(q_to_centroids, query_wrapper, candidate_id, candest);
}

// Stage 3: Full cost full_est for very promising candidates
if (candest.low_dist < threshold_2) {
    get_full_est(q_to_centroids, query_wrapper, candidate_id, candest);
}
```

**Expected Gain:** 15-25% faster query latency

---

### 7. Hot/Cold Data Separation
**Impact:** 7 | **Difficulty:** 5 | **Priority:** 1.4

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:137-138`

**Description:**
Separate frequently accessed BinData from rarely accessed ExData to improve cache efficiency.

**Current Layout:**
```
edges | cluster_id | external_id | BinData | ExData
```

**Optimized Layout:**
```
Array 1 (Hot): edges | cluster_id | external_id | BinData
Array 2 (Cold): ExData (loaded only when needed)
```

**Expected Gain:** 10-15% faster query latency for ex_bits > 0

---

### 8. Aggressive Prefetching Tuning
**Impact:** 6 | **Difficulty:** 3 | **Priority:** 2.0

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:772-784, 1218-1225`

**Description:**
Current prefetching is basic. Optimize prefetch distance and add prefetching for additional data structures.

**Implementation:**
- Prefetch 2-3 neighbors ahead instead of 1
- Prefetch ExData when ex_bits > 0 and candidate is promising
- Prefetch neighbor lists for next iteration
- Use `_mm_prefetch` with different temporal locality hints

**Expected Gain:** 8-12% faster query latency

---

### 9. Vectorize Error Bound Computations
**Impact:** 5 | **Difficulty:** 3 | **Priority:** 1.67

**Location:** `include/rabitqlib/quantization/rabitq_impl.hpp:103-137`

**Description:**
Error bound computation involves square roots and divisions that can be vectorized.

**Implementation:**
- Use SIMD for dot products in error computation
- Approximate `sqrt` with `rsqrt` + Newton-Raphson
- Batch error computations for multiple vectors

**Expected Gain:** 5-10% faster query latency

---

### 10. Centroid Distance Caching
**Impact:** 6 | **Difficulty:** 3 | **Priority:** 2.0

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:1101-1126`

**Description:**
Query-to-centroid distances are recomputed for every query. Cache when processing batches.

**Implementation:**
```cpp
// Add cache structure
struct CentroidCache {
    std::vector<float> distances;
    const float* last_query = nullptr;
};

// Reuse if same query
if (cache.last_query == rotated_query) {
    q_to_centroids = cache.distances;
} else {
    // Compute and cache
}
```

**Expected Gain:** 5-8% faster batch query throughput

---

## High Effort, High Reward

### 11. NUMA-Aware Memory Allocation
**Impact:** 9 | **Difficulty:** 7 | **Priority:** 1.29

**Location:** Global memory allocation strategy

**Description:**
For multi-socket systems, allocate data structures local to threads that will access them.

**Implementation:**
- Use `numa_alloc_onnode()` for graph partitions
- Pin threads to NUMA nodes with `pthread_setaffinity_np`
- Interleave quantized vectors across nodes
- Replicate centroids on each NUMA node

**Expected Gain:** 20-40% faster query latency on multi-socket systems

---

### 12. Lock-Free Graph Updates
**Impact:** 8 | **Difficulty:** 8 | **Priority:** 1.0

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:621, 643, 868`

**Description:**
Replace mutex-based synchronization with lock-free data structures during construction.

**Implementation:**
- Use atomic operations for neighbor list updates
- Implement RCU for neighbor list modifications
- Use versioned pointers for lock-free reads
- Careful memory ordering with C++11 atomics

**Expected Gain:** 30-50% faster parallel index construction

---

### 13. Dynamic EF Tuning
**Impact:** 7 | **Difficulty:** 5 | **Priority:** 1.4

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:1088-1175`

**Description:**
Adaptively adjust ef during search based on result quality.

**Implementation:**
```cpp
size_t current_ef = initial_ef;
while (result_quality < target && current_ef < max_ef) {
    // Expand search with higher ef
    current_ef *= 1.5;
    continue_search(current_ef);
}
```

**Expected Gain:** 10-20% better recall/latency tradeoff

---

### 14. Cluster-Aware Graph Navigation
**Impact:** 8 | **Difficulty:** 6 | **Priority:** 1.33

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:1183-1267`

**Description:**
Bias graph navigation toward candidates in nearby clusters.

**Implementation:**
- Precompute cluster-to-cluster distances
- Add cluster proximity bonus to candidate scores
- Use cluster hierarchy for pruning
- Implement cluster-based entry point selection

**Expected Gain:** 15-25% faster query latency with better recall

---

### 15. Adaptive Bit Allocation
**Impact:** 9 | **Difficulty:** 8 | **Priority:** 1.13

**Location:** `include/rabitqlib/quantization/rabitq_impl.hpp`

**Description:**
Allocate more quantization bits to high-variance dimensions.

**Implementation:**
- Analyze dimension-wise variance during training
- Allocate bits proportional to variance
- Use PCA to concentrate variance in first dimensions
- Implement non-uniform quantization scheme

**Expected Gain:** 20-30% better accuracy at same bit budget

---

## Advanced Optimizations

### 16. Query Batching with Shared LUT
**Impact:** 7 | **Difficulty:** 6 | **Priority:** 1.17

**Location:** `include/rabitqlib/fastscan/fastscan.hpp:232-245`

**Description:**
When processing multiple queries, reuse computational results.

**Implementation:**
- Batch rotate multiple queries together
- Share centroid distance computations
- Vectorize LUT construction across queries
- Process queries in SIMD width batches

**Expected Gain:** 25-40% faster batch query throughput

---

### 17. Compact Graph Representation
**Impact:** 6 | **Difficulty:** 7 | **Priority:** 0.86

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:269-281`

**Description:**
Use smaller data types for graph structure when possible.

**Implementation:**
```cpp
// For datasets < 16M points
using PID_Compact = uint24_t;  // Custom 24-bit type

// Pack edge count with first edge
struct CompactNeighborList {
    uint8_t count : 8;
    uint24_t first_neighbor : 24;
    uint24_t neighbors[];
};
```

**Expected Gain:** 20-30% memory reduction, 5-10% faster due to cache

---

### 18. SIMD Binary Search in SearchBuffer
**Impact:** 5 | **Difficulty:** 6 | **Priority:** 0.83

**Location:** `include/rabitqlib/utils/buffer.hpp:21-31`

**Description:**
Use SIMD for parallel comparisons during binary search.

**Implementation:**
- For buffers of 16+ elements, use AVX2 for 8-way parallel compare
- Use `_mm256_cmpgt_ps` and `_mm256_movemask_ps`
- Vectorize the memmove operation

**Expected Gain:** 5-8% faster query latency

---

### 19. Optimize Priority Queue in best_rescale_factor
**Impact:** 6 | **Difficulty:** 5 | **Priority:** 1.2

**Location:** `include/rabitqlib/quantization/rabitq_impl.hpp:295-330`

**Description:**
The priority queue can be optimized for the specific use case.

**Implementation:**
- Use fixed-size array with linear scan for dim < 128
- Add early termination when max_ip plateaus
- Cache previous optimization results for similar vectors
- Use approximate optimization for faster_quant mode

**Expected Gain:** 10-15% faster quantization (construction time)

---

### 20. Thread-Local Visited Lists
**Impact:** 5 | **Difficulty:** 4 | **Priority:** 1.25

**Location:** `include/rabitqlib/index/hnsw/hnsw.hpp:1192, 742`

**Description:**
Use thread-local storage for visited lists to avoid pool synchronization.

**Implementation:**
```cpp
thread_local HashBasedBooleanSet visited_list;

// No need to get from pool
// visited_list.reset() instead of get/release
```

**Expected Gain:** 3-5% faster parallel query throughput

---

## Ranked Summary (by Priority Score)

| Rank | Optimization | Impact | Difficulty | Priority | Quick Win |
|------|-------------|--------|------------|----------|-----------|
| 1 | Enable Faster Quantization by Default | 8 | 1 | 8.00 | ✅ |
| 2 | Cache-Line Align Quantized Data | 5 | 1 | 5.00 | ✅ |
| 3 | Optimize mask_ip_x0_q Function | 7 | 2 | 3.50 | ✅ |
| 4 | Batch Neighbor Distance Computations | 6 | 2 | 3.00 | ✅ |
| 5 | Optimize SearchBuffer Insertion | 6 | 2 | 3.00 | ✅ |
| 6 | Multi-Stage Distance Estimation | 8 | 4 | 2.00 | |
| 7 | Aggressive Prefetching Tuning | 6 | 3 | 2.00 | |
| 8 | Centroid Distance Caching | 6 | 3 | 2.00 | |
| 9 | Vectorize Error Bound Computations | 5 | 3 | 1.67 | |
| 10 | Hot/Cold Data Separation | 7 | 5 | 1.40 | |
| 11 | Dynamic EF Tuning | 7 | 5 | 1.40 | |
| 12 | Cluster-Aware Graph Navigation | 8 | 6 | 1.33 | |
| 13 | NUMA-Aware Memory Allocation | 9 | 7 | 1.29 | |
| 14 | Thread-Local Visited Lists | 5 | 4 | 1.25 | |
| 15 | Optimize Priority Queue | 6 | 5 | 1.20 | |
| 16 | Query Batching with Shared LUT | 7 | 6 | 1.17 | |
| 17 | Adaptive Bit Allocation | 9 | 8 | 1.13 | |
| 18 | Lock-Free Graph Updates | 8 | 8 | 1.00 | |
| 19 | Compact Graph Representation | 6 | 7 | 0.86 | |
| 20 | SIMD Binary Search | 5 | 6 | 0.83 | |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 weeks)
1. Enable Faster Quantization by Default
2. Cache-Line Align Quantized Data
3. Optimize mask_ip_x0_q Function
4. Batch Neighbor Distance Computations
5. Optimize SearchBuffer Insertion

**Expected Total Gain:** 40-60% overall performance improvement

---

### Phase 2: Medium Effort (3-4 weeks)
6. Multi-Stage Distance Estimation
7. Aggressive Prefetching Tuning
8. Centroid Distance Caching
9. Vectorize Error Bound Computations
10. Thread-Local Visited Lists

**Expected Total Gain:** Additional 25-35% improvement

---

### Phase 3: High Impact, High Effort (6-8 weeks)
11. Hot/Cold Data Separation
12. Dynamic EF Tuning
13. Cluster-Aware Graph Navigation
14. Optimize Priority Queue
15. Query Batching with Shared LUT

**Expected Total Gain:** Additional 30-50% improvement

---

### Phase 4: Advanced (Ongoing)
16. NUMA-Aware Memory Allocation
17. Lock-Free Graph Updates
18. Adaptive Bit Allocation
19. Compact Graph Representation
20. SIMD Binary Search

**Expected Total Gain:** Additional 40-80% improvement in specific scenarios

---

## Performance Testing Recommendations

For each optimization:

1. **Benchmark Before:** Record baseline metrics
   - Index construction time
   - Query latency (p50, p95, p99)
   - Memory usage
   - Recall@10, Recall@100

2. **Isolate Changes:** Implement one optimization at a time

3. **Test Across Datasets:**
   - Small (100K vectors, 128 dim)
   - Medium (1M vectors, 256 dim)
   - Large (10M+ vectors, 512+ dim)

4. **Profile Hotspots:**
   - Use `perf` for CPU profiling
   - Use `valgrind --tool=cachegrind` for cache analysis
   - Monitor cache misses, branch mispredictions

5. **Regression Testing:** Ensure accuracy metrics don't degrade

---

## Notes

- **Quick Wins** should be prioritized first - they provide significant gains with minimal risk
- **NUMA optimizations** only apply to multi-socket systems
- **Lock-free optimizations** are complex but crucial for high-concurrency scenarios
- **Adaptive techniques** (dynamic EF, bit allocation) require careful tuning but offer best long-term gains
- Many optimizations are **complementary** and can be combined for multiplicative effects

---

## References

- Current codebase: `/Users/mohammadtamimul.ehsan/Documents/personal/khorgosh`
- Key files analyzed:
  - `include/rabitqlib/index/hnsw/hnsw.hpp`
  - `include/rabitqlib/quantization/rabitq.hpp`
  - `include/rabitqlib/fastscan/fastscan.hpp`
  - `sample/hnsw_rabitq_indexing.cpp`
  - `sample/hnsw_rabitq_querying.cpp`
