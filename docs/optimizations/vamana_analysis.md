# Vamana Index Design Analysis - Intel ScalableVectorSearch (SVS)

## 1. Overview of Variants

SVS implements **three** Vamana variants:

| Variant | Class | Key File |
|---------|-------|----------|
| **Static Vamana** | `VamanaIndex<Graph, Data, Dist>` | `include/svs/index/vamana/index.h` |
| **Quantized Vamana (LVQ)** | Same class, with LVQ dataset + extensions | `include/svs/extensions/vamana/lvq.h` |
| **Dynamic Vamana** | `MutableVamanaIndex<Graph, Data, Dist>` | `include/svs/index/vamana/dynamic_index.h` |

All three share the same greedy search and pruning infrastructure. The differences lie in the data representation and the extension hooks.

---

## 2. Standard (Static) Vamana - Index Building

### Entry Point
The public API lives in `include/svs/orchestrators/vamana.h` (type-erased `Vamana` class). Under the hood, `VamanaBuilder` in `vamana_build.h` drives construction.

### Two-Pass Construction
The build runs **two full passes** over the dataset:

1. **Pass 1 (alpha = 1.0):** Builds initial graph connectivity with no distance relaxation. This gives a tight, greedy-optimal graph.
2. **Pass 2 (alpha = user-specified, e.g., 1.2 for L2):** Refines the graph with the MRNG heuristic using the relaxation parameter, allowing longer but more diverse edges.

This is consistent with the original DiskANN/Vamana paper's approach of double-pass construction for better recall.

### Per-Batch Processing
Each pass splits nodes into **~40+ batches** (minimum 40, or `N / (64*64)` batches, whichever is larger). Within each batch:

1. **Greedy search** from the entry point(s) using the current graph to find candidate neighbors for each node.
2. **Heuristic pruning** (MRNG rule) to select the final neighbor list.
3. **Reverse edge insertion**: For each new edge (u -> v), also try to add (v -> u). If v's degree is already at max, the overflow goes into a `BackedgeBuffer`.
4. **Backedge pruning**: All overflowed vertices get their neighbor lists re-pruned.

### Entry Point Selection
Uses **medioid** (the point closest to the dataset centroid), not a random point. Computed via `utils::find_medioid()`.

### Key Build Parameters
```
alpha:                    1.2 (L2) / 0.95 (IP, Cosine) - pruning relaxation
graph_max_degree:         Max neighbors per node (e.g., 64-128)
window_size:              Search window during construction (higher = better quality)
max_candidate_pool_size:  Cap on candidates before pruning
prune_to:                 Target degree after backedge pruning (<= graph_max_degree)
use_full_search_history:  If true, uses ALL visited nodes as candidates, not just the search buffer
```

---

## 3. Standard Vamana - Search

The search is a **greedy best-first traversal** implemented in `greedy_search.h`.

### Algorithm (lines 104-149 of greedy_search.h)
```
1. Initialize search buffer with entry point(s), compute their distances
2. Sort the buffer
3. While buffer has unvisited candidates:
   a. Pop the best unvisited node
   b. Get its adjacency list from the graph
   c. For each neighbor:
      - Skip if already visited (via visited set or bloom filter)
      - Prefetch next data items (lookahead)
      - Compute distance to query
      - Insert into search buffer (maintains sorted order)
4. Return top-k from buffer
```

### Search Buffer (`search_buffer.h`)
A cache-aligned, two-region data structure:
- **Active window**: The top `search_window_size` candidates, kept sorted
- **Overflow region**: Extra capacity for additional candidates

The `done()` method returns true when all candidates in the window have been visited.

### Visited Set Filter
An optional bloom-like filter with ~65K entries (128KB). Enabled for high-recall scenarios to avoid redundant distance computations. Supports prefetching of filter buckets.

---

## 4. Pruning Strategies - A Key Differentiator

SVS implements **three distinct pruning strategies**, selected automatically by distance type. This is a significant enhancement over standard Vamana.

### 4.1 ProgressivePruneStrategy (Default for L2)
**File:** `prune.h:178-234`

Instead of a boolean pruned/not-pruned state, tracks the **maximum distance ratio** seen for each candidate as a float. A candidate is pruned when:
```
max(candidate.distance / d(selected_neighbor, candidate)) >= current_alpha
```

The alpha relaxation increases progressively: `alpha^0, alpha^1, alpha^2, ...` This means:
- First pass (alpha^0 = 1.0): strict MRNG, only keep truly non-redundant edges
- Second pass (alpha^1 = 1.2): allow slightly redundant edges
- Third pass (alpha^2 = 1.44): allow even more redundant edges

This **progressive relaxation** is more nuanced than a single-threshold prune. It fills the neighbor list with the most valuable edges first, then progressively fills remaining slots with increasingly "redundant" but still useful edges.

### 4.2 IterativePruneStrategy (Default for IP/Cosine)
**File:** `prune.h:102-169`

Uses discrete states: `Available`, `Added`, `Pruned`. After each alpha level, all `Pruned` candidates are reset to `Available` for reconsideration. This is better suited for non-metric distance functions where the triangle inequality doesn't hold.

### 4.3 LegacyPruneStrategy
**File:** `prune.h:252-303`

Single-pass, simple boolean pruning. One scan through candidates with alpha-scaled MRNG rule. No iterative re-evaluation. Kept for backward compatibility.

### The MRNG Pruning Rule (All Strategies)
For a candidate `c` being considered as a neighbor of query `q`, with already-selected neighbor `n`:
```
Prune c if: alpha * d(n, c) < d(q, c)
```
Meaning: if an already-selected neighbor `n` is much closer to `c` than `q` is, then `c` is redundant because you can reach it through `n`.

---

## 5. Quantized Vamana (LVQ) - How It Differs

### What is LVQ?
**Locally-adaptive Vector Quantization** compresses each vector dimension into low-bit integers (typically 4 or 8 bits). Unlike product quantization (PQ), LVQ operates per-dimension with locally-adaptive scaling.

Core implementation: `include/svs/quantization/lvq/`

### Two Levels of Compression
LVQ supports **one-level** and **two-level** encoding:

- **One-level (primary only):** Each dimension quantized to N bits (e.g., 4-bit). Good compression, some accuracy loss.
- **Two-level (primary + residual):** Primary quantization plus a residual correction. The residual encodes the error from the primary quantization, giving better accuracy at the cost of more storage.

### How LVQ Integrates with Vamana

The integration is done through the **extension point system** (`extensions.h`), which uses tag-dispatched customization points (CPOs). The LVQ dataset hooks into the Vamana pipeline at multiple points:

#### During Build (`VamanaBuildAdaptor` in `lvq.h:56-136`)

```
VamanaBuildAdaptor<Distance, TwoLevel>:
  - graph_search_accessor:
      If TwoLevel: PrimaryAccessor (search using ONLY compressed primary vectors)
      If OneLevel: GetDatumAccessor (use fully decompressed vectors)
  - post_search_modify:
      If TwoLevel: Recompute distance using full reconstruction (primary + residual)
      If OneLevel: No modification needed
  - general_distance: Full precision for final pruning decisions
```

**Key insight:** During graph construction, the greedy search phase uses **compressed primary vectors** for speed, but the final neighbor selection uses **full-precision reconstructed vectors** for accuracy. This is a hybrid approach - fast traversal, accurate pruning.

#### During Search (`lvq.h:142-181`)

For **two-level** LVQ datasets, search is a two-phase process:

```
Phase 1 - Graph Traversal:
  Use PrimaryAccessor (compressed 4/8-bit vectors)
  Fast distance computations, small cache footprint
  Produces initial candidate set

Phase 2 - Reranking:
  For each candidate in the search buffer:
    Recompute distance using dataset.get_datum(id) (full reconstruction)
  Re-sort the buffer by new distances
  Return refined top-k
```

For **one-level** LVQ, there is no reranking phase since there's no higher-precision representation available.

#### Distance Computation
LVQ uses a `DecompressionAdaptor` that wraps the underlying distance function. It decompresses vectors on-the-fly during distance computation rather than storing decompressed copies, saving memory bandwidth.

### Prefetch Tuning for LVQ (`lvq.h:34-50`)
LVQ datasets get custom prefetch parameters:
- **4-bit Turbo strategy:** `{lookahead=3, step=1}` - aggressive, since data is tiny
- **4-bit Sequential:** `{lookahead=2, step=1}`
- **Default fallback:** `{lookahead=1, step=1}` - conservative

This is tuned because compressed vectors are much smaller per element, so more can be prefetched without cache thrashing.

---

## 6. Normal vs Quantized: Key Differences Summary

| Aspect | Normal Vamana | Quantized (LVQ) Vamana |
|--------|---------------|------------------------|
| **Vector storage** | Full precision (float32) | 4/8-bit per dimension |
| **Memory per vector** | `4 * dims` bytes | `bits * dims / 8` bytes (+ optional residual) |
| **Memory for 128-dim** | 512 bytes | 64 bytes (4-bit) or 128 bytes (8-bit) |
| **Graph search distance** | Full precision | Compressed primary only |
| **Build pruning** | Full precision | Full reconstruction (primary + residual) |
| **Search** | Single-phase | Two-phase (search + rerank) for two-level |
| **Distance computation** | Direct float ops | Decompress-on-the-fly via adaptor |
| **Prefetching** | Size-adaptive | Aggressively tuned for small data |
| **Accuracy** | Baseline | Slight loss (recoverable with reranking) |
| **Speed** | Baseline | Faster (smaller data = more cache hits) |

---

## 7. Why SVS Vamana Is Fast

### 7.1 Aggressive Prefetching (greedy_search.h:114-133)
The search loop uses a **software prefetcher** with two key behaviors:
- **Data prefetching:** While processing neighbor `i`, prefetch data for neighbor `i + lookahead`
- **Visited set prefetching:** Prefetch the bloom filter bucket for the next neighbor before checking the current one
- **Conditional prefetching:** Only prefetch data if the visited set says the node hasn't been seen (avoiding wasted prefetches)

The prefetch parameters are auto-tuned based on data element size:
- Small vectors (< 128B): lookahead=16, step=1 (aggressive)
- Large vectors (> 4KB): lookahead=0, step=0 (disabled)

### 7.2 Cache-Friendly Memory Layout
- **Graph storage** (`graph.h`): Packed adjacency lists in row-major layout. Each row is `[degree | neighbor_0 | neighbor_1 | ...]` - a single contiguous allocation. The degree is stored inline, so reading a node's neighbors is a single cache-line-aligned memory access.
- **Search buffer:** Uses `CacheAlignedAllocator` (64-byte alignment) to avoid false sharing in multi-threaded scenarios.
- **LVQ data:** 4-bit vectors are 8x smaller than float32, meaning 8x more vectors fit in L1/L2 cache during search.

### 7.3 Template-Based Zero-Cost Abstractions
Everything is templated: Graph type, Data type, Distance type, Accessor type. The compiler specializes each combination at compile time, eliminating virtual dispatch overhead. The extension point system uses `svs_invoke` with tag dispatch rather than virtual functions.

### 7.4 Efficient Visited Tracking
The bloom-like `VisitedFilter` uses 65K entries (128KB - fits in L2 cache) with direct hash indexing. The `emplace_visited` method atomically checks and marks a node as visited in one operation, avoiding redundant distance computations.

### 7.5 Batched Parallel Construction
- **Static partitioning** for search: each thread gets a contiguous chunk of nodes
- **Coarse-grained locking** for backedges: `BackedgeBuffer` uses ~1000-node buckets with one mutex per bucket, not per-node. This drastically reduces lock contention.
- **Per-vertex spinlocks** for direct edge insertion (only when the degree hasn't been exceeded)

### 7.6 LVQ-Specific Speed Gains
- **Two-phase search:** The graph traversal phase uses 4-bit compressed vectors. A 128-dim vector is only 64 bytes (fits in one cache line!) vs 512 bytes for float32. This means:
  - 8x fewer cache misses during the hot search loop
  - More of the dataset fits in LLC (last-level cache)
  - Memory bandwidth is the typical bottleneck in ANN search, and LVQ directly reduces it
- **Turbo strategy:** A specialized SIMD-friendly decompression path for 4-bit data
- **Decompression-on-the-fly:** No intermediate buffer allocation; the `DecompressionAdaptor` decompresses directly into the distance computation pipeline

### 7.7 Smart Build Decisions
- **Two-pass construction** produces higher-quality graphs than single-pass, meaning fewer hops are needed during search
- **Progressive pruning** (for L2) produces more diverse neighbor lists than simple threshold pruning, improving graph navigability
- **Full search history option:** Using ALL visited nodes during construction (not just the search buffer) produces better candidate pools for pruning

---

## 8. Architecture: The Extension Point System

The key architectural insight in SVS is the **extension point (CPO) system** in `extensions.h`. This allows different dataset types to customize the Vamana pipeline without modifying core code:

```
Extension Points:
  compute_entry_point     -> How to find the medioid (LVQ uses decompressor)
  estimate_prefetch_parameters -> Auto-tune prefetching for data size
  build_adaptor           -> Customize graph search vs general distance during build
  single_search_setup     -> Allocate search-time scratch (LVQ adapts distance)
  single_search           -> Override entire search (LVQ adds reranking phase)
  reconstruct_accessor    -> Custom decompression for result vectors
  UsesReranking           -> Query whether this dataset does reranking
```

This means adding a new quantization scheme (e.g., product quantization) would only require implementing these extension points - the core Vamana loop remains untouched.

---

## 9. Graph Storage Format

`SimpleGraph<Idx>` in `graph.h`:
- Row-major 2D array: `num_nodes x (max_degree + 1)`
- First element of each row stores the current degree
- Remaining elements store neighbor IDs
- Supports `uint32_t` or `uint16_t` index types
- `prefetch_node(id)` prefetches the entire adjacency list for a node

This is simpler than CSR format but wastes some space when nodes have fewer than `max_degree` neighbors. The trade-off is O(1) access to any node's neighbors without indirection through an offset array.

---

## 10. Summary

SVS's Vamana implementation is a heavily optimized, production-grade version of the DiskANN Vamana algorithm. Its speed comes from:

1. **Memory efficiency** (LVQ compresses vectors 4-8x, fitting more in cache)
2. **Prefetching** (software-driven, adaptive, covers both data and metadata)
3. **Zero-cost abstractions** (templates, not virtual dispatch)
4. **Smart algorithmic choices** (two-pass build, progressive pruning, two-phase search)
5. **Cache-aware data structures** (aligned allocators, packed graph, bloom filter in L2)
6. **Efficient parallelism** (coarse-grained locking, static partitioning, per-thread scratch)

The quantized variant adds a ~8x memory reduction with minimal accuracy loss, recovered through the two-phase search-then-rerank approach. The extension point architecture makes this modular and composable.
