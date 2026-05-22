# Python API Design for RaBitQLib

**Goal:** Define the exact Python interfaces exposed to users via `rabitqlib`

---

## Design Principles

1. **Match what users expect** — follow the patterns of hnswlib and faiss that Python ANN users already know
2. **Hide C++ complexity** — users shouldn't need to know about padded dimensions, rotators, or batch layouts
3. **NumPy in, NumPy out** — all data goes in as `np.ndarray`, results come back as `np.ndarray`
4. **Separate indexing from searching** — HNSW/IVF require preprocessing (clustering), SymQG does not
5. **Consistent API across index types** — same method names and patterns for HNSW, IVF, SymQG

---

## C++ API Surface (What We're Wrapping)

### HierarchicalNSW (HNSW + RaBitQ)

| C++ Method | Signature | Python Wrapper | Notes |
|------------|-----------|----------------|-------|
| Constructor | `HierarchicalNSW(max_elements, dim, total_bits, M, ef_construction, random_seed, metric)` | `HnswIndex(dim, max_elements, M, ef_construction, nbits, metric, random_seed)` | Reordered params — dim first for consistency |
| `construct()` | `construct(num_clusters, centroids*, num_points, data*, cluster_ids*, num_threads, faster)` | `HnswIndex.build(data, centroids, cluster_ids, num_threads, fast_quantization)` | `faster` → `fast_quantization` for clarity |
| `search()` | `search(queries*, nq, k, ef, num_threads) → vector<vector<pair<float, PID>>>` | `HnswIndex.search(queries, k, ef, num_threads) → (ids, distances)` | Returns numpy tuple instead of vector of pairs |
| `save()` | `save(filename)` | `HnswIndex.save(path)` | |
| `load()` | `load(filename, metric_type)` | `HnswIndex.load(path)` classmethod | Metric stored in file; returns new instance instead of mutating |

### IVF (IVF + RaBitQ)

| C++ Method | Signature | Python Wrapper | Notes |
|------------|-----------|----------------|-------|
| Constructor | `IVF(n, dim, num_clusters, total_bits, metric, rotator)` | `IvfIndex(dim, max_elements, num_clusters, nbits, metric)` | `n` → `max_elements` for consistency with HnswIndex; `rotator` hidden — always FhtKac |
| `construct()` | `construct(data*, centroids*, cluster_ids*, faster)` | `IvfIndex.build(data, centroids, cluster_ids, fast_quantization)` | |
| `search()` | `search(query*, k, nprobe, results*, high_accuracy)` | `IvfIndex.search(queries, k, nprobe, high_accuracy, num_threads) → (ids, distances)` | Single-query C++ → batched Python with threading |
| `save()` | `save(filename)` | `IvfIndex.save(path)` | |
| `load()` | `load(filename)` | `IvfIndex.load(path)` classmethod | |

### QuantizedGraph (SymQG)

| C++ Method | Signature | Python Wrapper | Notes |
|------------|-----------|----------------|-------|
| Constructor | `QuantizedGraph(num, dim, max_degree, metric, rotator)` | `SymqgIndex(dim, max_degree, metric)` | `num` inferred from data; `rotator` hidden |
| `QGBuilder` | `QGBuilder(qg, ef_build, data*, num_threads).build()` | `SymqgIndex.build(data, ef_construction, num_threads)` | Builder class hidden behind single method |
| `search()` + `set_ef()` | `set_ef(ef)` then `search(query*, k, results*)` | `SymqgIndex.search(queries, k, ef, num_threads) → (ids, distances)` | `set_ef()` merged into `search(ef=)` param; single-query → batched |
| `save()` | `save(filename)` | `SymqgIndex.save(path)` | |
| `load()` | `load(filename)` | `SymqgIndex.load(path)` classmethod | |

### RaBitQ Quantization (`rabitqlib::quant`)

| C++ Function | Signature | Python Wrapper | Notes |
|--------------|-----------|----------------|-------|
| No direct equivalent | N/A | `RaBitQuantizer(dim, nbits, metric, fast)` constructor | Python-only wrapper class — holds dim, nbits, metric, and internally calls `faster_config()` when `fast=True` |
| `faster_config()` | `faster_config(dim, total_bits) → RabitqConfig` | Called internally by `RaBitQuantizer.__init__(fast=True)` | Hidden — no standalone function needed |
| `quantize_compact_one_bit()` | `quantize_compact_one_bit(data*, centroid*, padded_dim, bin_data*, metric)` | `RaBitQuantizer.quantize(data, centroids)` when `nbits=1` | Merged into unified `quantize()` |
| `quantize_compact_ex_bits()` | `quantize_compact_ex_bits(data*, centroid*, padded_dim, ex_bits, ex_data*, metric, config)` | `RaBitQuantizer.quantize(data, centroids)` when `nbits>1` | Merged into unified `quantize()` |
| `quantize_full_single()` | `quantize_full_single(data*, [centroid*,] dim, total_bits, total_code*, f_add, f_rescale, f_error, metric, config)` | `RaBitQuantizer.quantize(data, centroids) → QuantizedResult` | Out-params → returned `QuantizedResult` object |
| `quantize_scalar()` | `quantize_scalar(data*, [centroid*,] dim, total_bits, total_code*, delta, vl, config, scalar_type)` | `RaBitQuantizer.quantize_scalar(data, centroids, mode) → ScalarQuantizedResult` | `scalar_type` enum → `mode` string (`'reconstruction'`/`'unbiased'`/`'plain'`) |
| `reconstruct_vec()` | `reconstruct_vec(quantized_vec*, delta, vl, dim, results*)` | `RaBitQuantizer.reconstruct(scalar_result) → np.ndarray` | Takes `ScalarQuantizedResult` directly instead of separate args |
| `quantize_one_batch()` | `quantize_one_batch(data*, centroid*, num, padded_dim, batch_data*, metric)` | Not exposed | Internal fastscan layout — used by index types only |
| `quantize_split_batch()` | `quantize_split_batch(data*, centroid*, num, padded_dim, ex_bits, batch_data*, ex_data*, metric, config)` | Not exposed | Internal fastscan layout — used by index types only |

### Distance Estimation (`rabitqlib::estimator`)

| C++ Function | Signature | Python Wrapper | Notes |
|--------------|-----------|----------------|-------|
| `split_single_estdist()` | `split_single_estdist(bin_data*, q_obj, padded_dim, ip_x0_qr, est_dist, low_dist, g_add, g_error)` | `RaBitQuantizer.estimate_distance_coarse(quantized_result, queries) → (distances, lower_bounds)` | 1-bit only distance estimation — fast but coarse. Works even when `nbits > 1` by using only the 1-bit base codes. Returns estimated distances and error lower bounds. |
| `split_distance_boosting()` | `split_distance_boosting(ex_data*, ip_func, q_obj, padded_dim, ex_bits, ip_x0_qr) → float` | `RaBitQuantizer.refine_distance(quantized_result, queries, coarse_intermediates) → distances` | Refines 1-bit estimates using extra-bits codes. Takes intermediate results from `estimate_distance_coarse()` and produces a more accurate distance. |
| `split_single_fulldist()` | `split_single_fulldist(bin_data*, ex_data*, ip_func, q_obj, padded_dim, ex_bits, est_dist, low_dist, ip_x0_qr, g_add, g_error)` | `RaBitQuantizer.estimate_distance(quantized_result, queries) → (distances, lower_bounds)` | Full multi-bit distance estimation in one pass (1-bit + extra-bits combined). |
| `split_batch_estdist()` | `split_batch_estdist(batch_data*, q_obj, padded_dim, est_distance*, low_distance*, ip_x0_qr*, high_accuracy)` | Used internally by `IvfIndex.search()` | Batch 1-bit estimation with fastscan — internal to IVF |
| `full_est_dist()` | `full_est_dist(quantized_vec*, query*, ip_func, dim, bits, f_add, f_rescale, g_add, k1xsumq) → float` | Used internally by `estimate_distance()` | Low-level distance formula — wrapped by higher-level methods |
| `qg_batch_estdist()` | `qg_batch_estdist(batch_data*, q_obj, padded_dim, est_distance*)` | Used internally by `SymqgIndex.search()` | SymQG-specific batch estimation — internal |

### Rotator (`rabitqlib::Rotator<T>` / `rabitqlib::rotator_impl`)

| C++ Method | Signature | Python Wrapper | Notes |
|------------|-----------|----------------|-------|
| `choose_rotator()` | `choose_rotator(dim, type, padded_dim) → Rotator<T>*` | `Rotator(dim, method)` constructor | Factory hidden — `method` string (`'fht_kac'`/`'matrix'`) replaces `RotatorType` enum |
| `Rotator::rotate()` | `rotate(src*, dst*)` | `Rotator.rotate(data) → np.ndarray` | Single-vector → batched; allocates output internally |
| `Rotator::save()` | `save(ofstream&)` | `Rotator.save(path)` | File path string instead of stream |
| `Rotator::load()` | `load(ifstream&)` | `Rotator.load(path)` classmethod | Returns new instance |
| `Rotator::size()` | `size() → size_t` | `Rotator.padded_dim` property | Renamed for clarity |
| `padding_requirement()` | `padding_requirement(dim, type) → size_t` | Not exposed | Internal — `padded_dim` property is sufficient |

### Supporting Types

| C++ Type | Header | Python Equivalent | Notes |
|----------|--------|-------------------|-------|
| `RabitqConfig` | `rabitq.hpp` | Not exposed | Hidden behind `fast=True` param |
| `MetricType` | `defines.hpp` | `rabitqlib.MetricType` enum + `'l2'`/`'ip'` strings | String accepted everywhere; enum available for explicit use |
| `ScalarQuantizerType` | `defines.hpp` | `mode` string: `'reconstruction'`/`'unbiased'`/`'plain'` | String instead of enum |
| `RotatorType` | `rotator.hpp` | `method` string: `'fht_kac'`/`'matrix'` | String instead of enum |
| `BatchDataMap<T>` | `data_layout.hpp` | Not exposed | Internal memory layout |
| `ExDataMap<T>` | `data_layout.hpp` | Not exposed | Internal memory layout |
| `BinDataMap<T>` | `data_layout.hpp` | Not exposed | Internal memory layout |
| `SplitBatchQuery<T>` | `query.hpp` | Not exposed | Internal query object for batch 1-bit estimation |
| `SplitSingleQuery<T>` | `query.hpp` | Not exposed | Internal query object for single-vector estimation |
| `BatchQuery<T>` | `query.hpp` | Not exposed | Internal query object for SymQG batch estimation |

### Result Objects

#### `QuantizedResult`

Returned by `RaBitQuantizer.quantize()`. Contains the quantized codes and the distance estimation factors needed by `estimate_distance()`, `estimate_distance_coarse()`, and `refine_distance()`.

| Attribute | C++ Name | Type | Shape | Description |
|-----------|----------|------|-------|-------------|
| `codes` | `total_code` / `bin_data` + `ex_data` | `np.ndarray`, uint8 | `(n, code_size)` | Packed binary quantization codes |
| `add_factor` | `f_add` | `np.ndarray`, float32 | `(n,)` | Additive factor for distance estimation |
| `scale_factor` | `f_rescale` | `np.ndarray`, float32 | `(n,)` | Multiplicative factor for distance estimation |
| `error_factor` | `f_error` | `np.ndarray`, float32 | `(n,)` | Error bound factor (used to compute distance lower bounds) |
| `nbits` | `total_bits` | `int` | — | Number of bits per dimension used |
| `has_extra_bits` | `ex_bits > 0` | `bool` | — | `True` if `nbits > 1` (extra-bits codes are present) |

#### `ScalarQuantizedResult`

Returned by `RaBitQuantizer.quantize_scalar()`. Contains per-dimension integer codes and the parameters needed to reconstruct the original vectors via `RaBitQuantizer.reconstruct()`.

| Attribute | C++ Name | Type | Shape | Description |
|-----------|----------|------|-------|-------------|
| `codes` | `total_code` | `np.ndarray`, uint8 | `(n, dim)` | Per-dimension quantized codes in `[0, 2^bits - 1]` |
| `delta` | `delta` | `np.ndarray`, float32 | `(n,)` | Bin width per vector (reconstruction: `value = code * delta + lower_bound`) |
| `lower_bound` | `vl` | `np.ndarray`, float32 | `(n,)` | Lower bound per vector |

#### `CoarseIntermediates`

Returned as part of `estimate_distance_coarse()` output. Holds intermediate values that `refine_distance()` needs to refine 1-bit estimates with extra-bits codes.

| Attribute | C++ Name | Type | Shape | Description |
|-----------|----------|------|-------|-------------|
| `ip_x0_qr` | `ip_x0_qr` | `np.ndarray`, float32 | `(nq, n)` | Intermediate inner product terms from 1-bit estimation |

---

## Proposed Python API

### Overview

```python
import rabitqlib
import numpy as np

# --- HNSW ---
index = rabitqlib.HnswIndex(dim=128, max_elements=100000, M=32,
                            ef_construction=200, nbits=5, metric='l2')
index.build(data, centroids, cluster_ids, num_threads=8)
ids, distances = index.search(queries, k=10, ef=100, num_threads=4)
index.save("hnsw.index")

index2 = rabitqlib.HnswIndex.load("hnsw.index")

# --- IVF ---
index = rabitqlib.IvfIndex(dim=128, max_elements=100000, num_clusters=1000,
                           nbits=5, metric='l2')
index.build(data, centroids, cluster_ids)
ids, distances = index.search(queries, k=10, nprobe=50)
index.save("ivf.index")

index2 = rabitqlib.IvfIndex.load("ivf.index")

# --- SymQG ---
index = rabitqlib.SymqgIndex(dim=128, max_degree=64, metric='l2')
index.build(data, ef_construction=300, num_threads=8)
ids, distances = index.search(queries, k=10, ef=100)
index.save("symqg.index")

index2 = rabitqlib.SymqgIndex.load("symqg.index")

# --- Utilities ---
centroids, cluster_ids = rabitqlib.cluster(data, num_clusters=1000, metric='l2')
recall = rabitqlib.recall_at_k(predicted_ids, ground_truth_ids, k=10)
```

---

### Detailed API: `rabitqlib.HnswIndex`

```python
class HnswIndex:
    """HNSW graph index with RaBitQ quantization.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    max_elements : int
        Maximum number of vectors the index can hold.
    M : int, default=16
        Number of bi-directional links per node per layer.
        Higher M → better recall, more memory, slower build.
    ef_construction : int, default=200
        Search width during construction. Higher → better graph quality, slower build.
    nbits : int, default=5
        Bits per dimension for RaBitQ quantization (1-9).
        1 = fastest/least accurate, 9 = slowest/most accurate.
    metric : str, default='l2'
        Distance metric: 'l2' (Euclidean) or 'ip' (inner product).
    random_seed : int, default=100
        Random seed for reproducibility.
    """

    def __init__(self, dim, max_elements, M=16, ef_construction=200,
                 nbits=5, metric='l2', random_seed=100): ...

    def build(self, data, centroids, cluster_ids,
              num_threads=0, fast_quantization=True):
        """Build the index: add vectors, quantize, and construct the HNSW graph.

        Parameters
        ----------
        data : np.ndarray, shape (n, dim), dtype float32
            Vectors to index.
        centroids : np.ndarray, shape (num_clusters, dim), dtype float32
            Cluster centroids (e.g., from k-means).
        cluster_ids : np.ndarray, shape (n,), dtype uint32
            Cluster assignment for each vector.
        num_threads : int, default=0
            Number of build threads. 0 = use all available cores.
        fast_quantization : bool, default=True
            Use faster (but slightly less accurate) quantization.
        """

    def search(self, queries, k=10, ef=None, num_threads=1):
        """Search for nearest neighbors.

        Parameters
        ----------
        queries : np.ndarray, shape (nq, dim), dtype float32
            Query vectors.
        k : int, default=10
            Number of nearest neighbors to return.
        ef : int, optional
            Search width. Higher ef → better recall, slower search.
            Default: max(k, 10).
        num_threads : int, default=1
            Number of search threads. Single-threaded by default since
            individual queries are latency-sensitive; set higher for
            throughput-oriented batch workloads. 0 = all available cores.

        Returns
        -------
        ids : np.ndarray, shape (nq, k), dtype uint32
            Indices of nearest neighbors.
        distances : np.ndarray, shape (nq, k), dtype float32
            Distances to nearest neighbors (sorted ascending for L2).
        """

    def save(self, path):
        """Save index to disk."""

    @classmethod
    def load(cls, path):
        """Load a saved index from disk.

        Metric type is read from the file (stored during save).

        Returns
        -------
        index : HnswIndex
            Loaded index ready for search.
        """

    # --- Properties ---
    @property
    def dim(self) -> int: ...

    @property
    def max_elements(self) -> int: ...

    @property
    def is_built(self) -> bool: ...

    def __repr__(self) -> str:
        """e.g. HnswIndex(dim=128, max_elements=100000, M=32, nbits=5, metric='l2', is_built=True)"""
```

**Usage:**
```python
import numpy as np
from sklearn.cluster import KMeans
import rabitqlib

# Prepare data
data = np.random.randn(100000, 128).astype('float32')

# Cluster (required for HNSW + RaBitQ)
km = KMeans(n_clusters=1000, n_init=1).fit(data)
centroids = km.cluster_centers_.astype('float32')
cluster_ids = km.labels_.astype('uint32')

# Build index
index = rabitqlib.HnswIndex(dim=128, max_elements=100000, M=32, nbits=5)
index.build(data, centroids, cluster_ids, num_threads=8)

# Search
queries = np.random.randn(100, 128).astype('float32')
ids, dists = index.search(queries, k=10, ef=100)

# Save / Load
index.save("my_index.bin")
index2 = rabitqlib.HnswIndex.load("my_index.bin")
```

---

### Detailed API: `rabitqlib.IvfIndex`

```python
class IvfIndex:
    """Inverted file index with RaBitQ quantization.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    max_elements : int
        Maximum number of vectors the index can hold.
    num_clusters : int
        Number of IVF clusters (partitions).
    nbits : int, default=5
        Bits per dimension for RaBitQ quantization (1-9).
    metric : str, default='l2'
        Distance metric: 'l2' or 'ip'.
    """

    def __init__(self, dim, max_elements, num_clusters, nbits=5, metric='l2'): ...

    def build(self, data, centroids, cluster_ids, fast_quantization=True):
        """Build the index: add vectors, quantize, and construct the IVF structure.

        Parameters
        ----------
        data : np.ndarray, shape (n, dim), dtype float32
        centroids : np.ndarray, shape (num_clusters, dim), dtype float32
        cluster_ids : np.ndarray, shape (n,), dtype uint32
        fast_quantization : bool, default=True
        """

    def search(self, queries, k=10, nprobe=10, high_accuracy=True, num_threads=1):
        """Search for nearest neighbors.

        Parameters
        ----------
        queries : np.ndarray, shape (nq, dim), dtype float32
            Query vectors.
        k : int, default=10
            Number of nearest neighbors.
        nprobe : int, default=10
            Number of clusters to probe. Higher → better recall, slower.
        high_accuracy : bool, default=True
            Use high-accuracy fastscan (int32 accumulation).
            Set False for faster but less accurate search.
        num_threads : int, default=1
            Number of search threads. Single-threaded by default since
            individual queries are latency-sensitive; set higher for
            throughput-oriented batch workloads. 0 = all available cores.

        Returns
        -------
        ids : np.ndarray, shape (nq, k), dtype uint32
        distances : np.ndarray, shape (nq, k), dtype float32
        """

    def save(self, path): ...

    @classmethod
    def load(cls, path):
        """Load saved IVF index. Metric type is stored in the file."""

    @property
    def dim(self) -> int: ...

    @property
    def max_elements(self) -> int: ...

    @property
    def num_clusters(self) -> int: ...

    @property
    def is_built(self) -> bool: ...

    def __repr__(self) -> str:
        """e.g. IvfIndex(dim=128, max_elements=100000, num_clusters=1000, nbits=5, metric='l2', is_built=True)"""
```

**Note on IVF search:** The C++ `IVF::search()` is single-query. The Python wrapper will batch queries internally:

```cpp
// In ivf_bindings.cpp
.def("search_batch",
     [](IVF& self, py::array_t<float> queries, size_t k, size_t nprobe, bool high_accuracy) {
         auto q = queries.request();
         size_t nq = q.shape[0], dim = q.shape[1];
         py::array_t<uint32_t> out_ids({nq, k});
         auto id_ptr = out_ids.mutable_data();
         py::gil_scoped_release release;
         #pragma omp parallel for
         for (size_t i = 0; i < nq; i++) {
             self.search((float*)q.ptr + i * dim, k, nprobe,
                         id_ptr + i * k, high_accuracy);
         }
         return out_ids;
     })
```

**Usage:**
```python
index = rabitqlib.IvfIndex(dim=128, max_elements=100000, num_clusters=1000, nbits=5)
index.build(data, centroids, cluster_ids)
ids, dists = index.search(queries, k=10, nprobe=50)
```

---

### Detailed API: `rabitqlib.SymqgIndex`

```python
class SymqgIndex:
    """SymphonyQG graph index with RaBitQ quantization.

    Unlike HNSW and IVF, SymQG does NOT require pre-computed clusters.
    It builds its own quantized graph directly from the data.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    max_degree : int, default=64
        Maximum out-degree per node. Must be a multiple of 32.
    metric : str, default='l2'
        Distance metric: 'l2' or 'ip'.
    """

    def __init__(self, dim, max_degree=64, metric='l2'): ...

    def build(self, data, ef_construction=300, num_threads=0):
        """Build the SymQG index from data.

        No clustering required — the graph is built directly.

        Parameters
        ----------
        data : np.ndarray, shape (n, dim), dtype float32
            Vectors to index.
        ef_construction : int, default=300
            Search width during graph construction.
        num_threads : int, default=0
            Build threads. 0 = all available cores.
        """

    def search(self, queries, k=10, ef=None, num_threads=1):
        """Search for nearest neighbors.

        Parameters
        ----------
        queries : np.ndarray, shape (nq, dim), dtype float32
        k : int, default=10
        ef : int, optional
            Search width. Default: max(k, 10).
        num_threads : int, default=1
            Single-threaded by default for latency; set higher for batch
            throughput. 0 = all available cores.

        Returns
        -------
        ids : np.ndarray, shape (nq, k), dtype uint32
        distances : np.ndarray, shape (nq, k), dtype float32
        """

    def save(self, path): ...

    @classmethod
    def load(cls, path):
        """Load saved SymQG index."""

    @property
    def dim(self) -> int: ...

    @property
    def num_vertices(self) -> int: ...

    @property
    def max_degree(self) -> int: ...

    @property
    def is_built(self) -> bool: ...

    def __repr__(self) -> str:
        """e.g. SymqgIndex(dim=128, max_degree=64, metric='l2', is_built=True)"""
```

**Note on SymQG search:** Like IVF, C++ `search()` is single-query. The wrapper batches internally and handles `set_ef()` per call.

**Note on SymQG build:** The C++ uses a separate `QGBuilder` class. The Python wrapper hides this — `build()` creates a `QGBuilder` internally:

```cpp
// In qg_bindings.cpp
.def("build",
     [](QuantizedGraph<float>& self, py::array_t<float> data,
        size_t ef_build, size_t num_threads) {
         auto d = data.request();
         py::gil_scoped_release release;
         QGBuilder builder(self, ef_build, (float*)d.ptr, num_threads);
         builder.build();
     })
```

**Usage:**
```python
# No clustering needed!
index = rabitqlib.SymqgIndex(dim=128, max_degree=64)
index.build(data, ef_construction=300)
ids, dists = index.search(queries, k=10, ef=100)
```

---

### Detailed API: `rabitqlib.cluster()`

```python
def cluster(data, num_clusters, metric='l2', verbose=False):
    """Run k-means clustering on data vectors.

    Convenience function for preprocessing before HNSW/IVF indexing.
    Uses faiss for clustering if available, falls back to sklearn.

    Parameters
    ----------
    data : np.ndarray, shape (n, dim), dtype float32
        Vectors to cluster.
    num_clusters : int
        Number of clusters (k).
    metric : str, default='l2'
        Distance metric: 'l2' or 'ip'.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    centroids : np.ndarray, shape (num_clusters, dim), dtype float32
        Cluster centroids.
    cluster_ids : np.ndarray, shape (n,), dtype uint32
        Cluster assignment for each vector.
    """
```

This wraps the existing `python/ivf.py` logic into a proper function. Uses faiss if installed (fast), otherwise falls back to sklearn KMeans.

---

### Detailed API: `rabitqlib.recall_at_k()`

```python
def recall_at_k(predicted, ground_truth, k):
    """Compute recall@k.

    Parameters
    ----------
    predicted : np.ndarray, shape (nq, n_results), dtype uint32
    ground_truth : np.ndarray, shape (nq, n_gt), dtype uint32
    k : int

    Returns
    -------
    recall : float
        Fraction of true neighbors found in top-k results.
    """
```

---

### Detailed API: `rabitqlib.RaBitQuantizer`

```python
class RaBitQuantizer:
    """RaBitQ vector quantizer.

    Quantizes float32 vectors into compact binary/multi-bit codes with
    distance estimation factors. This is the core quantization primitive
    that powers all rabitqlib index types. Use it standalone to build
    custom pipelines, integrate quantization into existing systems, or
    compress vector datasets for storage and transmission.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    nbits : int, default=1
        Bits per dimension (1-9).
        1 = pure binary quantization (fastest, least accurate).
        >1 = 1-bit base code + (nbits-1) extra-bit refinement codes.
    metric : str, default='l2'
        Distance metric: 'l2' or 'ip'.
    fast : bool, default=True
        Use pre-computed constant rescaling factors for faster quantization.
        Slightly less accurate but significantly faster for nbits > 1.
    random_seed : int, optional
        Seed for reproducibility. If None, uses random device.
    """

    def __init__(self, dim, nbits=1, metric='l2', fast=True,
                 random_seed=None): ...

    def quantize(self, data, centroids=None):
        """Quantize vectors into RaBitQ codes.

        Parameters
        ----------
        data : np.ndarray, shape (n, dim), dtype float32
            Vectors to quantize.
        centroids : np.ndarray, shape (n, dim), dtype float32, optional
            Per-vector centroids (residual quantization).
            If None, quantizes raw vectors (centroid = zero vector).

        Returns
        -------
        QuantizedResult
            Object containing:
            - codes : np.ndarray, shape (n, code_size), dtype uint8
                Quantized binary codes.
            - add_factor : np.ndarray, shape (n,), dtype float32
                Additive distance estimation factor per vector.
            - scale_factor : np.ndarray, shape (n,), dtype float32
                Multiplicative distance estimation factor per vector.
            - error_factor : np.ndarray, shape (n,), dtype float32
                Error bound factor per vector.
        """

    def quantize_scalar(self, data, centroids=None,
                        mode='reconstruction'):
        """Quantize vectors using scalar quantization.

        Scalar quantization maps each dimension to an integer in
        [0, 2^nbits - 1] using uniform binning.

        Parameters
        ----------
        data : np.ndarray, shape (n, dim), dtype float32
            Vectors to quantize.
        centroids : np.ndarray, shape (n, dim), dtype float32, optional
            Per-vector centroids. If None, quantizes raw vectors.
        mode : str, default='reconstruction'
            Quantizer objective:
            - 'reconstruction': minimize reconstruction error
            - 'unbiased': unbiased distance estimation
            - 'plain': simple uniform binning

        Returns
        -------
        ScalarQuantizedResult
            Object containing:
            - codes : np.ndarray, shape (n, dim), dtype uint8
                Per-dimension quantized codes.
            - delta : np.ndarray, shape (n,), dtype float32
                Bin width per vector (for reconstruction).
            - lower_bound : np.ndarray, shape (n,), dtype float32
                Lower bound per vector (for reconstruction).
        """

    def estimate_distance(self, quantized_result, queries):
        """Estimate distances using all available bits (1-bit + extra-bits).

        Full-precision distance estimation in one pass. Uses all bits
        from the quantized codes for the most accurate estimate.

        Parameters
        ----------
        quantized_result : QuantizedResult
            Output from quantize().
        queries : np.ndarray, shape (nq, dim), dtype float32
            Query vectors.

        Returns
        -------
        distances : np.ndarray, shape (nq, n), dtype float32
            Estimated distances between each query and each quantized vector.
        lower_bounds : np.ndarray, shape (nq, n), dtype float32
            Distance lower bounds (for filtering / early termination).
        """

    def estimate_distance_coarse(self, quantized_result, queries):
        """Estimate distances using only the 1-bit base codes.

        Fast but coarse distance estimation. Works regardless of nbits —
        even if quantized with nbits=5, this uses only the 1-bit codes
        for a quick initial ranking before optionally refining with
        refine_distance().

        Parameters
        ----------
        quantized_result : QuantizedResult
            Output from quantize(). Only the 1-bit codes are used.
        queries : np.ndarray, shape (nq, dim), dtype float32
            Query vectors.

        Returns
        -------
        distances : np.ndarray, shape (nq, n), dtype float32
            Coarse estimated distances (1-bit precision).
        lower_bounds : np.ndarray, shape (nq, n), dtype float32
            Distance lower bounds.
        intermediates : CoarseIntermediates
            Intermediate inner product values needed by refine_distance().
        """

    def refine_distance(self, quantized_result, queries, intermediates,
                       candidate_ids=None):
        """Refine 1-bit distance estimates using extra-bits codes.

        Takes the intermediate results from estimate_distance_coarse() and
        uses the extra-bits codes to produce a more accurate distance.
        Requires nbits > 1 (i.e., extra-bits must be present).

        Typical workflow:
            1. estimate_distance_coarse() on all candidates → coarse ranking
            2. refine_distance() on top-k candidates only → accurate re-ranking

        Parameters
        ----------
        quantized_result : QuantizedResult
            Output from quantize(). Must have nbits > 1.
        queries : np.ndarray, shape (nq, dim), dtype float32
            Query vectors.
        intermediates : CoarseIntermediates
            Output from estimate_distance_coarse().
        candidate_ids : np.ndarray, shape (nq, m), dtype uint32, optional
            Indices of candidates to refine per query. If None, refines
            all vectors (equivalent to estimate_distance with all bits).

        Returns
        -------
        distances : np.ndarray, shape (nq, m) or (nq, n), dtype float32
            Refined distance estimates (higher precision than 1-bit).
            Shape depends on whether candidate_ids is provided.
        """

    @staticmethod
    def reconstruct(scalar_result):
        """Reconstruct float vectors from scalar-quantized codes.

        Parameters
        ----------
        scalar_result : ScalarQuantizedResult
            Output from quantize_scalar().

        Returns
        -------
        reconstructed : np.ndarray, shape (n, dim), dtype float32
            Reconstructed vectors: value = code * delta + lower_bound
        """

    @property
    def dim(self) -> int: ...

    @property
    def nbits(self) -> int: ...

    @property
    def code_size(self) -> int:
        """Size of the quantized code per vector in bytes."""

    def __repr__(self) -> str:
        """e.g. RaBitQuantizer(dim=128, nbits=5, metric='l2', fast=True)"""
```

**Usage:**

```python
import numpy as np
import rabitqlib

data = np.random.randn(10000, 128).astype('float32')
queries = np.random.randn(100, 128).astype('float32')

# --- 1-bit quantization (fastest) ---
quantizer = rabitqlib.RaBitQuantizer(dim=128, nbits=1)
result = quantizer.quantize(data)
# result.codes: (10000, 16) uint8 — 128 bits packed into 16 bytes
# result.add_factor, result.scale_factor, result.error_factor: (10000,) float32

# --- Multi-bit quantization (better accuracy) ---
quantizer = rabitqlib.RaBitQuantizer(dim=128, nbits=5)
result = quantizer.quantize(data)

# --- Residual quantization (relative to centroids) ---
centroids, cids = rabitqlib.cluster(data, num_clusters=100)
per_vec_centroids = centroids[cids]  # shape (10000, 128)
result = quantizer.quantize(data, centroids=per_vec_centroids)

# --- Full distance estimation (all bits) ---
dists, lower_bounds = quantizer.estimate_distance(result, queries)

# --- Coarse-to-fine distance estimation ---
# Step 1: fast 1-bit ranking over all candidates
coarse_dists, _, intermediates = quantizer.estimate_distance_coarse(result, queries)

# Step 2: refine only the top-100 candidates per query with extra-bits
top100 = np.argsort(coarse_dists, axis=1)[:, :100]
refined_dists = quantizer.refine_distance(
    result, queries, intermediates, candidate_ids=top100
)
# refined_dists shape: (nq, 100) — distances for the top-100 candidates only

# --- Scalar quantization + reconstruction ---
sq_result = quantizer.quantize_scalar(data)
reconstructed = rabitqlib.RaBitQuantizer.reconstruct(sq_result)
error = np.linalg.norm(data - reconstructed, axis=1).mean()
print(f"Mean reconstruction error: {error:.4f}")
```

---

### Detailed API: `rabitqlib.Rotator`

```python
class Rotator:
    """Random rotation preprocessing for RaBitQ quantization.

    RaBitQ quantization quality depends on vectors being near-isotropic.
    Applying a random rotation before quantization spreads information
    uniformly across dimensions, improving recall in production workloads.

    The FHT-Kac rotation (default) uses the Fast Hadamard Transform with
    random sign flips — O(d log d) per vector instead of O(d²) for a
    dense random matrix. This is the same rotation used internally by
    the IVF and SymQG index types.

    Parameters
    ----------
    dim : int
        Original vector dimensionality.
    method : str, default='fht_kac'
        Rotation method:
        - 'fht_kac': Fast Hadamard Transform + random sign flips.
          Pads dim to next multiple of 64.
        - 'matrix': Dense random orthogonal matrix.
          No padding needed but O(d²) per rotation.
    random_seed : int, optional
        Seed for reproducibility. If None, uses random device.
    """

    def __init__(self, dim, method='fht_kac', random_seed=None): ...

    def rotate(self, data):
        """Apply rotation to vectors.

        Parameters
        ----------
        data : np.ndarray, shape (n, dim), dtype float32
            Input vectors.

        Returns
        -------
        rotated : np.ndarray, shape (n, padded_dim), dtype float32
            Rotated (and possibly zero-padded) vectors.
        """

    def save(self, path):
        """Save rotator state (random signs / matrix) to disk."""

    @classmethod
    def load(cls, path):
        """Load a saved rotator."""

    @property
    def dim(self) -> int:
        """Original dimensionality."""

    @property
    def padded_dim(self) -> int:
        """Dimensionality after padding (same as dim for matrix method)."""

    def __repr__(self) -> str:
        """e.g. Rotator(dim=128, padded_dim=128, method='fht_kac')"""
```

**Usage:**

```python
# Standalone quantization pipeline
rotator = rabitqlib.Rotator(dim=128, method='fht_kac')

# Rotate data before quantization
rotated_data = rotator.rotate(data)            # (10000, 128) → (10000, 128)
rotated_queries = rotator.rotate(queries)      # must use same rotator

# Quantize the rotated vectors
quantizer = rabitqlib.RaBitQuantizer(dim=rotator.padded_dim, nbits=5)
result = quantizer.quantize(rotated_data)

# Estimate distances on rotated space
est_dists, _ = quantizer.estimate_distance(result, rotated_queries)

# Save/load rotator for consistency between indexing and querying
rotator.save("rotator.bin")
rotator2 = rabitqlib.Rotator.load("rotator.bin")
```

### Module-Level Exports

```python
# rabitqlib/__init__.py

__version__ = "0.1.0"

# Index classes
from rabitqlib.index.hnsw import HnswIndex
from rabitqlib.index.ivf import IvfIndex
from rabitqlib.index.symqg import SymqgIndex

# Quantization
from rabitqlib.quantization.quantizer import RaBitQuantizer
from rabitqlib.quantization.rotator import Rotator
from rabitqlib.quantization.results import (
    QuantizedResult,
    ScalarQuantizedResult,
    CoarseIntermediates,
)

# Utilities
from rabitqlib.utils.clustering import cluster
from rabitqlib.utils.metrics import recall_at_k

# Enums (from C++ core)
from rabitqlib._core import MetricType

__all__ = [
    "HnswIndex",
    "IvfIndex",
    "SymqgIndex",
    "RaBitQuantizer",
    "Rotator",
    "QuantizedResult",
    "ScalarQuantizedResult",
    "CoarseIntermediates",
    "cluster",
    "recall_at_k",
    "MetricType",
]
```

---

## API Comparison with hnswlib and faiss

| Feature | rabitqlib (proposed) | hnswlib | faiss |
|---------|---------------------|---------|-------|
| Create index | `HnswIndex(dim, max_elements, M=16)` | `Index('l2', dim)` + `init_index(max_elements, M, ef)` | `index_factory(dim, "HNSW32")` |
| Build index | `index.build(data, centroids, cluster_ids)` | `index.add_items(data)` | `index.add(data)` |
| Search | `ids, dists = index.search(queries, k=10)` | `ids, dists = index.knn_query(queries, k=10)` | `dists, ids = index.search(queries, k=10)` |
| Save | `index.save(path)` | `index.save_index(path)` | `faiss.write_index(index, path)` |
| Load | `HnswIndex.load(path)` | `index.load_index(path)` | `faiss.read_index(path)` | Metric stored in file for all index types |
| Set search params | `search(ef=100)` arg | `index.set_ef(100)` | `index.hnsw.efSearch = 100` |
| Return order | `(ids, distances)` | `(ids, distances)` | `(distances, ids)` — reversed! |

Key differences from hnswlib:
- **Clustering step required** for HNSW/IVF (hnswlib doesn't quantize, so no clustering needed)
- **`search()` returns both ids and distances** consistently (matches hnswlib, not faiss)
- **`load()` is a classmethod** returning a new index (vs hnswlib's `load_index()` mutating self); metric is stored in the file for all index types
- **Search params passed per-call** via `ef=` argument (vs hnswlib's `set_ef()` side-effect)

---

## Binding Decisions

### What goes in C++ bindings (`_core`) vs Python wrappers

| C++ bindings (`_core`) | Python wrappers |
|------------------------|-----------------|
| Raw class constructors | Input validation (dtype, shape checks) |
| `construct()`, `search()`, `save()`, `load()` | Default parameter logic (`ef = max(k, 10)`) |
| GIL release for long operations | Batch wrapping for single-query C++ methods |
| NumPy ↔ raw pointer conversion | Classmethod `load()` pattern |
| `MetricType` enum | String → enum conversion (`'l2'` → `MetricType.L2`) |
| `quantize_full_single()`, `quantize_scalar()` | `RaBitQuantizer` class wrapping raw quant functions |
| `quantize_compact_one_bit()`, `quantize_compact_ex_bits()` | `QuantizedResult` / `ScalarQuantizedResult` containers |
| `reconstruct_vec()`, `full_est_dist()` | Batch loops over single-vector C++ quant functions |
| `FhtKacRotator`, `MatrixRotator` classes | `Rotator` class with string-based method selection |
| `RabitqConfig`, `faster_config()` | Hidden behind `fast=True` default in `RaBitQuantizer` |
| | `cluster()` utility |
| | `recall_at_k()` utility |
| | Properties (`dim`, `is_built`, etc.) |

### What to expose vs hide

| Expose | Hide |
|--------|------|
| `nbits` (1-9) — controls accuracy/speed tradeoff | `padded_dim` — internal padding detail |
| `M`, `ef_construction`, `ef` — standard HNSW params | `RabitqConfig` struct — hidden behind `fast` flag |
| `nprobe` (IVF) — standard IVF param | `fast` flag in construct — default True |
| `high_accuracy` (IVF) — meaningful accuracy tradeoff | `QGBuilder` class — hidden behind `build()` |
| `max_degree` (SymQG) — graph density control | Raw pointer APIs |
| `metric` ('l2'/'ip') | Internal buffer/visited-set details |
| `RaBitQuantizer` — standalone quantization | `BatchDataMap`, `ExDataMap`, `BinDataMap` — internal layout |
| `Rotator` with method selection | `RotatorType` enum — string selection instead |
| `quantize_scalar` + `reconstruct` — scalar quant roundtrip | `quantize_one_batch`, `quantize_qg_batch` — internal batch formats |
| `estimate_distance` — distance from codes | `quantize_split_batch`, `quantize_split_single` — index-internal |
| `ScalarQuantizerType` via string ('reconstruction'/'unbiased'/'plain') | `fastscan` internals |

---

## Error Handling

```python
# Dimension mismatch
index = rabitqlib.HnswIndex(dim=128, max_elements=1000)
index.build(np.zeros((100, 64)), ...)  # → ValueError: Expected dim=128, got 64

# Search before build
index.search(queries)  # → RuntimeError: Index not built. Call build() first.

# Invalid nbits
rabitqlib.HnswIndex(dim=128, max_elements=100, nbits=15)
# → ValueError: nbits must be between 1 and 9

# Invalid SymQG degree
rabitqlib.SymqgIndex(dim=128, max_degree=50)
# → ValueError: max_degree must be a multiple of 32

# Wrong dtype (auto-converts with warning)
data = np.random.randn(100, 128)  # float64
index.build(data, ...)
# → UserWarning: Input dtype float64 converted to float32. Pass float32 to suppress.
```

---

## Complete Example: End-to-End Pipeline

```python
import numpy as np
import rabitqlib

# 1. Load or generate data
data = np.random.randn(100_000, 128).astype('float32')
queries = np.random.randn(1000, 128).astype('float32')

# 2. Cluster (needed for HNSW and IVF, not for SymQG)
centroids, cluster_ids = rabitqlib.cluster(data, num_clusters=1000)

# 3. Build HNSW index
hnsw = rabitqlib.HnswIndex(dim=128, max_elements=100_000, M=32, nbits=5)
hnsw.build(data, centroids, cluster_ids, num_threads=8)

# 4. Build IVF index
ivf = rabitqlib.IvfIndex(dim=128, max_elements=100_000, num_clusters=1000, nbits=5)
ivf.build(data, centroids, cluster_ids)

# 5. Build SymQG index (no clustering needed)
symqg = rabitqlib.SymqgIndex(dim=128, max_degree=64)
symqg.build(data, ef_construction=300, num_threads=8)

# 6. Search all three
ids_hnsw, dist_hnsw = hnsw.search(queries, k=10, ef=100)
ids_ivf, dist_ivf = ivf.search(queries, k=10, nprobe=50)
ids_symqg, dist_symqg = symqg.search(queries, k=10, ef=100)

# 7. Evaluate
from scipy.spatial.distance import cdist
true_dists = cdist(queries, data)
gt = np.argsort(true_dists, axis=1)[:, :10].astype('uint32')

print(f"HNSW   recall@10: {rabitqlib.recall_at_k(ids_hnsw, gt, 10):.3f}")
print(f"IVF    recall@10: {rabitqlib.recall_at_k(ids_ivf, gt, 10):.3f}")
print(f"SymQG  recall@10: {rabitqlib.recall_at_k(ids_symqg, gt, 10):.3f}")

# 8. Save and reload
hnsw.save("hnsw.index")
hnsw2 = rabitqlib.HnswIndex.load("hnsw.index")
ids2, _ = hnsw2.search(queries[:5], k=10, ef=100)
np.testing.assert_array_equal(ids_hnsw[:5], ids2)
```

---

## Complete Example: Quantization Pipeline

```python
import numpy as np
import rabitqlib

data = np.random.randn(10_000, 128).astype('float32')
queries = np.random.randn(100, 128).astype('float32')

# --- Step 1: Rotate vectors (required preprocessing for RaBitQ) ---
rotator = rabitqlib.Rotator(dim=128, method='fht_kac')
rotated_data = rotator.rotate(data)
rotated_queries = rotator.rotate(queries)

# --- Step 2: Quantize with RaBitQ (multi-bit) ---
quantizer = rabitqlib.RaBitQuantizer(
    dim=rotator.padded_dim, nbits=5, metric='l2'
)
result = quantizer.quantize(rotated_data)

print(f"Code size per vector: {result.codes.shape[1]} bytes")
print(f"Compression ratio: {128 * 4 / result.codes.shape[1]:.1f}x")

# --- Step 3: Estimate distances from quantized codes ---
est_dists, _ = quantizer.estimate_distance(result, rotated_queries)
approx_nn = np.argmin(est_dists, axis=1)

# --- Step 4: Compare with exact distances ---
from scipy.spatial.distance import cdist
true_dists = cdist(queries, data)
exact_nn = np.argmin(true_dists, axis=1)
recall_1 = np.mean(approx_nn == exact_nn)
print(f"Recall@1 from quantized codes: {recall_1:.3f}")

# --- Step 5: Scalar quantization + reconstruction ---
sq_result = quantizer.quantize_scalar(rotated_data)
reconstructed = rabitqlib.RaBitQuantizer.reconstruct(sq_result)
mse = np.mean((rotated_data - reconstructed) ** 2)
print(f"Scalar quantization MSE: {mse:.6f}")

# --- Step 6: Residual quantization (with centroids) ---
centroids, cids = rabitqlib.cluster(data, num_clusters=100)
per_vec_centroids = centroids[cids]
rotated_centroids = rotator.rotate(per_vec_centroids)
result_residual = quantizer.quantize(rotated_data, centroids=rotated_centroids)
# Residual codes are more compact and accurate since residuals
# have smaller magnitude and more uniform distribution

# --- Save rotator for query-time use ---
rotator.save("rotator.bin")
```
