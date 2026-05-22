# Python API Design for RaBitQLib

**Goal:** Define the exact Python interfaces exposed to users via `rabitqlib`

See also: [C++ Binding Details](cpp_binding_details.md) for the C++ API surface mapping, result object definitions, and binding architecture decisions.

---

## Design Principles

1. **Match what users expect** — follow the patterns of hnswlib and faiss that Python ANN users already know
2. **Hide C++ complexity** — users shouldn't need to know about padded dimensions, rotators, or batch layouts
3. **NumPy in, NumPy out** — all data goes in as `np.ndarray`, results come back as `np.ndarray`
4. **Separate indexing from searching** — HNSW/IVF require preprocessing (clustering), SymQG does not
5. **Consistent API across index types** — same method names and patterns for HNSW, IVF, SymQG

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
        Higher M -> better recall, more memory, slower build.
    ef_construction : int, default=200
        Search width during construction. Higher -> better graph quality, slower build.
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
            Search width. Higher ef -> better recall, slower search.
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
            Number of clusters to probe. Higher -> better recall, slower.
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

    Optionally takes a Rotator and/or a normalizer (centroids) at
    construction time. If provided, rotation and centroid subtraction
    are applied automatically during quantize() and distance estimation
    calls — no manual preprocessing needed.

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
    rotator : Rotator, optional
        If provided, vectors are rotated before quantization and queries
        are rotated before distance estimation. Ensures consistency
        between indexing and query time.
    normalizer : np.ndarray, optional
        Cluster centroids for residual quantization.
        - None: no centroid subtraction (quantize raw vectors).
        - shape (dim,) or (1, dim): single centroid subtracted from
          all vectors. cluster_id not needed in quantize().
        - shape (num_clusters, dim): multiple centroids. If cluster_id
          is provided in quantize(), uses that centroid. If omitted,
          automatically finds the closest centroid to the input vector.
    """

    def __init__(self, dim, nbits=1, metric='l2', fast=True,
                 rotator=None, normalizer=None): ...

    def quantize(self, data, cluster_id=None):
        """Quantize a single vector into a RaBitQ code.

        Internally applies centroid subtraction (if normalizer was provided)
        and rotation (if rotator was provided) before quantizing.

        Parameters
        ----------
        data : np.ndarray, shape (dim,), dtype float32
            Vector to quantize.
        cluster_id : int, optional
            Cluster assignment for this vector. When normalizer has
            multiple centroids: if provided, uses that centroid directly;
            if omitted, automatically finds the closest centroid.
            Ignored when normalizer is None or a single centroid.

        Returns
        -------
        QuantizedResult
            Object containing:
            - codes : np.ndarray, shape (code_size,), dtype uint8
                Quantized binary code.
            - add_factor : float
                Additive distance estimation factor.
            - scale_factor : float
                Multiplicative distance estimation factor.
            - error_factor : float
                Error bound factor.
        """

    def quantize_scalar(self, data, cluster_id=None,
                        mode='reconstruction'):
        """Quantize a single vector using scalar quantization.

        Scalar quantization maps each dimension to an integer in
        [0, 2^nbits - 1] using uniform binning. Applies normalizer
        and rotator if configured.

        Parameters
        ----------
        data : np.ndarray, shape (dim,), dtype float32
            Vector to quantize.
        cluster_id : int, optional
            Cluster assignment for this vector. Same rules as quantize():
            optional when normalizer has multiple centroids (auto-finds
            closest if omitted).
        mode : str, default='reconstruction'
            Quantizer objective:
            - 'reconstruction': minimize reconstruction error
            - 'unbiased': unbiased distance estimation
            - 'plain': simple uniform binning

        Returns
        -------
        ScalarQuantizedResult
            Object containing:
            - codes : np.ndarray, shape (dim,), dtype uint8
                Per-dimension quantized codes.
            - delta : float
                Bin width (for reconstruction).
            - lower_bound : float
                Lower bound (for reconstruction).
        """

    def estimate_distance(self, quantized_result, query):
        """Estimate distance using all available bits (1-bit + extra-bits).

        Full-precision distance estimation in one pass. Internally handles
        all preprocessing of the query:
        - If normalizer is set, subtracts the centroid from the query
          using the cluster_id stored in quantized_result.
        - If rotator is set, rotates the adjusted query.

        Parameters
        ----------
        quantized_result : QuantizedResult
            Output from quantize(). The stored cluster_id is used to
            look up the centroid for query adjustment.
        query : np.ndarray, shape (dim,), dtype float32
            Query vector in original space.

        Returns
        -------
        DistanceResult
            Named tuple with fields:
            - distance : float
                Estimated distance between the query and the quantized vector.
            - lower_bound : float
                Distance lower bound (for filtering / early termination).
            Supports unpacking: ``dist, lb = quantizer.estimate_distance(...)``
        """

    def estimate_distance_coarse(self, quantized_result, query):
        """Estimate distance using only the 1-bit base codes.

        Fast but coarse distance estimation. Works regardless of nbits —
        even if quantized with nbits=5, this uses only the 1-bit codes
        for a quick initial estimate before optionally refining with
        refine_distance().

        Query preprocessing (centroid subtraction via cluster_id stored
        in quantized_result, then rotation) is handled internally.

        Parameters
        ----------
        quantized_result : QuantizedResult
            Output from quantize(). Only the 1-bit codes are used.
            The stored cluster_id is used for query adjustment.
        query : np.ndarray, shape (dim,), dtype float32
            Query vector in original space.

        Returns
        -------
        CoarseDistanceResult
            Named tuple with fields:
            - distance : float
                Coarse estimated distance (1-bit precision).
            - lower_bound : float
                Distance lower bound.
            - intermediates : CoarseIntermediates
                Intermediate inner product values needed by refine_distance().
            Supports unpacking: ``dist, lb, inter = quantizer.estimate_distance_coarse(...)``
        """

    def refine_distance(self, quantized_result, query, intermediates):
        """Refine a 1-bit distance estimate using extra-bits codes.

        Takes the intermediate results from estimate_distance_coarse() and
        uses the extra-bits codes to produce a more accurate distance.
        Requires nbits > 1 (i.e., extra-bits must be present).

        Query preprocessing (centroid subtraction + rotation) is handled
        internally, same as estimate_distance_coarse().

        Typical workflow:
            1. estimate_distance_coarse() on all candidates -> coarse ranking
            2. refine_distance() on top-k candidates only -> accurate re-ranking

        Parameters
        ----------
        quantized_result : QuantizedResult
            Output from quantize(). Must have nbits > 1.
            The stored cluster_id is used for query adjustment.
        query : np.ndarray, shape (dim,), dtype float32
            Query vector in original space.
        intermediates : CoarseIntermediates
            Output from estimate_distance_coarse().

        Returns
        -------
        distance : float
            Refined distance estimate (higher precision than 1-bit).
        """

    @staticmethod
    def reconstruct(scalar_result):
        """Reconstruct a float vector from scalar-quantized codes.

        Parameters
        ----------
        scalar_result : ScalarQuantizedResult
            Output from quantize_scalar().

        Returns
        -------
        reconstructed : np.ndarray, shape (dim,), dtype float32
            Reconstructed vector: value = code * delta + lower_bound
        """

    @property
    def dim(self) -> int: ...

    @property
    def nbits(self) -> int: ...

    @property
    def code_size(self) -> int:
        """Size of the quantized code per vector in bytes."""

    def save(self, path):
        """Save the quantizer to disk.

        Serializes all state needed to reconstruct this quantizer:
        dim, nbits, metric, fast, rotator state (if any), and
        normalizer centroids (if any).

        Parameters
        ----------
        path : str
            File path to write to.
        """

    @classmethod
    def load(cls, path):
        """Load a saved quantizer from disk.

        Returns a fully reconstructed RaBitQuantizer with the same
        rotator and normalizer that were present when save() was called.

        Parameters
        ----------
        path : str
            File path to read from.

        Returns
        -------
        quantizer : RaBitQuantizer
            Loaded quantizer ready for quantize() and estimate_distance*().
        """

    @property
    def rotator(self) -> 'Rotator | None':
        """The rotator used for preprocessing, or None."""

    @property
    def normalizer(self) -> 'np.ndarray | None':
        """The centroids used for normalization, or None."""

    def __repr__(self) -> str:
        """e.g. RaBitQuantizer(dim=128, nbits=5, metric='l2', rotator=Rotator(dim=128), normalizer=(100, 128))"""
```

**Usage:**

```python
import numpy as np
import rabitqlib

data = np.random.randn(10000, 128).astype('float32')
query = np.random.randn(128).astype('float32')

# --- Simplest: no rotation, no normalization ---
quantizer = rabitqlib.RaBitQuantizer(dim=128, nbits=1)
result = quantizer.quantize(data[0])

# --- With rotation (recommended) ---
rotator = rabitqlib.Rotator(dim=128)
quantizer = rabitqlib.RaBitQuantizer(dim=128, nbits=5, rotator=rotator)
result = quantizer.quantize(data[0])
# rotation is applied automatically — no manual rotate() call needed

# --- With rotation + single centroid normalization ---
mean_centroid = data.mean(axis=0)
quantizer = rabitqlib.RaBitQuantizer(
    dim=128, nbits=5, rotator=rotator, normalizer=mean_centroid
)
result = quantizer.quantize(data[0])
# cluster_id not needed — single centroid subtracted from the vector

# --- With rotation + multi-centroid normalization ---
centroids, cluster_ids = rabitqlib.cluster(data, num_clusters=100)
quantizer = rabitqlib.RaBitQuantizer(
    dim=128, nbits=5, rotator=rotator, normalizer=centroids
)
result = quantizer.quantize(data[0], cluster_id=cluster_ids[0])
# the vector's cluster centroid is subtracted before rotation + quantization

# --- Distance estimation (query rotated automatically) ---
# Unpacking style:
dist, lower_bound = quantizer.estimate_distance(result, query)
# Attribute access style:
result = quantizer.estimate_distance(result, query)
print(result.distance, result.lower_bound)

# --- Coarse-to-fine distance estimation ---
# Quantize multiple vectors, then estimate distances one at a time
results = [quantizer.quantize(data[i], cluster_id=cluster_ids[i])
           for i in range(len(data))]

# Step 1: fast 1-bit coarse estimate for each candidate
coarse = []
for r in results:
    cr = quantizer.estimate_distance_coarse(r, query)
    coarse.append(cr)

# Step 2: refine only the closest candidates with extra-bits
top_indices = np.argsort([c.distance for c in coarse])[:100]
for idx in top_indices:
    refined = quantizer.refine_distance(results[idx], query, coarse[idx].intermediates)

# --- Scalar quantization + reconstruction ---
sq_result = quantizer.quantize_scalar(data[0], cluster_id=cluster_ids[0])
reconstructed = rabitqlib.RaBitQuantizer.reconstruct(sq_result)
error = np.linalg.norm(data[0] - reconstructed)
print(f"Reconstruction error: {error:.4f}")
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
    random sign flips — O(d log d) per vector instead of O(d^2) for a
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
          No padding needed but O(d^2) per rotation.
    random_seed : int, optional
        Seed for reproducibility. If None, uses random device.
    """

    def __init__(self, dim, method='fht_kac', random_seed=None): ...

    def rotate(self, data):
        """Apply rotation to a single vector.

        Parameters
        ----------
        data : np.ndarray, shape (dim,), dtype float32
            Input vector.

        Returns
        -------
        rotated : np.ndarray, shape (padded_dim,), dtype float32
            Rotated (and possibly zero-padded) vector.
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
# Standalone usage — rotator is constructed separately, passed to quantizer
rotator = rabitqlib.Rotator(dim=128, method='fht_kac')

# Quantizer handles rotation automatically
quantizer = rabitqlib.RaBitQuantizer(dim=128, nbits=5, rotator=rotator)
result = quantizer.quantize(data)

# Queries are rotated automatically during distance estimation
est_dists, _ = quantizer.estimate_distance(result, queries)

# Save/load rotator for reuse across sessions
rotator.save("rotator.bin")
rotator2 = rabitqlib.Rotator.load("rotator.bin")
```

---

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
    DistanceResult,
    CoarseDistanceResult,
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
    "DistanceResult",
    "CoarseDistanceResult",
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

## Error Handling

```python
# Dimension mismatch
index = rabitqlib.HnswIndex(dim=128, max_elements=1000)
index.build(np.zeros((100, 64)), ...)  # -> ValueError: Expected dim=128, got 64

# Search before build
index.search(queries)  # -> RuntimeError: Index not built. Call build() first.

# Invalid nbits
rabitqlib.HnswIndex(dim=128, max_elements=100, nbits=15)
# -> ValueError: nbits must be between 1 and 9

# Invalid SymQG degree
rabitqlib.SymqgIndex(dim=128, max_degree=50)
# -> ValueError: max_degree must be a multiple of 32

# Wrong dtype (auto-converts with warning)
data = np.random.randn(100, 128)  # float64
index.build(data, ...)
# -> UserWarning: Input dtype float64 converted to float32. Pass float32 to suppress.
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
query = np.random.randn(128).astype('float32')

# --- Step 1: Set up quantizer with rotation + centroids ---
rotator = rabitqlib.Rotator(dim=128, method='fht_kac')
centroids, cluster_ids = rabitqlib.cluster(data, num_clusters=100)
quantizer = rabitqlib.RaBitQuantizer(
    dim=128, nbits=5, metric='l2', rotator=rotator, normalizer=centroids
)

# --- Step 2: Quantize each vector ---
results = [quantizer.quantize(data[i], cluster_id=cluster_ids[i])
           for i in range(len(data))]

print(f"Code size per vector: {results[0].codes.shape[0]} bytes")
print(f"Compression ratio: {128 * 4 / results[0].codes.shape[0]:.1f}x")

# --- Step 3: Coarse 1-bit ranking ---
coarse = []
for r in results:
    cr = quantizer.estimate_distance_coarse(r, query)
    coarse.append(cr)

top100 = np.argsort([c.distance for c in coarse])[:100]

# --- Step 4: Refine top candidates with extra-bits ---
refined = []
for idx in top100:
    d = quantizer.refine_distance(results[idx], query, coarse[idx].intermediates)
    refined.append((idx, d))

best = min(refined, key=lambda x: x[1])
print(f"Nearest neighbor: vector {best[0]}, distance {best[1]:.4f}")

# --- Step 5: Scalar quantization + reconstruction ---
sq_result = quantizer.quantize_scalar(data[0], cluster_id=cluster_ids[0])
reconstructed = rabitqlib.RaBitQuantizer.reconstruct(sq_result)
error = np.linalg.norm(data[0] - reconstructed)
print(f"Reconstruction error: {error:.4f}")

# --- Save rotator for reuse ---
rotator.save("rotator.bin")
```
