# C++ Binding Details for RaBitQLib

Implementation reference for the pybind11 binding layer: C++ API surface mapping, result object definitions, supporting types, and binding architecture decisions.

See also: [Python API](python_api.md) for the user-facing API design, class definitions, and usage examples.

---

## C++ API Surface (What We're Wrapping)

### HierarchicalNSW (HNSW + RaBitQ)

| C++ Method | Signature | Python Wrapper | Notes |
|------------|-----------|----------------|-------|
| Constructor | `HierarchicalNSW(max_elements, dim, total_bits, M, ef_construction, random_seed, metric)` | `HnswIndex(dim, max_elements, M, ef_construction, nbits, metric, random_seed)` | Reordered params — dim first for consistency |
| `construct()` | `construct(num_clusters, centroids*, num_points, data*, cluster_ids*, num_threads, faster)` | `HnswIndex.build(data, centroids, cluster_ids, num_threads, fast_quantization)` | `faster` -> `fast_quantization` for clarity |
| `search()` | `search(queries*, nq, k, ef, num_threads) -> vector<vector<pair<float, PID>>>` | `HnswIndex.search(queries, k, ef, num_threads) -> (ids, distances)` | Returns numpy tuple instead of vector of pairs |
| `save()` | `save(filename)` | `HnswIndex.save(path)` | |
| `load()` | `load(filename, metric_type)` | `HnswIndex.load(path)` classmethod | Metric stored in file; returns new instance instead of mutating |

### IVF (IVF + RaBitQ)

| C++ Method | Signature | Python Wrapper | Notes |
|------------|-----------|----------------|-------|
| Constructor | `IVF(n, dim, num_clusters, total_bits, metric, rotator)` | `IvfIndex(dim, max_elements, num_clusters, nbits, metric)` | `n` -> `max_elements` for consistency with HnswIndex; `rotator` hidden — always FhtKac |
| `construct()` | `construct(data*, centroids*, cluster_ids*, faster)` | `IvfIndex.build(data, centroids, cluster_ids, fast_quantization)` | |
| `search()` | `search(query*, k, nprobe, results*, high_accuracy)` | `IvfIndex.search(queries, k, nprobe, high_accuracy, num_threads) -> (ids, distances)` | Single-query C++ -> batched Python with threading |
| `save()` | `save(filename)` | `IvfIndex.save(path)` | |
| `load()` | `load(filename)` | `IvfIndex.load(path)` classmethod | |

### QuantizedGraph (SymQG)

| C++ Method | Signature | Python Wrapper | Notes |
|------------|-----------|----------------|-------|
| Constructor | `QuantizedGraph(num, dim, max_degree, metric, rotator)` | `SymqgIndex(dim, max_degree, metric)` | `num` inferred from data; `rotator` hidden |
| `QGBuilder` | `QGBuilder(qg, ef_build, data*, num_threads).build()` | `SymqgIndex.build(data, ef_construction, num_threads)` | Builder class hidden behind single method |
| `search()` + `set_ef()` | `set_ef(ef)` then `search(query*, k, results*)` | `SymqgIndex.search(queries, k, ef, num_threads) -> (ids, distances)` | `set_ef()` merged into `search(ef=)` param; single-query -> batched |
| `save()` | `save(filename)` | `SymqgIndex.save(path)` | |
| `load()` | `load(filename)` | `SymqgIndex.load(path)` classmethod | |

### RaBitQ Quantization (`rabitqlib::quant`)

| C++ Function | Signature | Python Wrapper | Notes |
|--------------|-----------|----------------|-------|
| No direct equivalent | N/A | `RaBitQuantizer(dim, nbits, metric, fast, rotator, normalizer)` constructor | Python-only wrapper class — holds config, optional `Rotator`, and optional normalizer (centroids). Calls `faster_config()` internally when `fast=True`. |
| `faster_config()` | `faster_config(dim, total_bits) -> RabitqConfig` | Called internally by `RaBitQuantizer.__init__(fast=True)` | Hidden — no standalone function needed |
| No direct equivalent | N/A | `RaBitQuantizer.save(path)` | Python-only — serializes dim, nbits, metric, fast, rotator state, and normalizer centroids to a single file |
| No direct equivalent | N/A | `RaBitQuantizer.load(path)` classmethod -> `RaBitQuantizer` | Python-only — reconstructs the full quantizer from a saved file |
| `quantize_compact_one_bit()` | `quantize_compact_one_bit(data*, centroid*, padded_dim, bin_data*, metric)` | `RaBitQuantizer.quantize(data, cluster_id)` when `nbits=1` | Merged into unified `quantize()`; centroid lookup via normalizer + cluster_id (auto-finds closest if omitted) |
| `quantize_compact_ex_bits()` | `quantize_compact_ex_bits(data*, centroid*, padded_dim, ex_bits, ex_data*, metric, config)` | `RaBitQuantizer.quantize(data, cluster_id)` when `nbits>1` | Merged into unified `quantize()` |
| `quantize_full_single()` | `quantize_full_single(data*, [centroid*,] dim, total_bits, total_code*, f_add, f_rescale, f_error, metric, config)` | `RaBitQuantizer.quantize(data, cluster_id) -> QuantizedResult` | Single vector; out-params -> returned `QuantizedResult`; centroid from normalizer |
| `quantize_scalar()` | `quantize_scalar(data*, [centroid*,] dim, total_bits, total_code*, delta, vl, config, scalar_type)` | `RaBitQuantizer.quantize_scalar(data, cluster_id, mode) -> ScalarQuantizedResult` | Single vector; `scalar_type` enum -> `mode` string (`'reconstruction'`/`'unbiased'`/`'plain'`) |
| `reconstruct_vec()` | `reconstruct_vec(quantized_vec*, delta, vl, dim, results*)` | `RaBitQuantizer.reconstruct(scalar_result) -> np.ndarray` | Takes `ScalarQuantizedResult` directly instead of separate args |
| `quantize_one_batch()` | `quantize_one_batch(data*, centroid*, num, padded_dim, batch_data*, metric)` | Not exposed | Internal fastscan layout — used by index types only |
| `quantize_split_batch()` | `quantize_split_batch(data*, centroid*, num, padded_dim, ex_bits, batch_data*, ex_data*, metric, config)` | Not exposed | Internal fastscan layout — used by index types only |

### Distance Estimation (`rabitqlib::estimator`)

| C++ Function | Signature | Python Wrapper | Notes |
|--------------|-----------|----------------|-------|
| `split_single_estdist()` | `split_single_estdist(bin_data*, q_obj, padded_dim, ip_x0_qr, est_dist, low_dist, g_add, g_error)` | `RaBitQuantizer.estimate_distance_coarse(quantized_result, query) -> (distance, lower_bound, intermediates)` | 1-bit only distance estimation — fast but coarse. Works even when `nbits > 1` by using only the 1-bit base codes. Single vector + single query. |
| `split_distance_boosting()` | `split_distance_boosting(ex_data*, ip_func, q_obj, padded_dim, ex_bits, ip_x0_qr) -> float` | `RaBitQuantizer.refine_distance(quantized_result, query, intermediates) -> distance` | Refines a 1-bit estimate using extra-bits codes. Takes intermediate result from `estimate_distance_coarse()`. Single vector + single query. |
| `split_single_fulldist()` | `split_single_fulldist(bin_data*, ex_data*, ip_func, q_obj, padded_dim, ex_bits, est_dist, low_dist, ip_x0_qr, g_add, g_error)` | `RaBitQuantizer.estimate_distance(quantized_result, query) -> (distance, lower_bound)` | Full multi-bit distance estimation in one pass (1-bit + extra-bits combined). Single vector + single query. |
| `split_batch_estdist()` | `split_batch_estdist(batch_data*, q_obj, padded_dim, est_distance*, low_distance*, ip_x0_qr*, high_accuracy)` | Used internally by `IvfIndex.search()` | Batch 1-bit estimation with fastscan — internal to IVF |
| `full_est_dist()` | `full_est_dist(quantized_vec*, query*, ip_func, dim, bits, f_add, f_rescale, g_add, k1xsumq) -> float` | Used internally by `estimate_distance()` | Low-level distance formula — wrapped by higher-level methods |
| `qg_batch_estdist()` | `qg_batch_estdist(batch_data*, q_obj, padded_dim, est_distance*)` | Used internally by `SymqgIndex.search()` | SymQG-specific batch estimation — internal |

### Rotator (`rabitqlib::Rotator<T>` / `rabitqlib::rotator_impl`)

| C++ Method | Signature | Python Wrapper | Notes |
|------------|-----------|----------------|-------|
| `choose_rotator()` | `choose_rotator(dim, type, padded_dim) -> Rotator<T>*` | `Rotator(dim, method)` constructor | Factory hidden — `method` string (`'fht_kac'`/`'matrix'`) replaces `RotatorType` enum |
| `Rotator::rotate()` | `rotate(src*, dst*)` | `Rotator.rotate(data) -> np.ndarray` | Single vector; allocates output internally instead of writing to dst pointer |
| `Rotator::save()` | `save(ofstream&)` | `Rotator.save(path)` | File path string instead of stream |
| `Rotator::load()` | `load(ifstream&)` | `Rotator.load(path)` classmethod | Returns new instance |
| `Rotator::size()` | `size() -> size_t` | `Rotator.padded_dim` property | Renamed for clarity |
| `padding_requirement()` | `padding_requirement(dim, type) -> size_t` | Not exposed | Internal — `padded_dim` property is sufficient |

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

---

## Result Objects

### `QuantizedResult`

Returned by `RaBitQuantizer.quantize()`. Contains the quantized code, distance estimation factors, and the cluster assignment for a single vector. The `cluster_id` is stored so that `estimate_distance*()` can automatically subtract the correct centroid from the query before computing the distance.

| Attribute | C++ Name | Type | Description |
|-----------|----------|------|-------------|
| `codes` | `total_code` / `bin_data` + `ex_data` | `np.ndarray`, uint8, shape `(code_size,)` | Packed binary quantization code |
| `add_factor` | `f_add` | `float` | Additive factor for distance estimation |
| `scale_factor` | `f_rescale` | `float` | Multiplicative factor for distance estimation |
| `error_factor` | `f_error` | `float` | Error bound factor (used to compute distance lower bounds) |
| `cluster_id` | N/A | `int` or `None` | Cluster assignment used during quantization. Set to the provided cluster_id, or the auto-detected closest centroid index when cluster_id was omitted. `None` if no normalizer or single centroid. Used by `estimate_distance*()` to look up the centroid and process the query as `rotate(q - centroid)`. |
| `nbits` | `total_bits` | `int` | Number of bits per dimension used |
| `has_extra_bits` | `ex_bits > 0` | `bool` | `True` if `nbits > 1` (extra-bits codes are present) |

### `ScalarQuantizedResult`

Returned by `RaBitQuantizer.quantize_scalar()`. Contains per-dimension integer codes and the parameters needed to reconstruct the vector via `RaBitQuantizer.reconstruct()`.

| Attribute | C++ Name | Type | Description |
|-----------|----------|------|-------------|
| `codes` | `total_code` | `np.ndarray`, uint8, shape `(dim,)` | Per-dimension quantized codes in `[0, 2^bits - 1]` |
| `delta` | `delta` | `float` | Bin width (reconstruction: `value = code * delta + lower_bound`) |
| `lower_bound` | `vl` | `float` | Lower bound |

### `CoarseIntermediates`

Returned as part of `estimate_distance_coarse()` output. Holds intermediate values that `refine_distance()` needs to refine the 1-bit estimate with extra-bits codes.

| Attribute | C++ Name | Type | Description |
|-----------|----------|------|-------------|
| `ip_x0_qr` | `ip_x0_qr` | `float` | Intermediate inner product term from 1-bit estimation |

---

## Binding Decisions

### What goes in C++ bindings (`_core`) vs Python wrappers

| C++ bindings (`_core`) | Python wrappers |
|------------------------|-----------------|
| Raw class constructors | Input validation (dtype, shape checks) |
| `construct()`, `search()`, `save()`, `load()` | Default parameter logic (`ef = max(k, 10)`) |
| GIL release for long operations | Batch wrapping for single-query C++ methods |
| NumPy <-> raw pointer conversion | Classmethod `load()` pattern |
| `MetricType` enum | String -> enum conversion (`'l2'` -> `MetricType.L2`) |
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
