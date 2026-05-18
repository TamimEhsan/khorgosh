# Python Bindings Implementation Plan for Khorgosh (RaBitQ)

**Goal:** Create pip-installable Python bindings (`pip install rabitqlib`) using pybind11 + setuptools (hnswlib-style source distribution)

**Reference implementations studied:** faiss (SWIG + scikit-build-core), hnswlib (pybind11 + setuptools), SVS (pybind11 + scikit-build + CMake)

---

## How faiss, hnswlib, and SVS Do It

### faiss
- **Binding tool:** SWIG — auto-generates Python wrappers from C++ headers
- **Build system:** scikit-build-core + CMake
- **Distribution:** Pre-built wheels on PyPI — user downloads a compiled binary, no compiler needed
- **CPU dispatch:** Builds multiple SWIG modules per ISA (generic, AVX2, AVX-512) and a runtime `loader.py` picks the best one based on detected CPU at import time
- **Extras:** Type stubs (`.pyi`), stable ABI (abi3) for cross-Python-version compatibility
- **Pros:** Great for large APIs (auto-wraps everything), instant install, no compiler needed
- **Cons:** Verbose generated code, harder to customize, complex CI for building wheels across platforms

### hnswlib
- **Binding tool:** pybind11 — single `bindings.cpp` file wrapping `Index<float>` and `BFIndex<float>`
- **Build system:** Plain setuptools with a custom `BuildExt` class that detects compiler flags (`-march=native`, Apple M1, OpenMP)
- **Distribution:** Source-only on PyPI — user's machine compiles the C++ code during `pip install`
- **CPU dispatch:** Compile-time only — `-march=native` lets the compiler auto-detect the user's CPU and emit the best SIMD instructions
- **Pattern:** Direct `py::class_<>` with `.def()` for each method. NumPy arrays via `py::array_t`. Pickle support via `py::pickle()`
- **Pros:** Simple, one file, easy to maintain, always gets optimal SIMD for the user's CPU
- **Cons:** User needs a C++ compiler, slower install (~30-60s), can break on misconfigured environments

### SVS (Scalable Vector Search)
- **Binding tool:** pybind11 v3.0.1 (fetched via CMake FetchContent)
- **Build system:** scikit-build + CMake — `setup.py` uses `skbuild.setup()`, CMake handles compilation
- **Distribution:** Pre-built wheels on PyPI
- **Module structure:** Central `PYBIND11_MODULE(_svs, m)` in `python_bindings.cpp`, with modular `wrap(m)` functions in separate .cpp files (core.cpp, vamana.cpp, flat.cpp, ivf.cpp, allocator.cpp)
- **Python side:** `svs/__init__.py` imports from `svs._svs` (the C++ module) and re-exports. Additional Python files (`common.py`, `upgrader.py`) add utilities
- **CPU dispatch:** Multi-architecture build with `archspec` detection at install time
- **Key files:**
  - `bindings/python/pyproject.toml` — scikit-build + cmake + numpy + archspec
  - `bindings/python/setup.py` — `skbuild.setup()` with cmake_args
  - `bindings/python/CMakeLists.txt` — FetchContent(pybind11), pybind11_add_module
  - `bindings/python/src/python_bindings.cpp` — module entry point
  - `bindings/python/src/svs/__init__.py` — Python package

---

## Distribution Approach Comparison

There are two fundamental approaches to distributing a pip package with compiled C++ code:

### Approach A: Source Distribution (hnswlib style)

**How it works:** Only source code (`.tar.gz`) is uploaded to PyPI. When a user runs `pip install rabitqlib`, pip downloads the source and compiles it on the user's machine.

```
pip install rabitqlib
      │
      ▼
  Downloads .tar.gz from PyPI (source + headers + setup.py)
      │
      ▼
  Runs setup.py on the USER'S machine
      │
      ▼
  Compiles C++ with -march=native
  (compiler auto-detects AVX-512 / AVX2 / NEON)
      │
      ▼
  Installs rabitqlib.cpython-311-x86_64.so into site-packages
```

| Pros | Cons |
|------|------|
| Always gets the best SIMD for the user's exact CPU | User must have a C++ compiler installed |
| No CPU dispatch logic needed — compiler handles it | Slower install (~30-60 seconds to compile) |
| Simple CI — just upload source, no wheel matrix | Can fail on misconfigured build environments |
| Tiny upload to PyPI (just source files) | Windows users often lack compilers |
| No risk of SIGILL from wrong architecture | OpenMP linking can be tricky on macOS |

### Approach B: Pre-built Wheels (faiss / SVS style)

**How it works:** CI builds compiled binaries (`.whl` files) for each platform/Python version and uploads them to PyPI. Users download the matching wheel — no compilation needed.

```
pip install rabitqlib
      │
      ▼
  Queries PyPI, matches platform + Python version
      │
      ├── Wheel found ──► Downloads .whl (~5-20MB pre-compiled)
      │                    Unzips into site-packages. Done in ~5 seconds.
      │
      └── No wheel ──► Falls back to source distribution (Approach A)
```

| Pros | Cons |
|------|------|
| Instant install, no compiler needed | CI must build for every platform × Python version |
| Works everywhere including Windows | Wheel is locked to the CI machine's CPU capabilities |
| Consistent, reproducible builds | Needs CPU dispatch logic for SIMD (multiple .so or runtime detection) |
| Better user experience | Larger PyPI upload (multiple wheels, each 5-20MB) |
| No build failures from environment issues | More complex release process |

### The SIMD problem with pre-built wheels

When CI builds a wheel on a machine with AVX2, that wheel uses AVX2 instructions. If a user with only SSE4.2 installs it → **crash (SIGILL)**. If a user with AVX-512 installs it → **works but slower than possible**.

Solutions used by libraries:

| Library | Strategy |
|---------|----------|
| **faiss** | Ships multiple `.so` per wheel (generic, AVX2, AVX-512). Python loader picks at import time |
| **SVS** | Uses `archspec` to detect CPU at install time, compiles matching variant |
| **hnswlib** | Avoids the problem entirely — always compiles from source with `-march=native` |

### Our choice: Source Distribution (Approach A)

For the initial release, we use the **hnswlib approach** (source distribution only) because:
1. Simpler to set up — no wheel-building CI matrix needed
2. Always gets optimal SIMD — AVX-512 users get VPOPCNTDQ, AVX2 users get PSHUFB popcount, automatically
3. Khorgosh's target audience (researchers, ML engineers) typically has compilers available
4. We can upgrade to pre-built wheels later without breaking the API

**Future upgrade path:** When the user base grows and frictionless install matters more, add pre-built wheels with the faiss multi-variant pattern (ship `_core.so`, `_core_avx2.so`, `_core_avx512.so` in one wheel with a Python loader).

---

## Chosen Approach for Khorgosh

Use the **hnswlib pattern** (pybind11 + plain setuptools + source distribution) because:
1. Simplest build setup — single `setup.py`, no CMake-to-pip bridge needed
2. Source-only distribution means users always get optimal SIMD for their CPU
3. pybind11 is simpler than SWIG and better for a focused API surface
4. Can upgrade to pre-built wheels later without changing the Python API

---

## What Gets Uploaded to PyPI vs What the User Gets

```
What you upload (once):
┌─────────────────────────────────────────────┐
│ rabitqlib-0.1.0.tar.gz                      │
│ ├── setup.py                                │
│ ├── pyproject.toml                          │
│ ├── include/rabitqlib/**/*.hpp  (C++ headers)│
│ ├── src/bindings/*.cpp         (pybind11)   │
│ ├── python/rabitqlib/**/*.py   (Python code)│
│ └── README.md, LICENSE                      │
└─────────────────────────────────────────────┘

What the user gets after pip install (compiled on their machine):
┌─────────────────────────────────────────────┐
│ site-packages/rabitqlib/                    │
│ ├── __init__.py                             │
│ ├── _core.cpython-311-x86_64.so  (compiled) │
│ ├── index/                                  │
│ │   ├── hnsw.py                             │
│ │   ├── ivf.py                              │
│ │   └── symqg.py                            │
│ └── utils/                                  │
│     └── metrics.py                          │
└─────────────────────────────────────────────┘
```

No C++ source code ends up in the user's site-packages — only the compiled `.so` and Python files.

---

## File Structure

```
khorgosh/
├── pyproject.toml                    # NEW - build requirements
├── setup.py                          # NEW - setuptools + pybind11 build
├── MANIFEST.in                       # NEW - include C++ sources in sdist
├── src/
│   └── bindings/
│       ├── module.cpp                # NEW - PYBIND11_MODULE entry point
│       ├── hnsw_bindings.cpp         # NEW - HierarchicalNSW binding
│       ├── ivf_bindings.cpp          # NEW - IVF binding
│       └── qg_bindings.cpp           # NEW - QuantizedGraph binding
├── python/
│   ├── rabitqlib/
│   │   ├── __init__.py               # NEW - package entry, re-exports
│   │   ├── _core.pyi                 # NEW - type stubs for C++ module
│   │   ├── index/
│   │   │   ├── __init__.py           # NEW
│   │   │   ├── hnsw.py              # NEW - Pythonic HNSW wrapper
│   │   │   ├── ivf.py               # NEW - Pythonic IVF wrapper
│   │   │   └── symqg.py             # NEW - Pythonic SymQG wrapper
│   │   └── utils/
│   │       ├── __init__.py           # NEW
│   │       ├── preprocessing.py      # MOVE from python/ivf.py
│   │       └── metrics.py            # NEW - recall@k, etc.
│   └── tests/
│       ├── conftest.py               # NEW - shared fixtures
│       ├── test_hnsw.py              # NEW
│       ├── test_ivf.py               # NEW
│       └── test_symqg.py             # NEW
├── .github/
│   └── workflows/
│       └── ci.yml                    # NEW - test on push (no wheel publishing)
├── include/rabitqlib/                # EXISTING - unchanged
└── CMakeLists.txt                    # EXISTING - unchanged (not used by pip)
```

---

## Implementation Details

### Step 1: Build System

#### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "pybind11>=2.11", "numpy>=1.20"]
build-backend = "setuptools.build_meta"

[project]
name = "rabitqlib"
version = "0.1.0"
description = "High-performance approximate nearest neighbor search with RaBitQ quantization"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "Apache-2.0"}
dependencies = ["numpy>=1.20"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: C++",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
]

[project.urls]
Homepage = "https://github.com/your-org/rabitqlib"
```

#### setup.py (hnswlib-style, with compiler flag detection)

```python
import os
import sys
import numpy as np
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.1.0"

include_dirs = [
    pybind11.get_include(),
    np.get_include(),
    "include",
]

source_files = [
    "src/bindings/module.cpp",
    "src/bindings/hnsw_bindings.cpp",
    "src/bindings/ivf_bindings.cpp",
    "src/bindings/qg_bindings.cpp",
]

ext_modules = [
    Extension(
        "rabitqlib._core",
        source_files,
        include_dirs=include_dirs,
        language="c++",
    ),
]


def has_flag(compiler, flagname):
    """Check if a compiler flag is supported."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except Exception:
            return False
    return True


class BuildExt(build_ext):
    """Custom build extension with platform-specific compiler flags."""
    c_opts = {
        "msvc": ["/EHsc", "/O2", "/std:c++17"],
        "unix": ["-O3", "-std=c++17", "-march=native"],
    }
    link_opts = {
        "unix": [],
        "msvc": [],
    }

    if os.environ.get("RABITQLIB_NO_NATIVE"):
        c_opts["unix"] = ["-O3", "-std=c++17"]

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
        link_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
    else:
        c_opts["unix"].append("-fopenmp")
        link_opts["unix"].extend(["-fopenmp", "-pthread"])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = list(BuildExt.c_opts.get(ct, []))

        if ct == "unix":
            opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

            if not os.environ.get("RABITQLIB_NO_NATIVE"):
                if not has_flag(self.compiler, "-march=native"):
                    opts = [o for o in opts if o != "-march=native"]
                    if sys.platform == "darwin":
                        if has_flag(self.compiler, "-mcpu=apple-m1"):
                            opts.append("-mcpu=apple-m1")

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(BuildExt.link_opts.get(ct, []))

        build_ext.build_extensions(self)


setup(
    name="rabitqlib",
    version=__version__,
    description="High-performance vector search with RaBitQ quantization",
    ext_modules=ext_modules,
    install_requires=["numpy>=1.20"],
    cmdclass={"build_ext": BuildExt},
    packages=["rabitqlib", "rabitqlib.index", "rabitqlib.utils"],
    package_dir={"": "python"},
    zip_safe=False,
)
```

Key points (matching hnswlib patterns):
- `has_flag()` tests compiler support before using a flag
- `-march=native` is default but can be disabled via `RABITQLIB_NO_NATIVE` env var
- macOS: uses `-stdlib=libc++`, detects Apple M1, no OpenMP (macOS ships without it)
- Linux: enables OpenMP for parallel construction/search

#### MANIFEST.in

```
include CMakeLists.txt LICENSE README.md
recursive-include include *.hpp
recursive-include src *.cpp
recursive-include python/rabitqlib *.py *.pyi
```

---

### Step 2: C++ Bindings

#### module.cpp — Entry Point

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "rabitqlib/defines.hpp"

namespace py = pybind11;

void init_hnsw(py::module_&);
void init_ivf(py::module_&);
void init_qg(py::module_&);

PYBIND11_MODULE(_core, m) {
    m.doc() = "RaBitQLib C++ core";
    m.attr("__version__") = "0.1.0";

    py::enum_<rabitqlib::MetricType>(m, "MetricType")
        .value("L2", rabitqlib::METRIC_L2)
        .value("IP", rabitqlib::METRIC_IP)
        .export_values();

    init_hnsw(m);
    init_ivf(m);
    init_qg(m);
}
```

#### hnsw_bindings.cpp — Key Patterns

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace py = pybind11;

void init_hnsw(py::module_& m) {
    using HNSW = rabitqlib::hnsw::HierarchicalNSW;

    py::class_<HNSW>(m, "HNSW")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, rabitqlib::MetricType>(),
             py::arg("max_elements"),
             py::arg("dim"),
             py::arg("total_bits"),
             py::arg("M"),
             py::arg("ef_construction"),
             py::arg("random_seed") = 100,
             py::arg("metric_type") = rabitqlib::METRIC_L2)

        // construct() — release GIL for multithreading
        .def("construct",
             [](HNSW& self,
                py::array_t<float, py::array::c_style> centroids,
                py::array_t<float, py::array::c_style> data,
                py::array_t<uint32_t, py::array::c_style> cluster_ids,
                size_t num_threads, bool faster) {
                 auto c = centroids.request();
                 auto d = data.request();
                 auto ids = cluster_ids.request();
                 size_t nc = c.shape[0], n = d.shape[0];
                 py::gil_scoped_release release;
                 self.construct(nc, (float*)c.ptr, n, (float*)d.ptr,
                                (uint32_t*)ids.ptr, num_threads, faster);
             },
             py::arg("centroids"), py::arg("data"), py::arg("cluster_ids"),
             py::arg("num_threads") = 0, py::arg("faster") = true)

        // search() — returns (ids, distances) numpy tuple
        .def("search",
             [](HNSW& self, py::array_t<float, py::array::c_style> queries,
                size_t nq, size_t k, size_t ef, size_t num_threads) {
                 auto q = queries.request();
                 std::vector<std::vector<std::pair<float, uint32_t>>> results;
                 {
                     py::gil_scoped_release release;
                     results = self.search((float*)q.ptr, nq, k, ef, num_threads);
                 }
                 py::array_t<uint32_t> out_ids({nq, k});
                 py::array_t<float> out_dists({nq, k});
                 auto id_ptr = out_ids.mutable_data();
                 auto dist_ptr = out_dists.mutable_data();
                 for (size_t i = 0; i < nq; i++)
                     for (size_t j = 0; j < k; j++) {
                         size_t idx = i * k + j;
                         if (j < results[i].size()) {
                             dist_ptr[idx] = results[i][j].first;
                             id_ptr[idx] = results[i][j].second;
                         } else {
                             dist_ptr[idx] = std::numeric_limits<float>::max();
                             id_ptr[idx] = UINT32_MAX;
                         }
                     }
                 return py::make_tuple(out_ids, out_dists);
             },
             py::arg("queries"), py::arg("nq"), py::arg("k"),
             py::arg("ef"), py::arg("num_threads") = 1)

        .def("save", [](const HNSW& self, const std::string& f) { self.save(f.c_str()); })
        .def("load", [](HNSW& self, const std::string& f, rabitqlib::MetricType mt) {
            self.load(f.c_str(), mt);
        });
}
```

IVF and SymQG bindings follow the same pattern, wrapping:
- `rabitqlib::ivf::IVF` — constructor, `construct()`, `search()`, `save()`, `load()`
- `rabitqlib::symqg::QuantizedGraph<float>` — `search()`, `save()`, `load()`

---

### Step 3: Python Wrappers

#### python/rabitqlib/__init__.py

```python
__version__ = "0.1.0"

from rabitqlib._core import MetricType
from rabitqlib.index.hnsw import HNSW
from rabitqlib.index.ivf import IVF
from rabitqlib.index.symqg import SymQG

__all__ = ["HNSW", "IVF", "SymQG", "MetricType", "__version__"]
```

#### python/rabitqlib/index/hnsw.py

```python
import os
import numpy as np
from typing import Tuple, Optional
from rabitqlib import _core

class HNSW:
    """HNSW index with RaBitQ quantization.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    max_elements : int
        Maximum number of vectors.
    M : int
        Number of connections per layer (default 16).
    ef_construction : int
        Construction search width (default 200).
    total_bits : int
        Bits per dimension for quantization (default 5).
    metric : str
        'l2' or 'ip' (default 'l2').

    Example
    -------
    >>> index = HNSW(dim=128, max_elements=10000)
    >>> index.add(vectors, centroids, cluster_ids)
    >>> ids, dists = index.search(queries, k=10)
    """

    def __init__(self, dim, max_elements, M=16, ef_construction=200,
                 total_bits=5, metric='l2', random_seed=100):
        self.dim = dim
        self.max_elements = max_elements
        metric_type = _core.MetricType.L2 if metric == 'l2' else _core.MetricType.IP
        self._index = _core.HNSW(max_elements, dim, total_bits, M,
                                  ef_construction, random_seed, metric_type)
        self._is_built = False

    def add(self, vectors, centroids, cluster_ids, num_threads=None, faster=True):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        cluster_ids = np.ascontiguousarray(cluster_ids, dtype=np.uint32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vectors.shape[1]}")
        if num_threads is None:
            num_threads = os.cpu_count() or 1
        self._index.construct(centroids, vectors, cluster_ids, num_threads, faster)
        self._is_built = True

    def search(self, queries, k=10, ef=None, num_threads=1):
        if not self._is_built:
            raise RuntimeError("Index not built. Call add() first.")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {queries.shape[1]}")
        if ef is None:
            ef = k
        return self._index.search(queries, queries.shape[0], k, ef, num_threads)

    def save(self, path):
        self._index.save(path)

    def load(self, path, metric='l2'):
        mt = _core.MetricType.L2 if metric == 'l2' else _core.MetricType.IP
        self._index.load(path, mt)
        self._is_built = True
```

IVF and SymQG wrappers follow the same pattern.

---

### Step 4: Tests

```python
# python/tests/test_hnsw.py
import numpy as np
import pytest
from rabitqlib import HNSW

def test_hnsw_build_and_search():
    np.random.seed(42)
    n, dim = 1000, 128
    vectors = np.random.randn(n, dim).astype('float32')

    # Simple clustering (10 clusters)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10, random_state=42, n_init=1).fit(vectors)
    centroids = km.cluster_centers_.astype('float32')
    cluster_ids = km.labels_.astype('uint32')

    index = HNSW(dim=dim, max_elements=n, M=16, total_bits=5)
    index.add(vectors, centroids, cluster_ids)

    ids, dists = index.search(vectors[:10], k=10)
    assert ids.shape == (10, 10)
    assert np.all(ids[:, 0] == np.arange(10))  # self is nearest neighbor

def test_hnsw_save_load(tmp_path):
    # ... build, save, load, verify same search results

def test_hnsw_wrong_dim():
    index = HNSW(dim=128, max_elements=100)
    with pytest.raises(ValueError):
        index.add(np.zeros((10, 64), dtype='float32'), ...)
```

---

### Step 5: CI (.github/workflows/ci.yml)

Since we're doing source-only distribution, CI just needs to test that the package builds and passes tests — no wheel publishing needed.

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Build and install
        run: pip install .
      - name: Run tests
        run: |
          pip install pytest numpy scikit-learn
          pytest python/tests -v
```

Publishing to PyPI is a simple `python -m build --sdist && twine upload dist/*` — just the source tarball.

---

## How `pip install rabitqlib` Works (End to End)

```
User runs: pip install rabitqlib
              │
              ▼
         PyPI has only .tar.gz (source distribution)
              │
              ▼
         pip downloads rabitqlib-0.1.0.tar.gz
              │
              ▼
         pip extracts and runs setup.py on the user's machine
              │
              ▼
         BuildExt detects compiler and CPU capabilities:
         ┌─────────────────────────────────────────────────┐
         │ Linux x86_64 with AVX-512?                      │
         │   → -march=native → compiler emits VPOPCNTDQ   │
         │                                                  │
         │ Linux x86_64 with AVX2 only?                    │
         │   → -march=native → compiler emits PSHUFB       │
         │                                                  │
         │ macOS Apple Silicon?                             │
         │   → -mcpu=apple-m1 → compiler emits NEON        │
         │                                                  │
         │ Old CPU without AVX?                             │
         │   → -march=native → compiler emits SSE4.2       │
         └─────────────────────────────────────────────────┘
              │
              ▼
         Compiled _core.cpython-311-x86_64.so installed
         into site-packages/rabitqlib/
              │
              ▼
         User can now: from rabitqlib import HNSW
```

---

## Implementation Order

| Phase | What | Estimate |
|-------|------|----------|
| 1 | Build system (pyproject.toml + setup.py + MANIFEST.in) | Day 1 |
| 2 | module.cpp with MetricType enum only — verify `pip install .` works | Day 1 |
| 3 | hnsw_bindings.cpp + Python HNSW wrapper — test with real data | Day 2-3 |
| 4 | ivf_bindings.cpp + Python IVF wrapper | Day 3-4 |
| 5 | qg_bindings.cpp + Python SymQG wrapper | Day 4 |
| 6 | Tests (pytest) | Day 5 |
| 7 | CI workflow + TestPyPI upload | Day 6 |
| 8 | Production PyPI release | Day 7 |

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Binding tool | pybind11 | Simpler than SWIG, great C++17 support, proven by hnswlib |
| Build system | setuptools + custom BuildExt | No CMake-to-pip bridge needed, matches hnswlib exactly |
| Distribution | Source-only (sdist) | Always optimal SIMD, simple CI, upgrade to wheels later |
| CPU optimization | `-march=native` at build time | Compiler auto-detects AVX-512/AVX2/NEON |
| GIL release | During construct() and search() | Enables Python threading |
| Module name | `_core` inside `rabitqlib` | Convention: underscore = C extension |
| NumPy integration | `py::array_t<float>` | Zero-copy when contiguous |

---

## Future Upgrade: Pre-built Wheels

When the user base grows and frictionless install becomes important, upgrade to pre-built wheels:

1. Add `cibuildwheel` to CI for building wheels across platforms
2. Ship multiple `.so` variants per wheel (faiss pattern):
   - `_core.so` — generic (SSE4.2)
   - `_core_avx2.so` — AVX2
   - `_core_avx512.so` — AVX-512 with VPOPCNTDQ
3. Add a Python loader that detects CPU and imports the best variant
4. Continue shipping sdist as fallback for unsupported platforms

This is a non-breaking upgrade — the Python API stays identical.

---

## Verification Checklist

- [ ] `pip install .` from khorgosh root succeeds
- [ ] `python -c "import rabitqlib; print(rabitqlib.__version__)"` prints 0.1.0
- [ ] `pytest python/tests/ -v` passes
- [ ] End-to-end: build HNSW from numpy, search, verify recall > 0.8 on 10K random vectors
- [ ] `python -m build --sdist` produces a .tar.gz
- [ ] `twine check dist/*.tar.gz` passes
- [ ] TestPyPI install works: `pip install --index-url https://test.pypi.org/simple/ rabitqlib`
- [ ] `RABITQLIB_NO_NATIVE=1 pip install .` builds without `-march=native`
