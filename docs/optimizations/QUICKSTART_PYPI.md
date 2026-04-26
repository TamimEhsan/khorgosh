# Quick Start: PyPI Packaging

**Goal:** Get a minimal working pip package in 1 week

This guide helps you create a basic pip-installable version of RaBitQLib quickly, focusing on essentials.

---

## Day 1: Setup (2-3 hours)

### Step 1: Create pyproject.toml

```bash
cd /Users/mohammadtamimul.ehsan/Documents/personal/khorgosh
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "pybind11>=2.11.1"]
build-backend = "setuptools.build_meta"

[project]
name = "rabitqlib"
version = "0.1.0"
description = "High-performance vector quantization and search"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "Apache-2.0"}
authors = [{name = "RaBitQ Team"}]
dependencies = ["numpy>=1.20.0"]

[project.urls]
Homepage = "https://github.com/yourusername/rabitqlib"
```

### Step 2: Install pybind11

```bash
pip install pybind11
```

### Step 3: Create Python package structure

```bash
mkdir -p python/rabitqlib/index
mkdir -p src/bindings
```

---

## Day 2: Minimal Bindings (4-5 hours)

### Create src/bindings/module.cpp

```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "RaBitQLib core module";
    m.attr("__version__") = "0.1.0";
}
```

### Update CMakeLists.txt

Add to your existing CMakeLists.txt:

```cmake
# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# Build Python module
pybind11_add_module(_core src/bindings/module.cpp)
target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR}/include)
```

---

## Day 3: Basic HNSW Binding (4-5 hours)

Update `src/bindings/module.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    py::class_<rabitqlib::hnsw::HierarchicalNSW>(m, "HNSW")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(),
             py::arg("max_elements"),
             py::arg("dim"),
             py::arg("total_bits"),
             py::arg("M"),
             py::arg("ef_construction"))
        .def("save", &rabitqlib::hnsw::HierarchicalNSW::save);
}
```

---

## Day 4: Python Wrapper (3-4 hours)

Create `python/rabitqlib/__init__.py`:

```python
from rabitqlib import _core

__version__ = "0.1.0"
__all__ = ["HNSW", "__version__"]

class HNSW:
    """Minimal HNSW wrapper"""
    def __init__(self, dim, max_elements, M=16, ef_construction=200, total_bits=5):
        self._index = _core.HNSW(max_elements, dim, total_bits, M, ef_construction)

    def save(self, filename):
        self._index.save(filename)
```

---

## Day 5: Local Testing (2-3 hours)

### Build and install locally

```bash
pip install -e .
```

### Test in Python

```python
from rabitqlib import HNSW

index = HNSW(dim=128, max_elements=1000)
print("Success!")
```

---

## Day 6: Basic Documentation (2-3 hours)

Update README.md with installation and usage:

```markdown
# RaBitQLib

## Installation

```bash
pip install rabitqlib
```

## Quick Start

```python
from rabitqlib import HNSW

# Create index
index = HNSW(dim=128, max_elements=10000)

# Coming soon: add, search, etc.
```
```

---

## Day 7: First Release (2-3 hours)

### Build wheel

```bash
pip install build twine
python -m build
```

### Test on TestPyPI

```bash
twine upload --repository testpypi dist/*
```

### Test installation

```bash
pip install --index-url https://test.pypi.org/simple/ rabitqlib
```

---

## Success Checklist

After 1 week, you should have:

- ✅ Working pybind11 bindings
- ✅ Minimal Python wrapper
- ✅ Local installation works
- ✅ Can import in Python
- ✅ Uploaded to TestPyPI
- ✅ Basic README

---

## Next Steps

Now follow the full roadmap to add:

- Week 2: Complete HNSW bindings (add, search)
- Week 3: IVF bindings
- Week 4: Tests
- Week 5-6: Full documentation
- Week 7: Production release

---

## Troubleshooting

### Issue: pybind11 not found

```bash
pip install "pybind11[global]"
export pybind11_DIR=$(python -m pybind11 --cmakedir)
```

### Issue: Build fails on macOS

```bash
# Install Xcode command line tools
xcode-select --install
```

### Issue: Import fails

```python
import sys
print(sys.path)  # Check if package is in path
```

---

## Minimal Working Example

Full `src/bindings/module.cpp` for quick copy-paste:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace py = pybind11;
using namespace rabitqlib;

PYBIND11_MODULE(_core, m) {
    m.doc() = "RaBitQLib - High-performance vector search";

    py::enum_<MetricType>(m, "MetricType")
        .value("L2", METRIC_L2)
        .value("IP", METRIC_IP);

    py::class_<hnsw::HierarchicalNSW>(m, "HNSW")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, MetricType>(),
             py::arg("max_elements"),
             py::arg("dim"),
             py::arg("total_bits"),
             py::arg("M"),
             py::arg("ef_construction"),
             py::arg("random_seed") = 100,
             py::arg("metric_type") = METRIC_L2)
        .def("save", &hnsw::HierarchicalNSW::save)
        .def("load", &hnsw::HierarchicalNSW::load);
}
```

That's it! You now have a minimal pip package. Follow the main roadmap for the complete implementation.
