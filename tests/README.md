# RaBitQ Testing Framework

This directory contains the comprehensive testing framework for the RaBitQ library with **automatic test discovery**, **test suites**, and **CI/CD integration**.

## ✨ Features

- ✅ **Automatic Test Discovery**: Tests are auto-discovered using `*_test.cpp` naming convention
- ✅ **Test Suite Grouping**: Run tests by category (quantization, utils, fastscan)
- ✅ **Multiple Executables**: Run all tests together or by suite
- ✅ **CI/CD Integration**: GitHub Actions workflows for Linux, macOS, sanitizers, and coverage
- ✅ **Google Test Framework**: Industry-standard testing with 31+ test cases

## Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler (GCC, Clang, or MSVC)
- Google Test (automatically downloaded via CMake FetchContent)

### Installing CMake (macOS)

```bash
brew install cmake
```

### Installing CMake (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install cmake
```

## Building and Running Tests

### Quick Start

From the project root directory:

```bash
# Create build directory
mkdir -p build
cd build

# Configure with tests enabled (tests are OFF by default)
cmake .. -DRABITQ_BUILD_TESTS=ON

# Build the tests
make -j$(nproc)

# Run all tests
./tests/rabitq_tests

# Or use CTest for detailed output
ctest --output-on-failure
```

### Building without Tests

By default, tests are **not built**. If you want to build only the library:

```bash
cmake ..
```

### Running Test Suites

With the new test suite grouping, you can run tests by category:

```bash
cd build/tests

# Run ALL tests (default)
./rabitq_tests

# Run only quantization tests
./rabitq_quantization_tests

# Run only utils tests
./rabitq_utils_tests

# Run only fastscan tests (when available)
./rabitq_fastscan_tests

# Or use make targets
cd ..
make test_all            # Run all tests
make test_quantization   # Run only quantization tests
make test_utils          # Run only utils tests
make test_fastscan       # Run only fastscan tests
```

### Running Specific Tests with Filters

```bash
# Run only quantization tests using gtest filter
./tests/rabitq_tests --gtest_filter="RabitQTest.*"

# Run only rotator tests
./tests/rabitq_tests --gtest_filter="RotatorTest.*"

# Run only space utility tests
./tests/rabitq_tests --gtest_filter="SpaceTest.*"

# List all available tests
./tests/rabitq_tests --gtest_list_tests
```

### Verbose Test Output

```bash
# Show detailed test output
./tests/rabitq_tests --gtest_print_time=1 --gtest_color=yes

# Run with verbose CTest output
ctest -V
```

## Test Structure

```
tests/
├── CMakeLists.txt              # Automatic test discovery & suite configuration
├── main.cpp                    # Test runner entry point
├── fixtures/                   # Test utilities and helpers
│   ├── test_data.hpp          # Test data generation utilities
│   ├── test_data.cpp
│   └── test_helpers.hpp       # Custom assertions and helpers
├── unit/                       # Unit tests (auto-discovered)
│   ├── quantization/
│   │   └── rabitq_test.cpp    # RaBitQ quantization tests (8 tests)
│   ├── utils/
│   │   ├── rotator_test.cpp   # Rotator tests (12 tests)
│   │   └── space_test.cpp     # Space utility tests (11 tests)
│   └── fastscan/              # FastScan tests (to be added)
├── integration/                # Integration tests (to be added)
└── benchmark/                  # Performance benchmarks (to be added)
```

## Test Coverage

### Current Tests

#### Quantization Tests ([rabitq_test.cpp](unit/quantization/rabitq_test.cpp:1))
- ✅ Basic scalar quantization and reconstruction
- ✅ Deterministic quantization
- ✅ Zero vector handling
- ✅ Bit width quality comparison (2-bit, 4-bit, 8-bit)
- ✅ Different data distributions (Gaussian, uniform)
- ✅ Delta and vl parameter relationship
- ✅ Edge value reconstruction

#### Rotator Tests ([rotator_test.cpp](unit/utils/rotator_test.cpp:1))
- ✅ Default rotator type selection
- ✅ Norm preservation
- ✅ Deterministic rotation
- ✅ Zero vector rotation
- ✅ Padding dimension correctness
- ✅ Save/load with file streams
- ✅ Save/load with memory buffers
- ✅ Value distribution after rotation
- ✅ Different input distributions
- ✅ Buffer size consistency

#### Space Utility Tests ([space_test.cpp](unit/utils/space_test.cpp:1))
- ✅ Scalar quantization with uint8_t and uint16_t
- ✅ Deterministic quantization
- ✅ Zero and constant value handling
- ✅ Ordering preservation
- ✅ AnnCandidate struct operations
- ✅ Different value ranges

### Planned Tests (Future)

- Integration tests for IVF+RaBitQ
- Integration tests for HNSW+RaBitQ
- Integration tests for SymQG
- FastScan implementation tests
- Performance benchmarks
- Memory leak detection
- SIMD instruction tests

## Adding New Tests

### Naming Convention

**All test files MUST follow the `*_test.cpp` naming convention** for automatic discovery.

Good examples:
- `rabitq_test.cpp` ✅
- `rotator_test.cpp` ✅
- `ivf_test.cpp` ✅
- `hnsw_rabitq_test.cpp` ✅

Bad examples:
- `test_rabitq.cpp` ❌ (old convention, not auto-discovered)
- `rabitq_tests.cpp` ❌ (plural, not the convention)
- `rabitq.cpp` ❌ (missing _test suffix)

### Creating a New Test File

1. Create a new file in the appropriate directory:
   - `tests/unit/` for unit tests
   - `tests/integration/` for integration tests
   - `tests/benchmark/` for performance tests

2. Follow the naming convention: `<component>_test.cpp`

3. Include necessary headers:
```cpp
#include <gtest/gtest.h>
#include "rabitqlib/your_module.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
```

4. Create a test fixture (optional but recommended):
```cpp
class MyModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
    }

    void TearDown() override {
        // Cleanup code
    }
};
```

5. Write test cases:
```cpp
TEST_F(MyModuleTest, TestSomething) {
    // Arrange
    auto data = TestDataGenerator::GenerateRandomVector(128);

    // Act
    auto result = my_function(data);

    // Assert
    EXPECT_GT(result, 0);
}
```

6. **That's it!** No need to update CMakeLists.txt - your test will be automatically discovered and built.

## Continuous Integration

CI/CD is currently disabled. A comprehensive GitHub Actions workflow is available at [.github/workflows/ci.yml.disabled](../.github/workflows/ci.yml.disabled).

To enable CI/CD, rename the file:
```bash
mv .github/workflows/ci.yml.disabled .github/workflows/ci.yml
```

### Available CI Pipelines (when enabled)

1. **Linux Tests** (GCC & Clang, Debug & Release)
2. **macOS Tests** (Debug & Release)
3. **Sanitizers** (Address Sanitizer & Undefined Behavior Sanitizer)
4. **Code Coverage** (with Codecov integration)
5. **Sample Build** (ensures library examples still work)

### Running CI Locally

```bash
# Mimic Linux GCC Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DRABITQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure

# Mimic Address Sanitizer build
export CC=clang CXX=clang++
cmake -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DRABITQ_BUILD_TESTS=ON
cmake --build build-asan -j$(nproc)
cd build-asan && ctest --output-on-failure
```

## Best Practices

1. **Write isolated tests**: Each test should be independent and not rely on other tests
2. **Use fixtures**: Group related tests using test fixtures
3. **Test edge cases**: Zero values, maximum values, empty inputs, etc.
4. **Use descriptive names**: Test names should clearly describe what they test
5. **Keep tests fast**: Unit tests should complete in milliseconds
6. **Use test helpers**: Leverage `test_helpers.hpp` for common operations
7. **Generate test data**: Use `TestDataGenerator` for consistent test data
8. **Follow naming convention**: Always use `*_test.cpp` for automatic discovery

## Troubleshooting

### CMake cannot find Google Test
Google Test is downloaded automatically via CMake's FetchContent. Ensure you have an internet connection during the first build.

### Test file not discovered
Make sure your test file follows the `*_test.cpp` naming convention and is in the `tests/unit/` directory or subdirectory.

### Compiler errors about AVX512 or SIMD instructions
The library uses SIMD optimizations. If your CPU doesn't support AVX512, the library should fall back to AVX2 or scalar implementations. Check your compiler flags.

### Tests fail with "Rotator dimension too big"
FhtKacRotator supports dimensions with log2 values from 6 to 11 (64 to 2048). For other dimensions, consider using MatrixRotator.

### Memory issues during tests
Some tests allocate large buffers. Ensure your system has sufficient RAM. You can run tests individually to reduce memory pressure.

## Contributing

When adding new features to RaBitQ:

1. Write tests for your feature FIRST (TDD approach recommended)
2. Use the `*_test.cpp` naming convention
3. Ensure all existing tests pass
4. Add integration tests if your feature affects multiple components
5. Update this README if you add new test categories

## Resources

- [Google Test Documentation](https://google.github.io/googletest/)
- [CMake Documentation](https://cmake.org/documentation/)
- [RaBitQ Paper](https://arxiv.org/abs/2409.09913)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
