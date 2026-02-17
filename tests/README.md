# RaBitQ Testing Framework

This directory contains the comprehensive testing framework for the RaBitQ library.

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

### Running Specific Tests

```bash
# Run only quantization tests
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
├── CMakeLists.txt              # Test build configuration
├── main.cpp                    # Test runner entry point
├── fixtures/                   # Test utilities and helpers
│   ├── test_data.hpp          # Test data generation utilities
│   ├── test_data.cpp
│   └── test_helpers.hpp       # Custom assertions and helpers
├── unit/                       # Unit tests
│   ├── quantization/
│   │   └── test_rabitq.cpp    # RaBitQ quantization tests
│   └── utils/
│       ├── test_rotator.cpp   # Rotator tests
│       └── test_space.cpp     # Space utility tests
├── integration/                # Integration tests (to be added)
└── benchmark/                  # Performance benchmarks (to be added)
```

## Test Coverage

### Current Tests

#### Quantization Tests (`test_rabitq.cpp`)
- ✅ Basic scalar quantization and reconstruction
- ✅ Deterministic quantization
- ✅ Zero vector handling
- ✅ Bit width quality comparison (2-bit, 4-bit, 8-bit)
- ✅ Different data distributions (Gaussian, uniform)
- ✅ Delta and vl parameter relationship
- ✅ Edge value reconstruction

#### Rotator Tests (`test_rotator.cpp`)
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

#### Space Utility Tests (`test_space.cpp`)
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

### Creating a New Test File

1. Create a new test file in the appropriate directory:
   - `tests/unit/` for unit tests
   - `tests/integration/` for integration tests
   - `tests/benchmark/` for performance tests

2. Include necessary headers:
```cpp
#include <gtest/gtest.h>
#include "rabitqlib/your_module.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
```

3. Create a test fixture (optional but recommended):
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

4. Write test cases:
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

5. Add the test file to `tests/CMakeLists.txt`:
```cmake
add_executable(rabitq_tests
    # ... existing files ...
    unit/your_module/test_your_module.cpp
)
```

## Continuous Integration

To set up CI/CD for automatic testing, create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        compiler: [gcc, clang]

    steps:
    - uses: actions/checkout@v2

    - name: Install dependencies
      run: |
        if [ "$RUNNER_OS" == "Linux" ]; then
          sudo apt-get update
          sudo apt-get install -y cmake
        elif [ "$RUNNER_OS" == "macOS" ]; then
          brew install cmake
        fi

    - name: Configure
      run: cmake -B build -DRABITQ_BUILD_TESTS=ON

    - name: Build
      run: cmake --build build -j$(nproc)

    - name: Test
      run: cd build && ctest --output-on-failure
```

## Best Practices

1. **Write isolated tests**: Each test should be independent and not rely on other tests
2. **Use fixtures**: Group related tests using test fixtures
3. **Test edge cases**: Zero values, maximum values, empty inputs, etc.
4. **Use descriptive names**: Test names should clearly describe what they test
5. **Keep tests fast**: Unit tests should complete in milliseconds
6. **Use test helpers**: Leverage `test_helpers.hpp` for common operations
7. **Generate test data**: Use `TestDataGenerator` for consistent test data

## Troubleshooting

### CMake cannot find Google Test
Google Test is downloaded automatically via CMake's FetchContent. Ensure you have an internet connection during the first build.

### Compiler errors about AVX512 or SIMD instructions
The library uses SIMD optimizations. If your CPU doesn't support AVX512, the library should fall back to AVX2 or scalar implementations. Check your compiler flags.

### Tests fail with "Rotator dimension too big"
FhtKacRotator supports dimensions with log2 values from 6 to 11 (64 to 2048). For other dimensions, consider using MatrixRotator.

### Memory issues during tests
Some tests allocate large buffers. Ensure your system has sufficient RAM. You can run tests individually to reduce memory pressure.

## Contributing

When adding new features to RaBitQ:

1. Write tests for your feature FIRST (TDD approach recommended)
2. Ensure all existing tests pass
3. Add integration tests if your feature affects multiple components
4. Update this README if you add new test categories

## Resources

- [Google Test Documentation](https://google.github.io/googletest/)
- [CMake Documentation](https://cmake.org/documentation/)
- [RaBitQ Paper](https://arxiv.org/abs/2409.09913)
