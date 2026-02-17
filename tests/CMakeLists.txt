cmake_minimum_required(VERSION 3.10)

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

# Compiler flags for tests (same as main project)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Test executable
add_executable(rabitq_tests
    main.cpp
    fixtures/test_data.cpp
    unit/quantization/test_rabitq.cpp
    unit/utils/test_rotator.cpp
    unit/utils/test_space.cpp
)

# Link against Google Test and pthread
target_link_libraries(rabitq_tests
    gtest
    gtest_main
    rabitq_headers
    pthread
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(rabitq_tests)

# Add custom target to run tests
add_custom_target(run_tests
    COMMAND rabitq_tests
    DEPENDS rabitq_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running tests..."
)
