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

# Auto-discover all test files using naming convention
file(GLOB_RECURSE ALL_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/*/*_test.cpp
)

# Always include fixtures and main
set(COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/test_data.cpp
)

# Single test executable with all tests (default approach)
add_executable(rabitq_tests
    ${COMMON_SRCS}
    ${ALL_TEST_SRCS}
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

# Create test suites for organized testing
# Suite: quantization tests
file(GLOB QUANTIZATION_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/unit/quantization/*_test.cpp)
if(QUANTIZATION_TEST_SRCS)
    add_executable(rabitq_quantization_tests
        ${COMMON_SRCS}
        ${QUANTIZATION_TEST_SRCS}
    )
    target_link_libraries(rabitq_quantization_tests
        gtest
        gtest_main
        rabitq_headers
        pthread
    )
    gtest_discover_tests(rabitq_quantization_tests)

    add_custom_target(test_quantization
        COMMAND rabitq_quantization_tests
        DEPENDS rabitq_quantization_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running quantization tests..."
    )
endif()

# Suite: utils tests
file(GLOB UTILS_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/*_test.cpp)
if(UTILS_TEST_SRCS)
    add_executable(rabitq_utils_tests
        ${COMMON_SRCS}
        ${UTILS_TEST_SRCS}
    )
    target_link_libraries(rabitq_utils_tests
        gtest
        gtest_main
        rabitq_headers
        pthread
    )
    gtest_discover_tests(rabitq_utils_tests)

    add_custom_target(test_utils
        COMMAND rabitq_utils_tests
        DEPENDS rabitq_utils_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running utils tests..."
    )
endif()

# Suite: fastscan tests (when added)
file(GLOB FASTSCAN_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/unit/fastscan/*_test.cpp)
if(FASTSCAN_TEST_SRCS)
    add_executable(rabitq_fastscan_tests
        ${COMMON_SRCS}
        ${FASTSCAN_TEST_SRCS}
    )
    target_link_libraries(rabitq_fastscan_tests
        gtest
        gtest_main
        rabitq_headers
        pthread
    )
    gtest_discover_tests(rabitq_fastscan_tests)

    add_custom_target(test_fastscan
        COMMAND rabitq_fastscan_tests
        DEPENDS rabitq_fastscan_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running fastscan tests..."
    )
endif()

# Add custom target to run all tests
add_custom_target(test_all
    COMMAND rabitq_tests
    DEPENDS rabitq_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all tests..."
)

# Print discovered tests
message(STATUS "Discovered ${CMAKE_CURRENT_LIST_LINE} test files:")
foreach(TEST_SRC ${ALL_TEST_SRCS})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SRC})
    message(STATUS "  - ${REL_PATH}")
endforeach()
