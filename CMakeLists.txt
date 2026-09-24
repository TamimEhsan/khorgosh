cmake_minimum_required(VERSION 3.20)

project(RaBitQLib VERSION 0.3.8 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# RABITQLIB_TARGET_X86 is a pure architecture fact: AVX2/AVX-512 intrinsics
# are not expressible source code on a non-x86 compiler, so the AVX2/AVX-512
# kernel source groups and flags (below) can only be compiled on x86 targets
# regardless of which dispatch body is active. Several tests reference
# _avx2/_avx512-suffixed functions directly (explicit-backend parity tests),
# so these sources must keep compiling on real x86 hardware even when
# RABITQ_FORCE_HIGHWAY_BACKEND is set.
option(
    RABITQ_FORCE_HIGHWAY_BACKEND
    "Use the portable (Highway) dispatch backend even on x86, for testing"
    OFF
)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" RABITQLIB_SYSTEM_PROCESSOR_LOWER)
if(RABITQLIB_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64|x86|i[3-6]86)$")
    set(RABITQLIB_TARGET_X86 ON)
else()
    set(RABITQLIB_TARGET_X86 OFF)
endif()

# What dispatch.cpp's #if actually reads: whether to use the x86 dispatch
# body. Distinct from RABITQLIB_TARGET_X86 above so that
# RABITQ_FORCE_HIGHWAY_BACKEND can swap the *dispatch* body without also
# excluding the AVX2/AVX-512 *kernel* sources those explicit-backend tests
# need.
if(RABITQLIB_TARGET_X86 AND NOT RABITQ_FORCE_HIGHWAY_BACKEND)
    set(RABITQLIB_USE_X86_DISPATCH ON)
else()
    set(RABITQLIB_USE_X86_DISPATCH OFF)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RABITQ_BUILD_SAMPLES "Build C++ sample programs" ON)
option(RABITQ_BUILD_TESTS "Build C++ tests" OFF)
option(RABITQ_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(RABITQ_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
# Enable native tuning for local builds; release wheels explicitly set this OFF.
option(
    RABITQ_ENABLE_NATIVE_OPTIMIZATION
    "Optimize generic code for the build machine"
    ON
)

find_package(OpenMP REQUIRED)

add_library(rabitq_compile_options INTERFACE)
target_compile_features(rabitq_compile_options INTERFACE cxx_std_17)
target_compile_options(rabitq_compile_options INTERFACE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-O3>
    $<$<AND:$<BOOL:${RABITQ_ENABLE_NATIVE_OPTIMIZATION}>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-march=native>
)
target_link_libraries(rabitq_compile_options INTERFACE OpenMP::OpenMP_CXX)
if(MSVC)
    # MSVC spells the GNU restrict extension differently. Public headers use
    # __restrict__, so consumers need the same definition as the library.
    target_compile_definitions(rabitq_compile_options INTERFACE __restrict__=__restrict)
    # Eigen's inline code must have one ISA-independent definition across the
    # generic, AVX2, and AVX-512 translation units on MSVC.
    target_compile_definitions(rabitq_compile_options INTERFACE EIGEN_DONT_VECTORIZE)
endif()
if(RABITQ_ENABLE_SANITIZERS)
    target_compile_options(rabitq_compile_options INTERFACE
        $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-fsanitize=address,undefined;-fno-omit-frame-pointer>
    )
    target_link_options(rabitq_compile_options INTERFACE
        $<$<CXX_COMPILER_ID:Clang,GNU>:-fsanitize=address,undefined>
    )
endif()

function(rabitq_enable_warnings target)
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-Wall;-Wextra>"
    )
endfunction()

set(RABITQ_COMMON_SOURCES
    src/index/ivf_search.cpp
    src/index/qg.cpp
    src/utils/cpu_features.cpp
    src/simd/dispatch.cpp
    src/simd/space_generic.cpp
    src/simd/fastscan_generic.cpp
    src/simd/quantization_generic.cpp
    src/simd/estimator_generic.cpp
    src/simd/matrix_generic.cpp
)

include(highway)

set(RABITQ_AVX2_SOURCES)
set(RABITQ_AVX512_SOURCES)
set(RABITQ_AVX512_POPCNT_SOURCES)
set(RABITQ_HNSW_AVX2_SOURCES)
set(RABITQ_HNSW_AVX512_CORE_SOURCES)
set(RABITQ_HNSW_AVX512_POPCNT_SOURCES)

if(RABITQLIB_TARGET_X86)
    set(RABITQ_AVX2_SOURCES
        src/simd/quantization_avx2.cpp
        src/simd/pack_excode_avx2.cpp
        src/simd/space_excode_avx2.cpp
        src/simd/space_avx2.cpp
        src/simd/fastscan_avx2.cpp
        src/simd/estimator_avx2.cpp
        src/simd/matrix_avx2.cpp
        src/simd/warmup_avx2.cpp
        src/simd/rotator_avx2.cpp
    )

    set(RABITQ_AVX512_SOURCES
        src/simd/quantization_avx512.cpp
        src/simd/pack_excode_avx512.cpp
        src/simd/space_excode_avx512.cpp
        src/simd/space_avx512.cpp
        src/simd/fastscan_avx512.cpp
        src/simd/estimator_avx512.cpp
        src/simd/matrix_avx512.cpp
        src/simd/rotator_avx512.cpp
    )

    set(RABITQ_AVX512_POPCNT_SOURCES
        src/simd/warmup_avx512.cpp
    )

    set(RABITQ_HNSW_AVX2_SOURCES
        src/index/hnsw_search_avx2.cpp
    )

    set(RABITQ_HNSW_AVX512_CORE_SOURCES
        src/index/hnsw_search_avx512_core.cpp
    )

    set(RABITQ_HNSW_AVX512_POPCNT_SOURCES
        src/index/hnsw_search_avx512_popcnt.cpp
    )
endif()
# When RABITQLIB_TARGET_X86 is OFF (a genuine non-x86 target, or
# RABITQ_FORCE_HIGHWAY_BACKEND), the AVX2/AVX-512 source lists above stay
# empty and dispatch.cpp's #include falls through to dispatch_highway.cpp
# instead — see docs/portability/highway-plan.md. Kernels that don't have a
# Highway implementation yet throw a descriptive error at call time rather
# than failing to configure/build.

add_library(rabitq_core STATIC
    ${RABITQ_COMMON_SOURCES}
    ${RABITQ_AVX2_SOURCES}
    ${RABITQ_AVX512_SOURCES}
    ${RABITQ_AVX512_POPCNT_SOURCES}
    ${RABITQ_HNSW_AVX2_SOURCES}
    ${RABITQ_HNSW_AVX512_CORE_SOURCES}
    ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
    $<TARGET_OBJECTS:rabitqlib_highway>
)

target_include_directories(rabitq_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# Highway's headers are a private implementation detail of rabitq_core (see
# cmake/highway.cmake): PRIVATE here means consumers linking rabitqlib do not
# also need hwy/highway.h on their include path.
target_include_directories(rabitq_core PRIVATE "${RABITQLIB_HIGHWAY_INCLUDE_DIR}")
target_link_libraries(rabitq_core PUBLIC rabitq_compile_options)
target_link_libraries(rabitq_core PRIVATE ${ATOMICS_LIBRARIES})
set_target_properties(rabitq_core PROPERTIES
    EXPORT_NAME core
    POSITION_INDEPENDENT_CODE ON
)
rabitq_enable_warnings(rabitq_core)

# src/simd/dispatch.cpp is an architecture-selected shell (see
# docs/portability/highway-plan.md): it #includes dispatch_x86.cpp when
# RABITQLIB_TARGET_X86 (the compile definition; note this reads
# RABITQLIB_USE_X86_DISPATCH, not the CMake variable of the same name — see
# the comment above that variable) is set, else dispatch_highway.cpp.
# Neither body is ever added to a source list directly — only the shell is
# a real translation unit; the bodies would duplicate-define every dispatch
# symbol if compiled on their own.
target_compile_definitions(rabitq_core PRIVATE
    RABITQLIB_TARGET_X86=$<BOOL:${RABITQLIB_USE_X86_DISPATCH}>
)

# RABITQ_AVX2_SOURCES etc. are already empty when RABITQLIB_TARGET_X86 is OFF
# (set above), so these calls are harmless no-ops on non-x86 without needing
# their own guard.
if(MSVC)
    set_source_files_properties(
        ${RABITQ_AVX2_SOURCES} ${RABITQ_HNSW_AVX2_SOURCES}
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
    )
    # MSVC enables the complete F/CD/VL/BW/DQ group. Keep the optional
    # backend's runtime guard in cpu_features.cpp aligned with this flag.
    set_source_files_properties(
        ${RABITQ_AVX512_SOURCES} ${RABITQ_AVX512_POPCNT_SOURCES}
        ${RABITQ_HNSW_AVX512_CORE_SOURCES} ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
    )
else()
    set_source_files_properties(${RABITQ_AVX2_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma"
    )
    set_source_files_properties(${RABITQ_AVX512_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mfma"
    )
    set_source_files_properties(${RABITQ_AVX512_POPCNT_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vpopcntdq;-mfma"
    )
    set_source_files_properties(${RABITQ_HNSW_AVX2_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma"
    )
    set_source_files_properties(${RABITQ_HNSW_AVX512_CORE_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx2;-mfma"
    )
    set_source_files_properties(${RABITQ_HNSW_AVX512_POPCNT_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vpopcntdq;-mfma"
    )
endif()

add_library(rabitq_headers INTERFACE)
add_library(rabitqlib::rabitqlib ALIAS rabitq_headers)
target_include_directories(rabitq_headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rabitq_headers INTERFACE rabitq_core)
set_target_properties(rabitq_headers PROPERTIES EXPORT_NAME rabitqlib)
set_target_properties(rabitq_compile_options PROPERTIES EXPORT_NAME compile_options)

install(TARGETS rabitq_headers rabitq_core rabitq_compile_options
    EXPORT rabitqlibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(rabitqlib_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/rabitqlib")
configure_package_config_file(
    cmake/rabitqlibConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfig.cmake"
    INSTALL_DESTINATION "${rabitqlib_cmake_dir}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(EXPORT rabitqlibTargets
    FILE rabitqlibTargets.cmake
    NAMESPACE rabitqlib::
    DESTINATION "${rabitqlib_cmake_dir}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfigVersion.cmake"
    DESTINATION "${rabitqlib_cmake_dir}"
)

if(RABITQ_BUILD_SAMPLES)
    add_subdirectory(sample/cpp)
endif()

if(RABITQ_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()

if(RABITQ_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
