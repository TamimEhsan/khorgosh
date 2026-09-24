# Builds the vendored Google Highway snapshot (a git submodule pinned to a
# tag at include/rabitqlib/third/highway; see `git submodule status` or
# .gitmodules for the current pin) as an OBJECT library whose object files
# are folded directly into rabitq_core's own archive (see CMakeLists.txt).
# An OBJECT library is used, rather than a normal STATIC library that rabitq_core
# links against, so that installed consumers of the exported rabitq_core target
# never need to separately find/link a vendored "rabitqlib_highway" target -
# it must not appear in the public install/export interface at all, per
# AGENTS.md's rule against exposing vendored types through public APIs.
#
# This is vendored third-party code per AGENTS.md's vendoring convention: it is
# not part of rabitqlib's public API/install interface and must not be
# reformatted or refactored (this submodule is a full upstream checkout, so
# there is nothing to reformat here in the first place).
#
# We deliberately do not add_subdirectory() Highway's own CMakeLists.txt: it
# only offers a STATIC or SHARED `hwy` target, either of which would need to
# be exported publicly (the opposite of the "invisible implementation detail"
# goal above) to satisfy `install(EXPORT rabitqlibTargets ...)`. Building the
# same handful of core sources ourselves as an OBJECT library avoids that,
# and also skips configuring Highway's own tests/examples/contrib.
#
# Deliberately isolated from `rabitq_compile_options`: Highway's dynamic
# dispatch relies on compiling each `foreach_target.h` translation unit once
# and letting Highway itself choose per-target code paths (function
# multiversioning) at runtime. Forcing -march=native onto these sources would
# bake in the build machine's ISA and defeat that "compile once, run
# anywhere" property, which is the entire reason this library exists.
#
# Sets RABITQLIB_HIGHWAY_INCLUDE_DIR and ATOMICS_LIBRARIES (from FindAtomics)
# in the includer's scope for use by rabitq_core in CMakeLists.txt.

find_package(Atomics REQUIRED)

set(RABITQLIB_HIGHWAY_ROOT "${PROJECT_SOURCE_DIR}/include/rabitqlib/third/highway")
set(RABITQLIB_HIGHWAY_INCLUDE_DIR "${RABITQLIB_HIGHWAY_ROOT}")

if(NOT EXISTS "${RABITQLIB_HIGHWAY_ROOT}/hwy/highway.h")
    message(FATAL_ERROR
        "rabitqlib: the Highway submodule at "
        "include/rabitqlib/third/highway is not initialized. Run:\n"
        "    git submodule update --init --recursive\n"
        "and re-configure."
    )
endif()

set(RABITQLIB_HIGHWAY_SOURCES
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/abort.cc
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/aligned_allocator.cc
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/nanobenchmark.cc
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/per_target.cc
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/perf_counters.cc
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/print.cc
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/profiler.cc
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/targets.cc
    ${RABITQLIB_HIGHWAY_ROOT}/hwy/timer.cc
)

add_library(rabitqlib_highway OBJECT ${RABITQLIB_HIGHWAY_SOURCES})
target_include_directories(rabitqlib_highway PUBLIC
    $<BUILD_INTERFACE:${RABITQLIB_HIGHWAY_INCLUDE_DIR}>
)
target_compile_features(rabitqlib_highway PUBLIC cxx_std_17)
target_compile_options(rabitqlib_highway PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-O3>
)
set_target_properties(rabitqlib_highway PROPERTIES POSITION_INDEPENDENT_CODE ON)
