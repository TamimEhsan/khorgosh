// Architecture-selected dispatch entry point. RABITQLIB_TARGET_X86 is a
// compile definition set by CMakeLists.txt from CMAKE_SYSTEM_PROCESSOR: it
// cannot be a runtime check, since dispatch_x86.cpp and (eventually)
// dispatch_highway.cpp would otherwise both define the same symbols.
#if RABITQLIB_TARGET_X86
#include "dispatch_x86.cpp"
#else
#error \
    "rabitqlib: portable dispatch is not implemented yet for this architecture; "
#endif
