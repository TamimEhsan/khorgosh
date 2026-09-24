// Architecture-selected dispatch entry point. RABITQLIB_TARGET_X86 is a
// compile definition set by CMakeLists.txt from CMAKE_SYSTEM_PROCESSOR (or
// forced to 0 by RABITQ_FORCE_HIGHWAY_BACKEND): it cannot be a runtime
// check, since dispatch_x86.cpp and dispatch_highway.cpp both define the
// same symbols. See docs/portability/highway-plan.md for the full design.
#if RABITQLIB_TARGET_X86
#include "dispatch_x86.cpp"
#else
#include "dispatch_highway.cpp"
#endif
