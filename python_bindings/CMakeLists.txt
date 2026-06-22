cmake_minimum_required(VERSION 3.15)

# 1. Tell pybind11 to use the modern FindPython module.
set(PYBIND11_FINDPYTHON ON)
if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# 2. Locate pybind11 matching the pip environment
find_package(pybind11 CONFIG REQUIRED)

# 3. Python executable is now reliably in Python_EXECUTABLE (new-style finder)
set(_PY_EXEC "${Python_EXECUTABLE}")

# 4. Grab the NumPy and pybind11 include paths from the active Python environment
execute_process(
    COMMAND "${_PY_EXEC}" "-c" "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND "${_PY_EXEC}" "-c" "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND "${_PY_EXEC}" "-c"
        "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR_FROM_EXEC
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using Python: ${_PY_EXEC}")
message(STATUS "pybind11 include: ${PYBIND11_INCLUDE_DIR}")
message(STATUS "Python include:   ${PYTHON_INCLUDE_DIR_FROM_EXEC}")
message(STATUS "NumPy include:    ${NUMPY_INCLUDE_DIR}")

# 5. Find OpenMP to ensure parallel batched searches compile correctly
find_package(OpenMP REQUIRED)

# 6. Define the shared pybind11 module 
pybind11_add_module(_rabitqlib
    rabitq_bindings.cpp
    hnsw_bindings.cpp
    ivf_bindings.cpp
    symqg_bindings.cpp
)

# 7. Explicitly add include paths so pip-installed pybind11 takes precedence
#    over any system-installed pybind11 (e.g. apt pybind11 2.9.x vs pip 2.13.x)
target_include_directories(_rabitqlib BEFORE PRIVATE
    ${PYBIND11_INCLUDE_DIR}              # pip pybind11 (must beat /usr/include/pybind11)
    ${PYTHON_INCLUDE_DIR_FROM_EXEC}      # Python from the active interpreter
    ${NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}          # For bindings_common.hpp
    ${PROJECT_SOURCE_DIR}/include        # For rabitqlib core headers
)

# 8. Link OpenMP flags natively
target_link_libraries(_rabitqlib PRIVATE 
    OpenMP::OpenMP_CXX
)

# 9. Map files to a 'rabitqlib' directory 
# inside the target wheel, completely bypassing your local layout.
install(TARGETS _rabitqlib DESTINATION rabitqlib)
install(FILES __init__.py DESTINATION rabitqlib)