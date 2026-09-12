cmake_minimum_required(VERSION 3.20)
project(TensorLibrary LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/bigobj)
endif()

# ---------------------------------------------------------------------------
# ggml (vendored under third_party/, CPU + Vulkan backends only)
# ---------------------------------------------------------------------------
set(GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml"
    CACHE PATH "Path to the ggml source tree")

set(GGML_VULKAN     ON  CACHE BOOL "ggml: use Vulkan")
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Build ggml (base + cpu + vulkan) as static archives and link them straight into
# our own shared objects, so we ship a single self-contained minitorch.dll and a
# self-contained _minitorch*.pyd instead of a pile of ggml-*.dll next to them.
# Flip to ON if you ever want the ggml DLLs back.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "ggml: build static (linked into minitorch)" FORCE)

add_subdirectory(${GGML_SOURCE_DIR} ${CMAKE_BINARY_DIR}/ggml)

# ---------------------------------------------------------------------------
# minitorch core
# ---------------------------------------------------------------------------
add_library(minitorch_core STATIC
    src/runtime.cpp
    src/graph.cpp
    src/tensor.cpp
    src/memory.cpp
    src/gguf_file.cpp
    src/ops.cpp)
target_include_directories(minitorch_core PUBLIC src)
target_link_libraries(minitorch_core PUBLIC ggml::ggml)

# ---------------------------------------------------------------------------
# shared C ABI library (for .NET / other FFI consumers)
# ---------------------------------------------------------------------------
add_library(minitorch SHARED src/c_api.cpp)
target_link_libraries(minitorch PRIVATE minitorch_core)
target_compile_definitions(minitorch PRIVATE MT_SHARED)

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE minitorch_core)

# ---------------------------------------------------------------------------
# python module (optional)
# ---------------------------------------------------------------------------
option(BUILD_PYTHON "Build the minitorch python module" ON)
if(BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        # the "minitorch" python package lives in minitorch/; the compiled
        # extension (_minitorch*.pyd) is emitted straight into that package.
        pybind11_add_module(minitorch_py minitorch/_minitorch.cpp)
        target_link_libraries(minitorch_py PRIVATE minitorch_core)
        set_target_properties(minitorch_py PROPERTIES
            OUTPUT_NAME _minitorch
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/minitorch"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/minitorch"
            # keep the module's import lib out of the way of minitorch.dll's
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pyimplib")
    else()
        message(WARNING "pybind11 not found (pass -Dpybind11_DIR=...); skipping python module")
    endif()
endif()
