cmake_minimum_required(VERSION 3.20)
project(vulkan-torch LANGUAGES C CXX)

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
# our own shared objects, so we ship a single self-contained vulkantorch.dll and a
# self-contained _vulkantorch*.pyd instead of a pile of ggml-*.dll next to them.
# Flip to ON if you ever want the ggml DLLs back.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "ggml: build static (linked into vulkantorch)" FORCE)

add_subdirectory(${GGML_SOURCE_DIR} ${CMAKE_BINARY_DIR}/ggml)

# ---------------------------------------------------------------------------
# vulkantorch core
# ---------------------------------------------------------------------------
add_library(vulkantorch_core STATIC
    src/runtime.cpp
    src/graph.cpp
    src/tensor.cpp
    src/memory.cpp
    src/gguf_file.cpp
    src/ops.cpp)
target_include_directories(vulkantorch_core PUBLIC src)
target_link_libraries(vulkantorch_core PUBLIC ggml::ggml)

# ---------------------------------------------------------------------------
# shared C ABI library (for .NET / other FFI consumers)
# ---------------------------------------------------------------------------
add_library(vulkantorch SHARED src/c_api.cpp)
target_link_libraries(vulkantorch PRIVATE vulkantorch_core)
target_compile_definitions(vulkantorch PRIVATE VT_SHARED)

# ---------------------------------------------------------------------------
# tests + benchmarks
# ---------------------------------------------------------------------------
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE vulkantorch_core)

# graph-cache replay nondeterminism reproducer (diagnostic)
add_executable(repro_graphcache tests/repro_graphcache.cpp)
target_link_libraries(repro_graphcache PRIVATE vulkantorch_core)

# mirrors bench/bench_py.py and example/CSharp/VulkanTorch.Bench
add_executable(bench_cpp bench/bench_cpp.cpp)
target_link_libraries(bench_cpp PRIVATE vulkantorch_core)

# ---------------------------------------------------------------------------
# python module (optional)
# ---------------------------------------------------------------------------
option(BUILD_PYTHON "Build the vulkantorch python module" ON)
if(BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        # the "vulkantorch" python package lives in vulkantorch/; the compiled
        # extension (_vulkantorch*.pyd) is emitted straight into that package.
        pybind11_add_module(vulkantorch_py vulkantorch/_vulkantorch.cpp)
        target_link_libraries(vulkantorch_py PRIVATE vulkantorch_core)
        set_target_properties(vulkantorch_py PROPERTIES
            OUTPUT_NAME _vulkantorch
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vulkantorch"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vulkantorch"
            # keep the module's import lib out of the way of vulkantorch.dll's
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pyimplib")
    else()
        message(WARNING "pybind11 not found (pass -Dpybind11_DIR=...); skipping python module")
    endif()
endif()
