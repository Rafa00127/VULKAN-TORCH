#!/bin/sh
# GCC / MinGW build -> ${VT_BUILD_DIR:-build} (same knob/layout as build_win.py).
# With pybind11 installed the python module is built, without it CMake warns and
# builds everything else. The extension always lands in vulkantorch/ (the source
# dir) whatever the build dir is -- MSVC and MinGW produce the same file name
# there, so the last build wins.
BUILD_DIR=${VT_BUILD_DIR:-build}

# Pick the interpreter whose pybind11 CMake config we can find; on Windows
# `python3` is usually the Store stub, hence the fallbacks. Both args below are
# omitted if nothing resolves -- CMake then falls back to its own detection.
PYTHON=${PYTHON:-python3}
PYBIND11_DIR=
PYTHON_EXE=
for p in "$PYTHON" python3 python; do
    if PYBIND11_DIR=$("$p" -c 'import pybind11;print(pybind11.get_cmake_dir())' 2>/dev/null); then
        PYTHON_EXE=$(command -v "$p")
        break
    fi
    PYBIND11_DIR=
done

cmake -G Ninja -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      ${PYTHON_EXE:+-DPython_EXECUTABLE="$PYTHON_EXE"} \
      ${PYBIND11_DIR:+-Dpybind11_DIR="$PYBIND11_DIR"}
cmake --build "$BUILD_DIR" --config Release -j
