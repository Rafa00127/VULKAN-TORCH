#!/usr/bin/env python
"""Configure + build TensorLibrary with MSVC + Ninja + the VS-bundled CMake.

Runs cmake configure and build, forcing UTF-8 output and English MSVC
diagnostics. Extra CLI args are forwarded to the cmake configure step.

    python build.py                       # Vulkan build -> build/
    MT_BUILD_DIR=build-hip python build.py -DGGML_HIP=ON -DGGML_VULKAN=OFF \
        -DHIP_PLATFORM=amd \
        -DCMAKE_C_COMPILER=<rocm>/bin/clang.exe \
        -DCMAKE_CXX_COMPILER=<rocm>/bin/clang++.exe \
        -DCMAKE_CXX_SCAN_FOR_MODULES=OFF
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, os.environ.get("MT_BUILD_DIR", "build"))

VS_CANDIDATES = [os.environ.get("MT_VS", ""), r"<vs>", r"<vs>"]


def find_vs():
    for p in VS_CANDIDATES:
        if p and os.path.isfile(os.path.join(p, r"VC\Auxiliary\Build\vcvars64.bat")):
            return p
    raise SystemExit("build: no Visual Studio install with vcvars64.bat found")


def msvc_env(vs):
    vcvars = os.path.join(vs, r"VC\Auxiliary\Build\vcvars64.bat")
    line = f'call "{vcvars}" >nul && chcp 65001 >nul && set'
    out = subprocess.check_output(line, shell=True)
    env = {}
    for entry in out.decode("utf-8", "replace").splitlines():
        if "=" in entry:
            k, v = entry.split("=", 1)
            env[k] = v
    return env


def run(cmd, env):
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for raw in proc.stdout:
        sys.stdout.write(raw.decode("utf-8", "replace"))
    return proc.wait()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    vs = find_vs()
    env = os.environ.copy()
    env.update(msvc_env(vs))
    env["PATH"] = os.path.join(vs, r"Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja") \
        + os.pathsep + env.get("PATH", "")
    env["VSLANG"] = "1033"  # force English MSVC diagnostics (no mojibake)
    env["PYTHONUTF8"] = "1"

    cmake = os.path.join(vs, r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")

    try:
        pybind11_dir = subprocess.check_output(
            [sys.executable, "-c", "import pybind11;print(pybind11.get_cmake_dir())"]
        ).decode().strip()
    except Exception:
        pybind11_dir = ""

    cfg = [cmake, "-G", "Ninja", "-S", ROOT, "-B", BUILD,
           "-DCMAKE_BUILD_TYPE=Release",
           f"-DPython_EXECUTABLE={sys.executable}"]
    if pybind11_dir:
        cfg.append(f"-Dpybind11_DIR={pybind11_dir}")
    cfg += sys.argv[1:]

    rc = run(cfg, env)
    if rc:
        sys.exit(rc)

    rc = run([cmake, "--build", BUILD], env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
