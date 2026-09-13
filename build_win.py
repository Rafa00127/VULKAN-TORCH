#!/usr/bin/env python
"""Configure + build vulkan-torch on Windows (MSVC + Ninja + the VS-bundled CMake).

Visual Studio is located automatically with vswhere; set VT_VS to point at a
specific install. Forces UTF-8 output and English MSVC diagnostics. Extra CLI
args are forwarded to the cmake configure step.

    python build_win.py                    # -> build/
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, os.environ.get("VT_BUILD_DIR", "build"))

VCVARS = r"VC\Auxiliary\Build\vcvars64.bat"
VC_TOOLS = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"


def _has_vcvars(vs_root):
    return bool(vs_root) and os.path.isfile(os.path.join(vs_root, VCVARS))


def find_vs():
    # 1) explicit override
    if _has_vcvars(os.environ.get("VT_VS", "")):
        return os.environ["VT_VS"]

    # 2) vswhere — ships with every VS 2017+ install
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           r"Microsoft Visual Studio\Installer\vswhere.exe")
    if os.path.isfile(vswhere):
        try:
            out = subprocess.check_output(
                [vswhere, "-latest", "-products", "*", "-requires", VC_TOOLS,
                 "-property", "installationPath"],
                encoding="utf-8", errors="replace").strip()
            if _has_vcvars(out):
                return out
        except OSError:
            pass

    # 3) last resort: the usual install locations
    # vswhere (step 2) already finds any VS install regardless of drive; this is a fallback
    candidates = []
    for base in (r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio"):
        for year in ("2026", "2022", "2019"):
            for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
                candidates.append(os.path.join(base, year, edition))
    for p in candidates:
        if _has_vcvars(p):
            return p

    raise SystemExit(
        "build_win: no Visual Studio with C++ tools found.\n"
        "  Install the 'Desktop development with C++' workload, or set VT_VS "
        "to the VS install root (the one containing VC\\Auxiliary\\Build\\vcvars64.bat).")


def msvc_env(vs):
    vcvars = os.path.join(vs, VCVARS)
    line = f'call "{vcvars}" >nul && chcp 65001 >nul && set'
    # stderr merged in: vcvars may chatter there, and we only parse the `K=V` lines
    out = subprocess.check_output(line, shell=True, stderr=subprocess.STDOUT)
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
    if sys.platform != "win32":
        raise SystemExit("build_win.py is for Windows/MSVC. On other platforms just use "
                         "cmake directly: cmake -B build -DCMAKE_BUILD_TYPE=Release && "
                         "cmake --build build")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    vs = find_vs()
    print(f"using Visual Studio: {vs}", flush=True)
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
