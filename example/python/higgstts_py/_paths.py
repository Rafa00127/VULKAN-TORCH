"""Repo path resolution for the higgstts_py example (robust to relocation)."""

import os
import sys


def find_root(start=None):
    p = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    while True:
        # repo root = has both CMakeLists.txt and src/ (stable markers, not a script name)
        if os.path.exists(os.path.join(p, "CMakeLists.txt")) and os.path.isdir(os.path.join(p, "src")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("vulkan-torch repo root not found (no CMakeLists.txt + src/)")
        p = parent


ROOT = find_root()
BUILD = os.path.join(ROOT, "build")
REFS = os.path.join(ROOT, "data", "higgstts")
REF_AUDIO = os.path.join(ROOT, "data", "ref_audio")
REF_WAV = os.path.join(REF_AUDIO, "melinaref_24k.wav")
REF_TOKENIZER = os.path.join(REF_AUDIO, "higgs_tts_v3_tokenizer.json")
EXAMPLE = os.path.join(ROOT, "example", "python")


def setup():
    """Put the repo root (vulkantorch) + example dir on sys.path and expose native DLLs.

    Called from the package __init__, so any ``import higgstts_py`` bootstraps the
    environment without every script repeating the sys.path / DLL-dir dance.
    """
    for p in (ROOT, EXAMPLE):
        if p not in sys.path:
            sys.path.insert(0, p)
    if sys.platform == "win32":
        bin_ = os.path.join(BUILD, "bin")
        if os.path.isdir(bin_):
            try:
                os.add_dll_directory(bin_)
            except OSError:
                pass
