"""Dump fixed decode inputs + the Python (minitorch) reference output.

The C# port (example/CSharp/HiggsTts.Net) reads dec_codes.i32, decodes, and
compares against dec_py.f32 so the two implementations are checked bit-for-bit.

    python higgstts_py/dev/dump_decode_ref.py
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "example", "python"))

from higgstts_py import _paths  # noqa: E402

BUILD = _paths.BUILD
REFS = _paths.REFS

import minitorch as mt  # noqa: E402
from higgstts_py import decode as DE  # noqa: E402
from higgstts_py.weights import HiggsWeights  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"


def main():
    codes = np.load(os.path.join(REFS, "ar_codes.npy")).astype(np.int32)
    print("codes", codes.shape)

    rt = mt.Runtime()
    print("backend:", rt.name())
    w = HiggsWeights(GGUF, rt.gpu())
    pcm = DE.dac_decode(rt, w, codes).astype(np.float32)
    print("pcm", pcm.shape, "range", float(pcm.min()), float(pcm.max()))

    codes.tofile(os.path.join(REFS, "dec_codes.i32"))
    pcm.tofile(os.path.join(REFS, "dec_py.f32"))
    print("wrote dec_codes.i32 / dec_py.f32")


if __name__ == "__main__":
    main()
