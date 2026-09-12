"""Stage 4 — DAC acoustic encoder. Validated against higgstts_py/ref.npz."""
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
import higgstts_py.model as M  # noqa: E402
from higgstts_py.weights import HiggsWeights  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"
REF = os.path.join(REFS, "ref.npz")


def main():
    ref = np.load(REF)
    ac_in = ref["ac_in"].astype(np.float32)[0][0]  # [364000]
    print("ac_in", ac_in.shape)

    rt = mt.Runtime()
    gpu = rt.gpu()
    w = HiggsWeights(GGUF, gpu)

    g = mt.Graph(rt, gpu)
    g.enter()
    x = g.input([ac_in.shape[0], 1], ac_in.tobytes())
    y = M.acoustic_encoder(w, x)
    yc = mt.contiguous(y)
    yc.mark_output()
    g.exit()

    got = np.frombuffer(yc.to_bytes(), dtype=np.float32)
    ref_a = ref["acoustic_encoder"][0].T  # [379,256]
    got = got.reshape(ref_a.shape).astype(np.float64)
    e = float(np.linalg.norm(ref_a.astype(np.float64) - got) / (np.linalg.norm(ref_a) + 1e-12))
    print(f"  [{'OK ' if e < 8e-2 else 'FAIL'}] acoustic_encoder: fro_rel={e:.3e}  {got.shape}")


if __name__ == "__main__":
    main()
