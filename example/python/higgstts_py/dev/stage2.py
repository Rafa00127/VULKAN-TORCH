"""Stage 1+2 — feature extractor, projection, WavLM 12-layer encoder (+stack&mean).
Validated against higgstts_py/ref.npz. Run with the main env python.
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
import higgstts_py.model as M  # noqa: E402
from higgstts_py.weights import HiggsWeights  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"
REF = os.path.join(REFS, "ref.npz")


def fro(name, ref_a, got, shape):
    ref_a = ref_a.reshape(shape).astype(np.float64)
    got = got.reshape(shape).astype(np.float64)
    e = float(np.linalg.norm(ref_a - got) / (np.linalg.norm(ref_a) + 1e-12))
    print(f"  [{'OK ' if e < 3e-2 else 'FAIL'}] {name}: fro_rel={e:.3e}  ref{ref_a.shape} got{got.shape}")


def main():
    ref = np.load(REF)
    sem_input = ref["sem_input"].astype(np.float32)[0]

    rt = mt.Runtime()
    gpu = rt.gpu()
    w = HiggsWeights(GGUF, gpu)
    M.build_pce_weight(w)

    g = mt.Graph(rt, gpu)
    g.enter()
    x = g.input([sem_input.shape[0], 1], sem_input.tobytes())
    fe = M.feature_extractor(w, x)
    fp = M.feature_projection(w, fe)
    wh = M.wavlm_encoder(w, fp)

    # everything we will read must be marked as a graph output (before the first
    # read), otherwise the scheduler may reuse its buffer for a later node
    fe_c, fp_c, wh_c = mt.contiguous(fe), mt.contiguous(fp), mt.contiguous(wh)
    for t in (fe_c, fp_c, wh_c):
        t.mark_output()

    fe_np = np.frombuffer(fe_c.to_bytes(), dtype=np.float32)
    fp_np = np.frombuffer(fp_c.to_bytes(), dtype=np.float32)
    wh_np = np.frombuffer(wh_c.to_bytes(), dtype=np.float32)
    g.exit()

    fro("fe_out", ref["fe_out"][0].T, fe_np, (379, 512))
    fro("fp_out", ref["fp_out"][0], fp_np, (379, 768))
    fro("wavlm_hidden", ref["wavlm_hidden"][0], wh_np, (379, 768))


if __name__ == "__main__":
    main()
