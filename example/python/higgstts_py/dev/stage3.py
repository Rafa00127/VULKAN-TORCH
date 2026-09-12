"""Stage 1+2+3 — through the semantic encoder. Validated against higgstts_py/ref.npz."""
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
    got = np.frombuffer(got, dtype=np.float32).reshape(shape).astype(np.float64)
    e = float(np.linalg.norm(ref_a - got) / (np.linalg.norm(ref_a) + 1e-12))
    print(f"  [{'OK ' if e < 3e-2 else 'FAIL'}] {name}: fro_rel={e:.3e}")


def main():
    ref = np.load(REF)
    sem = ref["sem_input"].astype(np.float32)[0]
    T = 379

    rt = mt.Runtime()
    gpu = rt.gpu()
    w = HiggsWeights(GGUF, gpu)
    M.build_pce_weight(w)

    g = mt.Graph(rt, gpu)
    g.enter()
    x = g.input([sem.shape[0], 1], sem.tobytes())
    fp = M.feature_projection(w, M.feature_extractor(w, x))
    wh = M.wavlm_encoder(w, fp)
    se = M.semantic_encoder(w, wh)

    se_c = mt.contiguous(se)
    se_c.mark_output()
    g.exit()

    fro("encoder_semantic", ref["encoder_semantic"][0].T, se_c.to_bytes(), (T, 768))


if __name__ == "__main__":
    main()
