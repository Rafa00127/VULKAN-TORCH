"""Stage 1 — HuBERT feature extractor (7x Conv1d + GroupNorm + GELU) and the
feature projection (LayerNorm + Linear). Validated against higgstts_py/ref.npz.

Run with the main env python.
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
from higgstts_py.weights import HiggsWeights  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"
REF = os.path.join(REFS, "ref.npz")

KERNELS = [10, 3, 3, 3, 3, 2, 2]
STRIDES = [5, 2, 2, 2, 2, 2, 2]


def norm_over_time(x):
    """Per-channel normalization over the time axis (HF GroupNorm(num_groups=C))."""
    xt = mt.contiguous(mt.transpose(x))   # [C, T]
    n = mt.layer_norm(xt, 1e-5)           # over last dim = T
    return mt.contiguous(mt.transpose(n))  # back to [T, C]


def feature_extractor(g, w, x):
    # x: PT [T, 1]
    for i in range(7):
        x = mt.conv1d(x, w[f"codec.sem.fe.cv.{i}.conv.weight"], STRIDES[i], 0, 1)
        if i == 0:
            x = norm_over_time(x)
            x = mt.add(mt.mul(x, w["codec.sem.fe.cv.0.ln.weight"]),
                       w["codec.sem.fe.cv.0.ln.bias"])
        x = mt.gelu(x)
    return x  # PT [T, 512]


def feature_projection(g, w, x):
    x = mt.layer_norm(x, 1e-5)                       # over channels
    x = mt.add(mt.mul(x, w["codec.sem.fp.ln.weight"]), w["codec.sem.fp.ln.bias"])  # affine
    x = mt.add(mt.linear(x, w["codec.sem.fp.projection.weight"]),
               w["codec.sem.fp.projection.bias"])
    return x  # PT [T, 768]


def main():
    ref = np.load(REF)
    sem_input = ref["sem_input"].astype(np.float32)[0]  # [121387]
    print("input", sem_input.shape)

    rt = mt.Runtime()
    gpu = rt.gpu()
    w = HiggsWeights(GGUF, gpu)

    g = mt.Graph(rt, gpu)
    g.enter()
    x = g.input([sem_input.shape[0], 1], sem_input.tobytes())
    fe = feature_extractor(g, w, x)
    fp = feature_projection(g, w, fe)

    fe_np = np.frombuffer(mt.contiguous(fe).to_bytes(), dtype=np.float32).reshape(-1, 512)
    fp_np = np.frombuffer(mt.contiguous(fp).to_bytes(), dtype=np.float32).reshape(-1, 768)
    g.exit()

    def cmp(name, ref_a, got):
        ref_a = ref_a.reshape(got.shape).astype(np.float64)
        got = got.astype(np.float64)
        e = float(np.linalg.norm(ref_a - got) / (np.linalg.norm(ref_a) + 1e-12))
        # reference runs in bf16 -> ~1e-2 is expected
        print(f"  [{'OK ' if e < 3e-2 else 'FAIL'}] {name}: fro_rel={e:.3e}  ref{ref_a.shape} got{got.shape}")

    # ref fe_out is [1,512,T] (C,T) -> [T,512]; fp_out is [1,T,768]
    cmp("fe_out", ref["fe_out"][0].T, fe_np)
    cmp("fp_out", ref["fp_out"][0], fp_np)


if __name__ == "__main__":
    main()
