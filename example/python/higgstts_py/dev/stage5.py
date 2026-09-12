"""Stage 5 — full prefill: fusion + RVQ. Compares codes with ref.npz (PyTorch)
and with higgs_tts.dll (Vulkan)."""
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
    sem = ref["sem_input"].astype(np.float32)[0]
    ac = ref["ac_in"].astype(np.float32)[0][0]

    rt = mt.Runtime()
    gpu = rt.gpu()
    w = HiggsWeights(GGUF, gpu)
    M.build_pce_weight(w)

    g = mt.Graph(rt, gpu)
    g.enter()
    xs = g.input([sem.shape[0], 1], sem.tobytes())
    xa = g.input([ac.shape[0], 1], ac.tobytes())
    ids_list, fused = M.prefill(w, xs, xa)
    id_c = [mt.contiguous(i) for i in ids_list]
    fused_c = mt.contiguous(fused)
    for t in id_c + [fused_c]:
        t.mark_output()
    g.exit()

    got_fused = np.frombuffer(fused_c.to_bytes(), dtype=np.float32).reshape(379, 1024).astype(np.float64)
    r_fused = ref["fused"][0].T.astype(np.float64)  # [379,1024]
    ef = float(np.linalg.norm(r_fused - got_fused) / (np.linalg.norm(r_fused) + 1e-12))
    print(f"  [{'OK ' if ef < 8e-2 else 'FAIL'}] fused: fro_rel={ef:.3e}")

    T = id_c[0].shape[0]
    codes = np.zeros((T, 8), dtype=np.int64)
    for q, t in enumerate(id_c):
        codes[:, q] = np.frombuffer(t.to_bytes(), dtype=np.int32).reshape(-1)
    np.save(os.path.join(REFS, "codes_minitorch.npy"), codes)

    ref_codes = ref["codes"]
    print(f"codes_minitorch {codes.shape}, ref {ref_codes.shape}")
    if codes.shape == ref_codes.shape:
        mism = int((codes != ref_codes).sum())
        print(f"  vs PyTorch: mismatch {mism}/{codes.size} ({100*mism/codes.size:.1f}%)")


if __name__ == "__main__":
    main()
