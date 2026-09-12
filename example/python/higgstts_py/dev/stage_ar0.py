"""AR backbone prefill: 36 layers + output_norm, compare every layer. ar_ref.npz."""
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
import higgstts_py.ar as AR  # noqa: E402
from higgstts_py.weights import HiggsWeights  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"
REF = os.path.join(REFS, "ar_ref.npz")
F16 = 1


def main():
    ref = np.load(REF)
    L = len(ref["prompt_ids"])
    max_ctx = L

    rt = mt.Runtime()
    gpu = rt.gpu()
    w = HiggsWeights(GGUF, gpu, backbone=True)

    kv_mem = mt.Memory(gpu, 2 * 36 * 8 * max_ctx * 128 * 2)
    kv_k = kv_mem.tensor([36, 8, max_ctx, 128], F16, b"")
    kv_v = kv_mem.tensor([36, 8, max_ctx, 128], F16, b"")

    mask_f = np.where(np.arange(L)[None, :] <= np.arange(L)[:, None], 0.0, -np.inf).astype(np.float32)

    g = mt.Graph(rt, gpu)
    g.enter()
    x = AR.build_prefill_embeds(g, w, ref["prompt_ids"], ref["ref_codes_delayed"])
    pos = g.input_i32([L], np.arange(L, dtype=np.int32).tobytes())
    mask = mt.cast(g.input([L, L], mask_f.tobytes()), F16)

    outs = []
    for li in range(36):
        x = AR.backbone_layer(g, w, li, x, kv_k, kv_v, pos, mask, 0, max_ctx, L)
        outs.append(x)
    xn = mt.mul(mt.rms_norm(x, AR.ECL), w["output_norm.weight"])
    c = [mt.contiguous(o) for o in outs]
    cc = mt.contiguous(xn)
    for t in c + [cc]:
        t.mark_output()
    g.exit()

    hs = ref["hidden_states"]  # [37,1,L,D]
    for li in range(36):
        got = np.frombuffer(c[li].to_bytes(), dtype=np.float32).astype(np.float64)
        r = hs[li + 1][0].reshape(-1).astype(np.float64)
        print(f"  layer{li:2d}: fro_rel={np.linalg.norm(r-got)/np.linalg.norm(r):.3e}")
    # hs[-1] for this model is the post-output_norm hidden state
    got = np.frombuffer(cc.to_bytes(), dtype=np.float32).astype(np.float64)
    r = hs[36][0].reshape(-1).astype(np.float64)
    print(f"  after output_norm: fro_rel={np.linalg.norm(r-got)/np.linalg.norm(r):.3e}")
    r = ref["last_hidden"][0].reshape(-1).astype(np.float64)
    print(f"  last_hidden: fro_rel={np.linalg.norm(r-got[-2560:])/np.linalg.norm(r):.3e}")


if __name__ == "__main__":
    main()
