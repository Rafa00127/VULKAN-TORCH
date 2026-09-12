"""Find what preprocessing sits between fp_out and encoder layer-0 attention."""
import os
import sys

import numpy as np
import torch

SGL = r"<local>/sglang\sglang-omni-main"
sys.path.insert(0, SGL)
from higgs_tts_standalone.loader import load_model  # noqa: E402

m = load_model(os.path.join(SGL, "models", "bosonai--higgs-audio-v3-tts-4b"),
               device="cuda", dtype="bfloat16")
sm = m.codec.model.semantic_model
enc = sm.encoder
print("encoder type:", type(enc).__name__)
print("encoder children:", [n for n, _ in enc.named_children()])
print("layer0 children:", [n for n, _ in enc.layers[0].named_children()])

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
REFS = os.path.join(_ROOT, "data", "higgstts")
ref = np.load(os.path.join(REFS, "ref.npz"))
fp_out = torch.from_numpy(ref["fp_out"]).to(device="cuda", dtype=torch.bfloat16)

caps = {}


def mk(name, idx=0):
    def h(_m, i, o):
        caps[name] = i[idx]
    return h


enc.register_forward_hook(mk("enc_in"))
enc.layers[0].register_forward_hook(mk("l0_in"))
enc.layers[0].self_attn.register_forward_hook(mk("l0_attn_in"))

with torch.no_grad():
    enc(fp_out)

fpo = fp_out[0].float().cpu().numpy()


def fro(a, b):
    a = a[0].float().cpu().numpy() if a.dim() == 3 else a.float().cpu().numpy()
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64))
                 / (np.linalg.norm(b.astype(np.float64)) + 1e-12))


for k, v in caps.items():
    print(f"{k} vs fp_out: {fro(v, fpo):.4f}   range[{float(v.min()):.3f},{float(v.max()):.3f}]")
