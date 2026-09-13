"""Dump the IndexTTS 2.5 s2mel CFM (25-step Euler + CFG) from the official PyTorch, for
validating example/CSharp/IndexTts.Net/Cfm.cs.

Run:  移植参考/.venv-indextts/Scripts/python.exe tools/dump_ref_cfm.py
"""
import os
import sys
from argparse import Namespace

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "移植参考", "index-tts-main"))

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2

MODEL = os.path.join(ROOT, "移植参考", "model", "index2.5")
OUT = os.path.join(ROOT, "data", "indextts_ref")
MEL, TP, TC = 80, 20, 40
SEED = 7
CFG = 0.7
STEPS = 25


def ns(d):
    if isinstance(d, dict):
        return Namespace(**{k: ns(v) for k, v in d.items()})
    return d


def main():
    cfg = ns(yaml.safe_load(open(os.path.join(MODEL, "config.yaml"), encoding="utf-8"))["s2mel"])
    model = MyModel(cfg, use_emovec=False, use_gpt_latent=False)
    load_checkpoint2(model, None, os.path.join(MODEL, "s2mel.pth"),
                     load_only_params=True, ignore_modules=[], is_distributed=False)
    model.eval()
    cfm = model.models["cfm"]
    cfm.estimator.setup_caches(max_batch_size=2, max_seq_length=8192)

    g = torch.Generator().manual_seed(3)
    prompt = torch.randn(1, MEL, TP, generator=g)
    mu = torch.randn(1, TP + TC, 512, generator=g)
    style = torch.randn(1, 192, generator=g)

    torch.manual_seed(SEED)
    z = torch.randn([1, MEL, TP + TC])
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = cfm.inference(mu, torch.LongTensor([mu.size(1)]), prompt, style, None,
                            STEPS, inference_cfg_rate=CFG)
    print("z", list(z.shape), "out", list(out.shape), "prompt_len", TP)

    os.makedirs(OUT, exist_ok=True)
    for name, t in [("cfm_z", z), ("cfm_prompt", prompt), ("cfm_mu", mu),
                    ("cfm_style", style), ("cfm_out", out)]:
        np.save(os.path.join(OUT, name + ".npy"), t.contiguous().numpy().astype(np.float32))
    print("wrote cfm_*.npy ->", OUT)


if __name__ == "__main__":
    main()
