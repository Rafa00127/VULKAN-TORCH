"""Dump the IndexTTS 2.5 BigVGAN generator (mel -> wav) from the official PyTorch, for
validating example/CSharp/IndexTts.Net/BigVgan.cs.

Run:  移植参考/.venv-indextts/Scripts/python.exe tools/dump_ref_bigvgan.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "移植参考", "index-tts-main"))

from indextts.s2mel.modules.bigvgan.bigvgan import BigVGAN
from indextts.s2mel.modules.bigvgan.env import AttrDict

MODEL = os.path.join(ROOT, "移植参考", "model", "index2.5")
OUT = os.path.join(ROOT, "data", "indextts_ref")
MELS, TT = 80, 24


def main():
    torch.manual_seed(0)
    with open(os.path.join(ROOT, "移植参考", "index-tts-main", "indextts", "s2mel",
                           "modules", "bigvgan", "config.json"), encoding="utf-8") as f:
        h = AttrDict(json.load(f))
    gen = BigVGAN(h, use_cuda_kernel=False)
    ck = torch.load(os.path.join(ROOT, "移植参考", "model", "bigvan", "bigvgan_generator.pt"),
                    map_location="cpu", weights_only=False)
    sd = ck["generator"]
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    print("bigvgan missing:", list(missing)[:6], " unexpected:", list(unexpected)[:6])
    gen.eval()

    x = torch.randn(1, MELS, TT, generator=torch.Generator().manual_seed(5))
    blk = {}
    rb = {}
    hooks = [gen.resblocks[0].register_forward_hook(lambda m, i, o: blk.__setitem__("rb0", o.detach().clone())),
             gen.conv_pre.register_forward_hook(lambda m, i, o: blk.__setitem__("pre", o.detach().clone())),
             gen.ups[0][0].register_forward_hook(lambda m, i, o: blk.__setitem__("up0", o.detach().clone())),
             gen.resblocks[0].activations[0].upsample.register_forward_hook(
                 lambda m, i, o: blk.__setitem__("a0up", o.detach().clone())),
             gen.resblocks[0].activations[0].act.register_forward_hook(
                 lambda m, i, o: blk.__setitem__("a0act", o.detach().clone())),
             gen.resblocks[0].activations[0].register_forward_hook(
                 lambda m, i, o: blk.__setitem__("a0", o.detach().clone())),
             gen.resblocks[0].convs1[0].register_forward_hook(
                 lambda m, i, o: blk.__setitem__("c10", o.detach().clone())),
             gen.resblocks[0].activations[2].register_forward_hook(
                 lambda m, i, o: blk.__setitem__("a2", o.detach().clone())),
             gen.resblocks[0].activations[3].register_forward_hook(
                 lambda m, i, o: blk.__setitem__("a3", o.detach().clone())),
             gen.resblocks[0].convs1[1].register_forward_hook(
                 lambda m, i, o: blk.__setitem__("c11", o.detach().clone())),
             *[gen.resblocks[0].convs2[q].register_forward_hook(
                 lambda m, i, o, q=q: blk.__setitem__(f"c2{q}", o.detach())) for q in range(3)],
             *[gen.resblocks[n].register_forward_hook(
                 lambda m, i, o, n=n: rb.__setitem__(n, o.detach().clone())) for n in range(18)],
             gen.activation_post.register_forward_hook(
                 lambda m, i, o: blk.__setitem__("actpost", o.detach()))]
    with torch.no_grad():
        out = gen(x)
    for hk in hooks:
        hk.remove()
    print("out", list(out.shape))

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "bv_x.npy"), x.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "bv_out.npy"), out.contiguous().numpy().astype(np.float32))
    for k, v in blk.items():
        np.save(os.path.join(OUT, f"bv_{k}.npy"), v.contiguous().numpy().astype(np.float32))
    gg = ck["generator"] if False else None
    np.save(os.path.join(OUT, "bv_up0w.npy"), (ck["generator"]["ups.0.0.weight_v"].float()
            * ck["generator"]["ups.0.0.weight_g"].float()
            / ck["generator"]["ups.0.0.weight_v"].float().norm(dim=(1, 2), keepdim=True)
            ).permute(1, 0, 2).flip(2).contiguous().numpy().astype(np.float32))
    for k in range(6):
        np.save(os.path.join(OUT, f"bv_stage{k}.npy"),
                ((rb[3*k] + rb[3*k+1] + rb[3*k+2]) / 3).contiguous().numpy().astype(np.float32))
    print("wrote bv_*.npy ->", OUT)


if __name__ == "__main__":
    main()
