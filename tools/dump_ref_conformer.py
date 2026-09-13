#!/usr/bin/env python
"""Dump Conformer (emo_conditioning_encoder) internals from the official model so
the C# port can be bisected stage by stage. Run with the dedicated venv:

    移植参考\\.venv-indextts\\Scripts\\python.exe tools\\dump_ref_conformer.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "移植参考", "index-tts-main")
MODEL = os.path.join(ROOT, "移植参考", "model")
OUT = os.path.join(ROOT, "data", "indextts_ref")

sys.path.insert(0, MAIN)


def done(name, arr):
    if hasattr(arr, "detach"):
        a = arr.detach().to(torch.float32).cpu().contiguous().numpy()
    else:
        a = np.ascontiguousarray(arr, np.float32)
    np.save(os.path.join(OUT, name + ".npy"), a)
    print(f"  OK   {name:30s} {list(a.shape)}")


def round_f16(module):
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() >= 2:
                p.copy_(p.half().float())
    return module


def main():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(MODEL, "index2.5", "config.yaml"))
    torch.manual_seed(1234)

    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    gpt = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
    load_checkpoint(gpt, os.path.join(MODEL, "index2.5", "gpt.pth"))
    round_f16(gpt)
    gpt.eval()

    enc = gpt.emo_conditioning_encoder
    cap = {}

    def hk(key, idx=0):
        def f(mod, inp, out):
            o = out[idx] if isinstance(out, tuple) else out
            cap[key] = o.detach()
        return f

    hs = [enc.embed.register_forward_hook(hk("emo_embed", 0)),   # (x, pos_emb, mask)
          enc.embed.register_forward_hook(hk("emo_pos", 1)),
          enc.embed.conv.register_forward_hook(hk("emo_conv", 0)),
          enc.embed.out.register_forward_hook(hk("emo_out", 0)),
          enc.register_forward_hook(hk("emo_conf", 0))]
    for i in range(len(enc.encoders)):
        hs.append(enc.encoders[i].register_forward_hook(hk(f"emo_l{i}", 0)))
    hs.append(enc.encoders[0].self_attn.register_forward_hook(hk("emo_l0_attn", 0)))
    hs.append(enc.encoders[0].conv_module.register_forward_hook(hk("emo_l0_conv", 0)))
    hs.append(enc.encoders[0].feed_forward.register_forward_hook(hk("emo_l0_ff", 0)))

    Te = 40
    emo_in = torch.randn(1, Te, 1024)
    done("gpt_emo_conf_in", emo_in)
    with torch.no_grad():
        enc(emo_in, torch.LongTensor([Te]))
    for h in hs:
        h.remove()

    for k, v in cap.items():
        done(k, v)

    # conditioning embeddings from the same emo_in (so all refs stay consistent)
    spk_in = torch.randn(1, 192)
    with torch.no_grad():
        spk = gpt.spk_emb_proj(spk_in)
        ev = gpt.get_emovec(emo_in, torch.LongTensor([Te]))
    done("emo_spk_in", spk_in)
    done("emo_spk", spk)
    done("emo_emovec", ev)
    print(f"\n{len(cap)} tensors -> {OUT}")


if __name__ == "__main__":
    main()
