"""Dump the IndexTTS 2.5 s2mel DiT estimator (one forward) from the official PyTorch, for
validating example/CSharp/IndexTts.Net/Dit.cs.

Run:  移植参考/.venv-indextts/Scripts/python.exe tools/dump_ref_dit.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "移植参考", "index-tts-main"))

from indextts.s2mel.modules.diffusion_transformer import DiT

MODEL = os.path.join(ROOT, "移植参考", "model", "index2.5")
OUT = os.path.join(ROOT, "data", "indextts_ref")
B, MEL, T = 1, 80, 60          # T = total mel frames (prompt + generated)
PROMPT_LEN = 20


def build_args():
    return SimpleNamespace(
        DiT=SimpleNamespace(
            hidden_dim=512, num_heads=8, depth=13, class_dropout_prob=0.1, block_size=8192,
            in_channels=80, style_condition=True, final_layer_type="wavenet", target="mel",
            content_dim=512, content_codebook_size=1024, content_type="discrete",
            f0_condition=False, n_f0_bins=512, content_codebooks=1, is_causal=False,
            long_skip_connection=True, zero_prompt_speech_token=False, time_as_token=False,
            style_as_token=False, uvit_skip_connection=True, add_resblock_in_transformer=False),
        wavenet=SimpleNamespace(hidden_dim=512, num_layers=8, kernel_size=5, dilation_rate=1,
                                p_dropout=0.2, style_condition=True),
        style_encoder=SimpleNamespace(dim=192),)


def main():
    torch.manual_seed(0)
    dit = DiT(build_args()).eval()
    ck = torch.load(os.path.join(MODEL, "s2mel.pth"), map_location="cpu", weights_only=False)
    cfm = ck["net"]["cfm"]
    sd = {k[len("estimator."):]: v for k, v in cfm.items() if k.startswith("estimator.")}
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    print("dit missing:", list(missing), " unexpected:", list(unexpected))
    dit.setup_caches(max_batch_size=1, max_seq_length=8192)

    g = torch.Generator().manual_seed(11)
    x = torch.randn(B, MEL, T, generator=g)
    prompt_x = torch.zeros(B, MEL, T)
    prompt_x[:, :, :PROMPT_LEN] = torch.randn(B, MEL, PROMPT_LEN, generator=g)
    x[:, :, :PROMPT_LEN] = 0
    t = torch.tensor([0.5])
    style = torch.randn(B, 192, generator=g)
    cond = torch.randn(B, T, 512, generator=g)
    x_lens = torch.tensor([T])

    hooks, blk, mid = [], {}, {}
    import indextts.s2mel.modules.gpt_fast.model as gm
    rope_cap = {}
    _orig_rope = gm.apply_rotary_emb

    def _rope_wrap(x, fc):
        out = _orig_rope(x, fc)
        if "in" not in rope_cap:
            rope_cap["in"] = x.detach().clone()
            rope_cap["out"] = out.detach().clone()
        return out

    gm.apply_rotary_emb = _rope_wrap
    for i, layer in enumerate(dit.transformer.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, i=i: blk.__setitem__(i, out.detach())))
    hooks.append(dit.cond_x_merge_linear.register_forward_hook(
        lambda m, inp, out: mid.__setitem__("merge", out.detach())))
    hooks.append(dit.transformer.layers[0].attention_norm.register_forward_hook(
        lambda m, inp, out: mid.__setitem__("an0", out.detach())))
    hooks.append(dit.transformer.layers[0].attention.register_forward_hook(
        lambda m, inp, out: mid.__setitem__("at0", out.detach())))
    with torch.no_grad():
        out = dit(x, prompt_x, x_lens, t, style, cond)
    for h in hooks:
        h.remove()

    print("out", list(out.shape), "blk0", list(blk[0].shape))
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "dit_x.npy"), x.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "dit_prompt_x.npy"), prompt_x.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "dit_style.npy"), style.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "dit_cond.npy"), cond.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "dit_out.npy"), out.contiguous().numpy().astype(np.float32))
    for i in (0, 6, 12):
        np.save(os.path.join(OUT, f"dit_blk{i}.npy"), blk[i].contiguous().numpy().astype(np.float32))
    for k, v in mid.items():
        np.save(os.path.join(OUT, f"dit_{k}.npy"), v.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "dit_qin.npy"), rope_cap["in"].contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "dit_qrope.npy"), rope_cap["out"].contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "dit_t.npy"), t.numpy().astype(np.float32))
    print("wrote dit_*.npy ->", OUT)


if __name__ == "__main__":
    main()
