"""Dump the IndexTTS 2.5 semantic codec decode chain from the official PyTorch, for validating
example/CSharp/IndexTts.Net/SemanticCodec.cs.

Run:  移植参考/.venv-indextts/Scripts/python.exe tools/dump_ref_codec.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "移植参考", "index-tts-main"))

from indextts.codec.models import EnhancedCodec

MODEL = os.path.join(ROOT, "移植参考", "model", "index2.5")
OUT = os.path.join(ROOT, "data", "indextts_ref")
B, T = 1, 40


def main():
    torch.manual_seed(0)
    model = EnhancedCodec(
        codebook_size=8192, hidden_size=1024, codebook_dim=8,
        vocos_dim=384, vocos_intermediate_dim=2048, vocos_num_layers=12,
        num_quantizers=1, downsample_scale=2,
    ).eval()

    ckpt = torch.load(os.path.join(MODEL, "codec.pth"), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # strip a possible "model." prefix
    fixed = {}
    for k, v in state.items():
        fixed[k[6:] if k.startswith("model.") else k] = v
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    print("missing:", list(missing)[:6], " unexpected:", list(unexpected)[:6])

    codes = torch.randint(0, 8192, (B, T), generator=torch.Generator().manual_seed(1234))
    blocks = {}
    hs = []
    for i, blk in enumerate(model.decoder[0].convnext):
        hs.append(blk.register_forward_hook(
            lambda mod, inp, out, i=i: blocks.__setitem__(i, out.detach())))
    finln = {}
    hs.append(model.decoder[0].final_layer_norm.register_forward_hook(
        lambda mod, inp, out: finln.__setitem__(0, out.detach())))
    with torch.no_grad():
        quant = model.quantizer.vq2emb(codes.unsqueeze(0))     # [1, 1024, T]
        dec = model.decoder(quant)                             # [1, T, 1024]
        x = dec.transpose(1, 2)                                # [1, 1024, T]
        xi = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")  # [1,1024,2T]
        xr = model.up(xi).transpose(1, 2)                      # [1, 2T, 1024]
        xr2 = model.decode(codes)
    for h in hs:
        h.remove()
    print("shapes:", list(quant.shape), list(dec.shape), list(xr.shape), "decode==chain:", bool(torch.allclose(xr, xr2)))
    print("blocks:", [tuple(blocks[i].shape) for i in sorted(blocks)][:3])

    os.makedirs(OUT, exist_ok=True)
    def save(name, t):
        np.save(os.path.join(OUT, name), t.contiguous().numpy().astype(np.float32)
                if t.dtype.is_floating_point else t.contiguous().numpy().astype(np.int32))
    save("codec_codes.npy", codes)
    save("codec_quant.npy", quant)
    save("codec_dec.npy", dec)
    save("codec_xr.npy", xr)
    save("codec_finln.npy", finln[0])
    for i in sorted(blocks):
        save(f"codec_blk{i}.npy", blocks[i])
    print("wrote codec_codes/quant/dec/xr.npy ->", OUT)


if __name__ == "__main__":
    main()
