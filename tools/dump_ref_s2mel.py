"""Dump the IndexTTS 2.5 s2mel front-end (length_regulator) from the official PyTorch, for
validating example/CSharp/IndexTts.Net/LengthRegulator.cs.

Run:  移植参考/.venv-indextts/Scripts/python.exe tools/dump_ref_s2mel.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "移植参考", "index-tts-main"))

from indextts.codec.models import EnhancedCodec
from indextts.s2mel.modules.length_regulator import InterpolateRegulator

MODEL = os.path.join(ROOT, "移植参考", "model", "index2.5")
OUT = os.path.join(ROOT, "data", "indextts_ref")
DURATION_FACTOR = 1.0


def main():
    torch.manual_seed(0)
    # --- S_infer: the semantic codec's output for the fixed code sequence ---
    codec = EnhancedCodec(
        codebook_size=8192, hidden_size=1024, codebook_dim=8,
        vocos_dim=384, vocos_intermediate_dim=2048, vocos_num_layers=12,
        num_quantizers=1, downsample_scale=2,
    ).eval()
    ck_c = torch.load(os.path.join(MODEL, "codec.pth"), map_location="cpu", weights_only=False)
    codec.load_state_dict({(k[6:] if k.startswith("model.") else k): v
                           for k, v in ck_c["model"].items()}, strict=False)
    codes = torch.from_numpy(
        np.load(os.path.join(OUT, "codec_codes.npy")).astype(np.int64)).reshape(1, -1)

    # --- length_regulator ---
    lr = InterpolateRegulator(
        channels=512, sampling_ratios=[1, 1, 1, 1], is_discrete=False, in_channels=1024,
        vector_quantize=False, codebook_size=2048, out_channels=None, groups=1,
        n_codebooks=1, quantizer_dropout=0.0, f0_condition=False, n_f0_bins=512,
    ).eval()
    ck_s = torch.load(os.path.join(MODEL, "s2mel.pth"), map_location="cpu", weights_only=False)
    lr_sd = ck_s["net"]["length_regulator"]     # keys are already bare (model.0.weight, ...)
    missing, unexpected = lr.load_state_dict(lr_sd, strict=False)
    print("lr missing:", list(missing), " unexpected:", list(unexpected))

    with torch.no_grad():
        S_infer = codec.decode(codes)                                   # [1, 2T, 1024]
        ylens = torch.LongTensor([int(S_infer.shape[1] * 1.72 * DURATION_FACTOR)])
        out = lr(S_infer, ylens=ylens, n_quantizers=3, f0=None)[0]      # [1, L, 512]
        # intermediates: the projection and the nearest-upsampled features
        proj = lr.content_in_proj(S_infer)
        up = torch.nn.functional.interpolate(proj.transpose(1, 2).contiguous(),
                                             size=int(ylens.max()), mode="nearest").transpose(1, 2)
    print("shapes: S_infer", list(S_infer.shape), "ylens", int(ylens[0]),
          "proj", list(proj.shape), "up", list(up.shape), "out", list(out.shape))

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "lr_sinfer.npy"), S_infer.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "lr_proj.npy"), proj.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "lr_up.npy"), up.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "lr_out.npy"), out.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "lr_ylens.npy"), ylens.numpy().astype(np.int32))
    print("wrote lr_sinfer/proj/up/out/ylens.npy ->", OUT)


if __name__ == "__main__":
    main()
