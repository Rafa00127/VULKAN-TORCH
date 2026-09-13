#!/usr/bin/env python
"""Convert PP-OCRv6 (PaddleX safetensors export) -> GGUF F32 for vulkan-torch.

Source (auto-downloaded by paddleocr, nothing fetched here):
    <PADDLEX_HOME>/PP-OCRv6_medium_det[_safetensors]/
    <PADDLEX_HOME>/PP-OCRv6_medium_rec[_safetensors]/

That ``*_safetensors`` dir is PaddleX's own pdparams->safetensors export: an
already-reparameterized inference graph, PyTorch-style names, all F32, with
BatchNorm ``running_mean/var`` present. We therefore *fold each BatchNorm into
its preceding conv* (folded conv gets a bias; BN tensors are dropped), which is
algebraically exact in F32 and removes the need for a runtime BN op.

Conv weights are written **as-is** (PyTorch ``[OC, IC, KH, KW]``): ggml reads
``ne`` reversed, so it lands as ``[KW, KH, IC, OC]`` — exactly what
``ggml_conv_2d`` wants. Same duality the Higgs/IndexTTS converters rely on.

    python tools/convert_ocr_to_gguf.py --dry-run
    python tools/convert_ocr_to_gguf.py            # writes model/ppocrv6/gguf/*.gguf
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PADDLEX = r"<PADDLEX_HOME>"

BN_EPS = 1e-5  # nn.BatchNorm2D default

MODELS = {
    # name -> (safetensors dir, gguf file, kind)
    "det": (os.path.join(PADDLEX, "PP-OCRv6_medium_det_safetensors"), "ppocrv6_det.f32.gguf", "det"),
    "rec": (os.path.join(PADDLEX, "PP-OCRv6_medium_rec_safetensors"), "ppocrv6_rec.f32.gguf", "rec"),
}

# ConvTranspose2d weights carry OC on dim *1* (PyTorch [IC, OC, KH, KW]); every
# other conv here is a plain Conv2d ([OC, IC, KH, KW]). Only conv_up (det head)
# is a conv-transpose and it is the only convT the BN folders touch.
CONVT_WEIGHTS = {"head.conv_up.convolution.weight"}


def load_safetensors(path):
    from safetensors.numpy import load_file
    return dict(load_file(path))


def fold_batchnorms(t):
    """Fold every BatchNorm (identified by a ``*.running_mean`` sibling) into the
    conv at ``<prefix>->convolution.weight``. Returns (folded_tensors, n_folded)."""
    out = dict(t)
    # BN prefixes: strip the trailing param name off a running_mean key
    bn_prefixes = [k[: -len(".running_mean")] for k in t if k.endswith(".running_mean")]
    n = 0
    for p in bn_prefixes:
        conv_key = p.rsplit(".", 1)[0] + ".convolution.weight"
        if conv_key not in t:
            raise KeyError(f"BN at {p} has no {conv_key}")
        gamma = t[p + ".weight"].astype(np.float64)
        beta = t[p + ".bias"].astype(np.float64)
        mean = t[p + ".running_mean"].astype(np.float64)
        var = t[p + ".running_var"].astype(np.float64)
        W = t[conv_key].astype(np.float64)

        # A few BN-wrapped convs are built with bias=True (e.g. the det intraclass
        # `conv_final`); BN runs *after* the conv, so its bias must be folded too.
        conv_bias_key = conv_key.rsplit(".", 1)[0] + ".bias"
        b_conv = t[conv_bias_key].astype(np.float64) if conv_bias_key in t else 0.0

        scale = gamma / np.sqrt(var + BN_EPS)
        if conv_key in CONVT_WEIGHTS:      # [IC, OC, KH, KW]
            Wf = W * scale.reshape(1, -1, 1, 1)
        else:                              # [OC, IC, KH, KW]
            Wf = W * scale.reshape(-1, 1, 1, 1)
        bias = (scale * b_conv + beta - mean * scale).astype(np.float32)

        out[conv_key] = np.ascontiguousarray(Wf.astype(np.float32))
        out[conv_key.rsplit(".", 1)[0] + ".bias"] = np.ascontiguousarray(bias)
        for suf in (".weight", ".bias", ".running_mean", ".running_var"):
            out.pop(p + suf, None)
        n += 1
    return out, n


def extract_char_dict(inference_yml):
    """Pull the rec character_dict out of a PaddleX inference.yml (one char/line)."""
    chars = []
    started = False
    with open(inference_yml, encoding="utf-8") as f:
        for line in f:
            if not started:
                if line.strip() == "character_dict:":
                    started = True
                continue
            if line.startswith("  - "):
                v = line[4:].rstrip("\n")
                # YAML single-quote escape: '' -> '
                if v.startswith("'") and v.endswith("'"):
                    v = v[1:-1].replace("''", "'")
                chars.append(v)
            elif line.strip() and not line.startswith("  - "):
                break  # next top-level key
    return chars


def convert(kind, dry_run, outtype="f32"):
    st_dir, gguf_name, _ = MODELS[kind]
    if outtype == "f16":
        gguf_name = gguf_name.replace(".f32.", ".f16.")
    cfg_path = os.path.join(st_dir, "config.json")
    st_path = os.path.join(st_dir, "model.safetensors")
    if not os.path.isfile(st_path):
        print(f"!! missing {st_path}")
        return 1
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    t = load_safetensors(st_path)
    n_raw = len(t)
    t, n_bn = fold_batchnorms(t)

    print(f"\n=== {kind}: {cfg['model_type']} ===")
    print(f"  raw tensors {n_raw} -> folded {len(t)}  ({n_bn} BatchNorms folded into convs)")
    # name-length sanity (GGML_MAX_NAME = 128 in this fork)
    longest = max(len(k) for k in t)
    print(f"  longest name {longest} chars (limit 128)")
    nparam = sum(v.size for v in t.values())
    print(f"  {nparam/1e6:.1f}M params, {sum(v.nbytes for v in t.values())/1e6:.1f} MB F32")

    if dry_run:
        return 0

    from gguf import GGUFWriter, GGMLQuantizationType
    out_dir = os.path.join(ROOT, "model", "ppocrv6", "gguf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, gguf_name)
    w = GGUFWriter(out_path, arch="ppocrv6")
    w.add_string("ocr.kind", kind)
    w.add_string("ocr.model_type", cfg["model_type"])
    from gguf import GGMLQuantizationType as QT
    for k, v in t.items():
        if outtype == "f16" and v.ndim > 1:      # conv/linear weights -> F16
            w.add_tensor(k, np.ascontiguousarray(v.astype(np.float16)), raw_dtype=QT.F16)
        else:                                     # norms/biases -> F32
            w.add_tensor(k, np.ascontiguousarray(v.astype(np.float32)), raw_dtype=QT.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

    if kind == "rec":
        yml = os.path.join(PADDLEX, "PP-OCRv6_medium_rec", "inference.yml")
        chars = extract_char_dict(yml)
        dict_path = os.path.join(out_dir, "ppocrv6_dict.txt")
        with open(dict_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chars))
        print(f"  wrote {dict_path}  ({len(chars)} chars)  head_out={cfg['head_out_channels']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["det", "rec", "both"])
    ap.add_argument("--outtype", default="f32", choices=["f32", "f16"],
                    help="f16 stores 2D+ weights (conv/linear) as F16, 1D (norm/bias) stay F32")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    kinds = ["det", "rec"] if a.which == "both" else [a.which]
    for k in kinds:
        rc = convert(k, a.dry_run, a.outtype)
        if rc:  # non-zero == failure (0 also means "dry-run ok")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
