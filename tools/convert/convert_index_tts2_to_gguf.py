#!/usr/bin/env python
"""Convert the IndexTTS 2.5 checkpoints into a single GGUF (f16) for vulkan-torch.

Sources (all local, nothing is downloaded):
    <model>/index2.5/gpt.pth            GPT AR backbone (0.8B)
    <model>/index2.5/codec.pth          semantic codec
    <model>/index2.5/s2mel.pth          s2mel (length regulator + DiT/CFM)
    <model>/w2v/model.safetensors       w2v-bert-2.0  (semantic features)
    <model>/camplus/campplus_cn_common.bin   CAMPPlus (speaker style)
    <model>/bigvan/bigvgan_generator.pt      BigVGAN vocoder

Tensor names keep the original checkpoint keys, prefixed by module
(`gpt.`, `codec.`, `s2mel.`, `w2v.`, `campplus.`, `bigvgan.`), so the C# port can
read them straight out of the existing GgufFile.

2D+ weights -> F16 (our matmul runs on fp16 tensor cores anyway, and this halves
the memory bandwidth); 1D tensors (norms, biases) stay F32.

    python tools/convert_index_tts2_to_gguf.py --dry-run          # just list
    python tools/convert_index_tts2_to_gguf.py --out E:/index2.5.gguf
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(ROOT, "移植参考", "model")


def flatten(obj, prefix="", out=None):
    """state_dict (possibly nested / lists / params) -> {flat_name: np.ndarray}."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}{k}.", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}{i}.", out)
    elif hasattr(obj, "detach"):  # torch.Tensor / Parameter
        out[prefix[:-1]] = obj.detach().to("cpu").float().contiguous().numpy()
    elif isinstance(obj, np.ndarray):
        out[prefix[:-1]] = np.ascontiguousarray(obj, dtype=np.float32)
    return out


def load_pth(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # some checkpoints wrap the state_dict (`{"model": ...}` / `{"state_dict": ...}`)
    while isinstance(obj, dict) and len(obj) == 1:
        k = next(iter(obj))
        if k in ("model", "state_dict", "gpt", "net") and hasattr(next(iter(obj.values())), "keys"):
            obj = obj[k]
        else:
            break
    return flatten(obj)


def load_safetensors(path):
    from safetensors.torch import load_file
    return flatten(load_file(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="dir containing index2.5/, w2v/, camplus/, bigvan/")
    ap.add_argument("--out", default=os.path.join(ROOT, "model", "indextts2.5", "indextts2.5.f16.gguf"))
    ap.add_argument("--dry-run", action="store_true", help="only list what would be written")
    a = ap.parse_args()

    m = a.model
    sources = [
        ("gpt",       os.path.join(m, "index2.5", "gpt.pth"), load_pth),
        ("codec",     os.path.join(m, "index2.5", "codec.pth"), load_pth),
        ("s2mel",     os.path.join(m, "index2.5", "s2mel.pth"), load_pth),
        ("w2v",       os.path.join(m, "w2v", "model.safetensors"), load_safetensors),
        ("campplus",  os.path.join(m, "camplus", "campplus_cn_common.bin"), load_pth),
        ("bigvgan",   os.path.join(m, "bigvan", "bigvgan_generator.pt"), load_pth),
    ]

    all_tensors = {}
    for prefix, path, loader in sources:
        if not os.path.isfile(path):
            print(f"!! missing: {path}")
            return 1
        t = loader(path)
        dropped = 0
        for name, arr in t.items():
            # checkpoints that carry their optimiser state would otherwise double the file
            if "optimizer" in name:
                dropped += 1
                continue
            all_tensors[f"{prefix}.{name}"] = arr
        if dropped:
            print(f"{prefix:9s}          (dropped {dropped} optimizer-state tensors)")
        n = sum(v.size for v in t.values())
        print(f"{prefix:9s} {len(t):5d} tensors  {n/1e6:8.1f}M params   <- {os.path.basename(path)}")

    # Emotion matrices for emo_vector control: per-emotion conditioning rows and the matching
    # speaker rows used to pick the closest one by cosine similarity. Small, and not part of any
    # network, but keeping them in the same file makes the GGUF self-contained.
    import torch
    for key, fn in (("emo.spk_matrix", "feat1.pt"), ("emo.emo_matrix", "feat2.pt")):
        t = torch.load(os.path.join(m, "index2.5", fn), map_location="cpu", weights_only=False)
        all_tensors[key] = t.detach().numpy().astype(np.float32)
        print(f"{'emo':9s} 1 tensor   {tuple(t.shape)}   <- {fn}")

    n_f32 = sum(1 for v in all_tensors.values() if v.ndim <= 1)
    n_f16 = len(all_tensors) - n_f32
    nbytes = sum(v.nbytes for v in all_tensors.values())
    print(f"\nTOTAL {len(all_tensors)} tensors, {sum(v.size for v in all_tensors.values())/1e6:.0f}M params")
    print(f"      f16: {n_f16}   f32(1D/norm): {n_f32}")
    print(f"      fp32 on disk: {nbytes/1e9:.2f} GB   -> f16 approx {nbytes/2e9:.2f} GB")

    if a.dry_run:
        print("\n--dry-run, nothing written. Sample names:")
        for name in list(all_tensors)[:12]:
            print(f"   {name:56s} {list(all_tensors[name].shape)}")
        return 0

    from gguf import GGUFWriter, GGMLQuantizationType
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    w = GGUFWriter(a.out, "indextts2.5")
    w.add_uint32("indextts.version", 25)
    for name, arr in all_tensors.items():
        # the writer records raw_dtype but writes the array's bytes as-is, so the
        # data itself has to be converted (a raw_dtype-only hint would lie)
        # the emo matrices are consumed as raw conditioning values, so keep them exact
        if arr.ndim > 1 and not name.startswith("emo."):
            arr = arr.astype(np.float16)
            dt = GGMLQuantizationType.F16
        else:
            dt = GGMLQuantizationType.F32
        w.add_tensor(name, arr, raw_dtype=dt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"\nwrote {a.out}  ({os.path.getsize(a.out)/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
