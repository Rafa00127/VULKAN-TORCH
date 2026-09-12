"""Dump fixed AR inputs + Python (minitorch) reference logits for the C# port.

Uses a synthetic prompt + synthetic codes so the backbone math is validated
without loading the codec. Writes into data/higgstts/:

    ar_prompt_ids.i32       int32 [L]
    ar_delayed_codes.i32    int32 [La, 8]   (delay-patterned ref codes)
    ar_step_codes.i32       int32 [S, 8]    (teacher-forced decode inputs)
    ar_prefill_logits.f32   float32 [N_CB * CB_VOCAB]
    ar_step_logits.f32      float32 [S * N_CB * CB_VOCAB]

    python higgstts_py/dev/dump_ar_ref.py
"""
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
from higgstts_py import ar as AR  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"

BACKBONE_PREFIXES = ("blk.", "token_embd", "fused_embed", "fused_head", "output_norm")


class BackboneWeights:
    """Backbone-only weight dict (avoids loading the codec)."""

    def __init__(self, path, dev):
        self.file = mt.GgufFile(path, dev, 6 << 30)
        self.t = {}
        for name in self.file.names():
            if name.startswith(BACKBONE_PREFIXES):
                self.t[name] = self.file.tensor(name)
        print(f"backbone tensors: {len(self.t)}")

    def __getitem__(self, name):
        return self.t[name]


def main():
    rng = np.random.default_rng(7)
    ref_codes = rng.integers(0, 1024, size=(5, 8)).astype(np.int32)
    delayed = AR._apply_delay_pattern(ref_codes).astype(np.int32)   # [La, 8]
    la = delayed.shape[0]
    prompt_ids = np.array([100, 200, 300, 400, 500, 600] + [-100] * la + [700, 800, 900],
                          dtype=np.int32)
    step_codes = rng.integers(0, 1024, size=(4, 8)).astype(np.int32)

    L = len(prompt_ids)
    La = delayed.shape[0]
    S = step_codes.shape[0]
    max_ctx = L + 16
    print(f"L={L} La={La} S={S} max_ctx={max_ctx}")

    rt = mt.Runtime()
    dev = rt.gpu()
    w = BackboneWeights(GGUF, dev)

    kv_mem = mt.Memory(dev, 2 * AR.N_LAYERS * AR.NKV * max_ctx * AR.HD * 2)
    kv_k = kv_mem.tensor([AR.N_LAYERS, AR.NKV, max_ctx, AR.HD], 1, b"")
    kv_v = kv_mem.tensor([AR.N_LAYERS, AR.NKV, max_ctx, AR.HD], 1, b"")

    # ---- prefill ----
    mask_f = np.where(np.arange(L)[None, :] <= np.arange(L)[:, None], 0.0, -np.inf).astype(np.float32)
    g = mt.Graph(rt, dev)
    g.enter()
    x = AR.build_prefill_embeds(g, w, prompt_ids, delayed)
    pos = g.input_i32([L], np.arange(L, dtype=np.int32).tobytes())
    mask = mt.cast(g.input([L, L], mask_f.tobytes()), 1)
    for li in range(AR.N_LAYERS):
        x = AR.backbone_layer(g, w, li, x, kv_k, kv_v, pos, mask, 0, max_ctx, L)
    hn = mt.mul(mt.rms_norm(x, AR.ECL), w["output_norm.weight"])
    last = mt.reshape(mt.view_2d(mt.contiguous(hn), AR.D, 1, AR.D * 4, (L - 1) * AR.D * 4),
                      [1, AR.D])
    pre = mt.contiguous(AR.fused_head_logits(w, last))
    pre.mark_output()
    g.exit()
    pre_lg = np.frombuffer(pre.to_bytes(), dtype=np.float32).reshape(AR.N_CB, AR.CB_VOCAB).copy()
    print("prefill logits", pre_lg.shape, "range", float(pre_lg.min()), float(pre_lg.max()))

    # ---- teacher-forced decode steps ----
    step_lgs = []
    n_past = L
    for s in range(S):
        cn = step_codes[s]
        g = mt.Graph(rt, dev)
        g.enter()
        e = AR.embed_codes(g, w, cn)
        pos = g.input_i32([1], np.array([n_past], dtype=np.int32).tobytes())
        for li in range(AR.N_LAYERS):
            e = AR.backbone_layer(g, w, li, e, kv_k, kv_v, pos, None, n_past, max_ctx, 1)
        hn = mt.mul(mt.rms_norm(e, AR.ECL), w["output_norm.weight"])
        lg = mt.contiguous(AR.fused_head_logits(w, hn))
        lg.mark_output()
        g.exit()
        step_lgs.append(np.frombuffer(lg.to_bytes(), dtype=np.float32).reshape(AR.N_CB, AR.CB_VOCAB).copy())
        n_past += 1
        print(f"  step {s}: codes={cn.tolist()}")

    # ---- save ----
    prompt_ids.tofile(os.path.join(REFS, "ar_prompt_ids.i32"))
    delayed.tofile(os.path.join(REFS, "ar_delayed_codes.i32"))
    step_codes.tofile(os.path.join(REFS, "ar_step_codes.i32"))
    pre_lg.astype(np.float32).tofile(os.path.join(REFS, "ar_prefill_logits.f32"))
    np.stack(step_lgs).astype(np.float32).tofile(os.path.join(REFS, "ar_step_logits.f32"))
    print("wrote ar_*.i32 / ar_*.f32")


if __name__ == "__main__":
    main()
