"""ERNIE-4.5-0.3B backbone as PaddleOCR-VL's language model.

Follows ``src/models/paddleocr.cpp`` (which is ``ernie4-5.cpp`` plus M-RoPE):
RMSNorm, no biases anywhere, SwiGLU, tied lm_head, and multi-section RoPE.

Two things here are not the usual "text LLM" recipe and are easy to get wrong:

* **M-RoPE positions.** Every token carries three positions (t, y, x) plus a zero
  fourth, as 4 blocks in one flat I32 buffer. For text tokens all three equal the
  running index; for image tokens ``t`` is pinned to where the image starts and
  ``y``/``x`` are the merged-grid row/column.
* **The causal mask is 3D lexicographic** on ``(t, y, x)``, not on token index
  (``llama_kv_cache``'s ``is_2d_gt``). Because image tokens share one ``t``, that
  makes the image causal in raster order while letting the text after it see the
  whole image.
"""
import numpy as np

import vulkantorch as mt

N_LAYER = 18
N_EMBD = 1024
N_HEAD = 16
N_HEAD_KV = 2
D_HEAD = 128
EPS = 1e-5

ROPE_MODE_MROPE = 8                 # GGML_ROPE_TYPE_MROPE
ROPE_SECTIONS = [16, 24, 24, 0]     # paddleocr.rope.dimension_sections
ROPE_N_CTX = 131072
ROPE_BASE = 500000.0

# decode-step graph cache: one captured graph per attention-window bucket
STEP_BUCKETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def init_kv(gpu, max_ctx):
    """F16 KV cache, PT [N_LAYER, N_HEAD_KV, max_ctx, D_HEAD]. Zero-filled: slots past
    n_past are masked out, and uninitialised device memory (possibly NaN) would
    otherwise poison the softmax."""
    zeros = np.zeros(N_LAYER * N_HEAD_KV * max_ctx * D_HEAD, dtype=np.float16).tobytes()
    mem = mt.Memory(gpu, 2 * N_LAYER * N_HEAD_KV * max_ctx * D_HEAD * 2)
    return mem, mem.tensor([N_LAYER, N_HEAD_KV, max_ctx, D_HEAD], 1, zeros), \
        mem.tensor([N_LAYER, N_HEAD_KV, max_ctx, D_HEAD], 1, zeros)


def rope_positions(t, y, x):
    """Pack (t, y, x) into the flat I32 buffer ggml_rope_multi reads: four blocks of
    n_tokens, index ``k*n + i`` for k = t, y, x, z."""
    n = len(t)
    z = np.zeros(n, dtype=np.int32)
    return np.concatenate([t, y, x, z]).astype(np.int32)


def causal_mask(t, y, x):
    """PT [n, n] additive mask: 0 where key j may be attended from query i, -inf
    otherwise, with the comparison done lexicographically on (t, y, x)."""
    # allowed(i, j)  <=>  (t_i, y_i, x_i) >=lex (t_j, y_j, x_j)
    gt = ((t[:, None] > t[None, :])
          | ((t[:, None] == t[None, :])
             & ((y[:, None] > y[None, :])
                | ((y[:, None] == y[None, :]) & (x[:, None] >= x[None, :])))))
    return np.where(gt, np.float32(0.0), np.float32(-np.inf))


def _to_heads(x, n_tok, n_head):
    """PT [T, nh*hd] -> PT [nh, T, hd] (ggml ne=[hd,T,nh])."""
    x4 = mt.reshape(x, [n_tok, n_head, D_HEAD, 1])
    return mt.reshape(mt.contiguous(mt.permute_pt(x4, 1, 0, 2, 3)), [n_head, n_tok, D_HEAD])


def _ffn(w, li, x):
    """Post-attention half of a layer: norm -> SwiGLU, with the residual."""
    residual = x
    h = mt.mul(mt.rms_norm(x, EPS), w[f"blk.{li}.ffn_norm.weight"])
    gate = mt.silu(mt.linear(h, w[f"blk.{li}.ffn_gate.weight"]))
    up = mt.linear(h, w[f"blk.{li}.ffn_up.weight"])
    return mt.add(residual, mt.linear(mt.mul(gate, up), w[f"blk.{li}.ffn_down.weight"]))


def _qkv(w, li, x, n_tok, pos):
    h = mt.mul(mt.rms_norm(x, EPS), w[f"blk.{li}.attn_norm.weight"])
    q = mt.reshape(mt.linear(h, w[f"blk.{li}.attn_q.weight"]), [n_tok, N_HEAD, D_HEAD])
    k = mt.reshape(mt.linear(h, w[f"blk.{li}.attn_k.weight"]), [n_tok, N_HEAD_KV, D_HEAD])
    v = mt.reshape(mt.linear(h, w[f"blk.{li}.attn_v.weight"]), [n_tok, N_HEAD_KV, D_HEAD])
    q = mt.rope_multi(q, pos, D_HEAD, ROPE_SECTIONS, ROPE_MODE_MROPE, ROPE_N_CTX, ROPE_BASE)
    k = mt.rope_multi(k, pos, D_HEAD, ROPE_SECTIONS, ROPE_MODE_MROPE, ROPE_N_CTX, ROPE_BASE)
    return (_to_heads(q, n_tok, N_HEAD), _to_heads(k, n_tok, N_HEAD_KV),
            _to_heads(v, n_tok, N_HEAD_KV))


def prefill_layer(w, li, x, kv_k, kv_v, pos, mask, n_tok, max_ctx):
    """One layer over the whole prompt; K/V go to rows [0, n_tok) of the cache."""
    nb1, nb2, nb3 = 2 * D_HEAD, 2 * D_HEAD * max_ctx, 2 * D_HEAD * max_ctx * N_HEAD_KV
    qh, kh, vh = _qkv(w, li, x, n_tok, pos)

    mt.cpy(kh, mt.view_4d(kv_k, D_HEAD, n_tok, N_HEAD_KV, 1, nb1, nb2, nb3, li * nb3))
    mt.cpy(vh, mt.view_4d(kv_v, D_HEAD, n_tok, N_HEAD_KV, 1, nb1, nb2, nb3, li * nb3))
    kf = mt.contiguous(mt.view_3d(kv_k, D_HEAD, n_tok, N_HEAD_KV, nb1, nb2, li * nb3))
    vf = mt.contiguous(mt.view_3d(kv_v, D_HEAD, n_tok, N_HEAD_KV, nb1, nb2, li * nb3))

    attn = mt.flash_attn(qh, kf, vf, mask, 1.0 / np.sqrt(D_HEAD))
    attn = mt.reshape(mt.contiguous(attn), [n_tok, N_HEAD * D_HEAD])
    x = mt.add(x, mt.linear(attn, w[f"blk.{li}.attn_output.weight"]))
    return _ffn(w, li, x)


def decode_layer(w, li, x, kv_k, kv_v, pos, row, mask, lk, max_ctx):
    """One layer for a single token; the KV write goes to the runtime row ``row`` and
    the attention reads a fixed ``lk``-row window, so the captured graph does not
    depend on n_past.

    ``pos`` (the M-RoPE buffer, 4 entries) and ``row`` (the KV cell index) are
    different quantities and must stay different inputs: image tokens share a
    position but never a cell.
    """
    nb1, nb2, nb3 = 2 * D_HEAD, 2 * D_HEAD * max_ctx, 2 * D_HEAD * max_ctx * N_HEAD_KV
    qh, kh, vh = _qkv(w, li, x, 1, pos)

    mt.set_rows(mt.view_3d(kv_k, D_HEAD, max_ctx, N_HEAD_KV, nb1, nb2, li * nb3), kh, row)
    mt.set_rows(mt.view_3d(kv_v, D_HEAD, max_ctx, N_HEAD_KV, nb1, nb2, li * nb3), vh, row)
    kf = mt.contiguous(mt.view_3d(kv_k, D_HEAD, lk, N_HEAD_KV, nb1, nb2, li * nb3))
    vf = mt.contiguous(mt.view_3d(kv_v, D_HEAD, lk, N_HEAD_KV, nb1, nb2, li * nb3))

    attn = mt.flash_attn(qh, kf, vf, mask, 1.0 / np.sqrt(D_HEAD))
    attn = mt.reshape(mt.contiguous(attn), [1, N_HEAD * D_HEAD])
    x = mt.add(x, mt.linear(attn, w[f"blk.{li}.attn_output.weight"]))
    return _ffn(w, li, x)


def head(w, x):
    """output_norm + tied lm_head -> PT [*, n_vocab]."""
    return mt.linear(mt.mul(mt.rms_norm(x, EPS), w["output_norm.weight"]), w["output.weight"])


def pick_bucket(lk_needed):
    for b in STEP_BUCKETS:
        if b >= lk_needed:
            return b
    return None
