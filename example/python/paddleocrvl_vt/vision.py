"""PaddleOCR-VL vision encoder: a modified SigLIP ViT plus the mlp_AR projector.

Mirrors ``tools/mtmd/models/paddleocr.cpp`` -> ``clip_graph_paddleocr::build`` with
``clip_graph::build_vit``/``build_patch_merge_permute`` inlined.

Layout convention throughout: tensors are PyTorch order, so a patch token plane is
``[n_patches, n_embd]`` and the patch index is row-major (``t = y * n_w + x``). That
is also the order llama.cpp's ``set_input`` writes the vision RoPE positions in.
"""
import numpy as np

import vulkantorch as mt

N_LAYER = 27
N_EMBD = 1152
N_HEAD = 16
D_HEAD = N_EMBD // N_HEAD          # 72
EPS = 1e-6
PATCH = 14
N_MERGE = 2                         # 2x2 patch merge in the projector
POS_GRID = 27                       # v.position_embd.weight is a 27x27 grid

ROPE_MODE_VISION = 24               # GGML_ROPE_TYPE_VISION
ROPE_SECTIONS = [D_HEAD // 4] * 4   # clip_graph_paddleocr: {d_head/4, ...}
ROPE_N_CTX = 32768
ROPE_BASE = 10000.0


def vision_positions(n_w, n_h):
    """The I32 position buffer for the ViT's mRoPE: 4 blocks of n_patches entries,
    each laid out as ggml reads it (``positions[k*n_patches + t]``).

    The C++ fills this with a doubly-nested 2x2-block walk, but that walk visits the
    patches in plain row-major order, so the buffer is just the patch row/column
    repeated twice (the model reads (y, x, y, x)).
    """
    ys, xs = np.divmod(np.arange(n_w * n_h, dtype=np.int32), n_w)
    return np.concatenate([ys, xs, ys, xs]).astype(np.int32)


def _patch_embed(w, x_img, n_w, n_h):
    """conv2d patch embedding + bias, as PT [n_patches, n_embd] in row-major order."""
    conv = mt.conv2d(w["v.patch_embd.weight"], x_img, PATCH, PATCH, 0, 0, 1, 1)
    # conv output is PT [1, n_embd, n_h, n_w]; move the spatial dims to the front so
    # that flattening (n_h, n_w) makes the patch index y*n_w + x.
    x = mt.reshape(mt.contiguous(mt.permute_pt(conv, 2, 3, 1, 0)), [n_h * n_w, N_EMBD])
    return mt.add(x, w["v.patch_embd.bias"])


def _to_heads(x, n_tok, n_head):
    """PT [T, nh*hd] -> PT [nh, T, hd] (ggml ne=[hd,T,nh]) for flash_attn."""
    x4 = mt.reshape(x, [n_tok, n_head, D_HEAD, 1])
    return mt.reshape(mt.contiguous(mt.permute_pt(x4, 1, 0, 2, 3)), [n_head, n_tok, D_HEAD])


def _layer(w, li, x, vpos, n_tok):
    p = f"v.blk.{li}."
    h = _affine_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])

    q = mt.add(mt.linear(h, w[p + "attn_q.weight"]), w[p + "attn_q.bias"])
    k = mt.add(mt.linear(h, w[p + "attn_k.weight"]), w[p + "attn_k.bias"])
    v = mt.add(mt.linear(h, w[p + "attn_v.weight"]), w[p + "attn_v.bias"])

    q3 = mt.reshape(q, [n_tok, N_HEAD, D_HEAD])
    k3 = mt.reshape(k, [n_tok, N_HEAD, D_HEAD])
    v3 = mt.reshape(v, [n_tok, N_HEAD, D_HEAD])
    q3 = mt.rope_multi(q3, vpos, D_HEAD // 2, ROPE_SECTIONS, ROPE_MODE_VISION, ROPE_N_CTX,
                       ROPE_BASE)
    k3 = mt.rope_multi(k3, vpos, D_HEAD // 2, ROPE_SECTIONS, ROPE_MODE_VISION, ROPE_N_CTX,
                       ROPE_BASE)

    # no attention mask: the encoder sees every patch (clip_graph_paddleocr passes none)
    attn = mt.flash_attn(_to_heads(q3, n_tok, N_HEAD), _to_heads(k3, n_tok, N_HEAD),
                         _to_heads(v3, n_tok, N_HEAD), None, 1.0 / np.sqrt(D_HEAD))
    attn = mt.reshape(mt.contiguous(attn), [n_tok, N_HEAD * D_HEAD])
    x = mt.add(x, mt.add(mt.linear(attn, w[p + "attn_out.weight"]), w[p + "attn_out.bias"]))

    h = _affine_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
    h = mt.add(mt.linear(h, w[p + "ffn_up.weight"]), w[p + "ffn_up.bias"])
    # FFN_GELU (clip.use_gelu), i.e. ggml_gelu's tanh approximation -- not gelu_erf
    h = mt.gelu(h)
    h = mt.add(mt.linear(h, w[p + "ffn_down.weight"]), w[p + "ffn_down.bias"])
    return mt.add(x, h)


def _affine_norm(x, weight, bias):
    return mt.add(mt.mul(mt.layer_norm(x, EPS), weight), bias)


def _patch_merge(x, n_w, n_h):
    """``build_patch_merge_permute`` with scale 2: folds each 2x2 patch block into the
    channel dim, giving PT [n_patches/4, 4*n_embd].

    Every reshape below is a plain (contiguous, row-major) regroup; only the two
    permutes need a copy. Output channel order is (dy, dx, e) with e fastest, which is
    what ``mm.1.weight`` expects.
    """
    x = mt.reshape(x, [n_h, n_w // 2, 2304])
    x = mt.reshape(mt.contiguous(mt.permute_pt(mt.reshape(x, [n_h, n_w // 2, 2304, 1]),
                                              1, 0, 2, 3)), [n_w // 2, n_h // 2, 4608])
    x = mt.reshape(mt.contiguous(mt.permute_pt(mt.reshape(x, [n_w // 2, n_h // 2, 4608, 1]),
                                              1, 0, 2, 3)), [(n_h // 2) * (n_w // 2), 4608])
    return x


def encode(w, x_img, pos_in, vpos, n_w, n_h):
    """Run the encoder.

    ``x_img`` is the ``inp_raw`` node (PT [1, 3, H, W]), ``pos_in`` the interpolated
    learned position embedding as a PT [n_patches, n_embd] node, ``vpos`` the I32
    position node from :func:`vision_positions`. Returns PT [n_tokens, 1024].
    """
    n_patches = n_w * n_h

    x = mt.add(_patch_embed(w, x_img, n_w, n_h), pos_in)
    for li in range(N_LAYER):
        x = _layer(w, li, x, vpos, n_patches)

    # post-layernorm, then the mlp_AR projector
    x = _affine_norm(x, w["v.post_ln.weight"], w["v.post_ln.bias"])
    x = _affine_norm(x, w["mm.input_norm.weight"], w["mm.input_norm.bias"])
    x = _patch_merge(x, n_w, n_h)
    x = mt.add(mt.linear(x, w["mm.1.weight"]), w["mm.1.bias"])
    x = mt.gelu(x)
    return mt.add(mt.linear(x, w["mm.2.weight"]), w["mm.2.bias"])
