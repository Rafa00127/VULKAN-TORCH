"""PP-OCRv6 recognition: LCNetV4 backbone + EncoderWithLightSVTR + CTC head.

Input  x: [3, 48, W]  (BGR, (px/255-0.5)/0.5)
Output    : softmax logits [T, 18710]  (T = W/8), CTC-decoded over the char dict
            with index 0 = blank.
"""
import vulkantorch as mt

from ocr_py import backbone
from ocr_py import nn as N

HIDDEN, NHEAD, DEPTH, EPS = 192, 8, 2, 1e-6
HEAD_DIM = HIDDEN // NHEAD


def _svtr_block(x, w, p, trace=None, d=0):
    r = x
    ln1 = N.layernorm_affine(x, w[p + "layer_norm1.weight"], w[p + "layer_norm1.bias"], EPS)
    if trace is not None:
        trace[f"sv{d}_ln1"] = ln1
    qkv = N.linear_bias(ln1, w[p + "self_attn.qkv.weight"], w[p + "self_attn.qkv.bias"])
    T = qkv.shape[0]
    nb1 = 3 * HIDDEN * 4
    q = mt.view_2d(qkv, HIDDEN, T, nb1, 0)
    k = mt.view_2d(qkv, HIDDEN, T, nb1, HIDDEN * 4)
    v = mt.view_2d(qkv, HIDDEN, T, nb1, 2 * HIDDEN * 4)
    scale = HEAD_DIM ** -0.5
    heads = None
    for h in range(NHEAD):
        off = h * HEAD_DIM * 4
        # nb1 = the parent's row stride (the full 3*HIDDEN row), NOT HEAD_DIM*4
        qh = mt.contiguous(mt.view_2d(q, HEAD_DIM, T, 3 * HIDDEN * 4, off))
        kh = mt.contiguous(mt.view_2d(k, HEAD_DIM, T, 3 * HIDDEN * 4, off))
        vh = mt.contiguous(mt.view_2d(v, HEAD_DIM, T, 3 * HIDDEN * 4, off))
        s = mt.soft_max(mt.scale(mt.matmul(qh, mt.transpose(kh)), scale))   # [T,T]
        oh = mt.matmul(s, vh)                                               # [T,HEAD_DIM]
        heads = oh if heads is None else mt.concat(heads, oh, 1)
    o = N.linear_bias(heads, w[p + "self_attn.projection.weight"],
                      w[p + "self_attn.projection.bias"])
    if trace is not None:
        trace[f"sv{d}_attn"] = o
    x = mt.add(r, o)

    r = x
    ln2 = N.layernorm_affine(x, w[p + "layer_norm2.weight"], w[p + "layer_norm2.bias"], EPS)
    if trace is not None:
        trace[f"sv{d}_ln2"] = ln2
    y = N.silu(N.linear_bias(ln2, w[p + "mlp.fc1.weight"], w[p + "mlp.fc1.bias"]))
    y = N.linear_bias(y, w[p + "mlp.fc2.weight"], w[p + "mlp.fc2.bias"])
    if trace is not None:
        trace[f"sv{d}_mlp"] = y
    return mt.add(r, y)


def _logits(x, w, trace=None):
    stages = backbone.forward(x, w, backbone.REC_BLOCKS)
    if trace is not None:
        trace["bb_last"] = stages[-1]
    h = N.avgpool2d(stages[-1], 3, 2, 3, 2)                    # [768,1,W2]
    if trace is not None:
        trace["post_avg"] = h

    hb = "head.encoder."
    res = N.silu(N.conv(h, w[hb + "conv_block.0.convolution.weight"],
                        w[hb + "conv_block.0.convolution.bias"], p=0))
    if trace is not None:
        trace["cb0"] = res
    t = N.silu(N.conv(h, w[hb + "conv_block.1.convolution.weight"],
                      w[hb + "conv_block.1.convolution.bias"], p=0))
    t = mt.add(t, N.silu(N.conv_dw(t, w[hb + "conv_block.2.convolution.weight"],
                                   w[hb + "conv_block.2.convolution.bias"], p=0, pw=3)))
    if trace is not None:
        trace["t"] = t

    C, _, W2 = t.shape
    seq = mt.contiguous(mt.transpose(mt.reshape(t, [C, W2])))  # [W2,192]
    if trace is not None:
        trace["sv_in"] = seq
    for d in range(DEPTH):
        seq = _svtr_block(seq, w, f"{hb}svtr_block.{d}.", trace, d)
        if trace is not None:
            trace[f"sv{d}_out"] = seq
    seq = N.layernorm_affine(seq, w[hb + "norm.weight"], w[hb + "norm.bias"], EPS)
    if trace is not None:
        trace["norm"] = seq

    r_seq = mt.transpose(mt.reshape(res, [C, W2]))             # [W2,192]
    seq = mt.add(seq, r_seq)
    logits = N.linear_bias(seq, w["head.head.weight"], w["head.head.bias"])   # [W2,18710]
    if trace is not None:
        trace["logits"] = logits
    return logits


def forward(x, w, trace=None):
    return mt.soft_max(_logits(x, w, trace))


def forward_ids(x, w):
    """CTC argmax ids [T,1] (I32) straight from the logits.

    Identical decode to ``argmax(forward(x, w))`` (softmax is monotonic), but the
    graph's output is T ints instead of a T×C float map — so only ~4·T bytes cross
    PCIe instead of 4·T·C (17 MB on a wide line). This is the production path."""
    return mt.argmax(_logits(x, w))
