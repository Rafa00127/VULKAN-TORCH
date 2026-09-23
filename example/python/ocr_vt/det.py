"""PP-OCRv6 text detection: LCNetV4 backbone + RepLKFPN neck + DB head.

Input  x: [3, H, W]  (BGR, ImageNet-normalized, both dims /32)
Output    : probability map [1, H, W] (sigmoid), DB-postprocessed into boxes.
"""
import vulkantorch as mt

from ocr_vt import backbone
from ocr_vt import nn as N

NECK_CH, REDUCE, N_IC = 256, 2, 4
SCALE_LIST = [1, 2, 4, 8]
# (name, kernel, pad) for the three strip-conv scales
_RATIOS = [("long", 7, 3), ("mid", 5, 2), ("short", 3, 1)]


def _conv(x, w, name, p, pw=None):
    b = w[name + ".bias"] if (name + ".bias") in w else None
    return N.conv_tiled(x, w[name + ".weight"], b, p=p, pw=pw)


def _intraclass(x, w, p):
    r = x
    x = _conv(x, w, p + "conv_reduce_channel", p=0)
    for name, k, pad in _RATIOS:
        sym = _conv(x, w, f"{p}symmetric_conv_long_{name}ratio", p=pad)
        ver = _conv(x, w, f"{p}vertical_long_to_small_conv_{name}ratio", p=pad, pw=0)
        hor = _conv(x, w, f"{p}horizontal_small_to_long_conv_{name}ratio", p=0, pw=pad)
        x = mt.add(mt.add(sym, ver), hor)
    x = N.relu(_conv(x, w, p + "conv_final.convolution", p=0))
    return mt.add(r, x)


def _neck(stages, w):
    p = "model.neck."
    n = len(stages)
    adj = [N.conv(s, w[f"{p}input_channel_adjustment_convolution.{i}.weight"], None, p=0)
           for i, s in enumerate(stages)]

    top = [None] * n
    top[-1] = adj[-1]
    for i in range(n - 2, -1, -1):
        top[i] = mt.add(adj[i], mt.upsample(top[i + 1], 2, 0))

    proj = []
    for i in range(n):
        h = top[i] if i < n - 1 else adj[-1]
        proj.append(_conv(h, w, f"{p}input_feature_projection_convolution.{i}", p=4))

    bu = [None] * n
    bu[0] = proj[0]
    for i in range(1, n):
        bu[i] = mt.add(proj[i], N.conv(bu[i - 1],
                      w[f"{p}path_aggregation_head_convolution.{i - 1}.weight"], None, sh=2, sw=2, p=1))

    ref = []
    for i in range(n):
        h = proj[0] if i == 0 else bu[i]
        h = _conv(h, w, f"{p}path_aggregation_lateral_convolution.{i}", p=4)
        h = _intraclass(h, w, f"{p}intraclass_blocks.{i}.")
        ref.append(h)

    ups = []
    for f, s in zip(ref, SCALE_LIST):
        ups.append(mt.upsample(f, s, 0) if s > 1 else f)
    out = ups[-1]
    for u in ups[-2::-1]:
        out = mt.concat(out, u, 0)                     # channel concat, hi-res first
    return out


def _head(x, w):
    x = N.relu(N.conv(x, w["head.conv_down.convolution.weight"],
                      w["head.conv_down.convolution.bias"], p=1))
    u = mt.conv_transpose_2d(w["head.conv_up.convolution.weight"], x, 2)
    u = N.relu(N._cadd(u, w["head.conv_up.convolution.bias"], w["head.conv_up.convolution.weight"].shape[1]))
    o = mt.conv_transpose_2d(w["head.conv_final.weight"], u, 2)
    o = mt.add(o, mt.reshape(w["head.conv_final.bias"], [1, 1, 1]))
    return mt.sigmoid(o)


def forward(x, w):
    stages = backbone.forward(x, w, backbone.DET_BLOCKS)
    return _head(_neck(stages, w), w)
