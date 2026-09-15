"""Small op helpers shared by the det/rec port.

Feature maps are PT [C, H, W] (batch N=1 is squeezed by the binding), which is
also how conv2d/conv2d_dw report their output. A per-channel weight/bias of
length C therefore broadcasts as a ggml ne=[C,1,1] tensor.
"""
import os

import vulkantorch as mt


def _cadd(y, b, oc):
    """Add a per-channel bias. A conv output is [OC,H,W] -> ggml ne=[W,H,OC], so the
    bias must be [OC,1,1] (ne=[1,1,OC]) to broadcast along the channel axis."""
    return mt.add(y, mt.reshape(b, [oc, 1, 1]))


# There are two conv2d paths. The fused one (`conv2d_direct`) emits a single implicit-GEMM
# node and materialises no im2col tensor at all, so past a certain output size it wins --
# on time, and again on capture, which is the dominant cost when the input size keeps
# changing (a hand-drawn screenshot is never the same twice). Same result either way; only
# the kernel and the shape of the graph differ.
#
# Which one to take is decided here rather than in the backend, so it stays visible and
# overridable; the backend only reports the fact that the choice depends on (whether the
# fused kernel got matrix cores on this device, i.e. `Device.conv_coopmat()`).
#
# With matrix cores live the fused kernel beats the tiled one for every OC <= 256 measured
# (ratio 0.27-0.98) and loses past 512 (1.06-1.13): coopmat only reaches the 64x32 tile,
# and once OC is big enough to select the 128x128 tile the tiled path's mul_mat GEMM is
# the faster kernel. Kernel size and spatial do NOT predict the crossover.
# Measured with `tools/bench_ops_compare.py --pair`; det -13%, rec -3%.
#
# Without matrix cores the fused kernel is scalar FMA and loses nearly everywhere, so fall
# back to the older rule below -- that one was tuned pre-coopmat and is still right there.
_FUSED_MAX_OC = 256
_FUSED_MIN_SPATIAL = 40000

# Force one path everywhere, for A/B of the fused kernel itself ("router" is the shipped
# behaviour; the others let a conv-shader change be measured without the tiled path's
# im2col->mul_mat rows muddying the comparison).
_CONV_PATH = os.environ.get("VT_CONV_PATH", "router")


def conv2d_router(x, w, b=None, sh=1, sw=1, p=1, d=1, pw=None, n_tiles=0):
    """The single conv2d entry point: picks the path by shape, then folds in the bias.

    ``n_tiles`` only concerns the tiled path -- the fused one needs no tiling."""
    pw = p if pw is None else pw
    kh, kw = w.shape[2], w.shape[3]
    if _CONV_PATH == "direct":
        fused = True
    elif _CONV_PATH == "tiled":
        fused = False
    elif _CONV_PATH == "oc":
        fused = w.shape[0] <= _FUSED_MAX_OC   # the rule above, ignoring the capability
    else:
        fused = (mt.conv_coopmat_available() and w.shape[0] <= _FUSED_MAX_OC) or (
            (kh, kw) != (3, 3) and
            (x.shape[1] + 2 * p - (kh - 1) * d) * (x.shape[2] + 2 * pw - (kw - 1) * d)
            >= _FUSED_MIN_SPATIAL)
    y = (mt.conv2d_direct(w, x, sw, sh, pw, p, d, d) if fused else
         mt.conv2d_tiled(w, x, sw, sh, pw, p, d, d, n_tiles))
    return _cadd(y, b, w.shape[0]) if b is not None else y


def conv(x, w, b, sh=1, sw=1, p=1, d=1, pw=None):
    """Pad H by p, W by pw (default = p)."""
    return conv2d_router(x, w, b, sh, sw, p, d, pw)


def conv_tiled(x, w, b, sh=1, sw=1, p=1, d=1, pw=None, n_tiles=0):
    """Same as ``conv``; both go through the router. Kept for call-site clarity."""
    return conv2d_router(x, w, b, sh, sw, p, d, pw, n_tiles)


def conv_dw(x, w, b, sh=1, sw=1, p=1, d=1, pw=None):
    pw = p if pw is None else pw
    y = mt.conv2d_dw(w, x, sw, sh, pw, p, d, d)
    return _cadd(y, b, w.shape[0]) if b is not None else y


def relu(x):
    return mt.relu(x)


def gelu(x):
    return mt.gelu_erf(x)


def silu(x):
    return mt.silu(x)


def hardsigmoid(x):
    # clamp(x/6 + 0.5, 0, 1)
    return mt.clamp(mt.scale_bias(x, 1.0 / 6.0, 0.5), 0.0, 1.0)


def sigmoid(x):
    return mt.sigmoid(x)


def linear_bias(x, w, b):
    return mt.add(mt.linear(x, w), b)


def layernorm_affine(x, w, b, eps):
    y = mt.layer_norm(x, eps)
    return mt.add(mt.mul(y, w), b)


def global_avg_pool(x):
    """[C,H,W] -> [C,1,1] via avg-pool with a full-size kernel."""
    _, h, w = x.shape
    return mt.pool_2d(x, 1, w, h, w, h)


def maxpool2d(x, k=2, s=1):
    return mt.pool_2d(x, 0, k, k, s, s)


def avgpool2d(x, kh, kw, sh, sw):
    return mt.pool_2d(x, 1, kw, kh, sw, sh)
