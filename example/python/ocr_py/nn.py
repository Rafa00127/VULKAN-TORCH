"""Small op helpers shared by the det/rec port.

Feature maps are PT [C, H, W] (batch N=1 is squeezed by the binding), which is
also how conv2d/conv2d_dw report their output. A per-channel weight/bias of
length C therefore broadcasts as a ggml ne=[C,1,1] tensor.
"""
import vulkantorch as mt


def _cadd(y, b, oc):
    """Add a per-channel bias. A conv output is [OC,H,W] -> ggml ne=[W,H,OC], so the
    bias must be [OC,1,1] (ne=[1,1,OC]) to broadcast along the channel axis."""
    return mt.add(y, mt.reshape(b, [oc, 1, 1]))


# There are two conv2d paths. The fused one (`conv2d_direct`) emits a single implicit-GEMM
# node and materialises no im2col tensor at all, so past a certain output size it wins --
# on time, and again on capture, which is the dominant cost when the input size keeps
# changing (a hand-drawn screenshot is never the same twice). Below that size the tiled
# im2col path is faster, and 3x3 is an exception at every size we measured.
# Same result either way; only the kernel and the shape of the graph differ.
_FUSED_MIN_SPATIAL = 40000


def conv2d_router(x, w, b=None, sh=1, sw=1, p=1, d=1, pw=None, n_tiles=0):
    """The single conv2d entry point: picks the path by shape, then folds in the bias.

    ``n_tiles`` only concerns the tiled path -- the fused one needs no tiling."""
    pw = p if pw is None else pw
    kh, kw = w.shape[2], w.shape[3]
    fused = (kh, kw) != (3, 3) and \
        (x.shape[1] + 2 * p - (kh - 1) * d) * (x.shape[2] + 2 * pw - (kw - 1) * d) \
        >= _FUSED_MIN_SPATIAL
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
