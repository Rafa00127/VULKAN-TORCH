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


def conv(x, w, b, sh=1, sw=1, p=1, d=1, pw=None):
    """Pad H by p, W by pw (default = p)."""
    pw = p if pw is None else pw
    y = mt.conv2d(w, x, sw, sh, pw, p, d, d)
    return _cadd(y, b, w.shape[0]) if b is not None else y


def conv_tiled(x, w, b, sh=1, sw=1, p=1, d=1, pw=None, n_tiles=0):
    """conv2d with the output width tiled so the im2col stays small (n_tiles=0: auto)."""
    pw = p if pw is None else pw
    y = mt.conv2d_tiled(w, x, sw, sh, pw, p, d, d, n_tiles)
    return _cadd(y, b, w.shape[0]) if b is not None else y


def conv_direct(x, w, b, sh=1, sw=1, p=1, d=1, pw=None):
    """conv2d as one fused (implicit-GEMM) node -- no im2col tensor is materialised,
    so it is not subject to the per-buffer size limit that conv_tiled works around.
    Identical result; it wins once the output map is large enough for the saved
    im2col traffic to pay for itself, and loses on small maps. See ``det._conv``."""
    pw = p if pw is None else pw
    y = mt.conv2d_direct(w, x, sw, sh, pw, p, d, d)
    return _cadd(y, b, w.shape[0]) if b is not None else y


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
