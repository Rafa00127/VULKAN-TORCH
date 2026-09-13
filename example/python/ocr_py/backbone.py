"""PP-LCNetV4 backbone (shared by PP-OCRv6 det and rec).

MetaFormer blocks: a depthwise token conv (+ fused SE) then a pointwise GELU
MLP, with a residual when the shape is preserved. Weights are already
reparameterized + BN-folded by tools/convert_ocr_to_gguf.py.
"""
import vulkantorch as mt

from ocr_py import nn as N

# [kernel, in_ch, out_ch, stride, use_se]; stride is int or (h, w)
DET_BLOCKS = [
    [[3, 128, 128, 1, True], [3, 128, 128, 1, False]],
    [[3, 128, 256, 2, False], [3, 256, 256, 1, True], [3, 256, 256, 1, False]],
    [[3, 256, 512, 2, False], [3, 512, 512, 1, True], [3, 512, 512, 1, False],
     [3, 512, 512, 1, True], [3, 512, 512, 1, False]],
    [[3, 512, 896, 2, False], [3, 896, 896, 1, True], [3, 896, 896, 1, False]],
]
REC_BLOCKS = [
    [[3, 128, 128, 1, True]],
    [[3, 128, 256, 1, False], [3, 256, 256, 1, False], [3, 256, 256, 1, True]],
    [[3, 256, 512, (2, 1), False], [3, 512, 512, 1, True], [3, 512, 512, 1, False],
     [3, 512, 512, 1, True], [3, 512, 512, 1, False], [3, 512, 512, 1, True],
     [3, 512, 512, 1, False]],
    [[3, 512, 768, (2, 1), False], [3, 768, 768, 1, True], [3, 768, 768, 1, False]],
]
STEM_STRIDES = [2, 1, 1, 2, 1]


def _hw(s):
    return (s, s) if isinstance(s, int) else (s[0], s[1])


def _cn(x, w, p, act, sh=1, sw=1, k=3):
    """PPLCNetV4ConvLayer: conv (BN folded into bias) then optional activation."""
    x = N.conv(x, w[p + ".convolution.weight"], w[p + ".convolution.bias"],
               sh=sh, sw=sw, p=(k - 1) // 2)
    return act(x) if act is not None else x


def _stem(x, w):
    p = "model.backbone.encoder.convolution."
    e = _cn(x, w, p + "stem1", N.relu, sh=STEM_STRIDES[0], sw=STEM_STRIDES[0])
    e = mt.pad(e, 1, 1)                                  # F.pad([0,1,0,1]): W right, H bottom
    a = _cn(e, w, p + "stem2a", N.relu, k=2)
    a = mt.pad(a, 1, 1)
    a = _cn(a, w, p + "stem2b", N.relu, k=2)
    pooled = N.maxpool2d(e, 2, 1)                        # pooled on the *padded* embedding
    e = mt.concat(pooled, a, 0)                          # channel concat ([C,H,W])
    e = _cn(e, w, p + "stem3", N.relu, sh=STEM_STRIDES[3], sw=STEM_STRIDES[3])
    e = _cn(e, w, p + "stem4", N.relu, sw=STEM_STRIDES[4], k=1)
    return e


def _block(x, w, p, k, stride, use_se, rep):
    sh, sw = _hw(stride)
    if rep:
        x = N.conv_dw(x, w[p + "token_conv.weight"], w[p + "token_conv.bias"], p=k // 2)
    else:
        x = N.conv_dw(x, w[p + "token_conv.convolution.weight"],
                      w[p + "token_conv.convolution.bias"], sh=sh, sw=sw, p=k // 2)
    if use_se:
        g = N.global_avg_pool(x)
        g = N.relu(N.conv(g, w[p + "token_squeeze_excitation.convolutions.0.weight"],
                          w[p + "token_squeeze_excitation.convolutions.0.bias"], p=0))
        g = N.hardsigmoid(N.conv(g, w[p + "token_squeeze_excitation.convolutions.2.weight"],
                                 w[p + "token_squeeze_excitation.convolutions.2.bias"], p=0))
        x = mt.mul(x, g)
    residual = x
    y = N.gelu(N.conv(x, w[p + "channel_conv1.convolution.weight"],
                      w[p + "channel_conv1.convolution.bias"], p=0))
    y = N.conv(y, w[p + "channel_conv2.convolution.weight"],
               w[p + "channel_conv2.convolution.bias"], p=0)
    return mt.add(residual, y) if rep else y


def forward(x, w, blocks):
    """x: [3,H,W] -> list of the 4 stage feature maps."""
    x = _stem(x, w)
    outs = []
    for s, stage in enumerate(blocks):
        for b, (k, cin, cout, stride, use_se) in enumerate(stage):
            p = f"model.backbone.encoder.blocks.{s}.blocks.{b}."
            rep = (stride == 1) and (cin == cout)
            x = _block(x, w, p, k, stride, use_se, rep)
        outs.append(x)
    return outs
