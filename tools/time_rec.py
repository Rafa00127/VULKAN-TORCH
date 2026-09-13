"""Time the rec net's parts on the GPU (backbone=conv vs head+SVTR).
Run: python tools/time_rec.py
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "example", "python"), os.path.join(ROOT, "tools")):
    sys.path.insert(0, p)
import numpy as np
import vulkantorch as mt
from ocr_py.ocr import Ocr, rec_preprocess
from ocr_py import rec, backbone, nn as N

ocr = Ocr()
IMG = r"<external>/_test_1.gif"
import segment
crop = next(segment.segment_lines(IMG))
bgr = np.ascontiguousarray(crop[:, :, ::-1])
x = rec_preprocess(bgr)
print("rec input", x.shape, flush=True)


def time_graph(fn, arr, n=60):
    a = np.ascontiguousarray(arr, np.float32)
    mem = mt.Memory(ocr.dev, a.nbytes)
    xt = mem.tensor(list(a.shape), a.tobytes())
    g = mt.Graph(ocr.rt, ocr.dev)
    g.enter()
    out = fn(xt, ocr.wrec)
    out.mark_output()
    g.exit()
    g.alloc_static()
    g.set_input(xt, a.tobytes())
    g.compute_static()
    t0 = time.perf_counter()
    for _ in range(n):
        g.set_input(xt, a.tobytes())
        g.compute_static()
    return (time.perf_counter() - t0) / n * 1000, np.frombuffer(out.to_bytes(), np.float32).reshape(list(out.shape))


full, _ = time_graph(lambda x, w: rec.forward(x, w), x)
print(f"  full rec        {full:7.2f} ms", flush=True)

bb_ms, bb_out = time_graph(lambda x, w: backbone.forward(x, w, backbone.REC_BLOCKS)[-1], x)
print(f"  backbone(conv)  {bb_ms:7.2f} ms   ({100*bb_ms/full:.0f}%)   out {bb_out.size} floats", flush=True)


def post_fn(h, w):
    hb = "head.encoder."
    h2 = N.avgpool2d(h, 3, 2, 3, 2)
    res = N.silu(N.conv(h2, w[hb + "conv_block.0.convolution.weight"],
                        w[hb + "conv_block.0.convolution.bias"], p=0))
    t = N.silu(N.conv(h2, w[hb + "conv_block.1.convolution.weight"],
                      w[hb + "conv_block.1.convolution.bias"], p=0))
    t = mt.add(t, N.silu(N.conv_dw(t, w[hb + "conv_block.2.convolution.weight"],
                                   w[hb + "conv_block.2.convolution.bias"], p=0, pw=3)))
    C, _, W2 = t.shape
    seq = mt.contiguous(mt.transpose(mt.reshape(t, [C, W2])))
    for d in range(rec.DEPTH):
        seq = rec._svtr_block(seq, w, f"{hb}svtr_block.{d}.")
    seq = N.layernorm_affine(seq, w[hb + "norm.weight"], w[hb + "norm.bias"], rec.EPS)
    r_seq = mt.transpose(mt.reshape(res, [C, W2]))
    seq = mt.add(seq, r_seq)
    return mt.soft_max(N.linear_bias(seq, w["head.head.weight"], w["head.head.bias"]))


post_ms, _ = time_graph(post_fn, bb_out)
print(f"  head+SVTR       {post_ms:7.2f} ms   ({100*post_ms/full:.0f}%)", flush=True)
print(f"  overhead        {full-bb_ms-post_ms:7.2f} ms", flush=True)
