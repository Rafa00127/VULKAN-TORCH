"""vulkantorch benchmark (Python) — mirrors bench/bench_cpp.cpp and
example/CSharp/VulkanTorch.Bench. Weights live in GPU Memory; each rep builds one
graph, runs the op, and reads back only a tiny reduction (sum_rows). Best of N.

    python bench/bench_py.py
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import vulkantorch as mt  # noqa: E402

REPS = 10
rt = mt.Runtime()
gpu = rt.gpu()
print("backend:", rt.name())
rng = np.random.default_rng(0)


def mm(M, K, N):
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    ta = mt.Memory(gpu, A.nbytes).tensor([M, K], A.tobytes())
    tb = mt.Memory(gpu, B.nbytes).tensor([K, N], B.tobytes())
    best = float("inf")
    for _ in range(REPS):
        t = time.perf_counter()
        g = mt.Graph(rt, gpu)
        g.enter()
        s = mt.sum_rows(mt.matmul(ta, tb))
        s.mark_output()
        g.exit()
        s.to_bytes()
        best = min(best, time.perf_counter() - t)
    return best


def conv(T, Cin, Cout, K, pad):
    x = rng.standard_normal((T, Cin), dtype=np.float32)
    w = rng.standard_normal((Cout, Cin, K), dtype=np.float32)
    tx = mt.Memory(gpu, x.nbytes).tensor([T, Cin], x.tobytes())
    tw = mt.Memory(gpu, w.nbytes).tensor([Cout, Cin, K], w.tobytes())
    best = float("inf")
    for _ in range(REPS):
        t = time.perf_counter()
        g = mt.Graph(rt, gpu)
        g.enter()
        s = mt.sum_rows(mt.conv1d(tx, tw, 1, pad, 1))
        s.mark_output()
        g.exit()
        s.to_bytes()
        best = min(best, time.perf_counter() - t)
    return best


print(f"\n{'matmul shape':>22} | {'ms':>10}")
print("-" * 38)
for (M, K, N) in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]:
    print(f"{f'{M}x{K}x{N}':>22} | {mm(M, K, N)*1e3:7.2f} ms")

print(f"\n{'conv1d T Cin->Cout':>22} | {'ms':>10}")
print("-" * 38)
for (T, Cin, Cout, K, pad) in [(8192, 256, 256, 7, 3), (16384, 512, 512, 7, 3)]:
    print(f"{f'{T} {Cin}->{Cout} K{K}':>22} | {conv(T, Cin, Cout, K, pad)*1e3:7.2f} ms")
