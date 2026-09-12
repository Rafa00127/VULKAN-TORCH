"""Clean prefill timing: minitorch (with/without RVQ) vs dll. Warm, best-of-N."""
import ctypes
import os
import sys
import time
import wave

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "example", "python"))

from higgstts_py import _paths  # noqa: E402

BUILD = _paths.BUILD
REFS = _paths.REFS

import minitorch as mt  # noqa: E402
import higgstts_py.model as M  # noqa: E402
from higgstts_py.weights import HiggsWeights  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"
REF = os.path.join(REFS, "ref.npz")


def main():
    ref = np.load(REF)
    sem = ref["sem_input"].astype(np.float32)[0]
    ac = ref["ac_in"].astype(np.float32)[0][0]

    rt = mt.Runtime()
    gpu = rt.gpu()
    w = HiggsWeights(GGUF, gpu)
    M.build_pce_weight(w)

    def run(with_rvq, reps=8):
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            g = mt.Graph(rt, gpu); g.enter()
            xs = g.input([sem.shape[0], 1], sem.tobytes())
            xa = g.input([ac.shape[0], 1], ac.tobytes())
            if with_rvq:
                ids, fused = M.prefill(w, xs, xa)
                out = mt.contiguous(ids[0])
            else:
                out = mt.contiguous(M.prefill_fused(w, xs, xa))
            out.mark_output()
            out.to_bytes()
            g.exit()
            best = min(best, time.perf_counter() - t)
        return best * 1e3

    tm_full = run(True)
    tm_norvq = run(False)

    # dll, warm
    R = r"<local>/external\HiggsTTSOnline\native\build-vk\bin\Release"
    os.add_dll_directory(R)
    dll = ctypes.CDLL(os.path.join(R, "higgs_tts.dll"))
    dll.higgs_tts_load.restype = ctypes.c_void_p
    dll.higgs_tts_load.argtypes = [ctypes.c_char_p]
    dll.higgs_tts_encode_ref.restype = ctypes.c_int
    dll.higgs_tts_encode_ref.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                         ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
    wf = wave.open(os.path.join(_ROOT, "data", "ref_audio", "melinaref_24k.wav"), "rb")
    a = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    h = dll.higgs_tts_load(GGUF.encode())
    out = (ctypes.c_int32 * (4000 * 8))()
    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    dll.higgs_tts_encode_ref(h, ap, len(a), out)
    best = 1e9
    for _ in range(8):
        t = time.perf_counter()
        dll.higgs_tts_encode_ref(h, ap, len(a), out)
        best = min(best, time.perf_counter() - t)
    td = best * 1e3

    print(f"minitorch prefill (with RVQ)   : {tm_full:6.1f} ms")
    print(f"minitorch prefill (no RVQ)     : {tm_norvq:6.1f} ms   (RVQ = {tm_full-tm_norvq:.1f} ms)")
    print(f"dll encode_ref (warm)          : {td:6.1f} ms")


if __name__ == "__main__":
    main()
