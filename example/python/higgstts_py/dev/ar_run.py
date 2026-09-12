"""Run the AR loop (minitorch) -> raw codes, save to higgstts_py/ar_codes.npy."""
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "example", "python"))

from higgstts_py import _paths  # noqa: E402

BUILD = _paths.BUILD
REFS = _paths.REFS

import minitorch as mt  # noqa: E402
import higgstts_py.ar as AR  # noqa: E402
from higgstts_py.weights import HiggsWeights  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"


def main():
    ref = np.load(os.path.join(REFS, "ar_ref.npz"))
    prompt_ids = ref["prompt_ids"]
    ref_codes = np.load(os.path.join(REFS, "dll_codes.npy")).astype(np.int32)

    rt = mt.Runtime()
    gpu = rt.gpu()
    w = HiggsWeights(GGUF, gpu, backbone=True)

    t = time.perf_counter()
    raw = AR.ar_generate(rt, w, prompt_ids, ref_codes, temperature=0.9, seed=42)
    print(f"AR done in {time.perf_counter()-t:.1f}s  raw codes {raw.shape}")
    np.save(os.path.join(REFS, "ar_codes.npy"), raw)
    print("first 5:", raw[:5].tolist())
    print("range:", int(raw.min()), int(raw.max()))


if __name__ == "__main__":
    main()
