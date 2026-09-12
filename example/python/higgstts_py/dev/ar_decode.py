"""Decode minitorch AR codes -> wav (PyTorch codec). venv only."""
import os
import sys
import wave

import numpy as np
import torch

SGL = r"<local>/sglang\sglang-omni-main"
sys.path.insert(0, SGL)
from higgs_tts_standalone.loader import load_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
REFS = os.path.join(_ROOT, "data", "higgstts")
MODEL = os.path.join(SGL, "models", "bosonai--higgs-audio-v3-tts-4b")


def save_wav(path, s, sr=24000):
    pcm = (np.clip(s, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    model = load_model(MODEL, device="cuda", dtype="bfloat16")
    codes = np.load(os.path.join(REFS, "ar_codes.npy")).astype(np.int64)
    print("codes", codes.shape)
    with torch.no_grad():
        y = model.codec.decode(torch.from_numpy(codes))
    y = y.float().numpy().reshape(-1)
    out = os.path.join(REFS, "ar_minitorch.wav")
    save_wav(out, y)
    print(f"wrote {out}  ({len(y)} samples, {len(y)/24000:.2f}s)")


if __name__ == "__main__":
    main()
