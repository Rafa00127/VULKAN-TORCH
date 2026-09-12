"""End-to-end HiggsTTS on minitorch: reference wav + text -> synthesized wav."""
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

from higgstts_py import HiggsTTS  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"
REF_WAV = _paths.REF_WAV
REF_TEXT = "I have no doubt you will become Elden lord, may you take the throne."
TEXT = "<|style:whispering|>Hello how you doing? Are you having fun these days?"


def load_wav(p):
    with wave.open(p, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return a, sr


def save_wav(path, s, sr=24000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(s, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ref, sr = load_wav(REF_WAV)
    tts = HiggsTTS(GGUF)
    print("loaded; synthesizing...")
    t = time.perf_counter()
    pcm = tts.synthesize(TEXT, ref, REF_TEXT, sample_rate=sr, temperature=0.9, seed=42)
    dt = time.perf_counter() - t
    out = os.path.join(REFS, "e2e_minitorch.wav")
    save_wav(out, pcm)
    print(f"done in {dt:.2f}s -> {out} ({len(pcm)/24000:.2f}s)")


if __name__ == "__main__":
    main()
