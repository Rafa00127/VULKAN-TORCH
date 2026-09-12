"""Time the PyTorch prefill and decode both minitorch & dll codes to wav.
Run with the sglang-omni venv."""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import wave

SGL = r"<local>/sglang\sglang-omni-main"
sys.path.insert(0, SGL)
from higgs_tts_standalone.loader import load_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
REFS = os.path.join(_ROOT, "data", "higgstts")
MODEL = os.path.join(SGL, "models", "bosonai--higgs-audio-v3-tts-4b")
WAV = os.path.join(_ROOT, "data", "ref_audio", "melinaref_24k.wav")


def save_wav(path, samples, sr=24000):
    z = np.clip(samples, -1, 1)
    pcm = (z * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    import librosa

    model = load_model(MODEL, device="cuda", dtype="bfloat16")
    cm = model.codec.model
    audio, sr = librosa.load(WAV, sr=24000, mono=True)
    wav = torch.from_numpy(audio).view(1, 1, -1).float()

    # ---- time PyTorch prefill (production encode_ref) ----
    with torch.no_grad():
        model.codec.encode_reference(wav, sample_rate=24000)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        with torch.no_grad():
            model.codec.encode_reference(wav, sample_rate=24000)
        best = min(best, time.perf_counter() - t)
    print(f"PyTorch encode_reference: {best*1e3:.1f}ms")

    # ---- decode minitorch & dll codes ----
    for name, fn in (("minitorch", "codes_minitorch.npy"), ("dll", "dll_codes.npy")):
        codes = np.load(os.path.join(REFS, fn)).astype(np.int64)
        with torch.no_grad():
            y = model.codec.decode(torch.from_numpy(codes))
        y = y.float().numpy().reshape(-1)
        out = os.path.join(REFS, f"dec_{name}.wav")
        save_wav(out, y)
        print(f"decoded {name} codes {codes.shape} -> {out}  ({len(y)} samples, {len(y)/24000:.2f}s)")


if __name__ == "__main__":
    main()
