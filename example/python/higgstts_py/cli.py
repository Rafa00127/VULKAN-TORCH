"""HiggsTTS command-line demo (minitorch, Python).

Two input modes:

  full  — reference wav + text -> wav (encode_ref + AR + DAC decode)
      python example/python/higgstts_py/cli.py --ref-wav ref.wav \
          --ref-text "..." --text "..." --out out.wav

  ar    — a pre-encoded prompt (token ids) + reference codes -> wav
          (AR + DAC decode only; this is the stage the C# CLI also runs)
      python example/python/higgstts_py/cli.py \
          --prompt-ids data/higgstts/ar_gen_prompt_ids.i32 \
          --ref-codes data/higgstts/ar_gen_ref_codes.i32 --out out.wav

Prints a per-stage timing breakdown so the Python and C# paths can be compared.
"""
import argparse
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # example/python

from higgstts_py import _paths  # noqa: E402  (runs setup: repo root + native DLL dirs)
from higgstts_py import ar as AR  # noqa: E402
from higgstts_py.tts import HiggsTTS  # noqa: E402

GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"
SR = 24000


def load_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return a, sr


def save_wav(path, s, sr=SR):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(s, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    p = argparse.ArgumentParser(description="HiggsTTS CLI (minitorch/python)")
    p.add_argument("--gguf", default=GGUF)
    p.add_argument("--ref-wav", default=_paths.REF_WAV,
                   help="reference wav (default: data/ref_audio/melinaref_24k.wav)")
    p.add_argument("--ref-text", default="")
    p.add_argument("--text")
    p.add_argument("--prompt-ids", help="int32 [L] file (ar mode)")
    p.add_argument("--ref-codes", help="int32 [R,8] file (ar mode)")
    p.add_argument("--out", default=os.path.join(_paths.REFS, "cli_py.wav"))
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--no-decode", action="store_true", help="stop after AR (codes only)")
    a = p.parse_args()

    ar_mode = a.prompt_ids is not None

    t0 = time.perf_counter()
    tts = HiggsTTS(a.gguf)
    print(f"load weights:        {(time.perf_counter()-t0)*1e3:8.1f} ms  "
          f"({len(tts.w)} tensors, {tts.rt.name()})")

    if ar_mode:
        prompt = np.fromfile(a.prompt_ids, dtype=np.int32)
        ref_codes = np.fromfile(a.ref_codes, dtype=np.int32).reshape(-1, 8)
        enc_ms = None
    else:
        if not a.text or not a.ref_wav:
            p.error("full mode needs --ref-wav and --text")
        wav, sr = load_wav(a.ref_wav)
        t0 = time.perf_counter()
        ref_codes = tts.encode_ref(wav, sr)
        enc_ms = (time.perf_counter() - t0) * 1e3
        prompt = tts.build_prompt(a.text, a.ref_text, ref_codes.shape[0] + AR.N_CB - 1)

    t0 = time.perf_counter()
    codes = AR.ar_generate(tts.rt, tts.w, prompt, ref_codes, temperature=a.temperature,
                           seed=a.seed, max_steps=a.max_steps, topk=a.topk)
    ar_ms = (time.perf_counter() - t0) * 1e3
    steps = codes.shape[0] + AR.N_CB - 1

    dec_ms = None
    pcm = None
    if not a.no_decode:
        t0 = time.perf_counter()
        pcm = tts.decode(codes)
        dec_ms = (time.perf_counter() - t0) * 1e3

    print(f"prompt L={len(prompt)}  ref_codes={ref_codes.shape}  ->  codes {codes.shape}")
    if enc_ms is not None:
        print(f"encode_ref:          {enc_ms:8.1f} ms")
    print(f"AR (prefill+{steps} steps): {ar_ms:8.1f} ms  ({ar_ms/steps:6.2f} ms/step)")
    if dec_ms is not None:
        print(f"DAC decode:          {dec_ms:8.1f} ms")
        print(f"total (excl. load):  {ar_ms + dec_ms + (enc_ms or 0):8.1f} ms")
        dur = len(pcm) / SR
        print(f"audio:               {len(pcm)} samples = {dur:.2f} s "
              f"({(ar_ms+dec_ms+(enc_ms or 0))/1e3/dur:.2f}x realtime)")
        save_wav(a.out, pcm)
        print(f"wrote {a.out}")
    else:
        np.save(a.out + ".npy", codes)
        print(f"wrote {a.out}.npy")


if __name__ == "__main__":
    main()
