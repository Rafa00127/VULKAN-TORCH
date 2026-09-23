"""HiggsTTS command-line demo (vulkantorch, Python) — self-contained.

    reference wav + text -> wav   (encode_ref + AR + DAC decode)

    python example/python/higgstts_vt/cli.py \
        --ref-wav ref_24k.wav --ref-text "..." --text "..." --out out.wav

Defaults: ``--ref-wav`` = data/ref_audio/melinaref_24k.wav, ``--out`` under
data/higgstts/. Pass ``--encode-only`` to stop after encode_ref (RVQ codes).
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

from higgstts_vt import _paths  # noqa: E402  (runs setup: repo root + native DLL dirs)
from higgstts_vt import ar as AR  # noqa: E402
from higgstts_vt.tts import HiggsTTS  # noqa: E402

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
    p = argparse.ArgumentParser(description="HiggsTTS CLI (vulkantorch/python)")
    p.add_argument("--model", required=True, help="HiggsTTS GGUF path")
    p.add_argument("--tokenizer", default=_paths.REF_TOKENIZER,
                   help="tokenizer.json (default: the copy in data/ref_audio/)")
    p.add_argument("--ref-wav", default=_paths.REF_WAV,
                   help="reference wav (default: data/ref_audio/melinaref_24k.wav)")
    p.add_argument("--ref-text", default="")
    p.add_argument("--text", default="")
    p.add_argument("--out", default=os.path.join(_paths.REFS, "cli_py.wav"))
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0,
                   help="AR step budget; 0 = predict from text length (12/token + 200)")
    p.add_argument("--encode-only", action="store_true", help="stop after encode_ref")
    p.add_argument("--no-graph-cache", action="store_true",
                   help="rebuild the decode graph every step (A/B against the default cached path)")
    a = p.parse_args()

    t0 = time.perf_counter()
    tts = HiggsTTS(a.model, a.tokenizer)
    print(f"load weights:        {(time.perf_counter()-t0)*1e3:8.1f} ms  "
          f"({len(tts.w)} tensors, {tts.rt.name()})")

    wav, sr = load_wav(a.ref_wav)
    t0 = time.perf_counter()
    ref_codes = tts.encode_ref(wav, sr)
    enc_ms = (time.perf_counter() - t0) * 1e3
    print(f"encode_ref:          {enc_ms:8.1f} ms  -> codes {ref_codes.shape}")
    if a.encode_only:
        return

    prompt = tts.build_prompt(a.text, a.ref_text, ref_codes.shape[0] + AR.N_CB - 1)
    t0 = time.perf_counter()
    codes = AR.ar_generate(tts.rt, tts.w, prompt, ref_codes, temperature=a.temperature,
                           seed=a.seed, max_steps=a.max_steps or None, topk=a.topk,
                           graph_cache=not a.no_graph_cache)
    ar_ms = (time.perf_counter() - t0) * 1e3
    steps = codes.shape[0] + AR.N_CB - 1

    t0 = time.perf_counter()
    pcm = tts.decode(codes)
    dec_ms = (time.perf_counter() - t0) * 1e3

    dur = len(pcm) / SR
    total = enc_ms + ar_ms + dec_ms
    print(f"Prefill: {ref_codes.shape[0]} frames x {AR.N_CB} codebooks ({enc_ms:.0f} ms)")
    print(f"Backbone AR: {steps} raw frames ({ar_ms:.0f} ms)")
    print(f"Decode: {len(pcm)} PCM samples ({dur:.2f} sec) ({dec_ms:.0f} ms)")

    # same summary block as the reference higgs_cli.exe and the C# CLI
    print()
    print("=== Timing ===")
    print(f"Prefill:      {enc_ms:8.0f} ms")
    print(f"Backbone AR:  {ar_ms:8.0f} ms")
    print(f"Decode:       {dec_ms:8.0f} ms")
    print(f"Total:        {total:8.0f} ms")
    print(f"Audio:        {dur:8.2f} sec")
    print(f"RTF:          {total / 1000 / dur:8.3f} x")

    save_wav(a.out, pcm)
    print(f"\nSaved: {a.out}")


if __name__ == "__main__":
    main()
