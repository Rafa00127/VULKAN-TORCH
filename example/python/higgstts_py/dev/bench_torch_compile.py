"""Try torch.compile on the notebook-logic prefill and time it. venv only."""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SGL = r"<local>/sglang\sglang-omni-main"
sys.path.insert(0, SGL)
from higgs_tts_standalone.loader import load_model  # noqa: E402

MODEL = os.path.join(SGL, "models", "bosonai--higgs-audio-v3-tts-4b")
WAV = _ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
WAV = os.path.join(_ROOT, "data", "ref_audio", "melinaref_24k.wav")


def main():
    import librosa
    import torchaudio

    model = load_model(MODEL, device="cuda", dtype="bfloat16")
    cm = model.codec.model
    sem_model = cm.semantic_model
    enc = sem_model.encoder

    audio, sr = librosa.load(WAV, sr=24000, mono=True)
    wav = torch.from_numpy(audio).view(1, 1, -1).float()
    dt = next(cm.parameters()).dtype

    def prefill():
        wd = wav.to(device="cuda", dtype=dt)
        sem = torchaudio.functional.resample(wd, 24000, cm.config.semantic_sample_rate)[:, 0, :]
        sem = F.pad(sem, (160, 160))
        sem = sem.to(dtype=next(sem_model.parameters()).dtype)
        fe = sem_model.feature_extractor(sem)
        fp = sem_model.feature_projection.projection(
            sem_model.feature_projection.layer_norm(fe.transpose(1, 2)))
        enc_out = enc(fp, output_hidden_states=True)
        wh = torch.stack(enc_out.hidden_states[1:], dim=0).mean(dim=0)
        se = cm.encoder_semantic(wh.transpose(1, 2))
        ac = cm.acoustic_encoder
        wg = wav.to(dtype=next(ac.parameters()).dtype)
        ao = ac(wg)
        if ao.shape[-1] != se.shape[-1]:
            hop = cm.config.hop_length
            pt = se.shape[-1] - ao.shape[-1]
            wg = F.pad(wg, ((pt // 2) * hop, (pt - pt // 2) * hop))
            ao = ac(wg)
        fused = cm.fc(torch.cat([ao, se], dim=1).transpose(1, 2)).transpose(1, 2)
        codes = cm.quantizer.encode(fused, bandwidth=8)
        return codes.squeeze(1).T if codes.dim() == 3 else codes.T

    try:
        cf = torch.compile(prefill)
        print("compiling (this may take a while)...", flush=True)
        with torch.no_grad():
            c = cf()
        torch.cuda.synchronize()
        print("codes shape:", tuple(c.shape))
        best = 1e9
        for _ in range(5):
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                cf()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        print(f"PyTorch torch.compile prefill: {best*1e3:.1f} ms")
    except Exception as e:
        print("torch.compile FAILED:", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    main()
