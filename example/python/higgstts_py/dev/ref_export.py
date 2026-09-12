"""Export per-stage reference tensors from the PyTorch model, mirroring the
`encoder学习.ipynb` manual pipeline (which matches higgs_tts.dll, T=379).

Run with the sglang-omni venv.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SGL = r"<local>/sglang\sglang-omni-main"
sys.path.insert(0, SGL)

from higgs_tts_standalone.loader import load_model  # noqa: E402

CHECKPOINT = os.path.join(SGL, "models", "bosonai--higgs-audio-v3-tts-4b")
WAV = os.path.join(_ROOT, "data", "ref_audio", "melinaref_24k.wav")
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
REFS = os.path.join(_ROOT, "data", "higgstts")
OUT = os.path.join(REFS, "ref.npz")

caps = {}


def cap(name, t):
    caps[name] = t.detach().float().cpu().numpy()


def main():
    import librosa

    model = load_model(CHECKPOINT, device="cuda", dtype="bfloat16")
    cm = model.codec.model
    sem_model = cm.semantic_model

    audio, sr = librosa.load(WAV, sr=24000, mono=True)
    wav = torch.from_numpy(audio).view(1, 1, -1).float()
    print("wav", tuple(wav.shape), "sr", sr)
    caps["wav_mono"] = audio.astype(np.float32)
    caps["sample_rate"] = np.array([sr])

    # ---- semantic input: resample 24k->16k, squeeze, pad(160,160) ----
    import torchaudio
    SEM_SR = cm.config.semantic_sample_rate
    wav_dev = wav.to(device="cuda", dtype=next(cm.parameters()).dtype)
    sem = torchaudio.functional.resample(wav_dev, 24000, SEM_SR)[:, 0, :]
    sem = F.pad(sem, (160, 160))
    sem_t = sem.to(device=next(sem_model.parameters()).device,
                   dtype=next(sem_model.parameters()).dtype)
    cap("sem_input", sem)

    with torch.no_grad():
        # ---- feature extractor + projection ----
        fe = sem_model.feature_extractor
        fe_out = fe(sem_t)                                  # [1,512,T]
        cap("fe_out", fe_out)
        fp = sem_model.feature_projection
        fp_out = fp.projection(fp.layer_norm(fe_out.transpose(1, 2)))
        cap("fp_out", fp_out)

        # ---- WavLM encoder 12L, stack & mean (NO downsample) ----
        enc = sem_model.encoder
        if hasattr(enc, "layers"):
            enc.layers[0].register_forward_hook(
                (lambda name: lambda _m, i, o: cap(name, i[0]))("enc_l0_in"))
            for li in (0, 1):
                att = getattr(enc.layers[li], "attention", None)
                if att is None:
                    continue
                att.register_forward_hook(
                    (lambda name: lambda _m, _i, o: cap(
                        name, o[0] if isinstance(o, tuple) else o))(f"l{li}_attn"))
                for proj in ("q_proj", "k_proj", "v_proj"):
                    if hasattr(att, proj):
                        getattr(att, proj).register_forward_hook(
                            (lambda name: lambda _m, _i, o: cap(name, o))(f"l{li}_{proj}"))
        # WavLM adds a positional conv embedding before the encoder
        if hasattr(sem_model, "pos_conv_embed"):
            pce = sem_model.pos_conv_embed(fp_out)
            cap("pce_out", pce)
            fp_in = fp_out + pce
        else:
            fp_in = fp_out
        enc_out = enc(fp_in, output_hidden_states=True)
        hidden = torch.stack(enc_out.hidden_states[1:], dim=0).mean(dim=0)  # [1,T,768]
        cap("wavlm_hidden", hidden)
        cap("wavlm_layers", torch.stack(enc_out.hidden_states[1:], dim=0))  # [12,1,T,768]

        # ---- semantic encoder ----
        se_out = cm.encoder_semantic(hidden.transpose(1, 2))   # [1,768,T]
        cap("encoder_semantic", se_out)

        # ---- acoustic encoder (pad wav to align T if needed) ----
        hop = cm.config.hop_length
        ac = cm.acoustic_encoder
        ac_dev = next(ac.parameters()).device
        ac_dtype = next(ac.parameters()).dtype
        ac.register_forward_hook(
            (lambda name: lambda _m, i, o: cap(name, i[0]))("ac_in"))
        wav_gpu = wav.to(device=ac_dev, dtype=ac_dtype)
        ac_out = ac(wav_gpu)
        T_sem = se_out.shape[-1]
        if ac_out.shape[-1] != T_sem:
            pad_total = T_sem - ac_out.shape[-1]
            pl, pr = pad_total // 2, pad_total - pad_total // 2
            wav_gpu = F.pad(wav_gpu, (pl * hop, pr * hop))
            ac_out = ac(wav_gpu)
        cap("acoustic_encoder", ac_out)

        # ---- concat + fc ----
        combined = torch.cat([ac_out, se_out], dim=1)          # [1,1024,T]
        fused = cm.fc(combined.transpose(1, 2)).transpose(1, 2)
        cap("fused", fused)

        # ---- RVQ encode ----
        codes = cm.quantizer.encode(fused, bandwidth=8)
        if codes.dim() == 3:
            codes = codes.squeeze(1).T
        elif codes.dim() == 2:
            codes = codes.T
        caps["codes"] = codes.cpu().numpy().astype(np.int64)

    np.savez(OUT, **caps)
    print("saved", OUT)
    for k, v in caps.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
