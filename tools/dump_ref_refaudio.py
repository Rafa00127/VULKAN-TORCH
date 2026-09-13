"""Dump the IndexTTS 2.5 reference-audio front-end (fbank / mel / w2v-bert / campplus) from the
official PyTorch, for validating example/CSharp/IndexTts.Net/{Mel,Fbank,W2vBert,CampPlus}.cs.

Run:  移植参考/.venv-indextts/Scripts/python.exe tools/dump_ref_refaudio.py
"""
import os
import sys

import numpy as np
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "移植参考", "index-tts-main"))

from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

W2V = os.path.join(ROOT, "移植参考", "model", "w2v")
CAMPLUS = os.path.join(ROOT, "移植参考", "model", "camplus", "campplus_cn_common.bin")
STATS = os.path.join(ROOT, "移植参考", "model", "index2.5", "wav2vec2bert_stats.pt")
WAV = os.path.join(ROOT, "data", "ref_audio", "melinaref_24k.wav")
OUT = os.path.join(ROOT, "data", "indextts_ref")
MAX_SECONDS = 15


def main():
    torch.manual_seed(0)
    wav, sr = torchaudio.load(WAV)
    wav = wav[:1, : sr * MAX_SECONDS]                     # mono, <=15 s

    a16 = torchaudio.transforms.Resample(sr, 16000)(wav)
    a22 = torchaudio.transforms.Resample(sr, 22050)(wav)

    fe = SeamlessM4TFeatureExtractor.from_pretrained(W2V, local_files_only=True)
    inputs = fe(a16, sampling_rate=16000, return_tensors="pt")
    feats = inputs["input_features"]
    mask = inputs["attention_mask"]

    w2v = Wav2Vec2BertModel.from_pretrained(W2V, local_files_only=True).eval()
    stats = torch.load(STATS, map_location="cpu", weights_only=False)
    mean = stats["mean"].float()
    std = torch.sqrt(stats["var"].float())

    campplus = CAMPPlus(feat_dim=80, embedding_size=192).eval()
    campplus.load_state_dict(torch.load(CAMPLUS, map_location="cpu", weights_only=False))

    cap = {}
    cap2 = {}
    for li in (0, 1):
        lay = w2v.encoder.layers[li]
        for nm, mod in (("ffn1", lay.ffn1), ("attn", lay.self_attn), ("conv", lay.conv_module),
                        ("ffn2", lay.ffn2)):
            mod.register_forward_hook(
                lambda mo, i, o, li=li, nm=nm: cap2.__setitem__(
                    f"l{li}_{nm}", (o[0] if isinstance(o, tuple) else o).detach().clone()))
    with torch.no_grad():
        hidden = w2v(input_features=feats, attention_mask=mask, output_hidden_states=True)
        h17 = hidden.hidden_states[17]
        spk = (h17 - mean) / std
        ref_mel = mel_spectrogram(a22.float(), n_fft=1024, num_mels=80, sampling_rate=22050,
                                  hop_size=256, win_size=1024, fmin=0, fmax=None)
        fbank = torchaudio.compliance.kaldi.fbank(a16, num_mel_bins=80, dither=0,
                                                  sample_frequency=16000)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
    campplus.head.register_forward_hook(lambda m, i, o: cap.__setitem__("head", o.detach().clone()))
    for idx, nm in ((0, "tdnn"), (1, "blk1"), (2, "tr1"), (3, "blk2"), (4, "tr2"), (5, "blk3"),
                    (7, "outnl"), (9, "dense")):
        campplus.xvector[idx].register_forward_hook(
            lambda m, i, o, nm=nm: cap.__setitem__(nm, o.detach().clone()))
    with torch.no_grad():
        style = campplus(fbank.unsqueeze(0))

    for k in (0, 1, 2, 5, 9, 17):
        np.save(os.path.join(OUT, f"ra_hs{k}.npy"),
                hidden.hidden_states[k].contiguous().numpy().astype(np.float32))
    print("wav", tuple(wav.shape), "sr", sr)
    print("input_features", tuple(feats.shape), "mask", tuple(mask.shape))
    print("h17", tuple(h17.shape), "spk", tuple(spk.shape))
    print("ref_mel", tuple(ref_mel.shape), "fbank", tuple(fbank.shape), "style", tuple(style.shape))

    os.makedirs(OUT, exist_ok=True)
    for k, v in cap.items():
        np.save(os.path.join(OUT, f"ra_cp_{k}.npy"), v.contiguous().numpy().astype(np.float32))
    for k, v in cap2.items():
        np.save(os.path.join(OUT, f"ra_{k}.npy"), v.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "ra_w2v_mean.npy"), mean.contiguous().numpy().astype(np.float32))
    np.save(os.path.join(OUT, "ra_w2v_std.npy"), std.contiguous().numpy().astype(np.float32))
    for name, t in [("ra_wav16k", a16), ("ra_wav22k", a22), ("ra_feats", feats), ("ra_mask", mask),
                    ("ra_h17", h17), ("ra_spk", spk), ("ra_mel", ref_mel), ("ra_fbank", fbank),
                    ("ra_style", style)]:
        np.save(os.path.join(OUT, name + ".npy"), t.contiguous().numpy().astype(np.float32))
    print("wrote ra_*.npy ->", OUT)


if __name__ == "__main__":
    main()
