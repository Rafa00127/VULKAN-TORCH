r"""Unified HiggsTTS API on minitorch.

    tts = HiggsTTS(gguf, tokenizer_json)
    codes = tts.encode_ref(ref_wav_24k)                 # reference audio -> codes [T,8]
    codes = tts.generate(text, codes, ref_text)         # text -> AR codes [T,8]
    pcm   = tts.decode(codes)                           # codes -> 24kHz PCM
    pcm   = tts.synthesize(text, ref_wav, ref_text)     # end to end
"""

import librosa
import numpy as np

import minitorch as mt

from higgstts_py import ar as AR
from higgstts_py import decode as DE
from higgstts_py import model as M
from higgstts_py.weights import HiggsWeights

_TOKENIZER_JSON = r"<local>/reader-app\models\higgs_tts_v3_tokenizer.json"

# prompt special token ids (from higgs_tts.h; GGUF metadata can override)
TOK_TTS = 151667
TOK_REF_TEXT = 151680
TOK_REF_AUDIO = 151679
TOK_TEXT = 151672
TOK_AUDIO = 151670
AUDIO_PLACEHOLDER = -100
N_CB = 8

_SEM_RATE = 16000
_AC_RATE = 24000
_HOP = int(np.prod(M.AC_STRIDES))
_FE_K = [10, 3, 3, 3, 3, 2, 2]
_FE_S = [5, 2, 2, 2, 2, 2, 2]


def _fe_len(L):
    for K, s in zip(_FE_K, _FE_S):
        L = (L - K) // s + 1
    return L


def _ac_len(L):
    def cl(L, K, s, p):
        return (L + 2 * p - (K - 1) - 1) // s + 1
    L = cl(L, 7, 1, 3)
    for K, s, p in zip([16, 10, 8, 4, 6], M.AC_STRIDES, M.AC_PADS):
        L = cl(L, K, s, p)
    return cl(L, 3, 1, 1)


class HiggsTTS:
    def __init__(self, gguf_path, tokenizer_json=_TOKENIZER_JSON, device=None, runtime=None):
        self.rt = runtime or mt.Runtime()
        self.dev = device or self.rt.gpu()
        self.w = HiggsWeights(gguf_path, self.dev, backbone=True)
        M.build_pce_weight(self.w)
        self.tok = None
        if tokenizer_json:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(tokenizer_json)

    # ---- 1. reference audio -> codes ----
    def encode_ref(self, waveform, sample_rate=24000):
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != _AC_RATE:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=_AC_RATE).astype(np.float32)
        sem = np.pad(librosa.resample(wav, orig_sr=_AC_RATE, target_sr=_SEM_RATE).astype(np.float32),
                     (160, 160))
        t_sem = _fe_len(len(sem))
        t_ac = _ac_len(len(wav))
        pad = t_sem - t_ac
        pcm = np.pad(wav, ((pad // 2) * _HOP, (pad - pad // 2) * _HOP)) if pad > 0 else wav

        g = mt.Graph(self.rt, self.dev)
        g.enter()
        xs = g.input([sem.shape[0], 1], sem.tobytes())
        xa = g.input([pcm.shape[0], 1], pcm.tobytes())
        ids, _ = M.prefill(self.w, xs, xa)
        id_c = [mt.contiguous(i) for i in ids]
        for t in id_c:
            t.mark_output()
        g.exit()
        t = id_c[0].shape[0]
        codes = np.zeros((t, N_CB), dtype=np.int32)
        for q, node in enumerate(id_c):
            codes[:, q] = np.frombuffer(node.to_bytes(), dtype=np.int32).reshape(-1)
        return codes

    # ---- prompt ----
    def build_prompt(self, text, ref_text, num_ref_tokens):
        if self.tok is None:
            raise RuntimeError("HiggsTTS: no tokenizer loaded")
        p = [TOK_TTS]
        if ref_text:
            p.append(TOK_REF_TEXT)
            p += self.tok.encode(ref_text, add_special_tokens=False).ids
        p.append(TOK_REF_AUDIO)
        p += [AUDIO_PLACEHOLDER] * num_ref_tokens
        p.append(TOK_TEXT)
        p += self.tok.encode(text, add_special_tokens=False).ids
        p.append(TOK_AUDIO)
        return np.asarray(p, dtype=np.int32)

    # ---- 2. text + ref codes -> AR codes ----
    def generate(self, text, ref_codes, ref_text="", temperature=0.9, seed=42, topk=50,
                 max_steps=400):
        num_ref = np.asarray(ref_codes).shape[0] + (N_CB - 1)   # + delay pattern
        prompt = self.build_prompt(text, ref_text, num_ref)
        return AR.ar_generate(self.rt, self.w, prompt, np.asarray(ref_codes, dtype=np.int32),
                              temperature=temperature, seed=seed, max_steps=max_steps, topk=topk)

    # ---- 3. codes -> PCM ----
    def decode(self, codes):
        return DE.dac_decode(self.rt, self.w, np.asarray(codes, dtype=np.int32))

    # ---- end to end ----
    def synthesize(self, text, ref_wav, ref_text="", sample_rate=24000, temperature=0.9,
                   seed=42, topk=50, max_steps=400):
        ref_codes = self.encode_ref(ref_wav, sample_rate)
        codes = self.generate(text, ref_codes, ref_text, temperature, seed, topk, max_steps)
        return self.decode(codes)
