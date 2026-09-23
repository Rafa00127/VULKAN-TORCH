"""Load the HiggsTTS prefill (encode_ref) weights from a GGUF into device Memory.

Only the encoder-side tensors are loaded; the acoustic/semantic *decoders* and
the Qwen3 backbone (`blk.*`) belong to the AR/decode stages and are skipped.
"""
import numpy as np

import vulkantorch as mt

PREFILL_PREFIXES = (
    "codec.ac_enc",   # acoustic encoder (DAC encoder)
    "codec.ac_dec",   # acoustic decoder (DAC decoder)
    "codec.enc_sem",  # semantic encoder conv blocks
    "codec.sem.",     # HuBERT feature extractor + encoder + pos-conv-embed
    "codec.quant.",   # RVQ quantizers
    "codec.fc",       # fusion FC
)

# Qwen3 backbone + fused embedding/head (for the AR stage)
BACKBONE_PREFIXES = (
    "blk.", "token_embd", "fused_embed", "fused_head", "output_norm",
)


class HiggsWeights:
    def __init__(self, path, device, arena_bytes=512 << 20, extra_bytes=128 << 20,
                 backbone=False):
        self.file = mt.GgufFile(path, device, arena_bytes if not backbone else (6 << 30))
        self.mem = mt.Memory(device, extra_bytes)  # for computed weights
        prefixes = PREFILL_PREFIXES + (BACKBONE_PREFIXES if backbone else ())
        self.t = {}
        for name in self.file.names():
            if name.startswith(prefixes):
                self.t[name] = self.file.tensor(name)

    def put(self, name, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        self.t[name] = self.mem.tensor(list(a.shape), a.tobytes())

    def __len__(self):
        return len(self.t)

    def __getitem__(self, name):
        return self.t[name]

    def get(self, name):
        return self.t.get(name)

    def has(self, name):
        return name in self.t
