"""Export AR-stage references (backbone prefill) from the PyTorch model.

Uses the DLL's ref codes so the AR comparison is isolated from any prefill gap.
Run with the sglang-omni venv.
"""
import os
import sys

import numpy as np
import torch

SGL = r"<local>/sglang\sglang-omni-main"
sys.path.insert(0, SGL)
from higgs_tts_standalone.loader import load_model                      # noqa: E402
from higgs_tts_standalone.delay_pattern import apply_delay_pattern      # noqa: E402
from higgs_tts_standalone.engine import _build_prefill_embeds           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
REFS = os.path.join(_ROOT, "data", "higgstts")
MODEL = os.path.join(SGL, "models", "bosonai--higgs-audio-v3-tts-4b")
OUT = os.path.join(REFS, "ar_ref.npz")
REF_TEXT = "I have no doubt you will become Elden lord, may you take the throne."
TARGET_TEXT = "<|style:whispering|>Hello how you doing? Are you having fun these days?"


def main():
    model = load_model(MODEL, device="cuda", dtype="bfloat16")
    codes = np.load(os.path.join(REFS, "dll_codes.npy")).astype(np.int64)  # [T,8]
    ref = torch.from_numpy(codes).to("cuda")
    delayed = apply_delay_pattern(ref)  # [T+7, 8]

    adapter = model.tokenizer
    prompt_ids = adapter.build_prompt(TARGET_TEXT, num_ref_tokens=delayed.shape[0],
                                      reference_text=REF_TEXT)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")

    backbone = model.backbone
    embed = backbone.get_input_embeddings()
    with torch.no_grad():
        ie = _build_prefill_embeds(prompt_t, delayed, embed, model.fused_embedding, model.dtype)
        out = backbone(inputs_embeds=ie, output_hidden_states=True, use_cache=True)

    caps = {
        "prompt_ids": np.array(prompt_ids, dtype=np.int32),
        "ref_codes_delayed": delayed.cpu().numpy().astype(np.int32),
        "inputs_embeds": ie.float().cpu().numpy(),
        "last_hidden": out.hidden_states[-1][:, -1:, :].float().cpu().numpy(),
        "hidden_states": torch.stack(out.hidden_states, 0).float().cpu().numpy(),  # [L+1,1,Lp,D]
    }
    pkv = out.past_key_values
    k = torch.stack([l.keys for l in pkv.layers], 0).float().cpu().numpy()  # [36,1,nkv,Lp,hd]
    v = torch.stack([l.values for l in pkv.layers], 0).float().cpu().numpy()
    caps["kv_k"] = k
    caps["kv_v"] = v
    np.savez(OUT, **caps)
    print("saved", OUT)
    for kk, vv in caps.items():
        print(f"  {kk}: {vv.shape}")


if __name__ == "__main__":
    main()
