#!/usr/bin/env python
"""Dump GPT autoregressive-step references (prefill + one decode step) from the
official model, so the C# AR loop can be validated without the text frontend.

    移植参考\\.venv-indextts\\Scripts\\python.exe tools\\dump_ref_gpt_ar.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "移植参考", "index-tts-main")
MODEL = os.path.join(ROOT, "移植参考", "model")
OUT = os.path.join(ROOT, "data", "indextts_ref")
sys.path.insert(0, MAIN)


def done(name, arr):
    a = arr.detach().to(torch.float32).cpu().contiguous().numpy() if hasattr(arr, "detach") \
        else np.ascontiguousarray(arr, np.float32)
    np.save(os.path.join(OUT, name + ".npy"), a)
    print(f"  OK   {name:28s} {list(a.shape)}")


def round_f16(module):
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() >= 2:
                p.copy_(p.half().float())
    return module


def main():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(MODEL, "index2.5", "config.yaml"))
    torch.manual_seed(1234)

    from indextts.gpt.model_v2_5 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    gpt = UnifiedVoice(**cfg.gpt, use_accel=False)
    load_checkpoint(gpt, os.path.join(MODEL, "index2.5", "gpt.pth"))
    gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True)   # builds inference_model
    round_f16(gpt)
    gpt.eval()

    B = 1
    with torch.no_grad():
        # conditioning: (spk + emo) then two zero rows -> the "soft prefix" [1,3,1280]
        spk = gpt.spk_emb_proj(torch.randn(B, 192)).unsqueeze(1)
        emo = gpt.get_emovec(torch.randn(B, 40, 1024), torch.LongTensor([40]))
        conds = torch.cat((spk + emo.unsqueeze(1), torch.zeros(B, 2, cfg.gpt.model_dim)), 1)
        done("ar_conds", conds)

        # pre-encoded text ids (no tokenizer needed for this test)
        L = 12
        text_inputs = torch.randint(1000, 60000, (B, L))
        langs = torch.zeros(B, dtype=torch.long)
        done("ar_text_inputs", text_inputs.float())
        fake_inputs, inputs_embeds, attn_mask = gpt.prepare_gpt_inputs(conds, text_inputs, langs)
        done("ar_fake_inputs", fake_inputs.float())
        done("ar_inputs_embeds", inputs_embeds)
        done("ar_attention_mask", attn_mask.float())
        print(f"  (mel_len={inputs_embeds.shape[1]}, prefix len={fake_inputs.shape[1]})")

        # ---- prefill ----
        gpt.inference_model.store_mel_emb(inputs_embeds)
        out = gpt.inference_model(fake_inputs, attention_mask=attn_mask, use_cache=True)
        done("ar_logits_prefill", out.logits)
        past = out.past_key_values

        # ---- one decode step (greedy argmax) ----
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        done("ar_next_token", nxt.float())
        step_mask = torch.cat([attn_mask, torch.ones(B, 1, dtype=attn_mask.dtype)], 1)
        out2 = gpt.inference_model(nxt, past_key_values=past, attention_mask=step_mask, use_cache=True)
        done("ar_logits_step1", out2.logits)
        print(f"  (decode step: token={int(nxt[0,0])}, total_len={step_mask.shape[1]})")


if __name__ == "__main__":
    main()
