#!/usr/bin/env python
"""Dump per-module reference tensors from the OFFICIAL IndexTTS 2.5 PyTorch model.

Run with the dedicated venv (Python 3.11 + torch 2.8 cpu + transformers 4.52):

    移植参考\\.venv-indextts\\Scripts\\python.exe tools\\dump_ref_index.py

Everything uses fixed, seeded inputs and is written to ``data/indextts_ref/*.npy``
so the C# port can be checked stage by stage (same idea as the HiggsTTS port).
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "移植参考", "index-tts-main")
MODEL = os.path.join(ROOT, "移植参考", "model")
OUT = os.path.join(ROOT, "data", "indextts_ref")

sys.path.insert(0, MAIN)

results = []


def done(name, arr):
    a = arr.detach().to(torch.float32).cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr, np.float32)
    np.save(os.path.join(OUT, name + ".npy"), a)
    results.append((name, list(a.shape)))
    print(f"  OK   {name:34s} {list(a.shape)}")


def round_f16(module):
    """Round 2D+ params to f16 precision in place, matching the GGUF the C# side
    loads (see convert_index_tts2_to_gguf.py). Without this the reference is fp32
    while the port is f16 and the comparison is meaningless."""
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() >= 2:
                p.copy_(p.half().float())
    return module


def stage(label, fn):
    print(f"[{label}]")
    try:
        fn()
    except Exception as e:
        import traceback
        print(f"  FAIL {label}: {type(e).__name__}: {str(e)[:120]}")
        traceback.print_exc()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    OUT = ap.parse_args().out
    os.makedirs(OUT, exist_ok=True)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(MODEL, "index2.5", "config.yaml"))
    torch.manual_seed(1234)
    np.random.seed(1234)

    # ---------------- GPT ----------------
    def gpt_stage():
        from indextts.gpt.model_v2 import UnifiedVoice
        from indextts.utils.checkpoint import load_checkpoint
        gpt = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
        load_checkpoint(gpt, os.path.join(MODEL, "index2.5", "gpt.pth"))
        round_f16(gpt); gpt.eval()

        B = 1
        # spk: campplus style [B, 192] -> spk_emb_proj -> [B, 1, 1280]
        spk_in = torch.randn(B, 192)
        spk = gpt.spk_emb_proj(spk_in).unsqueeze(1)
        done("gpt_spk_in", spk_in); done("gpt_spk_cond_emb", spk)

        # emo conformer: [B, T, 1024] -> [B, T, 512]  (returns (out, pos_emb))
        Te = 40
        emo_in = torch.randn(B, Te, 1024)
        with torch.no_grad():
            conf, _pos = gpt.emo_conditioning_encoder(emo_in, torch.LongTensor([Te]))
        done("gpt_emo_conf_in", emo_in); done("gpt_emo_conf_out", conf)

        # perceiver -> emovec [B, 1, 1280]
        with torch.no_grad():
            lat = gpt.emo_perceiver_encoder(conf)
        done("gpt_emo_perceiver_out", lat)

        # merge_emovec takes the RAW 1024-dim conditioning latents (it re-runs the
        # conformer+perceiver internally, [B,T,1024] -> transposed [B,1024,T])
        spk_raw = torch.randn(B, 30, 1024)
        emo_raw = emo_in
        with torch.no_grad():
            emovec = gpt.merge_emovec(spk_raw, emo_raw, torch.LongTensor([30]),
                                      torch.LongTensor([Te]), alpha=1.0)
        done("gpt_spk_raw", spk_raw); done("gpt_emovec", emovec)

        # GPT2 core: fixed inputs_embeds -> hidden states + head logits
        L = 24
        emb = torch.randn(B, L, cfg.gpt.model_dim)
        with torch.no_grad():
            o = gpt.gpt(inputs_embeds=emb, output_hidden_states=True)
        done("gpt_core_in", emb)
        done("gpt_core_last_hidden", o.last_hidden_state)
        # per-layer hidden states for debugging the C# port ([0]=embeddings)
        for i in (1, 2, 24):
            done(f"gpt_hs{i}", o.hidden_states[i])

        # ---- layer-0 internals (hook every stage of block 0) ----
        h0 = gpt.gpt.h[0]
        cap = {}

        def hk(key):
            def f(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                cap[key] = o.detach()
            return f

        hs = [h0.ln_1.register_forward_hook(hk("ln1")),
              h0.attn.c_attn.register_forward_hook(hk("qkv")),
              h0.attn.register_forward_hook(hk("attn")),
              h0.attn.c_proj.register_forward_hook(hk("attn_proj")),
              h0.ln_2.register_forward_hook(hk("ln2")),
              h0.mlp.c_fc.register_forward_hook(hk("fc")),
              h0.mlp.act.register_forward_hook(hk("act")),
              h0.mlp.register_forward_hook(hk("mlp"))]
        with torch.no_grad():
            gpt.gpt(inputs_embeds=emb)
        for h in hs:
            h.remove()
        for k, v in cap.items():
            done(f"gpt_l0_{k}", v)
        done("gpt_l0_m1", emb + cap["attn"])   # residual after the attention sub-block

        # attention internals, recomputed from the captured qkv (to isolate the port)
        qkv = cap["qkv"]
        q_, k_, v_ = qkv.split(1280, dim=2)
        B_, T_, _ = q_.shape
        def sp(x): return x.view(B_, T_, 20, 64).transpose(1, 2)
        qq, kk, vv = sp(q_), sp(k_), sp(v_)
        aw = (qq @ kk.transpose(-1, -2)) / 8.0
        cm = torch.tril(torch.ones(T_, T_, dtype=torch.bool))
        aw = torch.softmax(aw.masked_fill(~cm, float("-inf")), dim=-1)
        raw = (aw @ vv).transpose(1, 2).reshape(B_, T_, 1280)
        done("gpt_l0_attn_weights", aw)
        done("gpt_l0_attn_raw", raw)
        with torch.no_grad():
            h = gpt.final_norm(o.last_hidden_state)
            logits = gpt.mel_head(h)
        done("gpt_final_norm_out", h)
        done("gpt_mel_logits", logits)

    stage("GPT", gpt_stage)

    # ---------------- semantic codec ----------------
    def codec_stage():
        from indextts.codec.models import EnhancedCodec
        sc = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
        sc.load_checkpoint(os.path.join(MODEL, "index2.5", "codec.pth"))
        round_f16(sc); sc.eval()
        T = 12
        codes = torch.randint(0, 8192, (1, T))
        with torch.no_grad():
            S = sc.decode(codes)
        done("codec_codes", codes.to(torch.float32)); done("codec_decode_out", S)

    stage("semantic_codec", codec_stage)

    # ---------------- s2mel: length_regulator + CFM ----------------
    def s2mel_stage():
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        m = MyModel(cfg.s2mel)
        m, _, _, _ = load_checkpoint2(m, None, os.path.join(MODEL, "index2.5", "s2mel.pth"),
                                      load_only_params=True)
        round_f16(m); m.eval()
        lr = m.models["length_regulator"]

        Ti, d = 20, 1024
        S_in = torch.randn(1, Ti, d)
        ylen = torch.LongTensor([37])
        with torch.no_grad():
            cond = lr(S_in, ylens=ylen, n_quantizers=3, f0=None)[0]
        done("lr_in", S_in); done("lr_out", cond)

        cfm = m.models["cfm"]
        cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        style = torch.randn(1, cfg.s2mel.DiT.get("style_dim", 192))
        ref_mel = torch.randn(1, cfg.s2mel.preprocess_params.spect_params.n_mels, 50)
        cat_cond = torch.cat([torch.randn(1, 30, cond.shape[-1]), cond], dim=1)
        done("cfm_style", style); done("cfm_ref_mel", ref_mel); done("cfm_cat_cond", cat_cond)

        # capture the initial noise z so C# can reproduce the exact trajectory
        orig = torch.randn
        cap = {}

        def spy(*k, **kw):
            r = orig(*k, **kw)
            cap.setdefault("z", r)
            return r

        torch.randn = spy
        try:
            with torch.no_grad():
                vc = cfm.inference(cat_cond, torch.LongTensor([cat_cond.size(1)]),
                                   ref_mel, style, None, 25, inference_cfg_rate=0.7)
        finally:
            torch.randn = orig
        if "z" in cap:
            done("cfm_z0", cap["z"])
        done("cfm_out", vc)

    stage("s2mel", s2mel_stage)

    # ---------------- BigVGAN ----------------
    def bigvgan_stage():
        from indextts.s2mel.modules.bigvgan import bigvgan
        bv = bigvgan.BigVGAN.from_pretrained(os.path.join(MODEL, "bigvan"), use_cuda_kernel=False)
        bv.remove_weight_norm(); round_f16(bv); bv.eval()
        mel = torch.randn(1, 80, 64)
        with torch.no_grad():
            wav = bv(mel)
        done("bigvgan_mel", mel); done("bigvgan_wav", wav)

    stage("BigVGAN", bigvgan_stage)

    # ---------------- w2v-bert ----------------
    def w2v_stage():
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
        fx = SeamlessM4TFeatureExtractor.from_pretrained(os.path.join(MODEL, "w2v"))
        sm = Wav2Vec2BertModel.from_pretrained(os.path.join(MODEL, "w2v"))
        round_f16(sm); sm.eval()
        # feature extractor wants a 16k waveform; use a fixed synthetic one
        wav = (torch.randn(16000 * 3) * 0.1).numpy()
        inp = fx(wav, sampling_rate=16000, return_tensors="pt")
        feats, mask = inp["input_features"], inp["attention_mask"]
        with torch.no_grad():
            o = sm(input_features=feats, attention_mask=mask, output_hidden_states=True)
        st = torch.load(os.path.join(MODEL, "index2.5", "wav2vec2bert_stats.pt"), map_location="cpu")
        h = o.hidden_states[17]
        h = (h - st["mean"]) / torch.sqrt(st["var"])
        done("w2v_feats", feats); done("w2v_attention_mask", mask.to(torch.float32))
        done("w2v_hidden17_norm", h)

    stage("w2v-bert", w2v_stage)

    # ---------------- CAMPPlus ----------------
    def campplus_stage():
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        cp = CAMPPlus(feat_dim=80, embedding_size=192)
        cp.load_state_dict(torch.load(os.path.join(MODEL, "camplus", "campplus_cn_common.bin"),
                                      map_location="cpu"))
        round_f16(cp); cp.eval()
        T = 200
        fbank = torch.randn(T, 80)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)   # as infer_v2_5 does
        with torch.no_grad():
            style = cp(fbank.unsqueeze(0))
        done("campplus_fbank", fbank.unsqueeze(0)); done("campplus_style", style)

    stage("CAMPPlus", campplus_stage)

    print(f"\n{len(results)} tensors -> {OUT}")
    with open(os.path.join(OUT, "index.txt"), "w", encoding="utf-8") as f:
        for n, s in results:
            f.write(f"{n} {s}\n")


if __name__ == "__main__":
    main()
