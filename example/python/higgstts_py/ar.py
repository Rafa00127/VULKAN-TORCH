"""HiggsTTS AR (Qwen3 backbone) port — staged, validated against higgstts_py/ar_ref.npz."""
import numpy as np

import minitorch as mt

D = 2560
NH = 32
NKV = 8
HD = 128
N_LAYERS = 36
ROPE_THETA = 1e6
ECL = 1e-6
N_CB = 8
CB_VOCAB = 1026


def build_prefill_embeds(g, w, prompt_ids, delayed_codes):
    """text embedding at text positions + summed fused-audio embedding at the
    (-100) audio positions. Returns PT [L_prompt, D]."""
    L = len(prompt_ids)
    L_audio = delayed_codes.shape[0]
    pid = np.asarray(prompt_ids)
    audio_pos = pid == -100
    pre_len = int(np.argmax(audio_pos)) if audio_pos.any() else L

    safe = np.where(audio_pos, 0, pid).astype(np.int32)
    text_emb = mt.get_rows(w["token_embd.weight"], g.input_i32([L], safe.tobytes()))  # [L, D]

    audio_emb = None
    for c in range(N_CB):
        idx = (c * CB_VOCAB + delayed_codes[:, c]).astype(np.int32)
        e = mt.get_rows(w["fused_embed.weight"], g.input_i32([L_audio], idx.tobytes()))
        audio_emb = e if audio_emb is None else mt.add(audio_emb, e)

    mask = np.where(audio_pos, 0.0, 1.0).astype(np.float32).reshape(L, 1)
    masked = mt.mul(text_emb, g.input([L, 1], mask.tobytes()))

    zeros_pre = g.input([pre_len, D], np.zeros(pre_len * D, np.float32).tobytes())
    full = mt.concat(zeros_pre, audio_emb, 0)
    post = L - pre_len - L_audio
    if post > 0:
        full = mt.concat(full, g.input([post, D], np.zeros(post * D, np.float32).tobytes()), 0)
    return mt.add(masked, full)


def _to_heads(x, T, n):
    """PT [T, n*hd] -> PT [n, T, hd] (ne=[hd,T,n])."""
    x4 = mt.reshape(x, [T, n, HD, 1])
    y4 = mt.permute_pt(x4, 1, 0, 2, 3)
    return mt.reshape(mt.contiguous(y4), [n, T, HD])


def backbone_layer(g, w, li, x, kv_k, kv_v, pos, mask, n_past, max_ctx, T):
    """One Qwen3 layer with KV-cache write/read. x: PT [T, D] -> PT [T, D]."""
    p = f"blk.{li}."
    nb1, nb2, nb3 = 2 * HD, 2 * HD * max_ctx, 2 * HD * max_ctx * NKV   # F16 cache
    residual = x
    h = mt.mul(mt.rms_norm(x, ECL), w[p + "attn_norm.weight"])

    q3 = mt.reshape(mt.linear(h, w[p + "attn_q.weight"]), [T, NH, HD])
    k3 = mt.reshape(mt.linear(h, w[p + "attn_k.weight"]), [T, NKV, HD])
    v3 = mt.reshape(mt.linear(h, w[p + "attn_v.weight"]), [T, NKV, HD])
    q3 = mt.mul(mt.rms_norm(q3, ECL), w[p + "attn_q_norm.weight"])
    k3 = mt.mul(mt.rms_norm(k3, ECL), w[p + "attn_k_norm.weight"])

    rp = (HD, 2, 0, ROPE_THETA, 1.0, 0.0, 1.0, 32.0, 1.0)
    q3 = mt.rope(q3, pos, *rp); k3 = mt.rope(k3, pos, *rp)
    q = _to_heads_perm(q3, T, NH)  # -> [nh, T, hd] for flash-attn

    # write new K/V into the F16 cache at [n_past, n_past+T)
    kpo = _to_heads_perm(k3, T, NKV)
    vpo = _to_heads_perm(v3, T, NKV)
    kdst = mt.view_4d(kv_k, HD, T, NKV, 1, nb1, nb2, nb3, li * nb3 + n_past * nb1)
    vdst = mt.view_4d(kv_v, HD, T, NKV, 1, nb1, nb2, nb3, li * nb3 + n_past * nb1)
    mt.cpy(kpo, kdst); mt.cpy(vpo, vdst)

    Lk = n_past + T
    kf = mt.contiguous(mt.view_3d(kv_k, HD, Lk, NKV, nb1, nb2, li * nb3))
    vf = mt.contiguous(mt.view_3d(kv_v, HD, Lk, NKV, nb1, nb2, li * nb3))

    attn = mt.flash_attn(q, kf, vf, mask, 1.0 / (HD ** 0.5))
    attn = mt.reshape(mt.contiguous(attn), [T, NH * HD])
    x = mt.add(residual, mt.linear(attn, w[p + "attn_output.weight"]))

    residual = x
    h = mt.mul(mt.rms_norm(x, ECL), w[p + "ffn_norm.weight"])
    gate = mt.silu(mt.linear(h, w[p + "ffn_gate.weight"]))
    up = mt.linear(h, w[p + "ffn_up.weight"])
    x = mt.add(residual, mt.linear(mt.mul(gate, up), w[p + "ffn_down.weight"]))
    return x


def _to_heads_perm(x, T, n):
    """PT [T, n, hd] -> PT [n, T, hd] (ne=[hd,T,n]) for the cache write."""
    x4 = mt.reshape(x, [T, n, HD, 1])
    return mt.reshape(mt.contiguous(mt.permute_pt(x4, 1, 0, 2, 3)), [n, T, HD])


def fused_head_logits(w, hidden_last):
    """hidden_last: PT [1, D] (post output_norm) -> numpy [N_CB, 1026]."""
    lg = mt.linear(hidden_last, w["fused_head.weight"])   # PT [1, 8*1026]
    return mt.reshape(lg, [N_CB, CB_VOCAB])


def embed_codes(g, w, codes):
    """codes [N_CB] int -> PT [1, D] (sum of per-codebook fused embeddings)."""
    e = None
    for c in range(N_CB):
        idx = np.array([c * CB_VOCAB + int(codes[c])], dtype=np.int32)
        r = mt.get_rows(w["fused_embed.weight"], g.input_i32([1], idx.tobytes()))
        e = r if e is None else mt.add(e, r)
    return e


def sample_codes(logits, temperature, rng, topk=50):
    out = np.zeros(N_CB, dtype=np.int32)
    for c in range(N_CB):
        l = logits[c].astype(np.float64) / max(temperature, 1e-6)
        idx = np.argsort(-l)[:topk]
        v = l[idx]
        p = np.exp(v - v.max()); p /= p.sum()
        out[c] = idx[rng.choice(topk, p=p)]
    return out


def ar_generate(rt, w, prompt_ids, ref_codes, temperature=0.9, seed=42, max_steps=400, topk=50):
    """Returns raw codes PT [T, 8] (int32)."""
    delayed = _apply_delay_pattern(np.asarray(ref_codes, dtype=np.int32))   # [L_audio, 8]
    L = len(prompt_ids)
    max_ctx = L + max_steps + 8
    gpu = rt.gpu()
    kv_mem = mt.Memory(gpu, 2 * N_LAYERS * NKV * max_ctx * HD * 2)
    kv_k = kv_mem.tensor([N_LAYERS, NKV, max_ctx, HD], 1, b"")
    kv_v = kv_mem.tensor([N_LAYERS, NKV, max_ctx, HD], 1, b"")

    mask_f = np.where(np.arange(L)[None, :] <= np.arange(L)[:, None], 0.0, -np.inf).astype(np.float32)
    rng = np.random.default_rng(seed)

    # ---- prefill ----
    g = mt.Graph(rt, gpu); g.enter()
    x = build_prefill_embeds(g, w, prompt_ids, delayed)
    pos = g.input_i32([L], np.arange(L, dtype=np.int32).tobytes())
    mask = mt.cast(g.input([L, L], mask_f.tobytes()), 1)
    for li in range(N_LAYERS):
        x = backbone_layer(g, w, li, x, kv_k, kv_v, pos, mask, 0, max_ctx, L)
    hn = mt.mul(mt.rms_norm(x, ECL), w["output_norm.weight"])
    last = mt.reshape(mt.view_2d(mt.contiguous(hn), D, 1, D * 4, (L - 1) * D * 4), [1, D])
    logits = mt.contiguous(fused_head_logits(w, last)); logits.mark_output()
    g.exit()
    lg = np.frombuffer(logits.to_bytes(), dtype=np.float32).reshape(N_CB, CB_VOCAB)
    n_past = L

    codes_all = []
    delay_count = 0
    eoc_countdown = -1
    for step in range(max_steps):
        cn = sample_codes(lg, temperature, rng, topk)
        if delay_count < N_CB:
            if delay_count + 1 < N_CB:
                cn[delay_count + 1:] = 1024   # BOC
            delay_count += 1
        elif eoc_countdown >= 0:
            eoc_countdown -= 1
        elif cn[0] == 1025:                   # EOC in codebook 0
            eoc_countdown = N_CB - 2
        codes_all.append(cn.copy())
        if eoc_countdown == 0:
            break

        g = mt.Graph(rt, gpu); g.enter()
        e = embed_codes(g, w, cn)             # [1, D]
        pos = g.input_i32([1], np.array([n_past], dtype=np.int32).tobytes())
        for li in range(N_LAYERS):
            e = backbone_layer(g, w, li, e, kv_k, kv_v, pos, None, n_past, max_ctx, 1)
        hn = mt.mul(mt.rms_norm(e, ECL), w["output_norm.weight"])
        logits = mt.contiguous(fused_head_logits(w, hn)); logits.mark_output()
        g.exit()
        lg = np.frombuffer(logits.to_bytes(), dtype=np.float32).reshape(N_CB, CB_VOCAB)
        n_past += 1

    delayed_out = np.stack(codes_all, axis=0)   # [S, 8]
    return _reverse_delay_pattern(delayed_out)


def _apply_delay_pattern(codes):
    T, N = codes.shape
    out = np.full((T + N - 1, N), 1025, dtype=np.int32)
    for c in range(N):
        out[:c, c] = 1024
        out[c:c + T, c] = codes[:, c]
    return out


def _reverse_delay_pattern(delayed):
    N = delayed.shape[1]
    T = delayed.shape[0] - (N - 1)
    return np.stack([delayed[c:c + T, c] for c in range(N)], axis=1).astype(np.int32)
