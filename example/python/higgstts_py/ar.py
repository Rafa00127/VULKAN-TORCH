"""HiggsTTS AR (Qwen3 backbone) port — staged, validated against higgstts_py/ar_ref.npz."""
import numpy as np

import vulkantorch as mt

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


# Decode-step graph cache: one captured graph per Lk bucket. The graph is built
# once with Lk pinned to the bucket and the KV write/CausalMask supplied as
# runtime inputs, so it stays valid for every n_past inside the bucket — which
# lets us skip re-emitting ~400 ops through the binding layer on every token.
STEP_BUCKETS = (128, 256, 512, 1024, 2048)


def _pick_bucket(lk_needed):
    for b in STEP_BUCKETS:
        if b >= lk_needed:
            return b
    return None


def backbone_layer_cached(w, li, x, kv_k, kv_v, pos, mask, lk, max_ctx, T):
    """Like backbone_layer, but with no n_past baked in: the KV write uses
    set_rows at the runtime `pos`, and the attention reads a fixed `lk` window
    (slots past n_past are -inf in `mask`)."""
    p = f"blk.{li}."
    nb1, nb2, nb3 = 2 * HD, 2 * HD * max_ctx, 2 * HD * max_ctx * NKV
    residual = x
    h = mt.mul(mt.rms_norm(x, ECL), w[p + "attn_norm.weight"])

    q3 = mt.reshape(mt.linear(h, w[p + "attn_q.weight"]), [T, NH, HD])
    k3 = mt.reshape(mt.linear(h, w[p + "attn_k.weight"]), [T, NKV, HD])
    v3 = mt.reshape(mt.linear(h, w[p + "attn_v.weight"]), [T, NKV, HD])
    q3 = mt.mul(mt.rms_norm(q3, ECL), w[p + "attn_q_norm.weight"])
    k3 = mt.mul(mt.rms_norm(k3, ECL), w[p + "attn_k_norm.weight"])

    rp = (HD, 2, 0, ROPE_THETA, 1.0, 0.0, 1.0, 32.0, 1.0)
    q3 = mt.rope(q3, pos, *rp)
    k3 = mt.rope(k3, pos, *rp)
    q = _to_heads_perm(q3, T, NH)
    kpo = _to_heads_perm(k3, T, NKV)          # ne=[hd,T,nkv] — also the set_rows src
    vpo = _to_heads_perm(v3, T, NKV)

    # write new K/V into the cache; `pos` (= n_past) is the row index, supplied
    # at runtime so the captured graph does not depend on n_past
    kdst = mt.view_3d(kv_k, HD, max_ctx, NKV, nb1, nb2, li * nb3)
    vdst = mt.view_3d(kv_v, HD, max_ctx, NKV, nb1, nb2, li * nb3)
    mt.set_rows(kdst, kpo, pos)
    mt.set_rows(vdst, vpo, pos)

    kf = mt.contiguous(mt.view_3d(kv_k, HD, lk, NKV, nb1, nb2, li * nb3))
    vf = mt.contiguous(mt.view_3d(kv_v, HD, lk, NKV, nb1, nb2, li * nb3))

    attn = mt.flash_attn(q, kf, vf, mask, 1.0 / (HD ** 0.5))
    attn = mt.reshape(mt.contiguous(attn), [T, NH * HD])
    x = mt.add(residual, mt.linear(attn, w[p + "attn_output.weight"]))

    residual = x
    h = mt.mul(mt.rms_norm(x, ECL), w[p + "ffn_norm.weight"])
    gate = mt.silu(mt.linear(h, w[p + "ffn_gate.weight"]))
    up = mt.linear(h, w[p + "ffn_up.weight"])
    x = mt.add(residual, mt.linear(mt.mul(gate, up), w[p + "ffn_down.weight"]))
    return x


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


def ar_generate(rt, w, prompt_ids, ref_codes, temperature=0.9, seed=42, max_steps=None, topk=50,
                graph_cache=True):
    """Returns raw codes PT [T, 8] (int32).

    ``max_steps=None`` predicts the budget the same way the reference C++
    (`higgs_backbone_ar`) does: 12 frames per text token + 200 slack.
    ``graph_cache`` captures the decode step once per Lk bucket and replays it,
    which skips re-emitting the ~400-op graph through the binding layer.
    """
    delayed = _apply_delay_pattern(np.asarray(ref_codes, dtype=np.int32))   # [L_audio, 8]
    L = len(prompt_ids)
    if max_steps is None:
        n_text = max(1, L - delayed.shape[0] - 5)     # 5 = the prompt's special tokens
        max_steps = n_text * 12 + 200
    # the cache needs room for the largest bucket
    max_ctx = max(L + max_steps + 10, STEP_BUCKETS[-1]) if graph_cache else L + max_steps + 10
    gpu = rt.gpu()
    # zero-fill the cache: slots past n_past are masked to -inf, and uninitialised
    # device memory (possibly NaN) would otherwise poison the softmax
    zeros = np.zeros(N_LAYERS * NKV * max_ctx * HD, dtype=np.float16).tobytes() if graph_cache else b""
    kv_mem = mt.Memory(gpu, 2 * N_LAYERS * NKV * max_ctx * HD * 2)
    kv_k = kv_mem.tensor([N_LAYERS, NKV, max_ctx, HD], 1, zeros)
    kv_v = kv_mem.tensor([N_LAYERS, NKV, max_ctx, HD], 1, zeros)

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

    # ---- decode-step graph cache: one captured graph per Lk bucket ----
    step_cache = {}

    def build_step(lk):
        g = mt.Graph(rt, gpu)
        g.enter()
        ids = [g.input_i32([1], np.zeros(1, np.int32).tobytes()) for _ in range(N_CB)]
        pos_t = g.input_i32([1], np.zeros(1, np.int32).tobytes())
        mask_in = g.input([1, lk], np.zeros(lk, np.float32).tobytes())   # F32; updated per step
        mask_t = mt.cast(mask_in, 1)                                    # flash_attn wants F16
        e = None
        for c in range(N_CB):
            r = mt.get_rows(w["fused_embed.weight"], ids[c])
            e = r if e is None else mt.add(e, r)
        for li in range(N_LAYERS):
            e = backbone_layer_cached(w, li, e, kv_k, kv_v, pos_t, mask_t, lk, max_ctx, 1)
        hn = mt.mul(mt.rms_norm(e, ECL), w["output_norm.weight"])
        logits = mt.contiguous(fused_head_logits(w, hn))
        logits.mark_output()
        g.exit()
        g.alloc_static()          # allocate once; replays only re-upload inputs
        return {"g": g, "ids": ids, "pos": pos_t, "mask": mask_in, "logits": logits}

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

        lk = _pick_bucket(n_past + 1) if graph_cache else None
        if lk is not None:
            ent = step_cache.get(lk)
            if ent is None:
                ent = step_cache[lk] = build_step(lk)
            g = ent["g"]
            for c in range(N_CB):
                v = np.array([c * CB_VOCAB + int(cn[c])], dtype=np.int32)
                g.set_input(ent["ids"][c], v.tobytes())
            g.set_input(ent["pos"], np.array([n_past], dtype=np.int32).tobytes())
            m = np.full(lk, -np.inf, dtype=np.float32)
            m[: n_past + 1] = 0.0
            g.set_input(ent["mask"], m.tobytes())
            g.compute_static()
            lg = np.frombuffer(ent["logits"].to_bytes(), dtype=np.float32).reshape(N_CB, CB_VOCAB)
        else:
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
