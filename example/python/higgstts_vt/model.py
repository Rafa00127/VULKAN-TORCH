"""HiggsTTS prefill (encode_ref) port to vulkantorch — staged, validated against
higgstts_vt/ref.npz (PyTorch, bf16) per stage; final codes compared to higgs_tts.dll.
"""
import numpy as np

import vulkantorch as mt

STRIDES = [5, 2, 2, 2, 2, 2, 2]
WAVLM_LAYERS = 12
WAVLM_HEADS = 12
WAVLM_HD = 64

# pos_conv_embed (HubertPositionalConvEmbedding): grouped Conv1d(768,768,k=128,groups=16,pad=64)
PCE_K = 128
PCE_CG = 48
PCE_NG = 16
PCE_C = 768


def build_pce_weight(w):
    """Fuse weight_norm (dim=2, i.e. over the K axis) into a single grouped-conv
    weight, matching the reference: F[(i + j*K), oc] = g[i]*W[oc,j,i]/norm[i],
    where i=K, j=in-group channel, oc=out channel. Stored as PT [OC, K*Cg]."""
    g = np.frombuffer(w["codec.sem.encoder.pce.conv.pm.weight.orig0"].to_bytes(),
                      dtype=np.float16).astype(np.float32)                       # [K]
    W = np.frombuffer(w["codec.sem.encoder.pce.conv.pm.weight.orig1"].to_bytes(),
                      dtype=np.float16).astype(np.float32).reshape(PCE_C, PCE_CG, PCE_K)  # [oc,icg,k]
    norm = np.sqrt((W * W).sum(axis=(0, 1)))                                    # [k] over oc,icg
    a = np.arange(PCE_K * PCE_CG)
    i, j = a % PCE_K, a // PCE_K                                                # kernel / in-group index
    F = W[:, j, i].T * (g[i] / norm[i])[:, None]                                # [K*Cg, OC]
    w.put("__pce_fused", F.T)                                                   # PT [OC, K*Cg]


def pce(w, x, T):
    """Positional conv embedding: grouped conv + bias + GELU + SamePad crop.
    x: PT [T, 768] -> returns PT [T, 768]."""
    wf = w["__pce_fused"]                 # PT [768, 6144], ne=[6144,768]
    KC = PCE_K * PCE_CG
    NBB = PCE_C * 4                       # nb1 of x (stride between T)
    outs = []
    for gi in range(PCE_NG):
        xg = mt.contiguous(mt.view_2d(x, PCE_CG, T, NBB, gi * PCE_CG * 4))       # ne=[48,T]
        cols = mt.im2col_rafa(xg, PCE_K, 1, 64, 1)                               # ne=[6144,T+1]
        cols = mt.view_2d(cols, KC, T, KC * 4, 0)                                # SamePad: crop last col
        wg = mt.view_2d(wf, KC, PCE_CG, KC * 4, gi * PCE_CG * KC * 4)            # ne=[6144,48]
        outs.append(mt.mul_mat(cols, wg))                                        # ne=[T,48] -> PT [48,T]
    acc = outs[0]
    for o in outs[1:]:
        acc = mt.concat(acc, o, 0)                                               # PT [768, T]
    bias = mt.reshape(w["codec.sem.encoder.pce.conv.bias"], [PCE_C, 1])          # ne=[1,768]
    acc = mt.gelu(mt.add(acc, bias))
    return mt.transpose(acc)                                                     # PT [T, 768]


def norm_over_time(x):
    """Per-channel normalization over the time axis (HF GroupNorm(num_groups=C))."""
    xt = mt.contiguous(mt.transpose(x))
    n = mt.layer_norm(xt, 1e-5)
    return mt.contiguous(mt.transpose(n))


def layernorm_affine(x, wgt, bias, eps=1e-5):
    return mt.add(mt.mul(mt.layer_norm(x, eps), wgt), bias)


def feature_extractor(w, x):
    # x: PT [T, 1]
    for i in range(7):
        x = mt.conv1d(x, w[f"codec.sem.fe.cv.{i}.conv.weight"], STRIDES[i], 0, 1)
        if i == 0:
            x = norm_over_time(x)
            x = mt.add(mt.mul(x, w["codec.sem.fe.cv.0.ln.weight"]),
                       w["codec.sem.fe.cv.0.ln.bias"])
        x = mt.gelu(x)
    return x  # PT [T, 512]


def feature_projection(w, x):
    x = mt.layer_norm(x, 1e-5)
    x = mt.add(mt.mul(x, w["codec.sem.fp.ln.weight"]), w["codec.sem.fp.ln.bias"])
    return mt.add(mt.linear(x, w["codec.sem.fp.projection.weight"]),
                  w["codec.sem.fp.projection.bias"])  # PT [T, 768]


def _to_heads(x, T):
    """PT [T, nh*hd] -> PT [nh, T, hd] (ne=[hd,T,nh]) = flash_attn input layout."""
    x4 = mt.reshape(x, [T, WAVLM_HEADS, WAVLM_HD, 1])
    y4 = mt.permute_pt(x4, 1, 0, 2, 3)          # PT [nh, T, hd, 1]
    return mt.reshape(mt.contiguous(y4), [WAVLM_HEADS, T, WAVLM_HD])


def wavlm_attention(w, li, x):
    p = f"codec.sem.enc.{li}."
    T = x.shape[0]

    def proj(name):
        return mt.add(mt.linear(x, w[p + name + ".weight"]), w[p + name + ".bias"])

    q = _to_heads(proj("attn_q"), T)
    k = _to_heads(proj("attn_k"), T)
    v = _to_heads(proj("attn_v"), T)

    scale = 1.0 / (WAVLM_HD ** 0.5)
    attn = mt.flash_attn(q, k, v, None, scale)              # PT [T, nh, hd]
    attn = mt.reshape(mt.contiguous(attn), [T, WAVLM_HEADS * WAVLM_HD])
    return mt.add(mt.linear(attn, w[p + "attn_out.weight"]), w[p + "attn_out.bias"])


def wavlm_layer(w, li, x):
    p = f"codec.sem.enc.{li}."
    h = mt.add(x, wavlm_attention(w, li, x))
    h = layernorm_affine(h, w[p + "ln.weight"], w[p + "ln.bias"])
    ffn = mt.gelu(mt.add(mt.linear(h, w[p + "ffn1.weight"]), w[p + "ffn1.bias"]))
    ffn = mt.add(mt.linear(ffn, w[p + "ffn2.weight"]), w[p + "ffn2.bias"])
    out = mt.add(h, ffn)
    return layernorm_affine(out, w[p + "fin_ln.weight"], w[p + "fin_ln.bias"])


AC_STRIDES = [8, 5, 4, 2, 3]
AC_PADS = [4, 3, 2, 1, 2]
AC_DILS = [1, 3, 9]


def acoustic_encoder(w, x):
    """DAC encoder: conv1 + 5 blocks (3x ResUnit + down conv) + snake + conv2.
    x: PT [T_audio, 1] -> PT [T_out, 256]."""
    p = "codec.ac_enc."

    def conv(h, name, stride, pad, dil=1):
        return mt.conv1d(h, w[name], stride, pad, dil)

    x = mt.add(conv(x, p + "conv1.weight", 1, 3), w[p + "conv1.bias"])
    for bi in range(5):
        b = p + f"block.{bi}."
        for ri, dil in ((1, 1), (2, 3), (3, 9)):
            r = b + f"res_unit{ri}."
            residual = x
            h = mt.snake_1d(x, w[r + "snake1.alpha"])
            K1 = w[r + "conv1.weight"].shape[2]
            h = mt.add(conv(h, r + "conv1.weight", 1, (K1 - 1) // 2 * dil, dil),
                       w[r + "conv1.bias"])
            h = mt.snake_1d(h, w[r + "snake2.alpha"])
            h = mt.add(conv(h, r + "conv2.weight", 1, 0), w[r + "conv2.bias"])
            x = mt.add(residual, h)
        x = mt.snake_1d(x, w[b + "snake1.alpha"])
        x = mt.add(conv(x, b + "conv1.weight", AC_STRIDES[bi], AC_PADS[bi]),
                   w[b + "conv1.bias"])
    x = mt.snake_1d(x, w[p + "snake1.alpha"])
    return mt.add(conv(x, p + "conv2.weight", 1, 1), w[p + "conv2.bias"])
def semantic_encoder(w, x):
    """SemanticEncoder: Conv1d(k3) + 2 blocks of (2x ResidualUnit + Conv1d(k3)).
    x: PT [T, 768] -> PT [T, 768]."""
    x = mt.conv1d(x, w["codec.enc_sem.conv.weight"], 1, 1, 1)  # k=3, pad=1, no bias
    for bi in range(2):
        p = f"codec.enc_sem.blk.{bi}."
        for ri in range(2):
            rp = p + f"ru.{ri}."
            residual = x
            h = mt.elu(x)
            h = mt.conv1d(h, w[rp + "conv1.weight"], 1, 1, 1)  # k=3, dil=1, pad=1
            h = mt.elu(h)
            h = mt.conv1d(h, w[rp + "conv2.weight"], 1, 0, 1)  # 1x1
            x = mt.add(residual, h)
        x = mt.conv1d(x, w[p + "conv.weight"], 1, 1, 1)        # k=3, pad=1
        x = mt.add(x, w[p + "conv.bias"])
    return x  # PT [T, 768]


RVQ_N = 8


def rvq_encode(w, x):
    """8 residual quantizers; nearest neighbour via -||z-e||^2 = 2 z·e - ||z||² - ||e||².
    x: PT [T, hidden] -> list of 8 ids tensors (I32, PT [T,1])."""
    ids_list = []
    residual = x
    for q in range(RVQ_N):
        p = f"codec.quant.{q}."
        cb = mt.cast(w[p + "codebook.embed"], mt.F32)              # PT [cb_size, cb_dim]
        z = mt.add(mt.linear(residual, w[p + "project_in.weight"]), w[p + "project_in.bias"])
        dot = mt.linear(z, cb)                                    # [T, cb_size]
        x2 = mt.sum_rows(mt.mul(z, z))                            # [T, 1]
        e2t = mt.transpose(mt.sum_rows(mt.mul(cb, cb)))           # [1, cb_size]
        score = mt.sub(mt.sub(mt.scale(dot, 2.0), x2), e2t)       # [T, cb_size]
        ids = mt.argmax(score)                                    # [T, 1] I32
        ids_list.append(ids)
        zq = mt.get_rows(cb, ids)                                 # [T, cb_dim]
        pq = mt.add(mt.linear(zq, w[p + "project_out.weight"]), w[p + "project_out.bias"])
        residual = mt.sub(residual, pq)
    return ids_list


def prefill_fused(w, sem_input, ac_in):
    """Everything up to (and including) the fusion FC -> PT [T, 1024]."""
    fp = feature_projection(w, feature_extractor(w, sem_input))
    se = semantic_encoder(w, wavlm_encoder(w, fp))                # PT [T, 768]
    ae = acoustic_encoder(w, ac_in)                               # PT [T, 256]
    combined = mt.concat(ae, se, 1)                               # PT [T, 1024]
    return mt.add(mt.linear(combined, w["codec.fc.weight"]), w["codec.fc.bias"])


def prefill(w, sem_input, ac_in):
    """Full encode_ref -> 8 id tensors + fused."""
    fused = prefill_fused(w, sem_input, ac_in)
    return rvq_encode(w, fused), fused


def wavlm_encoder(w, x):
    T = x.shape[0]
    # encoder pre-step: pos_conv_embed + add + layer_norm (affine), then 12 layers
    x = mt.add(x, pce(w, x, T))
    x = layernorm_affine(x, w["codec.sem.encoder.ln.weight"], w["codec.sem.encoder.ln.bias"])
    outs = []
    cur = x
    for li in range(WAVLM_LAYERS):
        cur = wavlm_layer(w, li, cur)
        outs.append(cur)
    acc = outs[0]
    for o in outs[1:]:
        acc = mt.add(acc, o)
    return mt.scale(acc, 1.0 / WAVLM_LAYERS)  # PT [T, 768]
