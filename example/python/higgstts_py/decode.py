"""DAC decoder (codec.decode) in vulkantorch — codes [T,8] -> 24kHz PCM."""
import numpy as np

import vulkantorch as mt

AC_DEC_BLOCKS = [(8, 16), (5, 10), (4, 8), (2, 4), (3, 6)]


def _res_unit(g, w, x, base, dil):
    r = mt.snake_1d(x, w[base + "snake1.alpha"])
    K = w[base + "conv1.weight"].shape[2]
    r = mt.add(mt.conv1d(r, w[base + "conv1.weight"], 1, (K - 1) // 2 * dil, dil), w[base + "conv1.bias"])
    r = mt.snake_1d(r, w[base + "snake2.alpha"])
    r = mt.add(mt.conv1d(r, w[base + "conv2.weight"], 1, 0, 1), w[base + "conv2.bias"])
    return mt.add(x, r)


def dac_decode(rt, w, codes):
    """codes: [T_raw, 8] int -> float32 PCM samples."""
    gpu = rt.gpu()
    T = codes.shape[0]

    g = mt.Graph(rt, gpu)
    g.enter()
    dec = None
    for c in range(8):
        ids = np.ascontiguousarray(codes[:, c], dtype=np.int32)
        q = mt.get_rows(w[f"codec.quant.{c}.codebook.embed"], g.input_i32([T], ids.tobytes()))
        q = mt.add(mt.linear(q, w[f"codec.quant.{c}.project_out.weight"]),
                   w[f"codec.quant.{c}.project_out.bias"])
        dec = q if dec is None else mt.add(dec, q)

    x = mt.add(mt.linear(dec, w["codec.fc2.weight"]), w["codec.fc2.bias"])   # PT [T,256]
    x = mt.add(mt.conv1d(x, w["codec.ac_dec.conv1.weight"], 1, 3, 1),
               w["codec.ac_dec.conv1.bias"])                                  # PT [T,1024]

    for bi, (s, Kt) in enumerate(AC_DEC_BLOCKS):
        p = f"codec.ac_dec.block.{bi}."
        x = mt.snake_1d(x, w[p + "snake1.alpha"])
        Tin = x.shape[0]
        wp = w[p + "conv_t1.weight"]
        OC = wp.shape[0] // Kt
        ct = mt.conv_transpose_1d(x, wp, s, OC)                # PT [OC, Tu] (ne=[Tu,OC])
        Tu = (Tin - 1) * s + Kt
        Tout = Tu - s
        ct = mt.view_2d(ct, Tout, OC, Tu * 4, ((s + 1) // 2) * 4)   # crop -> ne=[Tout,OC]
        ct = mt.add(ct, mt.reshape(w[p + "conv_t1.bias"], [OC, 1]))
        x = mt.transpose(ct)                                   # PT [Tout, OC]
        for r in (1, 2, 3):
            x = _res_unit(g, w, x, p + f"res_unit{r}.", {1: 1, 2: 3, 3: 9}[r])

    # output: snake -> conv2(k7, Cout=1) -> waveform. No final tanh: the Higgs tokenizer
    # deliberately drops DAC's ("DAC in HiggsAudioV2Tokenizer ... removes the final nn.Tanh
    # activation function"), so these weights emit the waveform directly. Keeping upstream
    # DAC's tanh made our PCM tanh(the official) — 3e-2 off, see tools/check_higgs_ref.py.
    # (a PT [1,32,7] weight collapses its leading 1 under ggml, so do the
    #  single-output conv directly as im2col + mul_mat.)
    x = mt.snake_1d(x, w["codec.ac_dec.snake1.alpha"])
    im = mt.im2col_rafa(x, 7, 1, 3, 1)                                  # ne=[224, T]
    w2d = mt.reshape(w["codec.ac_dec.conv2.weight"], [1, 32 * 7])       # PT [1,224] ne=[224,1]
    x = mt.add(mt.mul_mat(im, w2d), w["codec.ac_dec.conv2.bias"])       # ne=[T,1]
    c = mt.contiguous(x)
    c.mark_output()
    g.exit()
    return np.frombuffer(c.to_bytes(), dtype=np.float32)
