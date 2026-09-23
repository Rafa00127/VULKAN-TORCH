using VulkanTorch;

namespace HiggsTts;

/// <summary>Port of higgstts_vt/decode.py — RVQ codes [T,8] -> 24 kHz PCM.</summary>
public static class DacDecoder
{
    private static readonly (int S, int Kt)[] Blocks =
        { (8, 16), (5, 10), (4, 8), (2, 4), (3, 6) };

    private static readonly int[] ResDil = { 1, 3, 9 };

    public static float[] Decode(Runtime rt, Device dev, HiggsWeights w, int[] codes, int t)
    {
        using var g = new Graph(rt, dev);
        g.Enter();

        Tensor? dec = null;
        for (int c = 0; c < 8; c++)
        {
            var ids = new int[t];
            for (int i = 0; i < t; i++) ids[i] = codes[i * 8 + c];
            var q = Ops.GetRows(w[$"codec.quant.{c}.codebook.embed"],
                                g.InputI32(new long[] { t }, ids));
            q = Ops.Add(Ops.Linear(q, w[$"codec.quant.{c}.project_out.weight"]),
                        w[$"codec.quant.{c}.project_out.bias"]);
            dec = dec is null ? q : Ops.Add(dec, q);
        }

        var x = Ops.Add(Ops.Linear(dec!, w["codec.fc2.weight"]), w["codec.fc2.bias"]);
        x = Ops.Add(Ops.Conv1d(x, w["codec.ac_dec.conv1.weight"], 1, 3, 1),
                    w["codec.ac_dec.conv1.bias"]);

        for (int bi = 0; bi < Blocks.Length; bi++)
        {
            var (s, kt) = Blocks[bi];
            string p = $"codec.ac_dec.block.{bi}.";
            x = Ops.Snake1d(x, w[p + "snake1.alpha"]);
            long tin = x.Shape[0];
            var wp = w[p + "conv_t1.weight"];
            long oc = wp.Shape[0] / kt;
            var ct = Ops.ConvTranspose1d(x, wp, s, (int)oc);
            long tu = (tin - 1) * s + kt;
            long tout = tu - s;
            ct = Ops.View2d(ct, tout, oc, (ulong)(tu * 4), (ulong)(((s + 1) / 2) * 4));
            ct = Ops.Add(ct, Ops.Reshape(w[p + "conv_t1.bias"], new long[] { oc, 1 }));
            x = Ops.Transpose(ct);
            for (int r = 0; r < 3; r++)
                x = ResUnit(w, x, p + $"res_unit{r + 1}.", ResDil[r]);
        }

        // output: snake -> conv2 (k7, Cout=1) -> waveform. No final tanh: the Higgs
        // tokenizer deliberately drops DAC's ("DAC in HiggsAudioV2Tokenizer ... removes the
        // final nn.Tanh activation function"), so these weights emit the waveform directly.
        // The [1,32,7] weight loses its leading 1 under ggml, so do the single-output conv
        // as im2col + mul_mat.
        x = Ops.Snake1d(x, w["codec.ac_dec.snake1.alpha"]);
        var im = Ops.Im2colRafa(x, 7, 1, 3, 1);
        var w2d = Ops.Reshape(w["codec.ac_dec.conv2.weight"], new long[] { 1, 32 * 7 });
        x = Ops.Add(Ops.MulMat(im, w2d), w["codec.ac_dec.conv2.bias"]);

        var outT = Ops.Contiguous(x).MarkOutput();
        var pcm = outT.ToFloats(g);
        g.Exit();
        return pcm;
    }

    private static Tensor ResUnit(HiggsWeights w, Tensor x, string b, int dil)
    {
        var r = Ops.Snake1d(x, w[b + "snake1.alpha"]);
        long k = w[b + "conv1.weight"].Shape[2];
        r = Ops.Add(Ops.Conv1d(r, w[b + "conv1.weight"], 1, (int)((k - 1) / 2 * dil), dil),
                    w[b + "conv1.bias"]);
        r = Ops.Snake1d(r, w[b + "snake2.alpha"]);
        r = Ops.Add(Ops.Conv1d(r, w[b + "conv2.weight"], 1, 0, 1), w[b + "conv2.bias"]);
        return Ops.Add(x, r);
    }
}
