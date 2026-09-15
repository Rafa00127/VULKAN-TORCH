using System;
using System.Collections.Generic;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 BigVGAN generator (mel -> waveform). 1536-channel pre-conv, six transposed-conv
/// upsampling stages (256x total, 22050 Hz) each followed by three anti-aliased multi-periodicity
/// (AMP) residual blocks, then a post-activation, a 1-channel conv and a clamp.
/// Ported from indextts/s2mel/modules/bigvgan/{bigvgan,activations}.py + alias_free_activation.
///
/// The SnakeBeta alpha/beta parameters are stored in log space (snake_logscale=True). The
/// alias-free upsample/downsample filters are fixed kaiser-sinc kernels kept in the checkpoint;
/// upstream upsamples use the time-reversed kernel (a transposed conv == a correlation with the
/// flipped kernel). Conv weights are weight-normalised and fused as g*v/||v|| on load.
/// </summary>
public sealed class BigVgan : IDisposable
{
    private const int Mel = 80, Init = 1536, ResN = 3, UpsN = 6, ActK = 12;
    private static readonly int[] Rates = { 4, 4, 2, 2, 2, 2 };
    private static readonly int[] UpKs = { 8, 8, 4, 4, 4, 4 };
    private static readonly int[] ResKs = { 3, 7, 11 };
    private static readonly int[] ResDs = { 1, 3, 5 };
    private const string P = "bigvgan.generator.";

    private readonly GgufWeights _w;
    private readonly Memory _mem;
    private readonly Dictionary<string, Tensor> _t = new();
    // The up-sampling kernels in the layout `Ops.ConvTranspose1d` wants: PT [K*OC, IC].
    // Built in FuseInner (load time, on the host) with the same permutation the Higgs
    // converter bakes into its GGUF -- this GGUF keeps the raw layout, so it happens here
    // instead (device tensors cannot be created while a graph capture is active).
    // The plain [OC, IC, K] form is not kept for these: only the wperm one is used.
    private readonly Dictionary<string, Tensor> _wp = new();
    private readonly Dictionary<string, (int OC, int K)> _upDim = new();
    private readonly float[] _upFilt = new float[ActK];
    private Dictionary<string, Tensor>? _dbg;
    private readonly float[] _dnFilt = new float[ActK];

    public BigVgan(Device dev, GgufWeights w)
    {
        _w = w;
        _mem = new Memory(dev, 600UL << 20);
        Fuse("conv_pre", false, Init, Mel, 7);
        Fuse("conv_post", false, 1, Init >> UpsN, 7);
        for (int i = 0; i < UpsN; i++)
            Fuse($"ups.{i}.0", true, Init >> (i + 1), Init >> i, UpKs[i]);
        for (int n = 0; n < UpsN * ResN; n++)
        {
            int ch = Init >> (n / ResN + 1);
            int k = ResKs[n % ResN];
            for (int l = 0; l < ResN; l++)
            {
                Fuse($"resblocks.{n}.convs1.{l}", false, ch, ch, k);
                Fuse($"resblocks.{n}.convs2.{l}", false, ch, ch, k);
            }
        }
        ReadFilter(P + "resblocks.0.activations.0.upsample.filter", _upFilt, reverse: true);
        ReadFilter(P + "resblocks.0.activations.0.downsample.lowpass.filter", _dnFilt, reverse: false);
        // Precompute every SnakeBeta pair, channel filter and residual flag up front: creating
        // device tensors while a Graph capture is active is unsafe.
        for (int n = 0; n < UpsN * ResN; n++)
        {
            int ch = Init >> (n / ResN + 1);
            DwFilter(ch, _upFilt, true);
            DwFilter(ch, _dnFilt, false);
            for (int a = 0; a < 2 * ResN; a++) AlphaBeta($"resblocks.{n}.activations.{a}");
        }
        DwFilter(Init >> UpsN, _upFilt, true);
        DwFilter(Init >> UpsN, _dnFilt, false);
        AlphaBeta("activation_post");
    }

    /// <summary>Fuse a weight_norm conv into the plain [Cout, Cin, K] form used here.
    /// The stored transposed-conv layout is [Cin, Cout, K]. Shapes are passed explicitly because
    /// Tensor.Shape collapses size-1 dims (e.g. a 1-channel conv_post reports rank 2).</summary>
    private void Fuse(string name, bool transposed, int outC, int inC, int k)
    {
        try { FuseInner(name, transposed, outC, inC, k); }
        catch (Exception e)
        {
            Console.Error.WriteLine($"FUSE FAIL {name} t={transposed} {outC}x{inC}x{k}: " +
                $"g={_w[P + name + ".weight_g"].Shape.Length} nbytes={_w[P + name + ".weight_g"].NBytes}/" +
                $"{_w[P + name + ".weight_v"].NBytes} :: {e.Message}");
            throw;
        }
    }

    private void FuseInner(string name, bool transposed, int outC, int inC, int k)
    {
        long c0 = transposed ? inC : outC;          // v's dim 0
        long c1 = transposed ? outC : inC;          // v's dim 1
        var gH = _w[P + name + ".weight_g"].ReadBytes();
        var vH = _w[P + name + ".weight_v"].ReadBytes();
        var W = new float[outC * inC * k];
        // PT [K*OC, IC] for Ops.ConvTranspose1d; only the transposed convs need it.
        var WP = transposed ? new float[k * outC * inC] : null;
        for (long a = 0; a < c0; a++)
        {
            double ss = 0;
            for (long b = 0; b < c1; b++)
                for (long kk = 0; kk < k; kk++)
                {
                    float x = (float)BitConverter.ToHalf(vH, (int)(((a * c1 + b) * k + kk) * 2));
                    ss += (double)x * x;
                }
            float inv = (float)((float)BitConverter.ToHalf(gH, (int)(a * 2)) / Math.Sqrt(ss));
            for (long b = 0; b < c1; b++)
                for (long kk = 0; kk < k; kk++)
                {
                    float val = (float)BitConverter.ToHalf(vH, (int)(((a * c1 + b) * k + kk) * 2)) * inv;
                    long o = transposed ? b : a, i = transposed ? a : b;
                    // A transposed conv replayed as a correlation needs the kernel time-reversed.
                    W[(o * inC + i) * k + (transposed ? k - 1 - kk : kk)] = val;
                    // The wperm form wants the kernel in its natural order: the op is a
                    // real transposed conv (mul_mat + col2im), not a correlation. Row order
                    // is `o*K + kk`, matching the converter's [IC,OC,K] -> reshape(IC,K*OC).T.
                    if (WP != null) WP[(o * k + kk) * inC + i] = val;
                }
        }
        // Store F16: the source weights are F16 and the convs run on the f16 path anyway, so
        // this halves the fused footprint (448 MB -> 224 MB) with no numerical difference.
        // A transposed conv gets only the wperm tensor (Up reads it plus _upDim); keeping
        // the other form too would cost another ~24 MB and nothing would read it.
        if (transposed)
        {
            _upDim[name] = (outC, k);
            _wp[name] = _mem.Tensor(new long[] { k * outC, inC }, Ops.F16, ToF16(WP!));
        }
        else
        {
            _t[name] = _mem.Tensor(new long[] { outC, inC, k }, Ops.F16, ToF16(W));
        }
    }

    private static byte[] ToF16(float[] src)
    {
        var bytes = new byte[src.Length * 2];
        for (int i = 0; i < src.Length; i++)
        {
            var h = BitConverter.GetBytes((Half)src[i]);
            bytes[2 * i] = h[0];
            bytes[2 * i + 1] = h[1];
        }
        return bytes;
    }

    private void ReadFilter(string name, float[] dst, bool reverse)
    {
        var f = _w[name];
        var raw = f.ReadBytes();
        int k = (int)f.Numel;
        for (int i = 0; i < k; i++)
            dst[i] = (float)BitConverter.ToHalf(raw, (reverse ? (k - 1 - i) : i) * 2);
    }

    /// <summary>mel: PT [T, 80] -> waveform PT [T*256, 1]. <paramref name="trace"/>, if given,
    /// collects [conv_pre out, first resblock out, activation_post out].</summary>
    public Tensor Forward(Graph g, Tensor x, List<Tensor>? trace = null, Dictionary<string, Tensor>? dbg = null)
    {
        _dbg = dbg;
        x = Ops.Add(Ops.Conv1d(x, _t["conv_pre"], 1, 3, 1), _w[P + "conv_pre.bias"]);
        trace?.Add(x.MarkOutput());
        for (int i = 0; i < UpsN; i++)
        {
            x = Up(g, x, $"ups.{i}.0", Rates[i]);
            if (i == 0) trace?.Add(x.MarkOutput());
            Tensor? xs = null;
            for (int j = 0; j < ResN; j++)
            {
                var r = ResBlock(g, x, $"resblocks.{i * ResN + j}", ResKs[j]);
                if (i == 0 && j == 0) trace?.Add(r.MarkOutput());
                xs = xs == null ? r : Ops.Add(xs, r);
            }
            x = Ops.Scale(xs!, 1f / ResN);
            if (_dbg != null) _dbg[$"stage{i}"] = x;
        }
        x = Activation(g, x, "activation_post");
        trace?.Add(x.MarkOutput());
        x = Ops.Conv1d(x, _t["conv_post"], 1, 3, 1);      // use_bias_at_final = False
        return Ops.Clamp(x, -1f, 1f);
    }

    /// <summary>Weight-normalised ConvTranspose1d: the real thing (mul_mat + col2im via the
    /// pre-permuted kernel), then trimmed to the reference output length. Zero-stuffing +
    /// a stock conv costs `rate`x the FLOPs and materialises a `rate`x longer im2col.</summary>
    public Tensor Up(Graph g, Tensor x, string name, int rate)
    {
        int t = (int)x.Shape[0];
        var (oc, k) = _upDim[name];
        int pad = (k - rate) / 2;
        int raw = (t - 1) * rate + k;                        // ConvTranspose1d_p0 length
        // Same shape dance as Higgs' DacDecoder. The op returns time-contiguous ne=[raw, OC],
        // i.e. PT [OC, raw], so: crop that axis (offset `pad`, `raw-2*pad` rows == torch's
        // ConvTranspose1d(padding=pad)), then transpose+collapse to the PT [T, OC] the rest
        // of the pipeline wants (every conv here is channel-contiguous).
        var ct = Ops.ConvTranspose1d(x, _wp[name], rate, oc);
        var view = Ops.View2d(ct, raw - 2 * pad, oc, (ulong)(raw * 4), (ulong)(pad * 4));
        var y = Ops.Contiguous(Ops.Transpose(view));
        return Ops.Add(y, _w[P + name + ".bias"]);
    }

    /// <summary>AMPBlock1: one fixed kernel size, dilations (1,3,5); each layer runs
    /// activation -> conv1(dilated) -> activation -> conv2 -> residual.</summary>
    private Tensor ResBlock(Graph g, Tensor x, string name, int k)
    {
        for (int l = 0; l < ResDs.Length; l++)
        {
            int dil = ResDs[l];
            var xt = Activation(g, x, $"{name}.activations.{2 * l}");
            xt = Ops.Add(Ops.Conv1d(xt, _t[$"{name}.convs1.{l}"], 1, (k * dil - dil) / 2, dil),
                         _w[P + $"{name}.convs1.{l}.bias"]);
            if (_dbg != null) _dbg[$"{name}.c1.{l}"] = xt;
            xt = Activation(g, xt, $"{name}.activations.{2 * l + 1}");
            xt = Ops.Add(Ops.Conv1d(xt, _t[$"{name}.convs2.{l}"], 1, (k - 1) / 2, 1),
                         _w[P + $"{name}.convs2.{l}.bias"]);
            if (_dbg != null) _dbg[$"{name}.c2.{l}"] = xt;
            x = Ops.Add(xt, x);
            if (_dbg != null) _dbg[$"{name}.x.{l}"] = x;
        }
        return x;
    }

    /// <summary>Activation1d: x2 kaiser upsample -> SnakeBeta -> x2 kaiser downsample.
    /// Input/output PT [T, C].</summary>
    private Tensor Activation(Graph g, Tensor x, string name)
    {
        int t = (int)x.Shape[0], c = (int)x.Shape[1];
        // upsample
        var xp = Replicate(g, x, 5, 5);
        var z = ZeroStuff(g, xp, 2);
        var up = Ops.Conv1dDw(z, DwFilter(c, _upFilt, true), 1, ActK - 1, 1);
        up = Ops.Scale(RowSlice(g, up, 15, 2 * t), 2f);
        if (_dbg != null) _dbg[name + ".up"] = up;
        // snake beta
        var act = SnakeBeta(g, up, name);
        if (_dbg != null) _dbg[name + ".act"] = act;
        // downsample
        var dp = Replicate(g, act, 5, 6);
        var down = Ops.Conv1dDw(dp, DwFilter(c, _dnFilt, false), 2, 0, 1);
        if (_dbg != null) _dbg[name + ".down"] = down;
        return down;
    }

    private Tensor SnakeBeta(Graph g, Tensor x, string name)
    {
        var (alpha, invbeta) = AlphaBeta(name);
        var s = Ops.Sin(Ops.Mul(x, alpha));
        return Ops.Add(x, Ops.Mul(Ops.Sqr(s), invbeta));
    }

    /// <summary>alpha = exp(raw), inv_beta = 1/(exp(raw)+1e-9), as [C] device tensors.</summary>
    private (Tensor, Tensor) AlphaBeta(string name)
    {
        string key = "ab:" + name;
        if (_t.TryGetValue(key, out var cached))
            return (cached, _t["ib:" + name]);
        var a = _w[P + name + ".act.alpha"];
        var b = _w[P + name + ".act.beta"];
        var aR = a.ReadFloats();
        var bR = b.ReadFloats();
        var alpha = new float[aR.Length];
        var inv = new float[aR.Length];
        for (int i = 0; i < alpha.Length; i++)
        {
            alpha[i] = MathF.Exp(aR[i]);
            inv[i] = 1f / (MathF.Exp(bR[i]) + 1e-9f);
        }
        _t[key] = Upload(aR.Length, alpha);
        _t["ib:" + name] = Upload(aR.Length, inv);
        return (_t[key], _t["ib:" + name]);
    }

    private Tensor Upload(int c, float[] v)
    {
        var bytes = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
        return _mem.Tensor(new long[] { c }, Ops.F32, bytes);
    }

    /// <summary>The shared kaiser kernel expanded to the depthwise weight form PT [C, 1, K].</summary>
    private Tensor DwFilter(int c, float[] filt, bool up)
    {
        string key = $"filt:{(up ? "up" : "dn")}:{c}";
        if (_t.TryGetValue(key, out var cached)) return cached;
        var v = new float[c * ActK];
        for (int i = 0; i < c; i++) Array.Copy(filt, 0, v, i * ActK, ActK);
        var bytes = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
        return _t[key] = _mem.Tensor(new long[] { c, 1, ActK }, Ops.F32, bytes);
    }

    /// <summary>Replicate (edge) padding along time: [T, C] -> [T+left+right, C].</summary>
    private static Tensor Replicate(Graph g, Tensor x, int left, int right)
    {
        int t = (int)x.Shape[0];
        var idx = new int[t + left + right];
        for (int i = 0; i < left; i++) idx[i] = 0;
        for (int i = 0; i < t; i++) idx[left + i] = i;
        for (int i = 0; i < right; i++) idx[left + t + i] = t - 1;
        return Ops.GetRows(x, g.InputI32(new long[] { idx.Length }, idx));
    }

    /// <summary>Interleave u-1 zeros after every sample: [T, C] -> [T*u, C].</summary>
    private static Tensor ZeroStuff(Graph g, Tensor x, int u)
    {
        if (u == 1) return x;
        int t = (int)x.Shape[0], c = (int)x.Shape[1];
        var zrow = g.Input(new long[] { t, 1, c }, new float[t * c]);
        var z = Ops.Reshape(x, new long[] { t, 1, c });
        for (int i = 1; i < u; i++) z = Ops.Concat(z, zrow, 1);
        return Ops.Reshape(z, new long[] { t * u, c });
    }

    /// <summary>Rows [from, from+count) of a PT [N, C] tensor, as an owned buffer. A ggml view
    /// aliases its parent's storage and marking the view as an output does NOT protect that
    /// parent, so later nodes can reuse it — scale by 1 to force a real copy.</summary>
    private static Tensor RowSlice(Graph g, Tensor x, int from, int count)
    {
        long c = x.Shape[1];
        var view = Ops.View2d(x, c, count, (ulong)(c * 4), (ulong)((long)from * c * 4));
        return Ops.Scale(view, 1f);
    }

    /// <summary>A fused weight, by base name (for tests).</summary>
    public Tensor Fused(string name) => _t[name];

    public void Dispose() => _mem.Dispose();
}
