using System;
using System.Collections.Generic;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 CAMPPlus speaker-style encoder: FCM (2D res-block front-end) + a TDNN x-vector
/// stack with CAM dense blocks, stats pooling and a 192-d dense head.
/// Ported from indextts/s2mel/modules/campplus/{DTDNN,layers}.py.
///
/// All BatchNorms are in eval mode, so each is folded on the host into a per-channel
/// scale/shift: y = x * (w / sqrt(var + eps)) + (b - mean * scale).
/// Activations are PT [T, C]; the 2D front-end uses PT [1, C, F, T].
/// </summary>
public sealed class CampPlus : IDisposable
{
    private const int Feat = 80, Embed = 192, MC = 32, InitCh = 128, Growth = 32, BnSize = 4;
    private const int SegLen = 100;
    private const string P = "campplus.";
    private static readonly (int Layers, int Dil)[] Blocks = { (12, 1), (24, 2), (16, 2) };

    private readonly GgufWeights _w;
    private readonly Memory _mem;
    private readonly Dictionary<string, (Tensor Scale, Tensor Shift)> _bn = new();

    private readonly Tensor _denseWMean, _denseWStd;

    public CampPlus(Device dev, GgufWeights w)
    {
        _w = w;
        _mem = new Memory(dev, 64UL << 20);   // folded BNs + the split dense head (192x512 F32 x2)
        // The dense head consumes pool = [mean ; std]. Split its weight into the two halves and
        // apply them separately: ggml reports a PT [1, c] tensor as rank 1, so it cannot be
        // concatenated along an axis. Reads the F16 GGUF weight, so this is a host-side split.
        var dw = _w[P + "xvector.dense.linear.weight"];          // PT [192, 1024, 1]
        long oc = dw.Shape[0], ic = dw.Shape[1];
        var raw = dw.ReadBytes();
        var wm = new float[oc * (ic / 2)];
        var ws = new float[oc * (ic / 2)];
        for (long o = 0; o < oc; o++)
            for (long i = 0; i < ic; i++)
            {
                float v = (float)BitConverter.ToHalf(raw, (int)((o * ic + i) * 2));
                if (i < ic / 2) wm[o * (ic / 2) + i] = v; else ws[o * (ic / 2) + (i - ic / 2)] = v;
            }
        _denseWMean = Upload(new long[] { oc, ic / 2 }, wm);
        _denseWStd = Upload(new long[] { oc, ic / 2 }, ws);

        // Precompute every folded BatchNorm up front: allocating device tensors while a Graph
        // capture is active is unsafe (and the folds are needed on the first forward anyway).
        // Channel counts come from the GGUF itself; the FCM head's BNs act on PT [1,C,F,T].
        const string suffix = ".running_mean";
        foreach (var name in w.File.Names())
        {
            if (!name.StartsWith(P, StringComparison.Ordinal) || !name.EndsWith(suffix)) continue;
            string bn = name.Substring(P.Length, name.Length - P.Length - suffix.Length);
            Bn(bn, (int)w[name].Numel, bn.StartsWith("head.", StringComparison.Ordinal));
        }
    }

    /// <summary>BatchNorm folded to scale/shift. <paramref name="rank4"/> shapes them for a
    /// PT [1, C, F, T] tensor (channel on dim 1); otherwise for PT [T, C].</summary>
    private (Tensor, Tensor) Bn(string name, int c, bool rank4)
    {
        string key = name + (rank4 ? ":4" : ":2");
        if (_bn.TryGetValue(key, out var hit)) return hit;
        // BatchNorm1d(affine=False) has no weight/bias — it is a pure normalizer.
        bool affine = _w.Has(P + name + ".weight");
        var wt = affine ? _w[P + name + ".weight"].ReadFloats() : Fill(c, 1f);
        var bs = affine ? _w[P + name + ".bias"].ReadFloats() : new float[c];
        var mu = _w[P + name + ".running_mean"].ReadFloats();
        var va = _w[P + name + ".running_var"].ReadFloats();
        var scale = new float[c];
        var shift = new float[c];
        for (int i = 0; i < c; i++)
        {
            scale[i] = (float)(wt[i] / Math.Sqrt(va[i] + 1e-5));
            shift[i] = bs[i] - mu[i] * scale[i];
        }
        long[] sh = rank4 ? new long[] { 1, c, 1, 1 } : new long[] { 1, c };
        var res = (Upload(sh, scale), Upload(sh, shift));
        _bn[key] = res;
        return res;
    }

    private static float[] Fill(int n, float v)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = v;
        return a;
    }

    private Tensor Upload(long[] shape, float[] v)
    {
        var b = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, b, 0, b.Length);
        return _mem.Tensor(shape, Ops.F32, b);
    }

    private Tensor BnApply(Tensor x, string name, int c, bool rank4)
    {
        var (s, sh) = Bn(name, c, rank4);
        return Ops.Add(Ops.Mul(x, s), sh);
    }

    /// <summary>x: PT [T, 80] -> PT [1, 192].</summary>
    private Dictionary<string, Tensor>? _dbg;

    public Tensor Forward(Graph g, Tensor x, Dictionary<string, Tensor>? dbg = null)
    {
        _dbg = dbg;
        int t = (int)x.Shape[0];
        // FCM: (T, 80) -> (1, 1, 80, T)
        var h = Ops.Reshape(Ops.Contiguous(Ops.Transpose(x)), new long[] { 1, 1, Feat, t });
        h = Ops.Relu(BnApply(Ops.Conv2d(_w[P + "head.conv1.weight"], h, 1, 1, 1, 1, 1, 1), "head.bn1", MC, true));
        h = ResBlock2d(g, h, "head.layer1.0", 2);
        h = ResBlock2d(g, h, "head.layer1.1", 1);
        h = ResBlock2d(g, h, "head.layer2.0", 2);
        h = ResBlock2d(g, h, "head.layer2.1", 1);
        h = Ops.Relu(BnApply(Ops.Conv2d(_w[P + "head.conv2.weight"], h, 1, 2, 1, 1, 1, 1), "head.bn2", MC, true));

        // (1, 32, 10, T) -> (T, 320)
        // Materialize: the im2col behind Conv1d needs dense input, and a transposed view is not.
        var y = Ops.Contiguous(Ops.Transpose(Ops.Reshape(h, new long[] { MC * (Feat / 8), t })));
        if (_dbg != null) _dbg["head"] = y;

        y = Tdnn(g, y, "xvector.tdnn", InitCh, 5, 2, 2);
        t = (int)y.Shape[0];
        if (_dbg != null) _dbg["tdnn"] = y;
        int ch = InitCh;
        for (int b = 0; b < Blocks.Length; b++)
        {
            var (layers, dil) = Blocks[b];
            y = CamDenseBlock(g, y, $"xvector.block{b + 1}", layers, ch, dil, t);
            if (_dbg != null) _dbg[$"blk{b + 1}"] = y;
            ch += layers * Growth;
            y = Transit(g, y, $"xvector.transit{b + 1}", ch, ch / 2);
            ch /= 2;
            if (_dbg != null) _dbg[$"tr{b + 1}"] = y;
        }
        y = Ops.Relu(BnApply(y, "xvector.out_nonlinear.batchnorm", ch, false));      // batchnorm-relu
        if (_dbg != null) _dbg["outnl"] = y;
        var (pmean, pstd) = StatsPool(g, y, t, ch);
        var dense = Ops.Add(Ops.Linear(pmean, _denseWMean), Ops.Linear(pstd, _denseWStd));
        dense = BnApply(dense, "xvector.dense.nonlinear.batchnorm", Embed, false);  // affine-free BN
        if (_dbg != null) _dbg["dense"] = dense;
        return dense;
    }

    // ---- x-vector pieces -------------------------------------------------------------------

    /// <summary>TDNNLayer: Conv1d + BN + ReLU (+ optional stride).</summary>
    private Tensor Tdnn(Graph g, Tensor x, string name, int outC, int k, int stride, int pad)
    {
        var y = Ops.Conv1d(x, _w[P + name + ".linear.weight"], stride, pad, 1);
        y = BnApply(y, name + ".nonlinear.batchnorm", outC, false);
        return Ops.Relu(y);
    }

    private Tensor Transit(Graph g, Tensor x, string name, int inC, int outC)
    {
        var y = Ops.Relu(BnApply(x, name + ".nonlinear.batchnorm", inC, false));
        return Ops.Conv1d(y, _w[P + name + ".linear.weight"], 1, 0, 1);
    }

    /// <summary>CAMDenseTDNNBlock: each layer's output is concatenated onto the running state.</summary>
    private Tensor CamDenseBlock(Graph g, Tensor x, string name, int layers, int inCh, int dil, int t)
    {
        int ch = inCh;
        for (int i = 0; i < layers; i++)
        {
            string p = $"{name}.tdnnd{i + 1}.";
            var h = Ops.Relu(BnApply(x, p + "nonlinear1.batchnorm", ch, false));
            h = Ops.Conv1d(h, _w[P + p + "linear1.weight"], 1, 0, 1);         // -> bn_channels
            h = Ops.Relu(BnApply(h, p + "nonlinear2.batchnorm", BnSize * Growth, false));
            h = Cam(g, h, p + "cam_layer", dil, t);
            x = Ops.Concat(x, h, 1);
            ch += Growth;
        }
        return x;
    }

    /// <summary>CAMLayer: y = local(x); m = sigmoid(linear2(relu(linear1(mean + seg_pool(x)))));
    /// return y * m.</summary>
    private Tensor Cam(Graph g, Tensor x, string name, int dil, int t)
    {
        int bc = BnSize * Growth;
        int pad = (3 - 1) / 2 * dil;
        var local = Ops.Conv1d(x, _w[P + name + ".linear_local.weight"], 1, pad, dil);

        var mean = Ops.Scale(Ops.Matmul(Ones(g, t), x), 1f / t);               // [1, bc]
        var seg = SegPool(g, x, t, bc);                                        // [T, bc]
        var ctx = Ops.Add(seg, Ops.Repeat(Ops.Reshape(mean, new long[] { 1, bc }),
                                          new long[] { t, bc }));
        // CAMLayer's linear1/linear2 are nn.Conv1d without bias=False, so they carry one
        // (linear_local is constructed with bias=False and does not).
        ctx = Ops.Add(Ops.Conv1d(ctx, _w[P + name + ".linear1.weight"], 1, 0, 1),
                      _w[P + name + ".linear1.bias"]);
        ctx = Ops.Relu(ctx);
        var m = Ops.Sigmoid(Ops.Add(Ops.Conv1d(ctx, _w[P + name + ".linear2.weight"], 1, 0, 1),
                                    _w[P + name + ".linear2.bias"]));
        return Ops.Mul(local, m);
    }

    /// <summary>avg_pool1d(kernel=100, stride=100, ceil_mode=True) then repeat each pooled value
    /// back over its window and truncate to T. Done as two small matmuls with a segment matrix.</summary>
    private Tensor SegPool(Graph g, Tensor x, int t, int c)
    {
        int nSeg = (t + SegLen - 1) / SegLen;
        var fwd = new float[nSeg * t];
        for (int s = 0; s < nSeg; s++)
        {
            int lo = s * SegLen, hi = Math.Min(lo + SegLen, t);
            float w = 1f / (hi - lo);
            for (int i = lo; i < hi; i++) fwd[s * t + i] = w;
        }
        var mF = g.Input(new long[] { nSeg, t }, fwd);
        var segs = Ops.Matmul(mF, x);                                          // [nSeg, c]
        var back = new float[t * nSeg];
        for (int s = 0; s < nSeg; s++)
            for (int i = s * SegLen; i < Math.Min((s + 1) * SegLen, t); i++)
                back[i * nSeg + s] = 1f;
        var mB = g.Input(new long[] { t, nSeg }, back);
        return Ops.Matmul(mB, segs);                                           // [T, c]
    }

    private Tensor Ones(Graph g, int t)
    {
        var v = new float[t];
        for (int i = 0; i < t; i++) v[i] = 1f;
        return g.Input(new long[] { 1, t }, v);
    }

    /// <summary>StatsPool: unbiased mean and std over time -> two PT [1, C] tensors.</summary>
    private (Tensor Mean, Tensor Std) StatsPool(Graph g, Tensor x, int t, int c)
    {
        var ones = Ones(g, t);
        var mean = Ops.Scale(Ops.Matmul(ones, x), 1f / t);                      // [1, c]
        var d = Ops.Sub(x, Ops.Repeat(mean, new long[] { t, c }));
        var varT = Ops.Scale(Ops.Matmul(ones, Ops.Mul(d, d)), 1f / (t - 1));    // unbiased
        return (mean, Ops.Sqrt(varT));
    }

    // ---- 2D front-end ----------------------------------------------------------------------

    private Tensor ResBlock2d(Graph g, Tensor x, string name, int stride)
    {
        string p = P + name + ".";
        var o = Ops.Relu(BnApply(Ops.Conv2d(_w[p + "conv1.weight"], x, 1, stride, 1, 1, 1, 1),
                                 name + ".bn1", MC, true));
        o = BnApply(Ops.Conv2d(_w[p + "conv2.weight"], o, 1, 1, 1, 1, 1, 1), name + ".bn2", MC, true);
        if (stride != 1)
            x = BnApply(Ops.Conv2d(_w[p + "shortcut.0.weight"], x, 1, stride, 0, 0, 1, 1),
                        name + ".shortcut.1", MC, true);
        return Ops.Relu(Ops.Add(o, x));
    }

    public void Dispose() => _mem.Dispose();
}
