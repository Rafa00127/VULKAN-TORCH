using System;
using System.Collections.Generic;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 s2mel diffusion transformer (the CFM estimator). 13 gpt_fast transformer blocks
/// with NeoX rotary embeddings, adaptive RMSNorm conditioning on the timestep, SwiGLU FFN, uvit
/// mid-skip connections, and a WaveNet final layer.
/// Ported from indextts/s2mel/modules/diffusion_transformer.py + gpt_fast/model.py + wavenet.py.
///
/// Operates on a single batch element; the CFG pass calls it twice (cond / uncond). All
/// activations are PT [T, C] (time-major).
/// </summary>
public sealed class Dit : IDisposable
{
    private const int Dim = 512, Heads = 8, HeadDim = 64, Depth = 13, Mel = 80;
    private const int Style = 192, CondDim = 512, Inter = 1536, Wn = 512, WnK = 5, WnLayers = 8;
    private const float NormEps = 1e-5f, FinalEps = 1e-6f, FreqBase = 10000f;
    private const string P = "s2mel.cfm.estimator.";

    // uvit mid-skip: layers < Depth/2 emit, layers > Depth/2 receive.
    private const int SkipMid = Depth / 2;

    private readonly GgufWeights _w;
    private readonly Memory _mem;
    private readonly Dictionary<string, Tensor> _fused = new();
    private readonly float[] _tfreq = new float[128];

    public Dit(Device dev, GgufWeights w)
    {
        _w = w;
        _mem = new Memory(dev, 256UL << 20);
        for (int i = 0; i < 128; i++)
            _tfreq[i] = MathF.Exp(-MathF.Log(10000f) * i / 128f);
        FuseTree("final_layer.linear");
        FuseTree("wavenet.cond_layer.conv.conv");
        for (int i = 0; i < WnLayers; i++)
        {
            FuseTree($"wavenet.in_layers.{i}.conv.conv");
            FuseTree($"wavenet.res_skip_layers.{i}.conv.conv");
        }
    }

    /// <summary>Fuse a weight_norm pair into a plain weight: W = g * v / ||v|| over every axis
    /// but 0. Cached under the base name (without the _g/_v suffix).</summary>
    private void FuseTree(string name)
    {
        var g = _w[P + name + ".weight_g"];
        var v = _w[P + name + ".weight_v"];
        var gH = g.ReadBytes();
        var vH = v.ReadBytes();
        var shape = v.Shape;                       // PT [Cout, ...]
        long cout = shape[0];
        long per = v.Numel / cout;
        var W = new float[v.Numel];
        for (long c = 0; c < cout; c++)
        {
            double ss = 0;
            for (long j = 0; j < per; j++)
            {
                float vv = (float)BitConverter.ToHalf(vH, (int)((c * per + j) * 2));
                ss += (double)vv * vv;
            }
            float inv = (float)((float)BitConverter.ToHalf(gH, (int)(c * 2)) / Math.Sqrt(ss));
            for (long j = 0; j < per; j++)
                W[c * per + j] = (float)BitConverter.ToHalf(vH, (int)((c * per + j) * 2)) * inv;
        }
        var bytes = new byte[W.Length * 4];
        Buffer.BlockCopy(W, 0, bytes, 0, bytes.Length);
        _fused[name] = _mem.Tensor(shape, Ops.F32, bytes);
    }

    private Tensor Wt(string name) => _fused[name];

    /// <summary>One estimator forward. x, promptX, cond: PT [T, *]; style: PT [192] or [1,192];
    /// t: scalar. Returns PT [T, 80]. <paramref name="trace"/>, if given, receives the running
    /// hidden state after each transformer block (for stage-wise validation).</summary>
    public Tensor Forward(Graph g, Tensor x, Tensor promptX, Tensor cond, Tensor style, float t,
                          Tensor pos, List<Tensor>? trace = null)
    {
        int T = (int)x.Shape[0];
        var t1 = TimeEmbed(g, t, "t_embedder");                        // [1, 512]
        var t2 = TimeEmbed(g, t, "t_embedder2");                       // [1, 512]

        var condP = Ops.Add(Ops.Linear(cond, _w[P + "cond_projection.weight"]),
                            _w[P + "cond_projection.bias"]);           // [T, 512]
        var styleT = Ops.Repeat(Ops.Reshape(style, new long[] { 1, Style }), new long[] { T, Style });
        var xIn = Ops.Concat(Ops.Concat(x, promptX, 1), condP, 1);     // [T, 672]
        xIn = Ops.Concat(xIn, styleT, 1);                              // [T, 864]
        xIn = Ops.Add(Ops.Linear(xIn, _w[P + "cond_x_merge_linear.weight"]),
                      _w[P + "cond_x_merge_linear.bias"]);             // [T, 512]

        var skipStack = new List<Tensor>();
        for (int i = 0; i < Depth; i++)
        {
            Tensor? skipIn = null;
            if (i > SkipMid) skipIn = skipStack[^1];
            xIn = Block(g, i, xIn, t1, skipIn, pos);
            if (i > SkipMid) skipStack.RemoveAt(skipStack.Count - 1);
            if (i < SkipMid) skipStack.Add(xIn);
            trace?.Add(xIn);
        }
        var xRes = AdaNorm(g, xIn, t1, "transformer.norm");            // [T, 512]

        // long skip: cat([x_res, x]) -> 592 -> 512
        xRes = Ops.Add(Ops.Linear(Ops.Concat(xRes, x, 1), _w[P + "skip_linear.weight"]),
                       _w[P + "skip_linear.bias"]);

        var h = Ops.Add(Ops.Linear(xRes, _w[P + "conv1.weight"]), _w[P + "conv1.bias"]);  // [T, 512]
        h = Ops.Add(Wavenet(g, h, t2), Ops.Add(Ops.Linear(xRes, _w[P + "res_projection.weight"]),
                                               _w[P + "res_projection.bias"]));
        h = FinalLayer(g, h, t1);                                      // [T, 512]
        return Ops.Add(Ops.Conv1d(h, _w[P + "conv2.weight"], 1, 0, 1), _w[P + "conv2.bias"]);  // [T, 80]
    }

    // ---- blocks -----------------------------------------------------------------------------

    private Tensor Block(Graph g, int i, Tensor x, Tensor c, Tensor? skipIn, Tensor pos)
    {
        string p = P + $"transformer.layers.{i}.";
        if (skipIn != null)
            x = Ops.Add(Ops.Linear(Ops.Concat(x, skipIn, 1), _w[p + "skip_in_linear.weight"]),
                        _w[p + "skip_in_linear.bias"]);
        var an = AdaNorm(g, x, c, $"transformer.layers.{i}.attention_norm");
        var at = Attention(g, i, an, pos);
        var h = Ops.Add(x, at);
        var fn = AdaNorm(g, h, c, $"transformer.layers.{i}.ffn_norm");
        return Ops.Add(h, FeedForward(g, i, fn));
    }

    private Tensor Attention(Graph g, int i, Tensor x, Tensor pos)
    {
        string p = P + $"transformer.layers.{i}.";
        var qkv = Ops.Linear(x, _w[p + "attention.wqkv.weight"]);       // [T, 1536]
        int t = (int)x.Shape[0];
        var q = Rope(Chan(qkv, t, 1536, 0, Dim), pos);
        var k = Rope(Chan(qkv, t, 1536, Dim, Dim), pos);
        var v = Attn.ToHeads(Chan(qkv, t, 1536, 2 * Dim, Dim), t, Heads, HeadDim);
        var o = Attn.Sdpa(q, k, v, 1f / MathF.Sqrt(HeadDim));           // [H, T, HD]
        return Ops.Linear(Attn.FromHeads(o, t, Heads, HeadDim), _w[p + "attention.wo.weight"]);
    }

    /// <summary>Rotary embedding over interleaved (x[2k], x[2k+1]) pairs. Takes the raw q/k
    /// slice PT [T, H*HD] and returns PT [H, T, HD]. In this ggml, rope mode 0 = interleaved
    /// pairs and mode 2 = split-half; the reference (reshape(-1, 2)) is the interleaved kind.</summary>
    private static Tensor Rope(Tensor qw, Tensor pos)
    {
        int t = (int)qw.Shape[0];
        var seq = Ops.Reshape(qw, new long[] { t, Heads, HeadDim });                 // [T, H, HD]
        seq = Ops.Rope(seq, pos, HeadDim, 0, 0, FreqBase);
        return Attn.ToHeads(Ops.Reshape(Ops.Contiguous(seq), new long[] { t, Heads * HeadDim }),
                            t, Heads, HeadDim);
    }

    private Tensor FeedForward(Graph g, int i, Tensor x)
    {
        string p = P + $"transformer.layers.{i}.";
        var a = Ops.Silu(Ops.Linear(x, _w[p + "feed_forward.w1.weight"]));
        var b = Ops.Linear(x, _w[p + "feed_forward.w3.weight"]);
        return Ops.Linear(Ops.Mul(a, b), _w[p + "feed_forward.w2.weight"]);
    }

    /// <summary>AdaptiveLayerNorm: weight * RMSNorm(x) + bias, where (weight, bias) = prog(t1).</summary>
    private Tensor AdaNorm(Graph g, Tensor x, Tensor c, string prefix)
    {
        var proj = Ops.Add(Ops.Linear(c, _w[P + prefix + ".project_layer.weight"]),
                           _w[P + prefix + ".project_layer.bias"]);      // [1, 1024]
        var w = Chan(proj, 1, 2 * Dim, 0, Dim);
        var b = Chan(proj, 1, 2 * Dim, Dim, Dim);
        var n = Ops.Mul(Ops.RmsNorm(x, NormEps), _w[P + prefix + ".norm.weight"]);
        return Ops.Add(Ops.Mul(n, w), b);
    }

    // ---- final layer / wavenet ---------------------------------------------------------------

    private Tensor FinalLayer(Graph g, Tensor x, Tensor t1)
    {
        var m = Ops.Linear(Ops.Silu(t1), _w[P + "final_layer.adaLN_modulation.1.weight"]);
        m = Ops.Add(m, _w[P + "final_layer.adaLN_modulation.1.bias"]);   // [1, 1024]
        var shift = Chan(m, 1, 2 * Dim, 0, Dim);
        var scale = Chan(m, 1, 2 * Dim, Dim, Dim);
        var n = Ops.LayerNorm(x, FinalEps);                             // affine-free
        n = Ops.Add(Ops.Mul(n, Ops.Add(scale, One(g, 1, Dim))), shift);  // x*(1+scale)+shift
        return Ops.Add(Ops.Linear(n, Wt("final_layer.linear")), _w[P + "final_layer.linear.bias"]);
    }

    private Tensor Wavenet(Graph g, Tensor x, Tensor t2)
    {
        int t = (int)x.Shape[0];
        var gconv = Ops.Add(Ops.Conv1d(t2, Wt("wavenet.cond_layer.conv.conv"), 1, 0, 1),
                            _w[P + "wavenet.cond_layer.conv.conv.bias"]);   // [1, 8192]
        var output = Zero(g, t, Wn);
        for (int i = 0; i < WnLayers; i++)
        {
            var xIn = Ops.Add(ConvReflect(g, x, Wt($"wavenet.in_layers.{i}.conv.conv"), WnK),
                              _w[P + $"wavenet.in_layers.{i}.conv.conv.bias"]);   // [T, 1024]
            var gL = Chan(gconv, 1, 2 * Wn * WnLayers, i * 2 * Wn, 2 * Wn);        // [1, 1024]
            // in_act = x_in + g_l (both T x 1024); tanh first half, sigmoid second half.
            var inAct = Ops.Add(Chan(xIn, t, 2 * Wn, 0, 2 * Wn), gL);              // [T, 1024]
            var acts = Ops.Mul(Ops.Tanh(Chan(inAct, t, 2 * Wn, 0, Wn)),
                               Ops.Sigmoid(Chan(inAct, t, 2 * Wn, Wn, Wn)));
            var rs = Ops.Add(Ops.Conv1d(acts, Wt($"wavenet.res_skip_layers.{i}.conv.conv"), 1, 0, 1),
                             _w[P + $"wavenet.res_skip_layers.{i}.conv.conv.bias"]);
            if (i < WnLayers - 1)
            {
                x = Ops.Add(x, Chan(rs, t, 2 * Wn, 0, Wn));
                output = Ops.Add(output, Chan(rs, t, 2 * Wn, Wn, Wn));
            }
            else
            {
                output = Ops.Add(output, rs);
            }
        }
        return output;
    }

    /// <summary>SConv1d (non-causal): reflect-pad by (k-1)*dil/2 on each side, then a zero-pad
    /// free conv. The pad is a gather with reflected time indices.</summary>
    private static Tensor ConvReflect(Graph g, Tensor x, Tensor w, int k)
    {
        int t = (int)x.Shape[0];
        int pad = (k - 1) / 2;
        var idx = new int[t + 2 * pad];
        for (int i = 0; i < pad; i++) idx[i] = pad - i;
        for (int i = 0; i < t; i++) idx[pad + i] = i;
        for (int i = 0; i < pad; i++) idx[pad + t + i] = t - 2 - i;
        var xp = Ops.GetRows(x, g.InputI32(new long[] { t + 2 * pad }, idx));
        return Ops.Conv1d(xp, w, 1, 0, 1);
    }

    // ---- timestep embedding ------------------------------------------------------------------

    /// <summary>Sinusoidal timestep embedding (host) then the 2-layer MLP: PT [1, 512].</summary>
    private Tensor TimeEmbed(Graph g, float t, string prefix)
    {
        var freq = new float[256];
        for (int i = 0; i < 128; i++)
        {
            float a = 1000f * t * _tfreq[i];
            freq[i] = MathF.Cos(a);
            freq[128 + i] = MathF.Sin(a);
        }
        var fe = g.Input(new long[] { 1, 256 }, freq);
        var h = Ops.Add(Ops.Linear(fe, _w[P + prefix + ".mlp.0.weight"]), _w[P + prefix + ".mlp.0.bias"]);
        h = Ops.Silu(h);
        return Ops.Add(Ops.Linear(h, _w[P + prefix + ".mlp.2.weight"]), _w[P + prefix + ".mlp.2.bias"]);
    }

    // ---- small helpers -----------------------------------------------------------------------

    /// <summary>Slice a channel range from a PT [T, total] tensor (channel is the fast axis).
    /// Materialized: the slice's stride is not dense, and downstream reshapes require it.</summary>
    private static Tensor Chan(Tensor x, int t, long total, long from, long count)
        => Ops.Contiguous(Ops.View2d(x, count, t, (ulong)(total * 4), (ulong)(from * 4)));

    /// <summary>A constant [1, c] tensor of ones / zeros on the device.</summary>
    private static Tensor One(Graph g, int t, int c)
    {
        var v = new float[t * c];
        for (int i = 0; i < v.Length; i++) v[i] = 1f;
        return g.Input(new long[] { t, c }, v);
    }

    private static Tensor Zero(Graph g, int t, int c)
        => g.Input(new long[] { t, c }, new float[t * c]);

    public void Dispose() => _mem.Dispose();
}
