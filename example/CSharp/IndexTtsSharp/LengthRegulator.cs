using System;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 s2mel length regulator (InterpolateRegulator): expands the semantic codec's
/// features [T, 1024] to the target mel length L by nearest-neighbour time resampling, then a
/// stack of Conv1d(k3) + GroupNorm + Mish blocks and a final 1x1 conv.
/// Ported from indextts/s2mel/modules/length_regulator.py.
///
/// GroupNorm(groups=1, C) normalizes over (C, T) jointly per sample, which here is every
/// element of the segment — implemented as a flatten + LayerNorm + per-channel affine.
/// The batch mask is all-ones for a single segment, so it is not applied.
/// </summary>
public sealed class LengthRegulator
{
    private const int C = 512;
    private const float Eps = 1e-5f;
    private const string P = "s2mel.length_regulator.";

    private readonly GgufWeights _w;

    public LengthRegulator(GgufWeights w) => _w = w;

    /// <summary>x: PT [T, 1024] -> PT [L, 512]. <paramref name="ylens"/> is the target length.</summary>
    public Tensor Forward(Graph g, Tensor x, int ylens)
    {
        var proj = Ops.Add(Ops.Linear(x, _w[P + "content_in_proj.weight"]),
                           _w[P + "content_in_proj.bias"]);           // [T, 512]
        int t = (int)proj.Shape[0];
        return Model(g, Nearest(g, proj, t, ylens));                  // [L, 512]
    }

    /// <summary>The projected features before resampling (for validation).</summary>
    public Tensor Project(Graph g, Tensor x)
        => Ops.Add(Ops.Linear(x, _w[P + "content_in_proj.weight"]), _w[P + "content_in_proj.bias"]);

    /// <summary>Nearest-neighbour resize along time: out[i] = x[min(floor(i*T/L), T-1)].
    /// Implemented as a row gather (the time axis is the row axis of a PT [T, C] tensor).</summary>
    public Tensor Nearest(Graph g, Tensor x, int t, int l)
    {
        var idx = new int[l];
        for (int i = 0; i < l; i++) idx[i] = Math.Min(i * t / l, t - 1);
        var idT = g.InputI32(new long[] { l }, idx);
        return Ops.GetRows(x, idT);                                   // [L, 512]
    }

    private Tensor Model(Graph g, Tensor x)
    {
        for (int i = 0; i < 4; i++)
        {
            x = Ops.Add(Ops.Conv1d(x, _w[P + $"model.{i * 3}.weight"], 1, 1, 1),
                        _w[P + $"model.{i * 3}.bias"]);
            x = Norm(g, x, $"model.{i * 3 + 1}");
            x = Ops.Mul(x, Ops.Tanh(Ops.Softplus(x)));                // Mish
        }
        return Ops.Add(Ops.Conv1d(x, _w[P + "model.12.weight"], 1, 0, 1), _w[P + "model.12.bias"]);
    }

    /// <summary>GroupNorm(groups=1, C): normalize over all C*T, then per-channel affine.</summary>
    private Tensor Norm(Graph g, Tensor x, string prefix)
    {
        int t = (int)x.Shape[0];
        var flat = Ops.Reshape(x, new long[] { 1, (long)t * C });     // ne = [C*T, 1]
        var normed = Ops.Reshape(Ops.LayerNorm(flat, Eps), new long[] { t, C });
        return Ops.Add(Ops.Mul(normed, _w[P + prefix + ".weight"]), _w[P + prefix + ".bias"]);
    }
}
