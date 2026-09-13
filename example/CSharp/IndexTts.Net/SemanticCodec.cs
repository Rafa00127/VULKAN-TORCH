using System;
using System.Collections.Generic;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 semantic codec (EnhancedCodec.decode): mel-code indices -> semantic features.
/// Ported from indextts/codec/models.py + kmeans/vocos.py.
///
/// codes [T] -> codebook lookup [T,8] -> out_project WNConv1d(8->1024) -> VocosBackbone
/// (1024 -> 384, 12 ConvNeXt blocks) -> Linear(384->1024) -> nearest x2 -> up Conv1d(1024,1024,k3)
/// -> [2T, 1024].  The time axis is doubled: one mel code becomes two semantic frames.
///
/// The quantizer's out_project is a weight-normed Conv1d whose g/v are NOT fused in the
/// checkpoint, so the effective weight g*v/||v|| is materialized once on the host.
/// </summary>
public sealed class SemanticCodec : IDisposable
{
    private const int Dim = 1024, VocosDim = 384, Inter = 2048, Layers = 12, CodebookDim = 8;
    private const string P = "codec.model.";

    private readonly GgufWeights _w;
    private readonly Memory _mem;
    private readonly Tensor _outProjW;

    public SemanticCodec(Device dev, GgufWeights w)
    {
        _w = w;
        _mem = new Memory(dev, 1UL << 20);
        _outProjW = FuseWeightNorm(_mem,
            w[P + "quantizer.quantizers.0.out_project.weight_g"],
            w[P + "quantizer.quantizers.0.out_project.weight_v"]);
    }

    /// <summary>WNConv1d without the fusion: W = g * v / ||v||, norm over every axis but 0.
    /// PT shapes: g [C,1,1], v [C,8,1] -> W [C,8,1].</summary>
    private static Tensor FuseWeightNorm(Memory mem, Tensor g, Tensor v)
    {
        var gH = g.ReadBytes();   // F16 [1024,1,1]
        var vH = v.ReadBytes();   // F16 [1024,8,1]
        int C = (int)v.Shape[0], D = (int)v.Shape[1], K = (int)v.Shape[2];
        var W = new float[C * D * K];
        for (int c = 0; c < C; c++)
        {
            float gc = (float)BitConverter.ToHalf(gH, c * 2);
            double ss = 0;
            for (int i = 0; i < D; i++)
                for (int k = 0; k < K; k++)
                {
                    float vv = (float)BitConverter.ToHalf(vH, ((c * D + i) * K + k) * 2);
                    ss += (double)vv * vv;
                }
            float inv = (float)(gc / Math.Sqrt(ss));
            for (int i = 0; i < D; i++)
                for (int k = 0; k < K; k++)
                    W[(c * D + i) * K + k] = (float)BitConverter.ToHalf(vH, ((c * D + i) * K + k) * 2) * inv;
        }
        var bytes = new byte[W.Length * 4];
        Buffer.BlockCopy(W, 0, bytes, 0, bytes.Length);
        return mem.Tensor(new long[] { C, D, K }, Ops.F32, bytes);
    }

    /// <summary>codes (I32 [T]) -> semantic features PT [2T, 1024].</summary>
    public Tensor Decode(Graph g, Tensor codes) => DecodeStages(g, codes).Rec;

    /// <summary>The decode chain's checkpoints: quantized embedding, the Vocos blocks, and the
    /// final upsampled features — for stage-wise validation against the reference.</summary>
    public (Tensor Quant, Tensor FinLn, List<Tensor> Blocks, Tensor Dec, Tensor Rec) DecodeStages(
        Graph g, Tensor codes)
    {
        var emb = Ops.GetRows(_w[P + "quantizer.quantizers.0.codebook.weight"], codes);  // [T, 8]
        var q = Ops.Add(Ops.Conv1d(emb, _outProjW, 1, 0, 1),
                        _w[P + "quantizer.quantizers.0.out_project.bias"]);              // [T, 1024]
        var (_, blocks, final) = VocosStages(g, q);
        var lin = Ops.Add(Ops.Linear(final, _w[P + "decoder.1.weight"]), _w[P + "decoder.1.bias"]);
        int t = (int)lin.Shape[0];
        // nearest x2 over time: [T,1024] -> [T,1,1024] -> repeat -> [T,2,1024] -> [2T,1024]
        var xi = Ops.Reshape(Ops.Repeat(Ops.Reshape(lin, new long[] { t, 1, Dim }),
                                        new long[] { t, 2, Dim }),
                             new long[] { 2 * t, Dim });
        var rec = Ops.Add(Ops.Conv1d(xi, _w[P + "up.weight"], 1, 1, 1), _w[P + "up.bias"]);  // [2T, 1024]
        return (q, final, blocks, lin, rec);
    }

    /// <summary>VocosBackbone with the per-block outputs exposed for stage-wise validation.</summary>
    private (Tensor Embed, List<Tensor> Blocks, Tensor Final) VocosStages(Graph g, Tensor x)
    {
        x = Ops.Add(Ops.Conv1d(x, _w[P + "decoder.0.embed.weight"], 1, 3, 1),
                    _w[P + "decoder.0.embed.bias"]);
        x = Norm(x, "decoder.0.norm");
        var blocks = new List<Tensor>(Layers);
        for (int i = 0; i < Layers; i++)
        {
            string b = P + $"decoder.0.convnext.{i}.";
            var h = Ops.Add(Ops.Conv1dDw(x, _w[b + "dwconv.weight"], 1, 3, 1), _w[b + "dwconv.bias"]);
            h = Norm(h, $"decoder.0.convnext.{i}.norm");
            h = Ops.Add(Ops.Linear(h, _w[b + "pwconv1.weight"]), _w[b + "pwconv1.bias"]);
            h = Ops.GeluErf(h);
            h = Ops.Add(Ops.Linear(h, _w[b + "pwconv2.weight"]), _w[b + "pwconv2.bias"]);
            h = Ops.Mul(h, _w[b + "gamma"]);
            x = Ops.Add(x, h);
            blocks.Add(x);
        }
        return (x, blocks, Norm(x, "decoder.0.final_layer_norm"));
    }

    /// <summary>ggml_norm is plain (no affine); apply LayerNorm weight/bias by hand.</summary>
    private Tensor Norm(Tensor x, string prefix)
        => Ops.Add(Ops.Mul(Ops.LayerNorm(x, 1e-6f), _w[P + prefix + ".weight"]),
                   _w[P + prefix + ".bias"]);

    public void Dispose()
    {
        _mem.Dispose();
    }
}
