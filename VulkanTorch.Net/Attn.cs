using System;

namespace VulkanTorch;

/// <summary>
/// Multi-head attention plumbing shared by the model ports: head-layout reshapes and
/// scaled dot-product attention.
/// </summary>
public static class Attn
{
    /// <summary>
    /// PT [T, n*hd] -> PT [n, T, hd]. Reshapes to an explicit 4-D [T,n,hd,1] first:
    /// a 3-D tensor's 4-D form is [1,n,T,hd], so permuting it directly would transpose
    /// the wrong pair.
    /// </summary>
    public static Tensor ToHeads(Tensor x, int t, int n, int hd)
    {
        var x4 = Ops.Reshape(x, new long[] { t, n, hd, 1 });
        return Ops.Reshape(Ops.Contiguous(Ops.PermutePt(x4, 1, 0, 2, 3)), new long[] { n, t, hd });
    }

    /// <summary>PT [n, T, hd] -> PT [T, n*hd] (inverse of <see cref="ToHeads"/>).</summary>
    public static Tensor FromHeads(Tensor x, int t, int n, int hd)
    {
        var x4 = Ops.Reshape(x, new long[] { n, t, hd, 1 });
        return Ops.Reshape(Ops.Contiguous(Ops.PermutePt(x4, 1, 0, 2, 3)), new long[] { t, n * hd });
    }

    /// <summary>
    /// Scaled dot-product attention. q PT [h, n_q, hd]; k, v PT [h, n_kv, hd] -> PT [h, n_q, hd].
    /// </summary>
    public static Tensor Sdpa(Tensor q, Tensor k, Tensor v, float scale)
        => Ops.Bmm(Ops.SoftMax(Ops.Scale(Ops.Bmm(q, Ops.Transpose(k)), scale)), v);
}
