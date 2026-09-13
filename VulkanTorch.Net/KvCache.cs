using System;

namespace VulkanTorch;

/// <summary>
/// An F16 key/value cache laid out PT [layers, heads, maxCtx, headDim], with the
/// strided views a cached decoder needs: a wide view to write new K/V into at runtime
/// positions (via set_rows) and a windowed, contiguous view to attend over.
/// </summary>
public sealed class KvCache : IDisposable
{
    public Memory Mem { get; }
    public Tensor K { get; }
    public Tensor V { get; }
    public int Layers { get; }
    public int Heads { get; }
    public int HeadDim { get; }
    public int MaxCtx { get; }

    public KvCache(Device dev, int layers, int heads, int headDim, int maxCtx)
    {
        Layers = layers; Heads = heads; HeadDim = headDim; MaxCtx = maxCtx;
        Mem = new Memory(dev, (ulong)(2L * layers * heads * maxCtx * headDim * 2));
        // Unwritten slots are masked to -inf, but zero them anyway: uninitialised device
        // memory can hold NaN, which survives a -inf mask add and poisons the softmax.
        var z = new byte[(long)layers * heads * maxCtx * headDim * 2];
        K = Mem.Tensor(new long[] { layers, heads, maxCtx, headDim }, 1, z);
        V = Mem.Tensor(new long[] { layers, heads, maxCtx, headDim }, 1, z);
    }

    /// <summary>Byte strides of the [heads, maxCtx, headDim] layout, for callers that
    /// need to build their own views (e.g. a Cpy at a baked-in position).</summary>
    public ulong Nb1 => (ulong)(HeadDim * 2);
    public ulong Nb2 => (ulong)(HeadDim * 2L * MaxCtx);
    public ulong Nb3 => (ulong)(HeadDim * 2L * MaxCtx * Heads);

    public ulong LayerOffset(int layer) => (ulong)layer * Nb3;

    /// <summary>Wide view PT [heads, maxCtx, headDim] to pass to <c>Ops.SetRows</c>.</summary>
    public (Tensor k, Tensor v) WriteView(int layer)
    {
        ulong off = LayerOffset(layer);
        return (Ops.View3d(K, HeadDim, MaxCtx, Heads, Nb1, Nb2, off),
                Ops.View3d(V, HeadDim, MaxCtx, Heads, Nb1, Nb2, off));
    }

    /// <summary>Contiguous window PT [heads, lk, headDim] holding the first <paramref name="lk"/> slots.</summary>
    public (Tensor k, Tensor v) ReadView(int layer, int lk)
    {
        ulong off = LayerOffset(layer);
        return (Ops.Contiguous(Ops.View3d(K, HeadDim, lk, Heads, Nb1, Nb2, off)),
                Ops.Contiguous(Ops.View3d(V, HeadDim, lk, Heads, Nb1, Nb2, off)));
    }

    public void Dispose() => Mem.Dispose();
}
