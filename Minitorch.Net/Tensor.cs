using System;
using System.Runtime.InteropServices;

namespace Minitorch;

/// <summary>A handle to a ggml tensor node (owned by a Graph or Memory/GgufFile).</summary>
public sealed class Tensor
{
    public IntPtr Handle { get; }

    public Tensor(IntPtr handle) => Handle = handle;

    /// <summary>PyTorch-order shape.</summary>
    public long[] Shape
    {
        get
        {
            int nd = Native.mt_tensor_dim(Handle);
            var s = new long[nd];
            Native.mt_tensor_shape(Handle, s);
            return s;
        }
    }

    public long Numel => Native.mt_tensor_numel(Handle);

    /// <summary>Mark as a graph output so its buffer is not reused (call before reading).</summary>
    public Tensor MarkOutput()
    {
        Native.mt_tensor_mark_output(Handle);
        return this;
    }

    public string BackendName => Marshal.PtrToStringAnsi(Native.mt_tensor_backend_name(Handle)) ?? "";

    public override string ToString() => $"Tensor(shape=[{string.Join(", ", Shape)}])";
}

public static class TensorExtensions
{
    /// <summary>Materialize an F32 tensor to a float[] (computes the graph if needed).</summary>
    public static float[] ToFloats(this Tensor t, Graph g)
    {
        var bytes = t.ToBytes(g);
        var f = new float[t.Numel];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }

    public static int[] ToInts(this Tensor t, Graph g)
    {
        var bytes = t.ToBytes(g);
        var v = new int[t.Numel];
        Buffer.BlockCopy(bytes, 0, v, 0, bytes.Length);
        return v;
    }

    public static byte[] ToBytes(this Tensor t, Graph g)
    {
        var buf = new byte[t.Numel * 4];
        var pin = GCHandle.Alloc(buf, GCHandleType.Pinned);
        try
        {
            Native.mt_graph_to_bytes(g.Handle, t.Handle, pin.AddrOfPinnedObject(), (UIntPtr)buf.Length);
        }
        finally { pin.Free(); }
        return buf;
    }
}
