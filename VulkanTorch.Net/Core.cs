using System;
using System.Runtime.InteropServices;

namespace VulkanTorch;

public sealed class Device
{
    internal IntPtr Handle { get; }
    public Device(IntPtr handle) => Handle = handle;
}

public sealed class Runtime : IDisposable
{
    internal IntPtr H { get; private set; }

    public Runtime() => H = Native.vt_runtime_new();
    public string Name => Marshal.PtrToStringAnsi(Native.vt_runtime_name(H)) ?? "";
    public Device Gpu() => new Device(Native.vt_runtime_gpu(H));
    public Device Cpu() => new Device(Native.vt_runtime_cpu(H));

    public void Dispose()
    {
        if (H != IntPtr.Zero) { Native.vt_runtime_free(H); H = IntPtr.Zero; }
    }
}

public sealed class GgufFile : IDisposable
{
    private IntPtr _h;
    internal IntPtr H => _h;

    public GgufFile(string path, Device device, ulong arenaBytes = 512UL << 20)
        => _h = Native.vt_gguf_new(path, device.Handle, (UIntPtr)arenaBytes);

    public bool Has(string name) => Native.vt_gguf_has(_h, name) != 0;

    public string[] Names()
    {
        int n = Native.vt_gguf_count(_h);
        var a = new string[n];
        for (int i = 0; i < n; i++)
            a[i] = Marshal.PtrToStringUTF8(Native.vt_gguf_name(_h, i)) ?? "";
        return a;
    }

    public Tensor Tensor(string name)
    {
        if (Native.vt_gguf_tensor(_h, name, out var t) != 0)
            throw new InvalidOperationException($"GgufFile: no tensor '{name}'");
        return new Tensor(t);
    }

    public long[] Shape(string name)
    {
        var s = new long[4];
        int nd = Native.vt_gguf_shape(_h, name, s, out _);
        if (nd < 0) throw new InvalidOperationException($"GgufFile: no tensor '{name}'");
        return s[..nd];
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero) { Native.vt_gguf_free(_h); _h = IntPtr.Zero; }
    }
}

public sealed class Memory : IDisposable
{
    private IntPtr _h;
    public Memory(Device device, ulong bytes) => _h = Native.vt_memory_new(device.Handle, (UIntPtr)bytes);

    public Tensor Tensor(long[] shape, int dtype = 0, byte[]? data = null)
    {
        IntPtr dataPtr = IntPtr.Zero;
        GCHandle pin = default;
        try
        {
            if (data != null) { pin = GCHandle.Alloc(data, GCHandleType.Pinned); dataPtr = pin.AddrOfPinnedObject(); }
            if (Native.vt_memory_tensor(_h, shape, shape.Length, dtype, dataPtr, (UIntPtr)(data?.Length ?? 0), out var t) != 0)
                throw new InvalidOperationException("Memory.Tensor failed");
            return new Tensor(t);
        }
        finally { if (data != null) pin.Free(); }
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero) { Native.vt_memory_free(_h); _h = IntPtr.Zero; }
    }
}

public sealed class Graph : IDisposable
{
    internal IntPtr Handle { get; private set; }

    /// <summary>Graph with an explicit node capacity. The default (16384) is enough for small
    /// models but NOT for wide vocoders — exceeding it silently corrupts memory.</summary>
    public Graph(Runtime rt, Device device, ulong maxNodes)
        => Handle = Native.vt_graph_new_n(rt.H, device.Handle, (UIntPtr)maxNodes);

    public Graph(Runtime rt, Device device) => Handle = Native.vt_graph_new(rt.H, device.Handle);

    /// <summary>Number of nodes in the captured graph (introspection).</summary>
    public int NodeCount => Native.vt_graph_n_nodes(Handle);

    public void Enter() => Native.vt_graph_enter(Handle);
    public void Exit() => Native.vt_graph_exit(Handle);

    public Tensor Input(long[] shape, byte[] data)
    {
        var pin = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            if (Native.vt_graph_input(Handle, shape, shape.Length, pin.AddrOfPinnedObject(),
                    (UIntPtr)data.Length, out var t) != 0)
                throw new InvalidOperationException("Graph.Input failed");
            return new Tensor(t);
        }
        finally { pin.Free(); }
    }

    public Tensor Input(long[] shape, float[] data)
    {
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return Input(shape, bytes);
    }

    public Tensor InputI32(long[] shape, int[] data)
    {
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        var pin = GCHandle.Alloc(bytes, GCHandleType.Pinned);
        try
        {
            if (Native.vt_graph_input_i32(Handle, shape, shape.Length, pin.AddrOfPinnedObject(),
                    (UIntPtr)bytes.Length, out var t) != 0)
                throw new InvalidOperationException("Graph.InputI32 failed");
            return new Tensor(t);
        }
        finally { pin.Free(); }
    }

    /// <summary>
    /// One-shot compute of the captured graph. NOT for replay: re-running the
    /// scheduler's reset+alloc changes the result on many graphs (the output then
    /// depends on the allocation pass, not on the inputs). For replay use
    /// <see cref="AllocStatic"/> + <see cref="ComputeStatic"/>.
    /// </summary>
    public void Compute() => Native.vt_graph_compute(Handle);

    /// <summary>
    /// Static replay path: allocate the graph ONCE with a graph allocator, then only
    /// re-upload inputs + compute. Cheaper than <see cref="Compute"/> (no reset/alloc
    /// per replay) and required for correctness: re-allocating on every replay makes
    /// the result depend on the allocation pass rather than on the inputs.
    /// Call after the graph is fully captured; then use <see cref="ComputeStatic"/>.
    /// </summary>
    public void AllocStatic() => Native.vt_graph_alloc_static(Handle);

    /// <summary>Replay a statically-allocated graph (see <see cref="AllocStatic"/>).</summary>
    public void ComputeStatic() => Native.vt_graph_compute_static(Handle);

    /// <summary>Update the host data of an already-created input tensor (for replay).</summary>
    public void SetInput(Tensor tensor, byte[] data)
    {
        var pin = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            if (Native.vt_graph_set_input(Handle, tensor.Handle, pin.AddrOfPinnedObject(),
                    (UIntPtr)data.Length) != 0)
                throw new InvalidOperationException("Graph.SetInput failed");
        }
        finally { pin.Free(); }
    }

    public void SetInput(Tensor tensor, float[] data)
    {
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        SetInput(tensor, bytes);
    }

    /// <summary>Update the host data of an already-created I32 input tensor (for replay).</summary>
    public void SetInput(Tensor tensor, int[] data)
    {
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        SetInput(tensor, bytes);
    }

    public void Dispose()
    {
        if (Handle != IntPtr.Zero) { Native.vt_graph_free(Handle); Handle = IntPtr.Zero; }
    }
}
