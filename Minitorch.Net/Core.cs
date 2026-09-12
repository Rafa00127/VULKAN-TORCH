using System;
using System.Runtime.InteropServices;

namespace Minitorch;

public sealed class Device
{
    internal IntPtr Handle { get; }
    public Device(IntPtr handle) => Handle = handle;
}

public sealed class Runtime : IDisposable
{
    internal IntPtr H { get; private set; }

    public Runtime() => H = Native.mt_runtime_new();
    public string Name => Marshal.PtrToStringAnsi(Native.mt_runtime_name(H)) ?? "";
    public Device Gpu() => new Device(Native.mt_runtime_gpu(H));
    public Device Cpu() => new Device(Native.mt_runtime_cpu(H));

    public void Dispose()
    {
        if (H != IntPtr.Zero) { Native.mt_runtime_free(H); H = IntPtr.Zero; }
    }
}

public sealed class GgufFile : IDisposable
{
    private IntPtr _h;
    internal IntPtr H => _h;

    public GgufFile(string path, Device device, ulong arenaBytes = 512UL << 20)
        => _h = Native.mt_gguf_new(path, device.Handle, (UIntPtr)arenaBytes);

    public bool Has(string name) => Native.mt_gguf_has(_h, name) != 0;

    public string[] Names()
    {
        int n = Native.mt_gguf_count(_h);
        var a = new string[n];
        for (int i = 0; i < n; i++)
            a[i] = Marshal.PtrToStringUTF8(Native.mt_gguf_name(_h, i)) ?? "";
        return a;
    }

    public Tensor Tensor(string name)
    {
        if (Native.mt_gguf_tensor(_h, name, out var t) != 0)
            throw new InvalidOperationException($"GgufFile: no tensor '{name}'");
        return new Tensor(t);
    }

    public long[] Shape(string name)
    {
        var s = new long[4];
        int nd = Native.mt_gguf_shape(_h, name, s, out _);
        if (nd < 0) throw new InvalidOperationException($"GgufFile: no tensor '{name}'");
        return s[..nd];
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero) { Native.mt_gguf_free(_h); _h = IntPtr.Zero; }
    }
}

public sealed class Memory : IDisposable
{
    private IntPtr _h;
    public Memory(Device device, ulong bytes) => _h = Native.mt_memory_new(device.Handle, (UIntPtr)bytes);

    public Tensor Tensor(long[] shape, int dtype = 0, byte[]? data = null)
    {
        IntPtr dataPtr = IntPtr.Zero;
        GCHandle pin = default;
        try
        {
            if (data != null) { pin = GCHandle.Alloc(data, GCHandleType.Pinned); dataPtr = pin.AddrOfPinnedObject(); }
            if (Native.mt_memory_tensor(_h, shape, shape.Length, dtype, dataPtr, (UIntPtr)(data?.Length ?? 0), out var t) != 0)
                throw new InvalidOperationException("Memory.Tensor failed");
            return new Tensor(t);
        }
        finally { if (data != null) pin.Free(); }
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero) { Native.mt_memory_free(_h); _h = IntPtr.Zero; }
    }
}

public sealed class Graph : IDisposable
{
    internal IntPtr Handle { get; private set; }

    public Graph(Runtime rt, Device device) => Handle = Native.mt_graph_new(rt.H, device.Handle);

    public void Enter() => Native.mt_graph_enter(Handle);
    public void Exit() => Native.mt_graph_exit(Handle);

    public Tensor Input(long[] shape, byte[] data)
    {
        var pin = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            if (Native.mt_graph_input(Handle, shape, shape.Length, pin.AddrOfPinnedObject(),
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
            if (Native.mt_graph_input_i32(Handle, shape, shape.Length, pin.AddrOfPinnedObject(),
                    (UIntPtr)bytes.Length, out var t) != 0)
                throw new InvalidOperationException("Graph.InputI32 failed");
            return new Tensor(t);
        }
        finally { pin.Free(); }
    }

    public void Dispose()
    {
        if (Handle != IntPtr.Zero) { Native.mt_graph_free(Handle); Handle = IntPtr.Zero; }
    }
}
