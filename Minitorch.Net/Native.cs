using System;
using System.Runtime.InteropServices;

namespace Minitorch;

// Raw P/Invoke surface over minitorch.dll (the C++ core's C ABI).
internal static class Native
{
    private const string Dll = "minitorch.dll";

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_runtime_new();
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_runtime_free(IntPtr rt);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_runtime_name(IntPtr rt);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_runtime_gpu(IntPtr rt);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_runtime_cpu(IntPtr rt);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_gguf_new([MarshalAs(UnmanagedType.LPUTF8Str)] string path, IntPtr device, UIntPtr arena);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_gguf_free(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int mt_gguf_has(IntPtr g, [MarshalAs(UnmanagedType.LPUTF8Str)] string name);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int mt_gguf_count(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_gguf_name(IntPtr g, int i);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int mt_gguf_tensor(IntPtr g, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out IntPtr tensor);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int mt_gguf_shape(IntPtr g, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, [Out] long[] shape, out int ndim);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_memory_new(IntPtr device, UIntPtr bytes);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_memory_free(IntPtr m);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int mt_memory_tensor(IntPtr m, long[] shape, int ndim, int dtype, IntPtr data, UIntPtr bytes, out IntPtr tensor);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_graph_new(IntPtr rt, IntPtr device);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_graph_free(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_graph_enter(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_graph_exit(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int mt_graph_input(IntPtr g, long[] shape, int ndim, IntPtr data, UIntPtr bytes, out IntPtr tensor);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int mt_graph_input_i32(IntPtr g, long[] shape, int ndim, IntPtr data, UIntPtr bytes, out IntPtr tensor);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_graph_to_bytes(IntPtr g, IntPtr tensor, IntPtr outBytes, UIntPtr bytes);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int mt_tensor_dim(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern long mt_tensor_numel(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_tensor_shape(IntPtr t, [Out] long[] outShape);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void mt_tensor_mark_output(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_tensor_backend_name(IntPtr t);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_matmul(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_mul_mat(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_add(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_sub(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_mul(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_scale(IntPtr a, float s);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_transpose(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_contiguous(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_reshape(IntPtr a, long[] shape, int ndim);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_repeat(IntPtr a, long[] shape, int ndim);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_gelu(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_elu(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_silu(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_tanh(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_soft_max(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_linear(IntPtr x, IntPtr w);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_rms_norm(IntPtr a, float eps);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_layer_norm(IntPtr a, float eps);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_concat(IntPtr a, IntPtr b, long dim);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_argmax(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_get_rows(IntPtr a, IntPtr ids);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_sum_rows(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_cast(IntPtr a, int type);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_cpy(IntPtr a, IntPtr dst);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_view_2d(IntPtr a, long ne0, long ne1, UIntPtr nb1, UIntPtr offset);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_view_3d(IntPtr a, long ne0, long ne1, long ne2, UIntPtr nb1, UIntPtr nb2, UIntPtr offset);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_view_4d(IntPtr a, long ne0, long ne1, long ne2, long ne3, UIntPtr nb1, UIntPtr nb2, UIntPtr nb3, UIntPtr offset);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_permute_pt(IntPtr a, int p0, int p1, int p2, int p3);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_rope(IntPtr a, IntPtr pos, int nDims, int mode, int nCtxOrig, float freqBase, float freqScale, float extFactor, float attnFactor, float betaFast, float betaSlow);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_flash_attn(IntPtr q, IntPtr k, IntPtr v, IntPtr mask, float scale, float maxBias, float logitSoftcap);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_conv1d(IntPtr x, IntPtr w, int stride, int pad, int dilation);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_conv_transpose_1d(IntPtr x, IntPtr wPerm, int stride, int oc);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_snake_1d(IntPtr x, IntPtr alpha);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_im2col_rafa(IntPtr x, int k, int s0, int p0, int d0);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr mt_col2im_1d(IntPtr col, int s0, int oc, int p0);
}
