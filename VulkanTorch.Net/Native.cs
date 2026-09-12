using System;
using System.Runtime.InteropServices;

namespace VulkanTorch;

// Raw P/Invoke surface over vulkantorch.dll (the C++ core's C ABI).
internal static class Native
{
    private const string Dll = "vulkantorch.dll";

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_runtime_new();
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_runtime_free(IntPtr rt);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_runtime_name(IntPtr rt);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_runtime_gpu(IntPtr rt);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_runtime_cpu(IntPtr rt);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_gguf_new([MarshalAs(UnmanagedType.LPUTF8Str)] string path, IntPtr device, UIntPtr arena);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_gguf_free(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_gguf_has(IntPtr g, [MarshalAs(UnmanagedType.LPUTF8Str)] string name);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_gguf_count(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_gguf_name(IntPtr g, int i);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_gguf_tensor(IntPtr g, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out IntPtr tensor);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_gguf_shape(IntPtr g, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, [Out] long[] shape, out int ndim);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_memory_new(IntPtr device, UIntPtr bytes);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_memory_free(IntPtr m);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_memory_tensor(IntPtr m, long[] shape, int ndim, int dtype, IntPtr data, UIntPtr bytes, out IntPtr tensor);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_graph_new(IntPtr rt, IntPtr device);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_graph_free(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_graph_enter(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_graph_exit(IntPtr g);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_graph_input(IntPtr g, long[] shape, int ndim, IntPtr data, UIntPtr bytes, out IntPtr tensor);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_graph_input_i32(IntPtr g, long[] shape, int ndim, IntPtr data, UIntPtr bytes, out IntPtr tensor);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_graph_to_bytes(IntPtr g, IntPtr tensor, IntPtr outBytes, UIntPtr bytes);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_tensor_dim(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern long vt_tensor_numel(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_tensor_shape(IntPtr t, [Out] long[] outShape);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_tensor_mark_output(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_tensor_backend_name(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern int vt_tensor_type(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern UIntPtr vt_tensor_nbytes(IntPtr t);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern void vt_tensor_to_bytes(IntPtr t, IntPtr outBytes, UIntPtr bytes);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_matmul(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_mul_mat(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_add(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_sub(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_mul(IntPtr a, IntPtr b);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_scale(IntPtr a, float s);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_transpose(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_contiguous(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_reshape(IntPtr a, long[] shape, int ndim);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_repeat(IntPtr a, long[] shape, int ndim);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_gelu(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_elu(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_silu(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_tanh(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_soft_max(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_linear(IntPtr x, IntPtr w);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_rms_norm(IntPtr a, float eps);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_layer_norm(IntPtr a, float eps);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_concat(IntPtr a, IntPtr b, long dim);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_argmax(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_get_rows(IntPtr a, IntPtr ids);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_sum_rows(IntPtr a);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_cast(IntPtr a, int type);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_cpy(IntPtr a, IntPtr dst);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_view_2d(IntPtr a, long ne0, long ne1, UIntPtr nb1, UIntPtr offset);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_view_3d(IntPtr a, long ne0, long ne1, long ne2, UIntPtr nb1, UIntPtr nb2, UIntPtr offset);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_view_4d(IntPtr a, long ne0, long ne1, long ne2, long ne3, UIntPtr nb1, UIntPtr nb2, UIntPtr nb3, UIntPtr offset);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_permute_pt(IntPtr a, int p0, int p1, int p2, int p3);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_rope(IntPtr a, IntPtr pos, int nDims, int mode, int nCtxOrig, float freqBase, float freqScale, float extFactor, float attnFactor, float betaFast, float betaSlow);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_flash_attn(IntPtr q, IntPtr k, IntPtr v, IntPtr mask, float scale, float maxBias, float logitSoftcap);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_conv1d(IntPtr x, IntPtr w, int stride, int pad, int dilation);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_conv_transpose_1d(IntPtr x, IntPtr wPerm, int stride, int oc);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_snake_1d(IntPtr x, IntPtr alpha);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_im2col_rafa(IntPtr x, int k, int s0, int p0, int d0);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr vt_col2im_1d(IntPtr col, int s0, int oc, int p0);
}
