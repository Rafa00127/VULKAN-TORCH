using System;
using VulkanTorch;

internal static class Program
{
    private static byte[] Bytes(float[] a)
    {
        var b = new byte[a.Length * 4];
        Buffer.BlockCopy(a, 0, b, 0, b.Length);
        return b;
    }

    private static float[] NaiveMatmul(float[] a, float[] b, int M, int K, int N)
    {
        var c = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                double s = 0;
                for (int k = 0; k < K; k++) s += a[m * K + k] * b[k * N + n];
                c[m * N + n] = (float)s;
            }
        return c;
    }

    private static float[] NaiveConv1d(float[] x, float[] w, int T, int Cin, int Cout, int K, int pad)
    {
        int Tout = T + 2 * pad - (K - 1);
        var y = new float[Tout * Cout];
        for (int t = 0; t < Tout; t++)
            for (int oc = 0; oc < Cout; oc++)
            {
                double s = 0;
                for (int k = 0; k < K; k++)
                {
                    int ti = t + k - pad;
                    if (ti < 0 || ti >= T) continue;
                    for (int ic = 0; ic < Cin; ic++)
                        s += x[ti * Cin + ic] * w[(oc * Cin + ic) * K + k];
                }
                y[t * Cout + oc] = (float)s;
            }
        return y;
    }

    private static float MaxErr(float[] refv, float[] got)
    {
        float e = 0;
        for (int i = 0; i < refv.Length; i++) e = Math.Max(e, Math.Abs(refv[i] - got[i]));
        return e;
    }

    private static int Main()
    {
        using var rt = new Runtime();
        var dev = rt.Gpu();
        Console.WriteLine($"backend: {rt.Name}");

        var rng = new Random(1);
        int fails = 0;

        // ---- matmul ----
        {
            int M = 8, K = 6, N = 5;
            var A = new float[M * K];
            var B = new float[K * N];
            for (int i = 0; i < A.Length; i++) A[i] = (float)rng.NextDouble();
            for (int i = 0; i < B.Length; i++) B[i] = (float)rng.NextDouble();

            using var wm = new Memory(dev, 1UL << 20);
            var b = wm.Tensor(new long[] { K, N }, Ops.F32, Bytes(B));
            using var g = new Graph(rt, dev);
            g.Enter();
            var a = g.Input(new long[] { M, K }, A);
            var c = Ops.Contiguous(Ops.Matmul(a, b)).MarkOutput();
            var got = c.ToFloats(g);
            g.Exit();

            float e = MaxErr(NaiveMatmul(A, B, M, K, N), got);
            Console.WriteLine($"[{(e < 1e-4 ? " OK " : "FAIL")}] matmul  max_err={e:E2}  buf={c.BackendName}");
            if (e >= 2e-3) fails++;
        }

        // ---- conv1d ----
        {
            int T = 32, Cin = 4, Cout = 6, K = 5, pad = 2;
            var X = new float[T * Cin];
            var W = new float[Cout * Cin * K];
            for (int i = 0; i < X.Length; i++) X[i] = (float)rng.NextDouble();
            for (int i = 0; i < W.Length; i++) W[i] = (float)rng.NextDouble();

            using var wm = new Memory(dev, 1UL << 20);
            var w = wm.Tensor(new long[] { Cout, Cin, K }, Ops.F32, Bytes(W));
            using var g = new Graph(rt, dev);
            g.Enter();
            var x = g.Input(new long[] { T, Cin }, X);
            var y = Ops.Contiguous(Ops.Conv1d(x, w, 1, pad, 1)).MarkOutput();
            var got = y.ToFloats(g);
            g.Exit();

            float e = MaxErr(NaiveConv1d(X, W, T, Cin, Cout, K, pad), got);
            Console.WriteLine($"[{(e < 2e-3 ? " OK " : "FAIL")}] conv1d  max_err={e:E2}");
            if (e >= 2e-3) fails++;
        }

        // ---- native errors must surface as managed exceptions ----
        // Before the C ABI boundary was guarded, each of these killed the process
        // outright: no exception, no stack, no output (exit 127). The intended flow is
        // still "report, then die" -- nothing catches these in real code -- but the
        // message now travels through .NET's error channel with a stack trace naming
        // the C# call site. Reaching the catch at all is the test; the message check
        // just confirms it is the right one.
        {
            using var wm = new Memory(dev, 1UL << 20);
            var a = wm.Tensor(new long[] { 2, 2 }, Ops.F32, Bytes(new float[] { 1, 2, 3, 4 }));
            var b = wm.Tensor(new long[] { 2, 2 }, Ops.F32, Bytes(new float[] { 1, 2, 3, 4 }));

            // Deliberately no Graph.Enter() here.
            ExpectFailure("op outside a Graph scope", "outside a Graph::Scope",
                () => Ops.Add(a, b), ref fails);
        }

        {
            using var g = new Graph(rt, dev);
            g.Enter();

            // 3-D operands trip the ggml_is_matrix guard (ne[2] == ne[3] == 1).
            var v1 = g.Input(new long[] { 2, 2, 2 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var v2 = g.Input(new long[] { 2, 2, 2 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            ExpectFailure("matmul on 3-D tensors", "expected 2D tensors",
                () => Ops.Matmul(v1, v2), ref fails);

            // A rank-1 operand slips past that guard -- a PT [K] vector and a PT [1,K]
            // row are the same ggml shape -- and used to reach
            // GGML_ASSERT(ggml_can_mul_mat), aborting the whole process.
            var r1 = g.Input(new long[] { 4 }, new float[] { 1, 2, 3, 4 });
            var r2 = g.Input(new long[] { 4 }, new float[] { 1, 2, 3, 4 });
            ExpectFailure("matmul on rank-1 tensors", "inner dimension mismatch",
                () => Ops.Matmul(r1, r2), ref fails);

            g.Exit();
        }

        ExpectFailure("GgufFile on a missing path", "failed to open",
            () => { using var f = new GgufFile("no-such-model.gguf", dev); }, ref fails);

        Console.WriteLine(fails == 0 ? "\nall passed" : "\nsome FAILED");
        return fails == 0 ? 0 : 1;
    }

    private static void ExpectFailure(string what, string wantSubstring, Action act, ref int fails)
    {
        try
        {
            act();
            Console.WriteLine($"[FAIL] {what}  expected an exception, none thrown");
            fails++;
        }
        catch (InvalidOperationException e) when (e.Message.Contains(wantSubstring))
        {
            Console.WriteLine($"[ OK ] {what}  \"{e.Message}\"");
        }
        catch (Exception e)
        {
            Console.WriteLine($"[FAIL] {what}  {e.GetType().Name}: \"{e.Message}\"");
            Console.WriteLine($"       expected an InvalidOperationException containing \"{wantSubstring}\"");
            fails++;
        }
    }
}
