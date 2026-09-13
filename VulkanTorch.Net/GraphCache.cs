using System;
using System.Collections.Generic;

namespace VulkanTorch;

/// <summary>
/// Policy helpers for a bucketed <b>graph cache</b>: a decode step is captured once per
/// window size and then replayed. Replaying only works if nothing position-dependent is
/// baked into the graph, so the attention window is pinned to a bucket
/// (<see cref="PickBucket"/>) and the causal mask is supplied at runtime
/// (<see cref="CausalMask"/>) -- the position itself goes through a runtime <c>set_rows</c>.
///
/// The KV storage itself lives in <see cref="KvCache"/>.
/// </summary>
public static class GraphCache
{
    public static readonly int[] Buckets = { 256, 512, 1024, 2048, 4096 };

    /// <summary>Smallest bucket &gt;= <paramref name="need"/>, or 0 if it exceeds every bucket.</summary>
    public static int PickBucket(int need)
    {
        foreach (var b in Buckets) if (b >= need) return b;
        return 0;
    }

    /// <summary>
    /// Causal F16 mask PT [t, lk]. For a single-token decode step pass t = 1 and
    /// <paramref name="nPast"/> = the index of the token being written.
    /// </summary>
    public static float[] CausalMask(int t, int lk, int nPast)
    {
        var m = new float[(long)t * lk];
        for (int q = 0; q < t; q++)
        {
            int qi = t == 1 ? nPast : q;
            for (int k = 0; k < lk; k++)
                m[(long)q * lk + k] = k <= qi ? 0f : float.NegativeInfinity;
        }
        return m;
    }
}

/// <summary>
/// A captured single-step decode graph: the graph itself, the input tensors the model has
/// to refresh each step (<typeparamref name="TInputs"/>, defined by the model), and the
/// output tensor to read back.
/// </summary>
public sealed class StepGraph<TInputs> : IDisposable where TInputs : class
{
    public Graph G { get; }
    public TInputs In { get; }
    public Tensor Output { get; }

    /// <summary>The window size (bucket) this graph was captured for.</summary>
    public int Window { get; }

    public StepGraph(Graph g, TInputs inputs, Tensor output, int window)
    {
        G = g; In = inputs; Output = output; Window = window;
    }

    /// <summary>Refresh one of this graph's inputs before a replay.</summary>
    public void Set(Tensor tensor, byte[] data) => G.SetInput(tensor, data);
    public void Set(Tensor tensor, int[] data) => G.SetInput(tensor, data);
    public void Set(Tensor tensor, float[] data) => G.SetInput(tensor, data);

    public float[] Read() => Output.ToFloats(G);

    public void Dispose() => G.Dispose();
}

/// <summary>
/// Runs a decode step through <b>one captured graph per window bucket</b>, replaying it
/// instead of re-emitting the graph every token. The model supplies only the build function
/// (its graph + input tensors) and a callback that refreshes the inputs; the bucket choice,
/// build-once-per-bucket, the fresh-graph fallback and compute/read all live here.
///
/// Requires the graph to be position-independent: the position goes through a runtime
/// <c>set_rows</c> and the mask is a runtime input (see <see cref="CausalMask"/>).
/// </summary>
public sealed class BucketedReplay<TInputs> : IDisposable where TInputs : class
{
    private readonly Func<int, StepGraph<TInputs>> _build;
    private readonly Dictionary<int, StepGraph<TInputs>> _byBucket = new();

    /// <summary>false forces a fresh graph every step (correct, slower) — the A/B lever.</summary>
    public bool Replay { get; set; } = true;

    /// <summary>Use the alloc-once static replay path (see <see cref="Graph.AllocStatic"/>).</summary>
    public bool Static { get; set; } = true;

    public BucketedReplay(Func<int, StepGraph<TInputs>> build) => _build = build;

    /// <summary>The captured graph for <paramref name="nPast"/>'s bucket, building it on first use
    /// (and rebuilding it when <see cref="Replay"/> is off).</summary>
    public StepGraph<TInputs> Get(int nPast)
    {
        int lk = GraphCache.PickBucket(nPast + 1);
        if (lk == 0)
            throw new InvalidOperationException($"position {nPast} exceeds the largest window bucket");

        if (!_byBucket.TryGetValue(lk, out var sg))
            _byBucket[lk] = sg = Build(lk);
        else if (!Replay)
            _byBucket[lk] = sg = Rebuild(lk, sg);
        return sg;
    }

    private StepGraph<TInputs> Build(int lk)
    {
        var sg = _build(lk);
        if (Static) sg.G.AllocStatic();
        return sg;
    }

    /// <summary>One step at position <paramref name="nPast"/>; returns the read-back output.</summary>
    public float[] Run(int nPast, Action<StepGraph<TInputs>> setInputs)
    {
        var sg = Get(nPast);
        setInputs(sg);
        if (Static) sg.G.ComputeStatic();
        else sg.G.Compute();
        return sg.Read();
    }

    private StepGraph<TInputs> Rebuild(int lk, StepGraph<TInputs> stale)
    {
        stale.Dispose();
        return Build(lk);
    }

    public void Dispose()
    {
        foreach (var s in _byBucket.Values) s.Dispose();
        _byBucket.Clear();
    }
}
