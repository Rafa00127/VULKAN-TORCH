using System;
using System.Collections.Generic;
using Minitorch;

namespace HiggsTts;

/// <summary>
/// Loads a subset of the HiggsTTS GGUF into device-resident Memory, keyed by name.
/// Mirrors higgstts_py/weights.py: only tensors whose name starts with one of the
/// given prefixes are copied, so the decode stage never touches the backbone.
/// </summary>
public sealed class HiggsWeights : IDisposable
{
    // decode: RVQ quantizers + fusion FC + DAC acoustic decoder.
    public static readonly string[] DecodePrefixes = { "codec.ac_dec", "codec.quant.", "codec.fc" };

    // AR: Qwen3 backbone + fused embedding / head (mirrors weights.py BACKBONE_PREFIXES).
    public static readonly string[] BackbonePrefixes =
        { "blk.", "token_embd", "fused_embed", "fused_head", "output_norm" };

    private readonly Dictionary<string, Tensor> _t = new();
    public GgufFile File { get; }

    public HiggsWeights(string path, Device dev, string[] prefixes, ulong arenaBytes = 512UL << 20)
    {
        File = new GgufFile(path, dev, arenaBytes);
        foreach (var name in File.Names())
            foreach (var p in prefixes)
                if (name.StartsWith(p, StringComparison.Ordinal))
                {
                    _t[name] = File.Tensor(name);
                    break;
                }
    }

    public Tensor this[string name] =>
        _t.TryGetValue(name, out var t)
            ? t
            : throw new KeyNotFoundException($"HiggsWeights: no tensor '{name}'");

    public bool Has(string name) => _t.ContainsKey(name);

    public int Count => _t.Count;

    public void Dispose() => File.Dispose();
}
