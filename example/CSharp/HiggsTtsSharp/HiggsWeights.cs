using System;
using System.Collections.Generic;
using VulkanTorch;

namespace HiggsTts;

/// <summary>
/// Loads a subset of the HiggsTTS GGUF into device-resident Memory, keyed by name.
/// Mirrors higgstts_vt/weights.py: only tensors whose name starts with one of the
/// given prefixes are copied, so each stage only pays for what it uses.
/// </summary>
public sealed class HiggsWeights : IDisposable
{
    // decode: RVQ quantizers + fusion FC + DAC acoustic decoder.
    public static readonly string[] DecodePrefixes = { "codec.ac_dec", "codec.quant.", "codec.fc" };

    // AR: Qwen3 backbone + fused embedding / head (mirrors weights.py BACKBONE_PREFIXES).
    public static readonly string[] BackbonePrefixes =
        { "blk.", "token_embd", "fused_embed", "fused_head", "output_norm" };

    // encode_ref: HuBERT feature extractor + semantic encoder + DAC acoustic encoder
    // + RVQ + fusion FC (mirrors weights.py PREFILL_PREFIXES).
    public static readonly string[] PrefillPrefixes =
        { "codec.ac_enc", "codec.ac_dec", "codec.enc_sem", "codec.sem.", "codec.quant.", "codec.fc" };

    private readonly Dictionary<string, Tensor> _t = new();
    private readonly Memory _mem;              // holds computed weights (e.g. fused PCE conv)
    public GgufFile File { get; }

    public HiggsWeights(string path, Device dev, string[] prefixes, ulong arenaBytes = 512UL << 20)
    {
        File = new GgufFile(path, dev, arenaBytes);
        _mem = new Memory(dev, 64UL << 20);
        foreach (var name in File.Names())
            foreach (var p in prefixes)
                if (name.StartsWith(p, StringComparison.Ordinal))
                {
                    _t[name] = File.Tensor(name);
                    break;
                }
    }

    /// <summary>Store a computed weight (PT-order shape, F32) in device memory.</summary>
    public Tensor Put(string name, long[] shape, float[] data)
    {
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        var t = _mem.Tensor(shape, Ops.F32, bytes);
        _t[name] = t;
        return t;
    }

    public Tensor this[string name] =>
        _t.TryGetValue(name, out var t)
            ? t
            : throw new KeyNotFoundException($"HiggsWeights: no tensor '{name}'");

    public bool Has(string name) => _t.ContainsKey(name);

    public int Count => _t.Count;

    public void Dispose()
    {
        _mem.Dispose();
        File.Dispose();
    }
}
