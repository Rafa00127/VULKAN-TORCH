using System;
using System.Collections.Generic;
using VulkanTorch;

namespace IndexTts;

/// <summary>
/// Weight accessor over the IndexTTS 2.5 GGUF (built by
/// tools/convert/convert_index_tts2_to_gguf.py). Names are the original checkpoint keys
/// prefixed by module (`gpt.`, `codec.`, `s2mel.`, `w2v.`, `campplus.`, `bigvgan.`),
/// so the ported code reads them straight out of the shared GgufFile loader.
/// </summary>
public sealed class GgufWeights : IDisposable
{
    private readonly Dictionary<string, Tensor> _t = new();
    public GgufFile File { get; }

    public GgufWeights(string path, Device dev, string[] prefixes, ulong arenaBytes = 4UL << 30)
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
            : throw new KeyNotFoundException($"GgufWeights: no tensor '{name}'");

    public bool Has(string name) => _t.ContainsKey(name);

    public int Count => _t.Count;

    public void Dispose() => File.Dispose();
}
