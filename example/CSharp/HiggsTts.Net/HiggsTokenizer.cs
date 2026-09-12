using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace HiggsTts;

/// <summary>
/// BPE tokenizer for the HiggsTTS tokenizer.json, built on Microsoft.ML.Tokenizers.
/// Validated to produce identical ids to the Python `tokenizers` library.
/// </summary>
public sealed class HiggsTokenizer
{
    private readonly BpeTokenizer _tok;
    public IReadOnlyDictionary<string, int> SpecialTokens { get; }

    public HiggsTokenizer(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root = doc.RootElement;

        var vocab = new Dictionary<string, int>();
        foreach (var kv in root.GetProperty("model").GetProperty("vocab").EnumerateObject())
            vocab[kv.Name] = kv.Value.GetInt32();

        var merges = root.GetProperty("model").GetProperty("merges").EnumerateArray()
            .Select(m => m.ValueKind == JsonValueKind.Array
                ? m[0].GetString() + " " + m[1].GetString()
                : m.GetString()!)
            .ToList();

        var special = new Dictionary<string, int>();
        foreach (var at in root.GetProperty("added_tokens").EnumerateArray())
            special[at.GetProperty("content").GetString()!] = at.GetProperty("id").GetInt32();
        SpecialTokens = special;

        // the JSON's pre_tokenizer is a Sequence ending in the GPT-2 regex splitter
        string pattern = null!;
        foreach (var pt in root.GetProperty("pre_tokenizer").GetProperty("pretokenizers").EnumerateArray())
            if (pt.TryGetProperty("pattern", out var p) && p.TryGetProperty("Regex", out var rx))
                pattern = rx.GetString()!;

        var opts = new BpeOptions(vocab)
        {
            Merges = merges,
            ByteLevel = true,
            SpecialTokens = special,
            PreTokenizer = new RegexPreTokenizer(new Regex(pattern), special),
        };
        _tok = BpeTokenizer.Create(opts);
    }

    /// <summary>Encode text without adding any special tokens (matches tokenizers add_special_tokens=False).</summary>
    public int[] Encode(string text) => _tok.EncodeToIds(text).ToArray();
}
