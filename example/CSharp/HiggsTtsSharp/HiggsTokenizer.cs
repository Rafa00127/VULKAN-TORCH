using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;
using VulkanTorch;

namespace HiggsTts;

/// <summary>
/// BPE tokenizer for HiggsTTS. Validated to produce identical ids to the Python
/// `tokenizers` library. Can be built either from an HF <c>tokenizer.json</c> or,
/// more conveniently, straight from the vocab + merges embedded in the model GGUF
/// (see <see cref="FromGguf"/>).
/// </summary>
public sealed class HiggsTokenizer
{
    private readonly BpeTokenizer _tok;
    public IReadOnlyDictionary<string, int> SpecialTokens { get; }

    // The GGUF stores only the base vocab + merges; the 84 added/special tokens
    // (ids 151643.., incl. <|tts|> / <|ref_audio|> / <|emotion:*|> / <|style:*|>)
    // and the pre-tokenizer splitter are NOT in the GGUF, so they live here — copied
    // verbatim from the official tokenizer.json.
    private const string SplitPattern =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    private static readonly (string Content, int Id)[] Specials =
    {
        new("<|endoftext|>", 151643),
        new("<|im_start|>", 151644),
        new("<|im_end|>", 151645),
        new("<|object_ref_start|>", 151646),
        new("<|object_ref_end|>", 151647),
        new("<|box_start|>", 151648),
        new("<|box_end|>", 151649),
        new("<|quad_start|>", 151650),
        new("<|quad_end|>", 151651),
        new("<|vision_start|>", 151652),
        new("<|vision_end|>", 151653),
        new("<|vision_pad|>", 151654),
        new("<|image_pad|>", 151655),
        new("<|video_pad|>", 151656),
        new("<tool_call>", 151657),
        new("</tool_call>", 151658),
        new("<|fim_prefix|>", 151659),
        new("<|fim_middle|>", 151660),
        new("<|fim_suffix|>", 151661),
        new("<|fim_pad|>", 151662),
        new("<|repo_name|>", 151663),
        new("<|file_sep|>", 151664),
        new("<|asr|>", 151665),
        new("<|streaming_asr|>", 151666),
        new("<|tts|>", 151667),
        new("<|streaming_tts|>", 151668),
        new("<|audio_cont_txt|>", 151669),
        new("<|audio|>", 151670),
        new("<|audio_end|>", 151671),
        new("<|text|>", 151672),
        new("<|text_end|>", 151673),
        new("<|eoc|>", 151674),
        new("<|user|>", 151675),
        new("<|assistant|>", 151676),
        new("<|system|>", 151677),
        new("<|await_audio|>", 151678),
        new("<|ref_audio|>", 151679),
        new("<|ref_text|>", 151680),
        new("<|emotion:elation|>", 151681),
        new("<|emotion:amusement|>", 151682),
        new("<|emotion:enthusiasm|>", 151683),
        new("<|emotion:determination|>", 151684),
        new("<|emotion:pride|>", 151685),
        new("<|emotion:contentment|>", 151686),
        new("<|emotion:affection|>", 151687),
        new("<|emotion:relief|>", 151688),
        new("<|emotion:contemplation|>", 151689),
        new("<|emotion:confusion|>", 151690),
        new("<|emotion:surprise|>", 151691),
        new("<|emotion:awe|>", 151692),
        new("<|emotion:longing|>", 151693),
        new("<|emotion:arousal|>", 151694),
        new("<|emotion:anger|>", 151695),
        new("<|emotion:fear|>", 151696),
        new("<|emotion:disgust|>", 151697),
        new("<|emotion:bitterness|>", 151698),
        new("<|emotion:sadness|>", 151699),
        new("<|emotion:shame|>", 151700),
        new("<|emotion:helplessness|>", 151701),
        new("<|env:music|>", 151702),
        new("<|env:noise|>", 151703),
        new("<|style:singing|>", 151704),
        new("<|style:shouting|>", 151705),
        new("<|style:whispering|>", 151706),
        new("<|sfx:cough|>", 151707),
        new("<|sfx:laughter|>", 151708),
        new("<|sfx:crying|>", 151709),
        new("<|sfx:screaming|>", 151710),
        new("<|sfx:burping|>", 151711),
        new("<|sfx:humming|>", 151712),
        new("<|sfx:sigh|>", 151713),
        new("<|sfx:sniff|>", 151714),
        new("<|sfx:sneeze|>", 151715),
        new("<|prosody:speed_very_slow|>", 151716),
        new("<|prosody:speed_slow|>", 151717),
        new("<|prosody:speed_fast|>", 151718),
        new("<|prosody:speed_very_fast|>", 151719),
        new("<|prosody:pitch_low|>", 151720),
        new("<|prosody:pitch_high|>", 151721),
        new("<|prosody:pause|>", 151722),
        new("<|prosody:long_pause|>", 151723),
        new("<|chatml|>", 151724),
        new("<|prosody:expressive_high|>", 151725),
        new("<|prosody:expressive_low|>", 151726),
    };

    private HiggsTokenizer(BpeTokenizer tok, Dictionary<string, int> special)
    {
        _tok = tok;
        SpecialTokens = special;
    }

    /// <summary>Build from an HF <c>tokenizer.json</c> (vocab, merges, specials and splitter all read from it).</summary>
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

        // the JSON's pre_tokenizer is a Sequence ending in the GPT-2 regex splitter
        string pattern = null!;
        foreach (var pt in root.GetProperty("pre_tokenizer").GetProperty("pretokenizers").EnumerateArray())
            if (pt.TryGetProperty("pattern", out var p) && p.TryGetProperty("Regex", out var rx))
                pattern = rx.GetString()!;

        _tok = Build(vocab, merges, special, pattern);
        SpecialTokens = special;
    }

    /// <summary>
    /// Build from the vocab + merges embedded in a Higgs model GGUF. The GGUF does not
    /// carry the added/special tokens or the splitter, so those come from the constants
    /// above. Removes the need for a separate tokenizer.json.
    /// </summary>
    public static HiggsTokenizer FromGguf(string ggufPath)
    {
        var meta = GgufMetadata.Read(ggufPath, "tokenizer.ggml.tokens", "tokenizer.ggml.merges");
        if (meta.TryGetValue("tokenizer.ggml.tokens", out var tv) is false || tv is not string[] tokens)
            throw new InvalidDataException($"GGUF has no tokenizer.ggml.tokens: {ggufPath}");
        if (meta.TryGetValue("tokenizer.ggml.merges", out var mv) is false || mv is not string[] mergeArr)
            throw new InvalidDataException($"GGUF has no tokenizer.ggml.merges: {ggufPath}");

        var vocab = new Dictionary<string, int>(tokens.Length);
        for (int i = 0; i < tokens.Length; i++) vocab[tokens[i]] = i;
        var special = Specials.ToDictionary(s => s.Content, s => s.Id);
        var tok = Build(vocab, mergeArr.ToList(), special, SplitPattern);
        return new HiggsTokenizer(tok, special);
    }

    private static BpeTokenizer Build(Dictionary<string, int> vocab, List<string> merges,
                                      Dictionary<string, int> special, string pattern)
        => BpeTokenizer.Create(new BpeOptions(vocab)
        {
            Merges = merges,
            ByteLevel = true,
            SpecialTokens = special,
            PreTokenizer = new RegexPreTokenizer(new Regex(pattern), special),
        });

    /// <summary>Encode text without adding any special tokens (matches tokenizers add_special_tokens=False).
    /// NFC-normalises first — the HF tokenizer this replaces sets <c>normalizer: NFC</c>, so é as
    /// one codepoint (NFC) and as e+combining-accent (NFD) must give the same ids.</summary>
    public int[] Encode(string text) => _tok.EncodeToIds(text.Normalize(NormalizationForm.FormC)).ToArray();
}
