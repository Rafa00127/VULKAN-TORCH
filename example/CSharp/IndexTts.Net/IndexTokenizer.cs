using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 text tokenizer: the whisper-style multilingual BPE plus the model's ~1680
/// special tokens. Ported from indextts/utils/tokenizer.py.
///
/// The special-token ORDER matters: their ids are assigned sequentially starting right
/// after the mergeable ranks, so the list below must stay in the reference's order.
///
/// This is only the tokenizer. The text normalization / pronunciation front-end is NOT
/// here — feed already-normalized text (see the README).
/// </summary>
public sealed class IndexTokenizer
{
    /// <summary>The 106 languages of LANGUAGES. Only the first <see cref="NumLanguages"/>
    /// get a &lt;|lang|&gt; special token; the rest are embedding-only.</summary>
    private static readonly string[] Languages =
    {
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
        "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
    };

    private const int NumLanguages = 99;
    /// <summary>Index of "common" — the fallback language.</summary>
    public const int CommonLangId = 105;

    private static readonly string[] AudioEvents =
    {
        "ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM",
        "Laughter", "/Laughter", "Applause", "/Applause",
    };
    private static readonly string[] Emotions = { "HAPPY", "SAD", "ANGRY", "NEUTRAL" };
    private static readonly string[] Tasks =
        { "translate", "transcribe", "startoflm", "startofprev", "nospeech", "notimestamps" };
    private static readonly string[] TtsVocal =
        { "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H" };

    // tiktoken's pre-tokenization pattern (indextts/utils/tokenizer.py pat_str).
    private const string PatStr =
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    private readonly TiktokenTokenizer _tok;

    /// <summary>The vocab file has one entry whose base64 token is a lone '=' — Python's
    /// base64.b64decode maps that to the EMPTY byte string (an inert token that still holds a
    /// rank slot, which is why <see cref="CountRanks"/> counts it), but ML.Tokenizers rejects
    /// the whole file over it. Swap it for 0xC0 0x80 — an overlong UTF-8 encoding that cannot
    /// occur in any .NET string, so the token stays just as unreachable and the rank count is
    /// unchanged.</summary>
    private static byte[] ReadVocabPatched(string vocabPath)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var line in File.ReadLines(vocabPath))
        {
            if (line.Length == 0) continue;
            var fields = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            sb.Append(fields[0] == "=" ? "wIA=" : fields[0]).Append(' ').Append(fields[1]).Append((char)10);
        }
        return System.Text.Encoding.UTF8.GetBytes(sb.ToString());
    }

    public IndexTokenizer(string vocabPath)
    {
        var specials = BuildSpecialTokens(CountRanks(vocabPath));
        var pre = new RegexPreTokenizer(new Regex(PatStr, RegexOptions.Compiled), specials);
        _tok = TiktokenTokenizer.Create(new MemoryStream(ReadVocabPatched(vocabPath)),
                                        pre, normalizer: null, specials, cacheSize: 100_000);
    }

    /// <summary>&lt;|endoftext|&gt;, &lt;|startoftranscript|&gt;, then 99 languages, 11 audio
    /// events, 4 emotions, 6 tasks, 30 SPECIAL_TOKENs, 7 TTS vocals, 13 TTS/SP, and the
    /// 1501 timestamps &lt;|0.00|&gt;..&lt;|30.00|&gt; in 0.02 steps.</summary>
    private static int CountRanks(string vocabPath)
    {
        int n = 0;
        foreach (var line in File.ReadLines(vocabPath)) if (line.Length > 0) n++;
        return n;
    }

    private static Dictionary<string, int> BuildSpecialTokens(int baseId)
    {
        var d = new Dictionary<string, int>();
        void Add(string s) => d[s] = baseId + d.Count;

        Add("<|endoftext|>");
        Add("<|startoftranscript|>");
        for (int i = 0; i < NumLanguages; i++) Add($"<|{Languages[i]}|>");
        foreach (var e in AudioEvents) Add($"<|{e}|>");
        foreach (var e in Emotions) Add($"<|{e}|>");
        foreach (var t in Tasks) Add($"<|{t}|>");
        for (int i = 1; i <= 30; i++) Add($"<|SPECIAL_TOKEN_{i}|>");
        foreach (var v in TtsVocal) Add($"<|{v}|>");
        for (int i = 1; i <= 13; i++) Add($"<|TTS/SP{i:00}|>");
        for (int i = 0; i <= 1500; i++) Add($"<|{i / 50}.{(i * 2) % 100:00}|>");
        return d;
    }

    /// <summary>LANGUAGE_DICT lookup: lowercase, fall back to "common".</summary>
    public static int LangToToken(string lang)
    {
        int i = Array.IndexOf(Languages, lang.ToLowerInvariant());
        return i < 0 ? CommonLangId : i;
    }

    /// <summary>The prefix the reference prepends to every text segment: "&lt;|lang|&gt; ".</summary>
    public static string LangPrefix(string lang) => $"<|{lang.ToLowerInvariant()}|> ";

    /// <summary>Text -> token ids (special tokens recognised).</summary>
    public int[] Encode(string text) => _tok.EncodeToIds(text).ToArray();
}
