using System;
using System.Collections.Generic;

namespace IndexTts;

/// <summary>
/// IndexTTS 2.5 text front-end: the v2.5 orchestration layer that turns raw user text into
/// per-segment token ids. Ported from audio.cpp's index_tts2/tokenizer_text.cpp
/// (encode_for_inference_v2_5), which in turn mirrors indextts/infer_v2_5.py.
///
/// Pipeline: pronunciation-annotation protection -> normalization (zh/en) or the punctuation
/// map (other languages) -> case fold -> expand &lt;word|pron&gt; annotations -> uppercase the
/// names inside &lt;|...|&gt; -> split to fit the model's token budget -> prefix each segment
/// with "&lt;|lang|&gt; " and append the trailing pad id 1.
///
/// Tokenization reuses <see cref="IndexTokenizer"/>; normalization reuses
/// <see cref="TextNormalization"/>. The &lt;|SPECIAL_TOKEN_n|&gt; spans produced by the
/// pronunciation expansion are kept atomic during splitting.
/// </summary>
public sealed class TextFrontend
{
    private readonly IndexTokenizer _tok;

    /// <summary>gpt.max_text_tokens (config.yaml) — the model's text-position capacity.</summary>
    public const int DefaultMaxTextTokens = 600;
    public const int DefaultMaxTokensPerSegment = 120;
    private const int SegmentPadTokenId = 1;

    private readonly int _maxTextTokens;

    public TextFrontend(IndexTokenizer tok, int maxTextTokens = DefaultMaxTextTokens)
    {
        _tok = tok;
        _maxTextTokens = maxTextTokens;
    }

    public sealed class Encoding
    {
        public string Lang = "";
        public string NormalizedText = "";
        public List<string> Segments = new();
        public List<int[]> SegmentTokenIds = new();
    }

    public Encoding EncodeForInference(string text, int maxTokensPerSegment = DefaultMaxTokensPerSegment,
                                       string lang = "", bool textNormalization = true)
    {
        if (maxTokensPerSegment <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxTokensPerSegment));

        string resolvedLang = TextNormalization.LowercaseAscii(lang);
        if (resolvedLang.Length == 0)
            resolvedLang = TextNormalization.ContainsHan(text) ? "zh" : "en";

        string processed = text;
        if (resolvedLang == "zh" || resolvedLang == "en")
        {
            if (textNormalization)
            {
                // Protect <word|pronunciation> from the normalizer, exactly as the official
                // TextNormalizer does inside normalize() (its digits/symbols would be rewritten).
                var (protectedText, placeholders) = ProtectPronunciationAnnotations(processed);
                protectedText = resolvedLang == "zh"
                    ? TextNormalization.NormalizeChineseText(protectedText)
                    : NormalizeEnglish(protectedText);
                processed = RestorePronunciationAnnotations(protectedText, placeholders);
            }
            else
            {
                // infer_v2_5.py line 702: the punctuation map always runs before the branch.
                processed = TextNormalization.NormalizeIndexTtsPunctuation(processed);
            }
        }
        else
        {
            // Other languages get no full TN, but the official front.normalize still applies
            // its punctuation map; Han text additionally maps "$" -> ".".
            bool han = TextNormalization.ContainsHan(processed);
            processed = TextNormalization.NormalizeIndexTtsPunctuation(processed);
            if (textNormalization && han) processed = TextNormalization.ReplaceAll(processed, "$", ".");
        }

        if (resolvedLang == "zh" || resolvedLang == "ja" || resolvedLang == "en")
            processed = TextNormalization.LowercaseAscii(processed);
        else if (resolvedLang == "es")
            processed = TextNormalization.UppercaseAscii(processed);

        processed = ApplyPronunciationAnnotations(processed);
        processed = UppercaseSpecialTokenNames(processed);

        string langPrefix = $"<|{resolvedLang}|> ";
        int prefixTokens = _tok.Encode(langPrefix).Length;
        long budget = Math.Min(maxTokensPerSegment, _maxTextTokens - 2) - prefixTokens;
        budget = Math.Max(budget, 1);

        int TokenLen(string value) => _tok.Encode(value).Length;

        var segments = new List<string>();
        if (TokenLen(processed) <= budget)
        {
            segments.Add(processed);
        }
        else
        {
            var chunks = new List<string>();
            foreach (var (piece, atomic) in SplitAtomicPieces(processed))
            {
                if (atomic) { chunks.Add(piece); continue; }
                foreach (string part in SplitAfterDelimiters(piece))
                {
                    if (TokenLen(part) <= budget) { chunks.Add(part); continue; }
                    string current = "";
                    foreach (char ch in part)
                    {
                        if (current.Length != 0 && TokenLen(current + ch) > budget)
                        {
                            chunks.Add(current);
                            current = ch.ToString();
                        }
                        else
                        {
                            current += ch;
                        }
                    }
                    if (current.Length != 0) chunks.Add(current);
                }
            }
            string acc = "";
            foreach (string chunk in chunks)
            {
                if (acc.Length != 0 && TokenLen(acc + chunk) > budget)
                {
                    segments.Add(acc);
                    acc = chunk;
                }
                else
                {
                    acc += chunk;
                }
            }
            if (acc.Length != 0) segments.Add(acc);
            if (segments.Count == 0) segments.Add(processed);
        }

        var enc = new Encoding { Lang = resolvedLang, NormalizedText = processed };
        foreach (string seg in segments)
        {
            enc.Segments.Add(seg);
            var ids = new List<int>(_tok.Encode(langPrefix + seg)) { SegmentPadTokenId };
            enc.SegmentTokenIds.Add(ids.ToArray());
        }
        return enc;
    }

    private string NormalizeEnglish(string text)
    {
        var opts = new TextNormalization.EnglishOptions
        {
            ExpandCommonContractions = true,
            SpellNumbers = true,
            IndexTtsPunctuation = true,
            UppercaseAscii = false,
            VerbalizeSymbols = true,
        };
        return TextNormalization.NormalizeEnglishText(text, opts);
    }

    // ---- pronunciation annotations ---------------------------------------------------------

    private readonly struct Annotation
    {
        public readonly int End, WordBegin, WordEnd, PronBegin, PronEnd;
        public readonly bool Ok;
        public Annotation(int end, int wb, int we, int pb, int pe)
        { End = end; WordBegin = wb; WordEnd = we; PronBegin = pb; PronEnd = pe; Ok = true; }
    }

    /// <summary>Matches &lt;([^|&gt;\n]+)\|([^&gt;\n]+)&gt; anchored at <paramref name="pos"/>.</summary>
    private static Annotation MatchPronunciationAnnotation(string text, int pos)
    {
        if (text[pos] != '<') return default;
        int cursor = pos + 1;
        int wordBegin = cursor;
        while (cursor < text.Length && text[cursor] != '|' && text[cursor] != '>' && text[cursor] != '\n')
            cursor++;
        if (cursor == wordBegin || cursor >= text.Length || text[cursor] != '|') return default;
        int wordEnd = cursor;
        int pronBegin = ++cursor;
        while (cursor < text.Length && text[cursor] != '>' && text[cursor] != '\n')
            cursor++;
        if (cursor == pronBegin || cursor >= text.Length) return default;
        return new Annotation(cursor + 1, wordBegin, wordEnd, pronBegin, cursor);
    }

    /// <summary>Base-26 spreadsheet index ("a".."z", "aa"..), mirroring the reference's
    /// PRONPLACEHOLDER tag naming.</summary>
    private static string AlphaPlaceholderIndex(int n)
    {
        string s = "";
        while (true)
        {
            s = (char)('a' + n % 26) + s;
            int q = n / 26;
            if (q == 0) break;
            n = q - 1;
        }
        return s;
    }

    private static (string, List<KeyValuePair<string, string>>) ProtectPronunciationAnnotations(string text)
    {
        var sb = new System.Text.StringBuilder(text.Length);
        var placeholders = new List<KeyValuePair<string, string>>();
        int pos = 0;
        while (pos < text.Length)
        {
            var m = MatchPronunciationAnnotation(text, pos);
            if (!m.Ok) { sb.Append(text[pos++]); continue; }
            string key = "PRONPLACEHOLDER" + AlphaPlaceholderIndex(placeholders.Count) + "PRONPLACEHOLDER";
            placeholders.Add(new(key, text.Substring(pos, m.End - pos)));
            sb.Append(key);
            pos = m.End;
        }
        return (sb.ToString(), placeholders);
    }

    private static string RestorePronunciationAnnotations(string text, List<KeyValuePair<string, string>> placeholders)
    {
        foreach (var (key, original) in placeholders)
        {
            int at = 0;
            while ((at = text.IndexOf(key, at, StringComparison.Ordinal)) >= 0)
                text = text.Substring(0, at) + original + text.Substring(at + key.Length);
        }
        return text;
    }

    /// <summary>True when every codepoint is Hiragana (U+3040..U+309F) or every one is
    /// Katakana (U+30A0..U+30FF).</summary>
    private static bool IsKana(string text)
    {
        if (text.Length == 0) return false;
        bool allHiragana = true, allKatakana = true;
        foreach (char c in text)
        {
            if (c < '぀' || c > 'ゟ') allHiragana = false;
            if (c < '゠' || c > 'ヿ') allKatakana = false;
        }
        return allHiragana || allKatakana;
    }

    /// <summary>Expands &lt;word|pron&gt; to special-token-wrapped PRON, or an inlined kana PRON.
    /// The enclosed pronunciation is uppercased.</summary>
    private static string ApplyPronunciationAnnotations(string text)
    {
        var sb = new System.Text.StringBuilder(text.Length);
        int pos = 0;
        while (pos < text.Length)
        {
            var m = MatchPronunciationAnnotation(text, pos);
            if (!m.Ok) { sb.Append(text[pos++]); continue; }
            string word = text.Substring(m.WordBegin, m.WordEnd - m.WordBegin);
            string pron = TextNormalization.UppercaseAscii(text.Substring(m.PronBegin, m.PronEnd - m.PronBegin));
            if (IsKana(pron))
            {
                sb.Append(' ').Append(pron).Append(' ');
            }
            else
            {
                string wrapper = TextNormalization.ContainsHan(word) ? "<|SPECIAL_TOKEN_2|>" : "<|SPECIAL_TOKEN_1|>";
                sb.Append(wrapper).Append(pron).Append(wrapper);
            }
            pos = m.End;
        }
        return sb.ToString();
    }

    /// <summary>Uppercases the name inside every &lt;|...|&gt; marker.</summary>
    private static string UppercaseSpecialTokenNames(string text)
    {
        var sb = new System.Text.StringBuilder(text.Length);
        int pos = 0;
        while (pos < text.Length)
        {
            if (text[pos] != '<' || pos + 1 >= text.Length || text[pos + 1] != '|')
            {
                sb.Append(text[pos++]);
                continue;
            }
            int cursor = pos + 2;
            while (cursor < text.Length && text[cursor] != '|') cursor++;
            if (cursor == pos + 2 || cursor + 1 >= text.Length || text[cursor + 1] != '>')
            {
                sb.Append(text[pos++]);
                continue;
            }
            sb.Append("<|").Append(TextNormalization.UppercaseAscii(text.Substring(pos + 2, cursor - (pos + 2)))).Append("|>");
            pos = cursor + 2;
        }
        return sb.ToString();
    }

    // ---- token-budget segmentation ----------------------------------------------------------

    private static bool IsSegmentDelimiter(char c) => c switch
    {
        ',' or '.' or '!' or '?' or ';' or ':' or '\n' => true,
        '，' or '。' or '！' or '？' or '、' or '；' or '：' => true,
        _ => false,
    };

    /// <summary>Split after each delimiter (the delimiter stays with the preceding part).</summary>
    private static List<string> SplitAfterDelimiters(string piece)
    {
        var parts = new List<string>();
        var current = new System.Text.StringBuilder();
        foreach (char c in piece)
        {
            current.Append(c);
            if (IsSegmentDelimiter(c))
            {
                parts.Add(current.ToString());
                current.Clear();
            }
        }
        if (current.Length != 0) parts.Add(current.ToString());
        return parts;
    }

    /// <summary>Matches "&lt;|SPECIAL_TOKEN_&lt;digits&gt;|&gt;" at <paramref name="pos"/>; returns its
    /// length or 0.</summary>
    private static int MatchSpecialTokenMarker(string text, int pos)
    {
        const string prefix = "<|SPECIAL_TOKEN_";
        if (pos + prefix.Length > text.Length) return 0;
        if (string.CompareOrdinal(text, pos, prefix, 0, prefix.Length) != 0) return 0;
        int cursor = pos + prefix.Length;
        int digitsBegin = cursor;
        while (cursor < text.Length && text[cursor] >= '0' && text[cursor] <= '9') cursor++;
        if (cursor == digitsBegin || cursor + 1 >= text.Length || text[cursor] != '|' || text[cursor + 1] != '>')
            return 0;
        return cursor + 2 - pos;
    }

    /// <summary>Splits into (piece, isAtomic) where an atomic piece is a
    /// &lt;|SPECIAL_TOKEN_n|&gt;...&lt;|SPECIAL_TOKEN_n|&gt; span kept whole during segmentation.</summary>
    private static List<(string Piece, bool Atomic)> SplitAtomicPieces(string text)
    {
        var pieces = new List<(string, bool)>();
        int pos = 0;
        while (pos < text.Length)
        {
            int opener = -1, openerLen = 0;
            for (int i = pos; i < text.Length; i++)
            {
                int len = MatchSpecialTokenMarker(text, i);
                if (len > 0) { opener = i; openerLen = len; break; }
            }
            if (opener < 0) break;
            int closer = -1, closerLen = 0;
            for (int i = opener + openerLen; i < text.Length; i++)
            {
                int len = MatchSpecialTokenMarker(text, i);
                if (len > 0) { closer = i; closerLen = len; break; }
            }
            if (closer < 0) break;
            if (opener > pos) pieces.Add((text.Substring(pos, opener - pos), false));
            pieces.Add((text.Substring(opener, closer + closerLen - opener), true));
            pos = closer + closerLen;
        }
        if (pos < text.Length) pieces.Add((text.Substring(pos), false));
        return pieces;
    }
}
