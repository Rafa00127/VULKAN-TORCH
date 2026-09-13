using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace IndexTts;

/// <summary>
/// Text normalization front-end: the wetext-style English/Chinese normalizers plus the
/// shared IndexTTS punctuation map. Ported from engine::text (text_normalization.cpp /
/// chinese_normalization.cpp).
///
/// Pure string transforms only — no I/O, no logging.
/// </summary>
public static class TextNormalization
{
    /// <summary>Mirror of EnglishTextNormalizationOptions. The parameterless constructor
    /// reproduces the C++ default member initializers (spell_numbers = true).</summary>
    public struct EnglishOptions
    {
        public bool ExpandCommonContractions;
        public bool SpellNumbers;
        public bool IndexTtsPunctuation;
        public bool UppercaseAscii;
        public bool VerbalizeSymbols;

        public EnglishOptions()
        {
            ExpandCommonContractions = false;
            SpellNumbers = true;
            IndexTtsPunctuation = false;
            UppercaseAscii = false;
            VerbalizeSymbols = false;
        }
    }

    // std::regex defaults to the ECMAScript grammar, whose \b, \d and \s are ASCII-only.
    // .NET's Unicode-aware \b / \d / \w would not treat a digit next to CJK ("2024年") as a
    // word boundary, so every reference \b is spelled out as an ASCII lookaround and every
    // \d as [0-9]; the semantics are then identical in both engines.
    private const RegexOptions Rx = RegexOptions.Compiled | RegexOptions.CultureInvariant;
    private const RegexOptions Rxi = Rx | RegexOptions.IgnoreCase;

    private const string AsciiWord = "A-Za-z0-9_";
    private const string Wb = "(?<![" + AsciiWord + "])";   // \b before a word char
    private const string Wa = "(?![" + AsciiWord + "])";    // \b after a word char
    private const string Sp = "[ \\t\\r\\n\\f\\v]";         // C++ \s in the default locale

    // ---- English regexes -------------------------------------------------------------------

    private static readonly Regex UrlRegex = new(
        Wb + @"(https?)://([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)", Rxi);
    private static readonly Regex DomainDotRegex = new(@"\.", Rx);
    private static readonly Regex PhoneRegex = new(
        Wb + @"([0-9]{3})-([0-9]{4})-([0-9]{4})" + Wa, Rx);
    private static readonly Regex TimeRegex = new(Wb + @"([0-9]{1,2}):00" + Wa, Rx);
    private static readonly Regex NumDateRegex = new(
        Wb + @"([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})" + Wa, Rx);
    private static readonly Regex CurrencyRegex = new(@"\$([0-9]+(?:\.[0-9]+)?)", Rx);
    private static readonly Regex NegativeRegex = new(@"(^|[^A-Za-z0-9])-([0-9]+)", Rx);
    private static readonly Regex PercentRegex = new(Wb + @"([0-9]+(?:\.[0-9]+)?)%", Rx);
    private static readonly Regex DecimalInnerRegex = new(@"([0-9]+)\.([0-9]+)", Rx);
    private static readonly Regex FractionRegex = new(Wb + @"([0-9]+)/([0-9]+)" + Wa, Rx);
    private static readonly Regex DecimalRegex = new(Wb + @"([0-9]+)\.([0-9]+)" + Wa, Rx);
    private static readonly Regex OrdinalRegex = new(
        Wb + @"([0-9]+)(st|nd|rd|th)" + Wa, Rxi);
    private static readonly Regex LetterDigitRegex = new(@"([A-Za-z])([0-9])", Rx);
    private static readonly Regex DigitLetterRegex = new(@"([0-9])([A-Za-z])", Rx);
    private static readonly Regex CardinalRegex = new(Wb + @"([0-9]+)" + Wa, Rx);
    private static readonly Regex EnglishDateRegex = new(
        Wb + @"(January|February|March|April|May|June|July|August|September|October|November|December)" +
        Sp + @"+([0-9]{1,2})(?:st|nd|rd|th)?(?:," + Sp + @"*([0-9]{4}))?", Rxi);
    private static readonly Regex ContractionRegex = new(
        @"(what|where|who|which|how|t?here|it|s?he|that|this)'s", Rxi);

    // ---- Chinese regexes -------------------------------------------------------------------

    private static readonly Regex CnDateSepRegex = new(@"([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})", Rx);
    private static readonly Regex CnPercentRegex = new(@"([0-9]+(?:\.[0-9]+)?)%", Rx);
    private static readonly Regex CnFractionRegex = new(@"([0-9]+)/([0-9]+)", Rx);
    private static readonly Regex CnTimeRegex = new(@"([0-9]{1,2}):00", Rx);
    private static readonly Regex CnPhoneRegex = new(@"([0-9]{3})-([0-9]{4})-([0-9]{4})", Rx);
    // [\xE4-\xE9][\x80-\xBF][\x80-\xBF] in the reference's byte regex == codepoints U+4000..U+9FFF.
    private static readonly Regex NameRegex = new(
        @"([䀀-鿿]+(?:[-·—][䀀-鿿]+){1,2})", Rx);
    private static readonly Regex TechTermRegex = new(@"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+", Rx);
    private static readonly Regex PinyinToneRegex = new(@"[A-Za-zÜüvV]+[1-5]", Rx);
    private static readonly Regex HRestoreRegex = new(Sp + @"*<H>" + Sp + "*", Rx);

    // ---- tables ----------------------------------------------------------------------------

    private static readonly string[] EnglishMonthsLower =
    {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    };

    private static readonly (string From, string To)[] EnglishPunctuationMap =
    {
        ("：", ","), ("；", ","), (";", ","), ("，", ","), ("。", "."), ("！", "!"),
        ("？", "?"), ("\n", " "), ("·", "-"), ("、", ","), ("...", "…"), (",,,", "…"),
        ("，，，", "…"), ("……", "…"), ("“", "'"), ("”", "'"), ("\"", "'"), ("‘", "'"),
        ("’", "'"), ("（", "'"), ("）", "'"), ("(", "'"), (")", "'"), ("《", "'"),
        ("》", "'"), ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"), ("—", "-"),
        ("～", "-"), ("~", "-"), ("「", "'"), ("」", "'"), (":", ","),
    };

    private static readonly (string From, string To)[] ChinesePunctuationMap =
    {
        ("$", "."), ("：", ","), ("；", ","), (";", ","), ("，", ","), ("。", "."),
        ("！", "!"), ("？", "?"), ("\n", " "), ("·", "-"), ("、", ","), ("...", "…"),
        (",,,", "…"), ("，，，", "…"), ("……", "…"), ("“", "'"), ("”", "'"), ("\"", "'"),
        ("‘", "'"), ("’", "'"), ("（", "'"), ("）", "'"), ("(", "'"), (")", "'"),
        ("《", "'"), ("》", "'"), ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"),
        ("—", "-"), ("～", "-"), ("~", "-"), ("「", "'"), ("」", "'"), (":", ","),
    };

    private static readonly string[] ChineseMeasureWords =
    {
        "个", "位", "只", "条", "张", "本", "件", "次", "台", "辆", "块", "颗",
        "杯", "瓶", "碗", "套", "把", "支", "根", "座", "间", "群", "双", "对",
        "种", "类", "项", "层", "栋", "扇", "面", "盏", "袋", "箱", "包", "桶",
    };

    private static readonly string[] PinyinInitials =
    {
        "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "j", "q", "x", "z", "c", "s", "r", "y", "w",
    };

    private static readonly string[] PinyinFinals =
    {
        "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "i", "ia", "ian", "iang",
        "iao", "ie", "in", "ing", "iong", "iu", "o", "ong", "ou", "u", "ua", "uai", "uan",
        "uang", "ue", "ui", "un", "uo", "v", "van", "ve", "vn", "ng",
    };

    // ---- public API ------------------------------------------------------------------------

    public static string ReplaceAll(string text, string from, string to)
    {
        if (from.Length == 0)
        {
            return text;
        }
        int pos = text.IndexOf(from, StringComparison.Ordinal);
        while (pos >= 0)
        {
            text = text.Substring(0, pos) + to + text.Substring(pos + from.Length);
            int next = pos + to.Length;
            if (next > text.Length)
            {
                break;
            }
            pos = text.IndexOf(from, next, StringComparison.Ordinal);
        }
        return text;
    }

    public static string CollapseAsciiWhitespace(string text)
    {
        var sb = new StringBuilder(text.Length);
        bool previousSpace = false;
        foreach (char ch in text)
        {
            bool space = ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
            if (space)
            {
                if (!previousSpace)
                {
                    sb.Append(' ');
                }
            }
            else
            {
                sb.Append(ch);
            }
            previousSpace = space;
        }
        return TrimAsciiWhitespace(sb.ToString());
    }

    /// <summary>Applies the IndexTTS punctuation replacement map (、→"," , 。→"." etc.),
    /// matching the char_rep_map the official front.TextNormalizer applies to every language path.</summary>
    public static string NormalizeIndexTtsPunctuation(string text)
        => ApplyPunctuationMap(text, EnglishPunctuationMap);

    public static string NormalizeEnglishNumbers(string text)
    {
        // URLs before anything else: "https://example.com" ->
        // "https comma slash slash example dot com" (wetext en verbalizes the
        // punctuation). The domain dots must not become decimal points later.
        text = UrlRegex.Replace(text, m =>
        {
            string domain = m.Groups[2].Value;
            return m.Groups[1].Value + " comma slash slash " + DomainDotRegex.Replace(domain, " dot ");
        });
        // Telephone runs: "135-4567-8900" -> digit-by-digit groups (wetext en).
        text = PhoneRegex.Replace(text, m =>
            EnglishDigitsIndividually(m.Groups[1].Value) + ", " +
            EnglishDigitsIndividually(m.Groups[2].Value) + ", " +
            EnglishDigitsIndividually(m.Groups[3].Value));
        // o'clock times: "3:00" -> "three,oh zero" (wetext en output, comma included).
        text = TimeRegex.Replace(text, m => EnglishCardinalFromDigits(m.Groups[1].Value) + ",oh zero");
        // Numeric dates: "2024/10/01" -> "the first of october twenty twenty four".
        text = NumDateRegex.Replace(text, m =>
        {
            if (!TryParseNonNegativeI64(m.Groups[1].Value, out long year) ||
                !TryParseNonNegativeI64(m.Groups[2].Value, out long month) || month < 1 || month > 12 ||
                !TryParseNonNegativeI64(m.Groups[3].Value, out long day) || day < 1 || day > 31)
            {
                return m.Value;
            }
            return "the " + EnglishOrdinal(day) + " of " + EnglishMonthsLower[month - 1] + " " + EnglishYear((int)year);
        });
        // Currency: "$50" -> "fifty dollars".
        text = CurrencyRegex.Replace(text, m =>
        {
            string value = m.Groups[1].Value;
            int dot = value.IndexOf('.');
            string amount = dot < 0
                ? EnglishCardinalFromDigits(value)
                : EnglishCardinalFromDigits(value.Substring(0, dot)) + " point " +
                  EnglishDigitsIndividually(value.Substring(dot + 1));
            return amount + (value == "1" ? " dollar" : " dollars");
        });
        // Negative numbers: "-5 degrees" -> "negative five degrees". The minus is
        // only a sign when not attached to a word/number ("GPT-5" stays intact).
        text = NegativeRegex.Replace(text, m =>
            m.Groups[1].Value + "negative " + EnglishCardinalFromDigits(m.Groups[2].Value));
        text = NormalizeEnglishDates(text);
        text = PercentRegex.Replace(text, m =>
            DecimalInnerRegex.Replace(m.Groups[1].Value, d =>
                EnglishCardinalFromDigits(d.Groups[1].Value) + " point " +
                EnglishDigitsIndividually(d.Groups[2].Value)) + " percent");
        text = FractionRegex.Replace(text, m => EnglishFraction(m.Groups[1].Value, m.Groups[2].Value));
        text = DecimalRegex.Replace(text, m =>
            EnglishCardinalFromDigits(m.Groups[1].Value) + " point " +
            EnglishDigitsIndividually(m.Groups[2].Value));
        text = OrdinalRegex.Replace(text, m => EnglishOrdinalFromDigits(m.Groups[1].Value));
        // Split letter<->digit boundaries so digits attached to letters verbalize
        // like the official wetext English normalizer ("DS4" -> "DS four",
        // "R2D2" -> "R two D two", "4K" -> "four K"). Runs after dates/decimals/
        // ordinals so "1st", "2.5" and friends have already been expanded; the
        // standalone cardinal rule below then spells out each digit group.
        text = LetterDigitRegex.Replace(text, "$1 $2");
        text = DigitLetterRegex.Replace(text, "$1 $2");
        text = CardinalRegex.Replace(text, m => EnglishCardinalFromDigits(m.Groups[1].Value));
        return text;
    }

    public static string NormalizeEnglishText(string text, EnglishOptions options)
    {
        string outText = CollapseAsciiWhitespace(text);
        if (options.ExpandCommonContractions)
        {
            outText = ContractionRegex.Replace(outText, "$1 is");
        }
        if (options.SpellNumbers)
        {
            outText = NormalizeEnglishNumbers(outText);
        }
        if (options.VerbalizeSymbols)
        {
            // Match the official wetext English normalizer: standalone ASCII
            // symbols are verbalized ("a_b" -> "a underscore b",
            // "C++" -> "C plus plus", "a=b" -> "a equal sign b"). Runs after
            // number spelling so "50%" has already become "fifty percent".
            (string From, string To)[] symbolWords =
            {
                ("_", " underscore "),
                ("+", " plus "),
                ("=", " equal sign "),
                ("*", " asterisk "),
                ("&", " and "),
                ("#", " hash "),
                ("%", " percent "),
                ("|", " vertical bar "),
                ("~", " tilde "),
            };
            foreach (var (from, to) in symbolWords)
            {
                outText = ReplaceAll(outText, from, to);
            }
            outText = CollapseAsciiWhitespace(outText);
        }
        if (options.IndexTtsPunctuation)
        {
            outText = ApplyPunctuationMap(outText, EnglishPunctuationMap);
        }
        if (options.UppercaseAscii)
        {
            outText = UppercaseAscii(outText);
        }
        return outText;
    }

    /// <summary>
    /// Warning: this is only a lightweight kana/width/punctuation cleanup helper.
    /// It does not perform Japanese word segmentation or Kanji-to-reading conversion,
    /// so it is not ready for model frontends that require full Japanese normalization.
    /// </summary>
    public static string NormalizeJapaneseText(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (uint cp in JapaneseCodepoints(CollapseAsciiWhitespace(text)))
        {
            uint c = cp;
            switch (c)
            {
                case 0xFF01: c = 0x0021; break;
                case 0xFF1F: c = 0x003F; break;
                case 0xFF0C: c = 0x3001; break;
                case 0xFF0E: c = 0x3002; break;
                case 0xFF61: c = 0x3002; break;
                case 0xFF64: c = 0x3001; break;
                case 0xFF62: c = 0x300C; break;
                case 0xFF63: c = 0x300D; break;
                case 0x3000: c = 0x0020; break;
            }
            AppendCodepoint(sb, c);
        }
        string mapped = sb.ToString();
        mapped = ReplaceAll(mapped, "……", "…");
        mapped = ReplaceAll(mapped, "...", "…");
        mapped = ReplaceAll(mapped, "..", "…");
        return CollapseAsciiWhitespace(mapped);
    }

    public static string NormalizeChineseText(string text, bool indexTtsTarget = true)
        => indexTtsTarget ? NormalizeIndexTtsChineseText(text) : NormalizeConfucius4ChineseText(text);

    // ---- English number spelling -----------------------------------------------------------

    private static string NormalizeEnglishDates(string text)
        => EnglishDateRegex.Replace(text, m =>
        {
            string outText = m.Groups[1].Value + " " +
                (TryParseNonNegativeI64(m.Groups[2].Value, out long day)
                    ? EnglishOrdinal(day)
                    : EnglishDigitsIndividually(m.Groups[2].Value));
            if (m.Groups[3].Success)
            {
                outText += " ";
                outText += TryParseNonNegativeI64(m.Groups[3].Value, out long year)
                    ? EnglishYear((int)year)
                    : EnglishDigitsIndividually(m.Groups[3].Value);
            }
            return outText;
        });

    private static bool TryParseNonNegativeI64(string text, out long value)
    {
        value = 0;
        if (text.Length == 0)
        {
            return false;
        }
        long v = 0;
        foreach (char ch in text)
        {
            if (ch < '0' || ch > '9')
            {
                return false;
            }
            int digit = ch - '0';
            if (v > (long.MaxValue - digit) / 10)
            {
                return false;
            }
            v = v * 10 + digit;
        }
        value = v;
        return true;
    }

    private static string EnglishCardinalUnder1000(int value)
    {
        string[] units =
        {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen",
        };
        string[] tens =
        {
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        };

        if (value < 20)
        {
            return units[value];
        }
        if (value < 100)
        {
            int ten = value / 10;
            int one = value % 10;
            return one == 0 ? tens[ten] : tens[ten] + " " + units[one];
        }
        int hundred = value / 100;
        int rest = value % 100;
        return rest == 0
            ? units[hundred] + " hundred"
            : units[hundred] + " hundred " + EnglishCardinalUnder1000(rest);
    }

    private static string EnglishDigitsIndividually(string digits)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < digits.Length; ++i)
        {
            if (i != 0)
            {
                sb.Append(' ');
            }
            sb.Append(EnglishCardinalUnder1000(digits[i] - '0'));
        }
        return sb.ToString();
    }

    private static string EnglishCardinal(long value)
    {
        if (value < 0)
        {
            return "minus " + EnglishCardinal(-value);
        }
        if (value < 1000)
        {
            return EnglishCardinalUnder1000((int)value);
        }
        if (value < 1000000)
        {
            long thousands = value / 1000;
            long rest = value % 1000;
            return rest == 0
                ? EnglishCardinal(thousands) + " thousand"
                : EnglishCardinal(thousands) + " thousand " + EnglishCardinal(rest);
        }
        return value.ToString(CultureInfo.InvariantCulture);
    }

    private static string EnglishCardinalFromDigits(string digits)
        => TryParseNonNegativeI64(digits, out long value) ? EnglishCardinal(value) : EnglishDigitsIndividually(digits);

    private static string EnglishYear(int value)
    {
        if (value >= 1900 && value <= 1999)
        {
            int rest = value - 1900;
            return rest == 0 ? "nineteen hundred" : "nineteen " + EnglishCardinalUnder1000(rest);
        }
        if (value >= 2000 && value <= 2099)
        {
            int rest = value - 2000;
            if (rest == 0)
            {
                return "two thousand";
            }
            if (rest < 10)
            {
                return "two thousand " + EnglishCardinalUnder1000(rest);
            }
            return "twenty " + EnglishCardinalUnder1000(rest);
        }
        return EnglishCardinal(value);
    }

    private static string EnglishOrdinal(long value)
    {
        string[] ordinals =
        {
            "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
            "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
            "sixteenth", "seventeenth", "eighteenth", "nineteenth",
        };
        string[] tensOrdinals =
        {
            "", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
            "sixtieth", "seventieth", "eightieth", "ninetieth",
        };
        if (value < 20)
        {
            return ordinals[(int)value];
        }
        if (value < 100)
        {
            long ten = value / 10;
            long one = value % 10;
            return one == 0
                ? tensOrdinals[(int)ten]
                : EnglishCardinalUnder1000((int)(ten * 10)) + " " + EnglishOrdinal(one);
        }
        if (value % 100 == 0)
        {
            return EnglishCardinal(value / 100) + " hundredth";
        }
        return EnglishCardinal(value - (value % 100)) + " " + EnglishOrdinal(value % 100);
    }

    private static string EnglishOrdinalFromDigits(string digits)
        => TryParseNonNegativeI64(digits, out long value) ? EnglishOrdinal(value) : EnglishDigitsIndividually(digits);

    private static string EnglishFraction(string numeratorDigits, string denominatorDigits)
    {
        if (!TryParseNonNegativeI64(numeratorDigits, out long numerator) ||
            !TryParseNonNegativeI64(denominatorDigits, out long denominator))
        {
            return EnglishDigitsIndividually(numeratorDigits) + " over " + EnglishDigitsIndividually(denominatorDigits);
        }
        if (denominator == 2)
        {
            return numerator == 1 ? "one half" : EnglishCardinal(numerator) + " halves";
        }
        string outText = EnglishCardinal(numerator) + " " + EnglishOrdinal(denominator);
        if (numerator != 1)
        {
            outText += "s";
        }
        return outText;
    }

    /// <summary>Uppercases [a-z] only, leaving every other character untouched.</summary>
    public static string UppercaseAscii(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char ch in text)
        {
            sb.Append(AsciiUpper(ch));
        }
        return sb.ToString();
    }

    /// <summary>Lowercases [A-Z] only, leaving every other character untouched.</summary>
    public static string LowercaseAscii(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char ch in text)
        {
            sb.Append(AsciiLower(ch));
        }
        return sb.ToString();
    }

    /// <summary>True when the string contains any CJK Unified Ideograph (U+4E00..U+9FFF).</summary>
    public static bool ContainsHan(string text)
    {
        foreach (char ch in text)
        {
            if (ch >= '一' && ch <= '鿿') return true;
        }
        return false;
    }

    // ---- IndexTTS Chinese normalization ----------------------------------------------------

    private static string NormalizeIndexTtsChineseText(string text)
    {
        string outText = ContractionRegex.Replace(text, "$1 is");

        ProtectTechTerms(ref outText);
        var pinyinTones = SavePinyinTones(ref outText);
        var names = SaveRegexMatches(ref outText, NameRegex, "name");

        outText = NormalizeTelephoneRuns(outText);
        outText = NormalizeDateSeparators(outText);
        outText = NormalizeTimeZeroMinutes(outText);
        outText = NormalizePercentages(outText);
        outText = NormalizeFractions(outText);
        outText = NormalizeChineseNumbers(outText);

        RestoreSaved(ref outText, names);
        RestoreSaved(ref outText, pinyinTones);
        outText = HRestoreRegex.Replace(outText, "-");
        return ApplyPunctuationMap(outText, ChinesePunctuationMap);
    }

    private static string NormalizeChineseNumbers(string text)
    {
        // Measure words after which a standalone "2" reads as 两 (wetext zh
        // cardinal rule): 两个, 两位, 两只, ...
        var sb = new StringBuilder(text.Length);
        int i = 0;
        while (i < text.Length)
        {
            // "$50" -> "五十美元" (wetext zh currency rule); consumes the "$".
            if (text[i] == '$' && i + 1 < text.Length && IsAsciiDigit(text[i + 1]))
            {
                int begin = ++i;
                while (i < text.Length && IsAsciiDigit(text[i]))
                {
                    ++i;
                }
                sb.Append(ChineseCardinalFromDigits(text.Substring(begin, i - begin)));
                if (i < text.Length && text[i] == '.' && i + 1 < text.Length && IsAsciiDigit(text[i + 1]))
                {
                    int decimalEnd = i + 1;
                    while (decimalEnd < text.Length && IsAsciiDigit(text[decimalEnd]))
                    {
                        ++decimalEnd;
                    }
                    sb.Append('点');
                    sb.Append(ChineseDigitsIndividually(text.Substring(i + 1, decimalEnd - i - 1)));
                    i = decimalEnd;
                }
                sb.Append("美元");
                continue;
            }
            if (!IsAsciiDigit(text[i]))
            {
                sb.Append(text[i++]);
                continue;
            }

            int start = i;
            while (i < text.Length && IsAsciiDigit(text[i]))
            {
                ++i;
            }
            string digits = text.Substring(start, i - start);
            // "100km/h" -> "每小时一百公里" (wetext zh measure rule).
            if (StartsWithAt(text, i, "km/h"))
            {
                sb.Append("每小时" + ChineseCardinalFromDigits(digits) + "公里");
                i += 4;
                continue;
            }
            if (i < text.Length && text[i] == '.' && i + 1 < text.Length && IsAsciiDigit(text[i + 1]))
            {
                int decimalEnd = i + 1;
                while (decimalEnd < text.Length && IsAsciiDigit(text[decimalEnd]))
                {
                    ++decimalEnd;
                }
                sb.Append(ChineseCardinalFromDigits(digits));
                sb.Append('点');
                sb.Append(ChineseDigitsIndividually(text.Substring(i + 1, decimalEnd - i - 1)));
                i = decimalEnd;
                continue;
            }
            if (digits.Length == 4 && StartsWithAt(text, i, "年"))
            {
                sb.Append(ChineseDigitsIndividually(digits));
            }
            else if (digits == "2" && ChineseMeasureWords.Any(w => StartsWithAt(text, i, w)))
            {
                sb.Append('两');
            }
            else
            {
                sb.Append(ChineseCardinalFromDigits(digits));
            }
        }
        return sb.ToString();
    }

    private static string NormalizeDateSeparators(string text)
        => CnDateSepRegex.Replace(text, "$1年$2月$3日");

    private static string NormalizePercentages(string text)
        => CnPercentRegex.Replace(text, m => "百分之" + NormalizeChineseNumbers(m.Groups[1].Value));

    private static string NormalizeFractions(string text)
        => CnFractionRegex.Replace(text, m =>
            ChineseCardinalFromDigits(m.Groups[2].Value) + "分之" + ChineseCardinalFromDigits(m.Groups[1].Value));

    private static string NormalizeTimeZeroMinutes(string text)
        => CnTimeRegex.Replace(text, "$1点");

    private static string NormalizeTelephoneRuns(string text)
        => CnPhoneRegex.Replace(text, m =>
            ChineseDigitsIndividually(m.Groups[1].Value + m.Groups[2].Value + m.Groups[3].Value));

    // ---- Chinese number / pinyin helpers ---------------------------------------------------

    private static bool IsAsciiDigit(char ch) => ch >= '0' && ch <= '9';

    private static bool IsAsciiAlpha(char ch) => (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');

    private static bool StartsWithAt(string text, int offset, string needle)
        => offset + needle.Length <= text.Length &&
           string.CompareOrdinal(text, offset, needle, 0, needle.Length) == 0;

    private static string ChineseDigitWord(char digit, bool telephoneOne = false)
    {
        string[] digits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };
        if (telephoneOne && digit == '1')
        {
            return "幺";
        }
        return digits[digit - '0'];
    }

    private static string ChineseDigitsIndividually(string digits, bool telephoneOne = false)
    {
        var sb = new StringBuilder();
        foreach (char digit in digits)
        {
            sb.Append(ChineseDigitWord(digit, telephoneOne));
        }
        return sb.ToString();
    }

    private static string ChineseCardinalUnder10000(int value)
    {
        if (value == 0)
        {
            return "零";
        }

        string[] units = { "", "十", "百", "千" };
        var sb = new StringBuilder();
        bool pendingZero = false;
        for (int unitIndex = 3; unitIndex >= 0; --unitIndex)
        {
            int divisor = 1;
            for (int i = 0; i < unitIndex; ++i)
            {
                divisor *= 10;
            }
            int digit = value / divisor;
            value %= divisor;
            if (digit == 0)
            {
                if (sb.Length != 0 && value > 0)
                {
                    pendingZero = true;
                }
                continue;
            }
            if (pendingZero)
            {
                sb.Append('零');
                pendingZero = false;
            }
            if (!(digit == 1 && unitIndex == 1 && sb.Length == 0))
            {
                if (digit == 2 && unitIndex >= 2)
                {
                    sb.Append('两');
                }
                else
                {
                    sb.Append(ChineseDigitWord((char)('0' + digit)));
                }
            }
            sb.Append(units[unitIndex]);
        }
        return sb.ToString();
    }

    private static string ChineseCardinalFromDigits(string digits)
    {
        if (digits.Length == 0)
        {
            return "";
        }
        if (digits.Length > 4)
        {
            return ChineseDigitsIndividually(digits, true);
        }
        return ChineseCardinalUnder10000(int.Parse(digits, CultureInfo.InvariantCulture));
    }

    /// <summary>j/q/x + ü is written with 'v' and the whole candidate is uppercased. The
    /// official correct_pinyin only uppercases these j/q/x ü->v cases; other pinyin keeps
    /// its original casing.</summary>
    private static string CorrectIndexTtsPinyin(string value)
    {
        if (value.Length >= 3)
        {
            char initial = AsciiLower(value[0]);
            if (initial == 'j' || initial == 'q' || initial == 'x')
            {
                char next = AsciiLower(value[1]);
                if (next == 'u')
                {
                    value = value.Substring(0, 1) + "v" + value.Substring(2);
                }
                else if (value.Length >= 4 && value[1] == 'ü')
                {
                    value = value.Substring(0, 1) + "v" + value.Substring(3);
                }
                return UppercaseAsciiAndV(value);
            }
        }
        return value;
    }

    private static bool IsIndexTtsPinyinCandidate(string candidate)
    {
        if (candidate.Length < 2 || !IsAsciiDigit(candidate[candidate.Length - 1]) ||
            candidate[candidate.Length - 1] < '1' || candidate[candidate.Length - 1] > '5')
        {
            return false;
        }
        string body = candidate.Substring(0, candidate.Length - 1);
        body = UppercaseAsciiAndV(body);
        body = AsciiLowerAll(body);
        foreach (string initial in PinyinInitials)
        {
            if (body.StartsWith(initial, StringComparison.Ordinal))
            {
                return PinyinFinals.Contains(body.Substring(initial.Length));
            }
        }
        return PinyinFinals.Contains(body);
    }

    private static string UppercaseAsciiAndV(string value)
    {
        var sb = new StringBuilder(value.Length);
        for (int i = 0; i < value.Length;)
        {
            if (value[i] == 'ü' || value[i] == 'Ü')
            {
                sb.Append('V');
                ++i;
                continue;
            }
            sb.Append(AsciiUpper(value[i]));
            ++i;
        }
        return sb.ToString();
    }

    private static char AsciiUpper(char ch) => ch >= 'a' && ch <= 'z' ? (char)(ch - 32) : ch;

    private static char AsciiLower(char ch) => ch >= 'A' && ch <= 'Z' ? (char)(ch + 32) : ch;

    private static string AsciiLowerAll(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (char ch in s)
        {
            sb.Append(AsciiLower(ch));
        }
        return sb.ToString();
    }

    private static void ProtectTechTerms(ref string text)
    {
        var matches = new List<string>();
        foreach (Match m in TechTermRegex.Matches(text))
        {
            matches.Add(m.Value);
        }
        matches.Sort(StringComparer.Ordinal);
        matches = DedupeSorted(matches);
        matches.Sort((a, b) => b.Length.CompareTo(a.Length));
        foreach (string term in matches)
        {
            text = ReplaceAll(text, term, ReplaceAll(term, "-", "<H>"));
        }
    }

    private static List<KeyValuePair<string, string>> SavePinyinTones(ref string text)
    {
        var matches = new List<string>();
        foreach (Match m in PinyinToneRegex.Matches(text))
        {
            int pos = m.Index;
            if (pos > 0 && IsAsciiAlpha(text[pos - 1]))
            {
                continue;
            }
            string value = m.Value;
            if (IsIndexTtsPinyinCandidate(value))
            {
                matches.Add(value);
            }
        }
        matches.Sort(StringComparer.Ordinal);
        matches = DedupeSorted(matches);

        var saved = new List<KeyValuePair<string, string>>();
        for (int i = 0; i < matches.Count; ++i)
        {
            // Letter suffix like the official TextNormalizer: a digit suffix would be
            // rewritten by the number normalizer and the placeholder would leak.
            string placeholder = "<pinyin_" + (char)('a' + i % 26) + ">";
            text = ReplaceAll(text, matches[i], placeholder);
            saved.Add(new KeyValuePair<string, string>(placeholder, CorrectIndexTtsPinyin(matches[i])));
        }
        return saved;
    }

    private static List<KeyValuePair<string, string>> SaveRegexMatches(ref string text, Regex pattern, string placeholderPrefix)
    {
        var matches = new List<string>();
        foreach (Match m in pattern.Matches(text))
        {
            matches.Add(m.Value);
        }
        matches.Sort(StringComparer.Ordinal);
        matches = DedupeSorted(matches);
        matches.Sort((a, b) => b.Length.CompareTo(a.Length));

        var saved = new List<KeyValuePair<string, string>>();
        for (int i = 0; i < matches.Count; ++i)
        {
            // Letter suffix like the official TextNormalizer: a digit suffix would be
            // rewritten by the number normalizer and the placeholder would leak.
            string placeholder = "<" + placeholderPrefix + "_" + (char)('a' + i % 26) + ">";
            text = ReplaceAll(text, matches[i], placeholder);
            saved.Add(new KeyValuePair<string, string>(placeholder, matches[i]));
        }
        return saved;
    }

    private static List<string> DedupeSorted(List<string> sorted)
    {
        var result = new List<string>(sorted.Count);
        for (int i = 0; i < sorted.Count; ++i)
        {
            if (i == 0 || !string.Equals(sorted[i], sorted[i - 1], StringComparison.Ordinal))
            {
                result.Add(sorted[i]);
            }
        }
        return result;
    }

    private static void RestoreSaved(ref string text, List<KeyValuePair<string, string>> saved)
    {
        foreach (var pair in saved)
        {
            text = ReplaceAll(text, pair.Key, pair.Value);
        }
    }

    private static string ApplyPunctuationMap(string text, (string From, string To)[] replacements)
    {
        foreach (var (from, to) in replacements)
        {
            text = ReplaceAll(text, from, to);
        }
        return text;
    }

    // ---- Confucius4TTS Chinese normalization -----------------------------------------------

    private readonly record struct CodepointSpan(int Start, int End, string Text);

    private static List<CodepointSpan> SplitCodepoints(string text)
    {
        var spans = new List<CodepointSpan>();
        for (int pos = 0; pos < text.Length;)
        {
            int width = char.IsSurrogatePair(text, pos) ? 2 : 1;
            spans.Add(new CodepointSpan(pos, pos + width, text.Substring(pos, width)));
            pos += width;
        }
        return spans;
    }

    private static string SubstringCodepoints(string text, List<CodepointSpan> spans, int start, int end)
        => start >= end ? "" : text.Substring(spans[start].Start, spans[end - 1].End - spans[start].Start);

    /// <summary>A single ASCII, non-space codepoint.</summary>
    private static bool IsAsciiNonSpace(string ch) => ch.Length == 1 && ch[0] <= 0x7F && ch[0] != ' ';

    private static string RemoveBlankBetweenChinese(string text)
    {
        var spans = SplitCodepoints(text);
        var sb = new StringBuilder(text.Length);
        for (int i = 0; i < spans.Count; ++i)
        {
            if (spans[i].Text == " " && i > 0 && i + 1 < spans.Count)
            {
                if (IsAsciiNonSpace(spans[i - 1].Text) && IsAsciiNonSpace(spans[i + 1].Text))
                {
                    sb.Append(' ');
                }
                continue;
            }
            sb.Append(spans[i].Text);
        }
        return sb.ToString();
    }

    private static string NormalizeConfucius4ChineseText(string text)
    {
        string outText = RemoveBlankBetweenChinese(text);
        outText = ReplaceAll(outText, "²", "平方");
        outText = ReplaceAll(outText, "³", "立方");
        outText = ReplaceAll(outText, ".", "。");
        outText = ReplaceAll(outText, " - ", "，");
        outText = ReplaceAll(outText, "（", "");
        outText = ReplaceAll(outText, "）", "");
        outText = ReplaceAll(outText, "【", "");
        outText = ReplaceAll(outText, "】", "");
        outText = ReplaceAll(outText, "`", "");
        outText = ReplaceAll(outText, "——", " ");

        var spans = SplitCodepoints(outText);
        int trimPos = spans.Count;
        while (trimPos > 0 &&
               (spans[trimPos - 1].Text == "，" || spans[trimPos - 1].Text == "," || spans[trimPos - 1].Text == "、"))
        {
            --trimPos;
        }
        if (trimPos != spans.Count)
        {
            return SubstringCodepoints(outText, spans, 0, trimPos) + "。";
        }
        return outText;
    }

    // ---- Japanese kana / width cleanup -----------------------------------------------------

    private static List<uint> JapaneseCodepoints(string text)
    {
        var list = new List<uint>();
        for (int pos = 0; pos < text.Length;)
        {
            uint cp = (uint)char.ConvertToUtf32(text, pos);
            pos += char.IsSurrogatePair(text, pos) ? 2 : 1;
            if (cp >= 0x3041U && cp <= 0x3096U)
            {
                cp += 0x60U;
            }
            else if (cp == 0x309DU)
            {
                cp = 0x30FDU;
            }
            else if (cp == 0x309EU)
            {
                cp = 0x30FEU;
            }
            else
            {
                cp = HalfwidthKatakanaToFullwidth(cp);
            }
            list.Add(cp);
        }
        for (int i = 1; i < list.Count;)
        {
            if (list[i] == 0x3099U || list[i] == 0x309AU || list[i] == 0xFF9EU || list[i] == 0xFF9FU)
            {
                uint voiced = ApplyKatakanaVoicing(list[i - 1], list[i]);
                if (voiced != list[i - 1])
                {
                    list[i - 1] = voiced;
                    list.RemoveAt(i);
                    continue;
                }
            }
            ++i;
        }
        return list;
    }

    private static uint HalfwidthKatakanaToFullwidth(uint cp) => cp switch
    {
        0xFF66 => 0x30F2, 0xFF67 => 0x30A1, 0xFF68 => 0x30A3, 0xFF69 => 0x30A5,
        0xFF6A => 0x30A7, 0xFF6B => 0x30A9, 0xFF6C => 0x30E3, 0xFF6D => 0x30E5,
        0xFF6E => 0x30E7, 0xFF6F => 0x30C3, 0xFF70 => 0x30FC, 0xFF71 => 0x30A2,
        0xFF72 => 0x30A4, 0xFF73 => 0x30A6, 0xFF74 => 0x30A8, 0xFF75 => 0x30AA,
        0xFF76 => 0x30AB, 0xFF77 => 0x30AD, 0xFF78 => 0x30AF, 0xFF79 => 0x30B1,
        0xFF7A => 0x30B3, 0xFF7B => 0x30B5, 0xFF7C => 0x30B7, 0xFF7D => 0x30B9,
        0xFF7E => 0x30BB, 0xFF7F => 0x30BD, 0xFF80 => 0x30BF, 0xFF81 => 0x30C1,
        0xFF82 => 0x30C4, 0xFF83 => 0x30C6, 0xFF84 => 0x30C8, 0xFF85 => 0x30CA,
        0xFF86 => 0x30CB, 0xFF87 => 0x30CC, 0xFF88 => 0x30CD, 0xFF89 => 0x30CE,
        0xFF8A => 0x30CF, 0xFF8B => 0x30D2, 0xFF8C => 0x30D5, 0xFF8D => 0x30D8,
        0xFF8E => 0x30DB, 0xFF8F => 0x30DE, 0xFF90 => 0x30DF, 0xFF91 => 0x30E0,
        0xFF92 => 0x30E1, 0xFF93 => 0x30E2, 0xFF94 => 0x30E4, 0xFF95 => 0x30E6,
        0xFF96 => 0x30E8, 0xFF97 => 0x30E9, 0xFF98 => 0x30EA, 0xFF99 => 0x30EB,
        0xFF9A => 0x30EC, 0xFF9B => 0x30ED, 0xFF9C => 0x30EF, 0xFF9D => 0x30F3,
        _ => cp,
    };

    private static uint ApplyKatakanaVoicing(uint cp, uint mark)
    {
        if (mark == 0x3099U || mark == 0xFF9EU)
        {
            return cp switch
            {
                0x30A6 => 0x30F4, 0x30AB => 0x30AC, 0x30AD => 0x30AE, 0x30AF => 0x30B0,
                0x30B1 => 0x30B2, 0x30B3 => 0x30B4, 0x30B5 => 0x30B6, 0x30B7 => 0x30B8,
                0x30B9 => 0x30BA, 0x30BB => 0x30BC, 0x30BD => 0x30BE, 0x30BF => 0x30C0,
                0x30C1 => 0x30C2, 0x30C4 => 0x30C5, 0x30C6 => 0x30C7, 0x30C8 => 0x30C9,
                0x30CF => 0x30D0, 0x30D2 => 0x30D3, 0x30D5 => 0x30D6, 0x30D8 => 0x30D9,
                0x30DB => 0x30DC, 0x30EF => 0x30F7, 0x30F0 => 0x30F8, 0x30F1 => 0x30F9,
                0x30F2 => 0x30FA,
                _ => cp,
            };
        }
        if (mark == 0x309AU || mark == 0xFF9FU)
        {
            return cp switch
            {
                0x30CF => 0x30D1, 0x30D2 => 0x30D4, 0x30D5 => 0x30D7,
                0x30D8 => 0x30DA, 0x30DB => 0x30DD,
                _ => cp,
            };
        }
        return cp;
    }

    private static void AppendCodepoint(StringBuilder sb, uint cp)
    {
        if (cp <= 0xFFFFU)
        {
            sb.Append((char)cp);
        }
        else
        {
            cp -= 0x10000U;
            sb.Append((char)(0xD800U + (cp >> 10)));
            sb.Append((char)(0xDC00U + (cp & 0x3FFU)));
        }
    }

    // ---- shared helpers --------------------------------------------------------------------

    private static string TrimAsciiWhitespace(string v)
    {
        const string ws = " \t\r\n";
        int b = 0;
        while (b < v.Length && ws.IndexOf(v[b]) >= 0)
        {
            ++b;
        }
        if (b == v.Length)
        {
            return "";
        }
        int e = v.Length - 1;
        while (e >= b && ws.IndexOf(v[e]) >= 0)
        {
            --e;
        }
        return v.Substring(b, e - b + 1);
    }
}
