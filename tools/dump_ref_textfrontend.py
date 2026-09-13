"""Dump the IndexTTS 2.5 text front-end (segmentation + pronunciation annotations + token ids)
from the official implementation, for validating example/CSharp/IndexTts.Net/TextFrontend.cs.

Text normalization is deliberately SKIPPED (the corpus is chosen so the normalizer is a no-op):
the point is to validate the deterministic front-end stages, not the wetext port.

Run:  移植参考/.venv-indextts/Scripts/python.exe tools/dump_ref_textfrontend.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "移植参考", "index-tts-main"))

from indextts.utils.tokenizer import get_encoding

# front.py / infer_v2_5.py pull in sentencepiece + torch, which this venv does not fully have.
# The two functions we need are pure and self-contained, so exec the reference's own source.
_INFER = os.path.join(ROOT, "移植参考", "index-tts-main", "indextts", "infer_v2_5.py")
_src = open(_INFER, encoding="utf-8").read().splitlines()
_ns = {"re": re}
exec("\n".join(_src[38:73]), _ns)          # PRONUNCIATION_ANNOTATION_PATTERN .. apply_pronunciation_annotations
apply_pronunciation_annotations = _ns["apply_pronunciation_annotations"]

# char_rep_map + clean_pattern, verbatim from indextts/utils/front.py TextNormalizer.__init__.
CHAR_REP_MAP = {
    "：": ",", "；": ",", ";": ",", "，": ",", "。": ".", "！": "!", "？": "?",
    "\n": " ", "·": "-", "、": ",", "...": "…", ",,,": "…", "，，，": "…", "……": "…",
    "“": "'", "”": "'", '"': "'", "‘": "'", "’": "'", "（": "'", "）": "'", "(": "'",
    ")": "'", "《": "'", "》": "'", "【": "'", "】": "'", "[": "'", "]": "'", "—": "-",
    "～": "-", "~": "-", "「": "'", "」": "'", ":": ",",
}
CLEAN_PATTERN = re.compile("|".join(re.escape(p) for p in CHAR_REP_MAP.keys()))

MODEL_DIR = os.path.join(ROOT, "移植参考", "model", "index2.5")
OUT = os.path.join(ROOT, "data", "indextts_ref", "textfrontend_ref.json")

ENC = get_encoding("multilingual_zh_ja_yue_char_del", 99, MODEL_DIR)
CAPACITY = 602                 # gpt.text_pos_embedding.emb.num_embeddings
PAD_ID = 1

SPLIT_PROTECTED_PATTERN = re.compile(r'<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>')
SPECIAL_UPPER = re.compile(r'<\|([^|]+)\|>')
DELIM_SPLIT = re.compile(r'(?<=[，。！？、；：,\.!\?;:\n])')


def token_len(text):
    return len(ENC.encode(text, allowed_special="all"))


def split_atomic_pieces(text):
    pieces, pos = [], 0
    for m in SPLIT_PROTECTED_PATTERN.finditer(text):
        if m.start() > pos:
            pieces.append((text[pos:m.start()], False))
        pieces.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    return pieces


def split_text_by_tokens(text, max_tokens, lang_prefix):
    """Verbatim from indextts/infer_v2_5.py IndexTTS2.split_text_by_tokens."""
    budget = min(max_tokens, CAPACITY - 2) - token_len(lang_prefix)
    budget = max(1, budget)
    if token_len(text) <= budget:
        return [text]
    chunks = []
    for piece, atomic in split_atomic_pieces(text):
        if atomic:
            chunks.append(piece)
            continue
        for part in DELIM_SPLIT.split(piece):
            if not part:
                continue
            if token_len(part) <= budget:
                chunks.append(part)
                continue
            current = ""
            for ch in part:
                if current and token_len(current + ch) > budget:
                    chunks.append(current)
                    current = ch
                else:
                    current += ch
            if current:
                chunks.append(current)
    segments, current = [], ""
    for chunk in chunks:
        if current and token_len(current + chunk) > budget:
            segments.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        segments.append(current)
    return segments or [text]


def process(text, lang, max_tokens=120):
    lang_prefix = f'<|{lang.lower()}|> '
    # line 702 of infer_v2_5.py: the punctuation map runs for every language, always.
    text = CLEAN_PATTERN.sub(lambda x: CHAR_REP_MAP[x.group()], text)
    # text_normalization is skipped on purpose (see module docstring)
    if lang.lower() in ("ja", "zh", "zhen", "en"):
        text = text.lower()
    elif lang.lower() == "es":
        text = text.upper()
    text = apply_pronunciation_annotations(text)
    text = SPECIAL_UPPER.sub(lambda m: f'<|{m.group(1).upper()}|>', text)
    segments = split_text_by_tokens(text, max_tokens, lang_prefix)
    ids = []
    for seg in segments:
        toks = ENC.encode(lang_prefix + seg, allowed_special="all")
        ids.append(list(toks) + [PAD_ID])
    return text, segments, ids


# Plain ASCII letters + spaces (avoid contractions/symbols so the normalizer is a no-op),
# sentence delimiters, pronunciation annotations, and special-token spans.
FILLER = ("the quick brown fox jumps over the lazy dog while a gentle breeze drifts across "
          "the quiet meadow and the distant hills fade into a soft and hazy golden light")
LONG3 = " ".join([FILLER] * 3)
LONG4 = " ".join([FILLER] * 4)
CASES = [
    ("en", "Hello world."),
    ("en", "The quick brown fox jumps over the lazy dog!"),
    ("en", "<going|G OW1 . IH0 NG> to the store."),
    ("en", "I said <read|R EH D> it again and again, please."),
    ("en", "say <going|g ow1 . ih0 ng> now."),      # lower -> upper
    ("ja", "これは <word|テスト> です。"),            # katakana pron -> inlined
    ("ja", "これは <word|てすと> です。"),            # hiragana pron -> inlined
    ("en", "prefix <|SPECIAL_TOKEN_1|>middle part<|SPECIAL_TOKEN_1|> suffix, ok."),
    ("en", "tags like <|startoftranscript|> and <|en|> stay put."),
    ("en", FILLER),
    ("en", FILLER + " " + FILLER),
    ("en", LONG3),
    ("en", LONG4),
    ("en", "alpha, beta! gamma? delta. epsilon; zeta: eta\n" + FILLER),
    ("en", FILLER + " <going|G OW1 . IH0 NG> " + FILLER),
    ("en", LONG3 + " and a short tail."),
    ("en", "the quick brown fox jumps over the lazy dog " * 6),   # tiny alphabet, no delimiters
    ("zh", "你好，今天天气不错。"),
    ("zh", "<行|XING2>走在这条路上。"),
    ("zh", "清晨拉开窗帘，阳光洒在窗台上，远处传来几声鸟鸣，空气里带着露水的味道。" * 4),
    ("ja", "これはテストです。"),
    ("ja", "これはテストです。" * 20),
]


def main():
    out = []
    for lang, text in CASES:
        processed, segments, ids = process(text, lang)
        out.append({
            "lang": lang,
            "text": text,
            "processed": processed,
            "segments": segments,
            "ids": ids,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} cases -> {OUT}")
    for rec in out:
        print(f"  [{rec['lang']}] segs={len(rec['segments'])} "
              f"ids={[len(i) for i in rec['ids']]}  {rec['processed'][:60]!r}")


if __name__ == "__main__":
    main()
