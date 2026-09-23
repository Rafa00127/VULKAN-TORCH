"""SentencePiece tokenizer, ported from llama.cpp's ``llm_tokenizer_spm``.

This is *not* the Viterbi unigram algorithm from the ``sentencepiece`` package.
llama.cpp implements sentencepiece's BPE-style merge directly: start from UTF-8
characters, then repeatedly merge the adjacent pair with the highest score until
no adjacent pair concatenates to a known token. Matching that algorithm matters —
the two disagree on real text, and the reference outputs are what we compare
against.

Two details that are easy to get wrong and are replicated here deliberately:

* The text is **whitespace-escaped** (``" "`` -> U+2581) before merging, and the
  escaping is applied per fragment *after* special-token splitting.
* Special tokens (types CONTROL / USER_DEFINED / UNKNOWN) are cut out of the raw
  text first, longest text first, and are never merged or byte-split.
"""
import heapq

# enum llama_token_type
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6

SPACE_MARK = "▁".encode("utf-8")


def _utf8_len(b):
    """Length of the UTF-8 sequence starting at byte ``b`` (1 when invalid/continuation)."""
    if b < 0x80:
        return 1
    if b >= 0xF0:
        return 4
    if b >= 0xE0:
        return 3
    if b >= 0xC0:
        return 2
    return 1  # stray continuation byte: treat as its own symbol


class SpmTokenizer:
    """Built from the GGUF metadata; both lookup directions are byte-keyed."""

    def __init__(self, tokens, scores, token_types, add_space_prefix=False):
        self.tokens = tokens                       # list[bytes]
        self.scores = scores
        self.types = token_types
        self.add_space_prefix = add_space_prefix
        self._text_to_id = {t: i for i, t in enumerate(tokens)}
        # byte-fallback tokens, "<0xNN>" -> single byte
        self._byte_to_id = {}
        for i, t in enumerate(tokens):
            if token_types[i] == BYTE and len(t) == 6 and t.startswith(b"<0x"):
                self._byte_to_id[int(t[3:5], 16)] = i
        # specials, longest first: a shorter special that is a prefix of a longer
        # one must not shadow it.
        self._specials = sorted(
            ((tokens[i], i) for i in range(len(tokens))
             if token_types[i] in (CONTROL, USER_DEFINED, UNKNOWN)),
            key=lambda p: len(p[0]), reverse=True,
        )

    @classmethod
    def from_meta(cls, meta):
        return cls(meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.scores"],
                   meta["tokenizer.ggml.token_type"],
                   bool(meta.get("tokenizer.ggml.add_space_prefix", False)))

    # ---- special-token splitting -------------------------------------------------

    def _partition(self, raw, parse_special):
        """Split ``raw`` on special tokens -> list of (bytes, token_id|None).

        Mirrors ``tokenizer_st_partition``: each special scans the whole buffer, and
        the fragments it produces are themselves rescanned by the specials that come
        later in the list.
        """
        frags = [(raw, None)]
        for text, sid in self._specials:
            if not parse_special and self.types[sid] in (CONTROL, UNKNOWN):
                continue
            out = []
            for buf, tok in frags:
                if tok is not None or text not in buf:
                    out.append((buf, tok))
                    continue
                start = 0
                while True:
                    hit = buf.find(text, start)
                    if hit < 0:
                        break
                    if hit > start:
                        out.append((buf[start:hit], None))
                    out.append((text, sid))
                    start = hit + len(text)
                if start < len(buf):
                    out.append((buf[start:], None))
            frags = out
        return frags

    # ---- the SPM merge itself ----------------------------------------------------

    def _encode_fragment(self, text):
        """``llm_tokenizer_spm_session::tokenize`` for one raw-text fragment."""
        # symbol chain over (u8 char) ranges; `n == 0` means "merged away"
        syms = []          # [start, n, prev, next]
        offs = 0
        while offs < len(text):
            n = min(_utf8_len(text[offs]), len(text) - offs)
            syms.append([offs, n, len(syms) - 1, len(syms) + 1])
            offs += n
        if not syms:
            return []
        syms[-1][3] = -1

        # max-heap on score, ties broken by smaller left index (llama.cpp's
        # comparator is `(l.score < r.score) || (l.score == r.score && l.left > r.left)`)
        heap = []
        merged = {}        # merged text -> (left_index, right_index); every key is a token

        def try_add(left, right):
            if left < 0 or right < 0:
                return
            blob = text[syms[left][0]:syms[left][0] + syms[left][1] + syms[right][1]]
            tok = self._text_to_id.get(blob)
            if tok is None:
                return
            merged[blob] = (left, right)
            heapq.heappush(heap, (-self.scores[tok], left, right, len(blob)))

        for i in range(1, len(syms)):
            try_add(i - 1, i)

        while heap:
            neg_score, left, right, size = heapq.heappop(heap)
            ls, rs = syms[left], syms[right]
            if ls[1] == 0 or rs[1] == 0 or ls[1] + rs[1] != size:
                continue
            ls[1] += rs[1]
            rs[1] = 0
            ls[3] = rs[3]
            if rs[3] >= 0:
                syms[rs[3]][2] = left
            try_add(ls[2], left)
            try_add(left, ls[3])

        out = []
        i = 0
        while i != -1:
            start, n, _p, nxt = syms[i]
            blob = text[start:start + n]
            tok = self._text_to_id.get(blob)
            if tok is not None:
                out.append(tok)
            else:
                # Not a token: emit its bytes one by one. (llama.cpp also consults its
                # rev_merge map here, but that map only ever holds texts that *are*
                # tokens -- try_add_bigram refuses the rest -- so the recursion is
                # unreachable and the byte fallback is the whole story.)
                out.extend(self._byte_to_id[b] for b in blob)
            i = nxt
        return out

    # ---- public API --------------------------------------------------------------

    def encode(self, text, parse_special=True):
        """str -> list[int]. ``add_special`` is not implemented: this model's GGUF sets
        ``add_bos_token = False``, and the chat template spells out the BOS itself."""
        if isinstance(text, str):
            text = text.encode("utf-8")
        out = []
        for frag, tok in self._partition(text, parse_special):
            if tok is not None:
                out.append(tok)
            else:
                out.extend(self._encode_fragment(frag.replace(b" ", SPACE_MARK)))
        return out

    def piece(self, tok):
        """Token id -> raw bytes (▁ unescaped, control tokens render as nothing)."""
        t = self.tokens[tok]
        if self.types[tok] == CONTROL:
            return b""
        if self.types[tok] == BYTE:
            return bytes([self._byte_to_id_rev(t)])
        return t.replace(SPACE_MARK, b" ")

    @staticmethod
    def _byte_to_id_rev(t):
        return int(t[3:5], 16)

    def decode(self, ids):
        """Token ids -> str, the same way llama.cpp detokenizes a completion."""
        return b"".join(self.piece(i) for i in ids).decode("utf-8", errors="replace")
