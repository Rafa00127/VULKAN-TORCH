"""PaddleOCR-VL end to end: image -> patches -> ViT -> projector -> ERNIE -> text.

The prompt is assembled the way llama-server assembles it, because that is the
reference we compare against. Rendering the GGUF chat template for one user message
whose content contains an image concatenates the content parts into a single string
with the image replaced by a random media marker, which the template then emits
verbatim; ``mtmd`` splits on that marker and wraps the image in
``<|IMAGE_START|>`` / ``<|IMAGE_END|>``. Net result, token ids:

    <|begin_of_sentence|> User: <|IMAGE_START|> <image embeddings> <|IMAGE_END|> OCR:\\n Assistant:\\n
"""
import numpy as np

import vulkantorch as mt

from . import gguf_meta, llm, preproc, vision
from .tokenizer import SpmTokenizer

ARENA = 512 << 20

BOS = 100273              # <|begin_of_sentence|>
IMAGE_START = 101305
IMAGE_END = 101306
# Verified against llama-server's stop behaviour; <|end_of_sentence|> is ERNIE's chat
# EOS and is what this model actually emits.
STOP_IDS = (100272, 2)


class PaddleOCRVL:
    def __init__(self, model_path, mmproj_path, arena=ARENA):
        self.rt = mt.Runtime()
        self.gpu = self.rt.gpu()

        meta = gguf_meta.read_metadata(model_path)
        self.tok = SpmTokenizer.from_meta(meta)
        vmeta = gguf_meta.read_metadata(mmproj_path)
        self.min_pixels = int(vmeta["clip.vision.image_min_pixels"])
        self.max_pixels = int(vmeta["clip.vision.image_max_pixels"])

        self.f = mt.GgufFile(model_path, self.gpu, arena)
        self.vf = mt.GgufFile(mmproj_path, self.gpu, arena)
        self.w = {n: self.f.tensor(n) for n in self.f.names()}
        self.vw = {n: self.vf.tensor(n) for n in self.vf.names()}

        # read the ViT's learned position grid back to the host once: it is resized
        # per image size in numpy (see preproc.resize_position_embd)
        self._pos_grid = np.frombuffer(
            self.vw["v.position_embd.weight"].to_bytes(), dtype=np.float32
        ).reshape(vision.POS_GRID ** 2, vision.N_EMBD)

    # ---- prompt -----------------------------------------------------------------

    def _prompt_ids(self, text, with_image):
        """(prefix_ids, suffix_ids). The image embeddings sit between them."""
        if with_image:
            pre = f"<|begin_of_sentence|>User: <|IMAGE_START|>"
            suf = f"<|IMAGE_END|>{text}\nAssistant:\n"
            return self.tok.encode(pre), self.tok.encode(suf)
        return self.tok.encode(f"<|begin_of_sentence|>User: {text}\nAssistant:\n"), []

    # ---- vision -----------------------------------------------------------------

    def image_embeds(self, img_hwc):
        """HWC uint8 -> (PT [n_tokens, 1024] float32, n_tok_x, n_tok_y)."""
        n_h, n_w = img_hwc.shape[0] // vision.PATCH, img_hwc.shape[1] // vision.PATCH
        if n_w % vision.N_MERGE or n_h % vision.N_MERGE:
            raise ValueError(f"patch grid {n_w}x{n_h} is not a multiple of {vision.N_MERGE}")

        pos_np = preproc.resize_position_embd(self._pos_grid, n_w, n_h)
        vpos_np = vision.vision_positions(n_w, n_h)
        shape, data = preproc.to_input_tensor(img_hwc)

        g = mt.Graph(self.rt, self.gpu)
        g.enter()
        x_img = g.input(shape, data)
        pos_in = g.input([n_w * n_h, vision.N_EMBD], pos_np.tobytes())
        vpos = g.input_i32([4 * n_w * n_h], vpos_np.tobytes())
        out = vision.encode(self.vw, x_img, pos_in, vpos, n_w, n_h)
        out.mark_output()
        g.exit()
        emb = np.frombuffer(out.to_bytes(), dtype=np.float32).reshape(-1, llm.N_EMBD)
        return emb, n_w // vision.N_MERGE, n_h // vision.N_MERGE

    # ---- language model ---------------------------------------------------------

    def _sequence(self, prefix_ids, img_emb, suffix_ids, n_tok_x, n_tok_y):
        """Token embeddings, M-RoPE positions and the 3D-lexicographic causal mask."""
        P, S = len(prefix_ids), len(suffix_ids)
        N = 0 if img_emb is None else img_emb.shape[0]

        t = np.empty(P + N + S, dtype=np.int32)
        y = np.empty_like(t)
        x = np.empty_like(t)
        t[:P] = y[:P] = x[:P] = np.arange(P, dtype=np.int32)

        if N:
            rows, cols = np.divmod(np.arange(N, dtype=np.int32), n_tok_x)
            t[P:P + N] = P
            y[P:P + N] = P + rows
            x[P:P + N] = P + cols
            cur = P + max(n_tok_x, n_tok_y)
        else:
            cur = P
        t[P + N:] = y[P + N:] = x[P + N:] = cur + np.arange(S, dtype=np.int32)
        return t, y, x, P, N, S, cur + S

    def _prefill(self, prefix_ids, img_emb, suffix_ids, n_tok_x, n_tok_y, kv_k, kv_v,
                 max_ctx):
        t, y, x, P, N, S, next_pos = self._sequence(prefix_ids, img_emb, suffix_ids,
                                                    n_tok_x, n_tok_y)
        T = P + N + S
        pos_np = llm.rope_positions(t, y, x)
        mask_np = llm.causal_mask(t, y, x)

        g = mt.Graph(self.rt, self.gpu)
        g.enter()
        parts = [mt.get_rows(self.w["token_embd.weight"],
                             g.input_i32([P], np.asarray(prefix_ids, np.int32).tobytes()))]
        if N:
            parts.append(g.input([N, llm.N_EMBD], img_emb.tobytes()))
        if S:
            parts.append(mt.get_rows(self.w["token_embd.weight"],
                                     g.input_i32([S], np.asarray(suffix_ids, np.int32).tobytes())))
        emb = parts[0]
        for p in parts[1:]:
            emb = mt.concat(emb, p, 0)

        pos = g.input_i32([4 * T], pos_np.tobytes())
        mask = mt.cast(g.input([T, T], mask_np.tobytes()), mt.F16)
        last = g.input_i32([1], np.array([T - 1], np.int32).tobytes())

        for li in range(llm.N_LAYER):
            emb = llm.prefill_layer(self.w, li, emb, kv_k, kv_v, pos, mask, T, max_ctx)
            if li == llm.N_LAYER - 1:
                emb = mt.get_rows(emb, last)
        logits = mt.contiguous(llm.head(self.w, emb))
        logits.mark_output()
        g.exit()
        # only one row comes back, so this is n_vocab floats, not T x n_vocab
        out = np.frombuffer(logits.to_bytes(), dtype=np.float32).reshape(-1)
        return out, next_pos

    def _decode_graph(self, kv_k, kv_v, lk, max_ctx):
        g = mt.Graph(self.rt, self.gpu)
        g.enter()
        tok = g.input_i32([1], np.zeros(1, np.int32).tobytes())
        pos = g.input_i32([4], np.zeros(4, np.int32).tobytes())
        row = g.input_i32([1], np.zeros(1, np.int32).tobytes())
        mask_in = g.input([1, lk], np.zeros(lk, np.float32).tobytes())
        mask = mt.cast(mask_in, mt.F16)

        emb = mt.get_rows(self.w["token_embd.weight"], tok)
        for li in range(llm.N_LAYER):
            emb = llm.decode_layer(self.w, li, emb, kv_k, kv_v, pos, row, mask, lk, max_ctx)
        logits = mt.contiguous(llm.head(self.w, emb))
        logits.mark_output()
        g.exit()
        g.alloc_static()
        return {"g": g, "tok": tok, "pos": pos, "row": row, "mask": mask_in,
                "logits": logits}

    # ---- public -----------------------------------------------------------------

    def generate(self, text, img_hwc=None, max_new_tokens=4096):
        """Greedy decode. ``img_hwc`` is a preprocessed uint8 HWC array (or None for a
        text-only prompt). Returns (text, n_prompt_tokens, n_generated)."""
        if img_hwc is None:
            img_emb, n_tok_x, n_tok_y = None, 1, 1
        else:
            img_emb, n_tok_x, n_tok_y = self.image_embeds(img_hwc)
        prefix_ids, suffix_ids = self._prompt_ids(text, img_hwc is not None)

        n_img = 0 if img_emb is None else img_emb.shape[0]
        T = len(prefix_ids) + n_img + len(suffix_ids)
        # The decode graph reads a whole bucket-sized window, so the cache must be at
        # least as large as the biggest bucket we can land in -- not just the prompt
        # length plus the token budget.
        max_ctx = llm.pick_bucket(T + max_new_tokens + 1)
        if max_ctx is None:
            raise ValueError(f"prompt {T} + {max_new_tokens} tokens exceeds the largest "
                             f"KV bucket {llm.STEP_BUCKETS[-1]}")
        _mem, kv_k, kv_v = llm.init_kv(self.gpu, max_ctx)

        logits, pos_next = self._prefill(prefix_ids, img_emb, suffix_ids, n_tok_x, n_tok_y,
                                         kv_k, kv_v, max_ctx)
        out_ids = []
        cache = {}
        n_past = T
        for _ in range(max_new_tokens):
            tok = int(np.argmax(logits))
            if tok in STOP_IDS:
                break
            out_ids.append(tok)

            lk = llm.pick_bucket(n_past + 1)
            ent = cache.get(lk)
            if ent is None:
                ent = cache[lk] = self._decode_graph(kv_k, kv_v, lk, max_ctx)
            g = ent["g"]
            g.set_input(ent["tok"], np.array([tok], np.int32).tobytes())
            g.set_input(ent["pos"], np.array([pos_next] * 3 + [0], np.int32).tobytes())
            g.set_input(ent["row"], np.array([n_past], np.int32).tobytes())
            m = np.full(lk, -np.inf, dtype=np.float32)
            # the new token is written to cell n_past, and must be able to attend to
            # itself -- so the window is inclusive of that cell
            m[:n_past + 1] = 0.0
            g.set_input(ent["mask"], m.tobytes())
            g.compute_static()
            logits = np.frombuffer(ent["logits"].to_bytes(), dtype=np.float32).reshape(-1)
            pos_next += 1
            n_past += 1
        return self.tok.decode(out_ids), T, len(out_ids)

    def read(self, image, prompt="OCR:", max_new_tokens=4096):
        """PIL image / uint8 HWC array / path -> ``(text, n_prompt, n_generated)``.

        The one-call entry point: prepares the image (see preproc.prepare_image) and
        greedy-decodes. ``prompt`` is one of the task prefixes, e.g. ``"OCR:"``.
        """
        from PIL import Image
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            with Image.open(image) as im:
                img = preproc.prepare_image(im, self.min_pixels, self.max_pixels)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            img = preproc.prepare_image(image, self.min_pixels, self.max_pixels)
        return self.generate(prompt, img, max_new_tokens)

    def generate_image(self, image_path, prompt="OCR:", max_new_tokens=4096):
        return self.read(image_path, prompt, max_new_tokens)
