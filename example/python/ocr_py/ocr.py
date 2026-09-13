"""Segmentation-based PP-OCRv6 OCR, mirroring ocr_server_gui.py:gif_to_text.

    ocr = Ocr()
    text = ocr.read_page("chapter.gif")   # invert -> line bands -> per-line det+rec
"""
import numpy as np
import cv2
from PIL import Image, ImageOps

import vulkantorch as mt

from ocr_py import det, rec, weights, postproc, _paths

# gif_to_text segmentation constants
LINE_TOP, LINE_GAP, LINE_H, CUT_PINYIN = 10, 38, 28, 10
# det preprocess
DET_LIMIT, DET_MAX_SIDE = 736, 4000
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)      # BGR order (matches inference.yml)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)
REC_H, REC_W = 48, 320


def det_preprocess(bgr):
    """bgr uint8 HWC -> ([3,H,W], ratio_h, ratio_w); DetResizeForTest(min 736, cap 4000)."""
    h, w = bgr.shape[:2]
    ratio = DET_LIMIT / min(h, w) if min(h, w) < DET_LIMIT else 1.0
    rh, rw = int(h * ratio), int(w * ratio)
    if max(rh, rw) > DET_MAX_SIDE:
        r = DET_MAX_SIDE / max(rh, rw)
        rh, rw = int(rh * r), int(rw * r)
    rh = max(int(round(rh / 32) * 32), 32)
    rw = max(int(round(rw / 32) * 32), 32)
    x = cv2.resize(bgr, (rw, rh)).astype(np.float32) / 255.0
    x = (x - DET_MEAN) / DET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)), rh / h, rw / w


def rec_preprocess(bgr):
    """bgr uint8 HWC -> [3,48,W]; OCRReisizeNormImg (dynamic width)."""
    h, w = bgr.shape[:2]
    max_wh = max(REC_W / REC_H, w / h)
    imgw = int(REC_H * max_wh)
    rw = int(np.ceil(REC_H * (w / h)))
    rw = min(rw, imgw)
    x = cv2.resize(bgr, (rw, REC_H)).astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    pad = np.zeros((3, REC_H, imgw), np.float32)
    pad[:, :, :rw] = x
    return np.ascontiguousarray(pad)


class Ocr:
    def __init__(self, device=None):
        self.rt = mt.Runtime()
        self.dev = device or self.rt.gpu()
        self.wdet = weights.Weights(_paths.DET_GGUF, self.dev, arena_bytes=2 << 30)
        self.wrec = weights.Weights(_paths.REC_GGUF, self.dev)
        with open(_paths.DICT_TXT, encoding="utf-8") as f:
            self.chars = f.read().split("\n")
        self._cache = {}                    # (net, shape) -> (graph, x, out, mem), bounded

    _CACHE_MAX = 24

    def _run(self, net, arr):
        """Device-resident input (not g.input(), which pins to CPU) + a small shape
        cache. Capturing a graph is cheap (~2-3 ms) so this only avoids the CPU
        round-trip and the repeated scheduler alloc; the cache is bounded so many
        distinct rec widths can't blow GPU memory."""
        arr = np.ascontiguousarray(arr, np.float32)
        key = (net, arr.shape)
        fn, w = (det.forward, self.wdet) if net == "det" else (rec.forward, self.wrec)

        ent = self._cache.get(key)
        if ent is not None:
            g, x, out, mem = ent
            g.set_input(x, arr.tobytes())
            g.compute_static()
            return np.frombuffer(out.to_bytes(), np.float32).reshape(list(out.shape))

        mem = mt.Memory(self.dev, arr.nbytes)              # outside the capture
        x = mem.tensor(list(arr.shape), arr.tobytes())
        g = mt.Graph(self.rt, self.dev)
        g.enter()
        out = fn(x, w)
        out.mark_output()
        g.exit()

        if len(self._cache) < self._CACHE_MAX:
            try:
                g.alloc_static()
            except RuntimeError:
                pass                                       # not cacheable (e.g. det: ~3 GB)
            else:
                # Once alloc_static() is used, EVERY read must replay through
                # compute_static() -- calling compute() (via to_bytes()) would re-run
                # the scheduler's alloc and clobber the gallocr buffers, corrupting
                # all later replays. Hence we drive the first read statically too.
                self._cache[key] = (g, x, out, mem)
                g.set_input(x, arr.tobytes())
                g.compute_static()
                return np.frombuffer(out.to_bytes(), np.float32).reshape(list(out.shape))

        return np.frombuffer(out.to_bytes(), np.float32).reshape(list(out.shape))

    def read_line(self, rgb, det=False):
        """rgb uint8 HWC (as passed to PaddleOCR, i.e. BGR-interpreted) -> text.

        det=False (default): the line is already cropped, so run recognition only —
        this is ~50x faster than the full det+rec pipeline and matches the user's
        workflow (lines are pre-cut by the segmenter)."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if not det:
            logits = self._run("rec", rec_preprocess(bgr))
            return "".join(ch for ch, _ in postproc.ctc_decode(logits, self.chars))
        det_in, rh, rw = det_preprocess(bgr)
        prob = self._run("det", det_in)
        texts = []
        for box in postproc.boxes_from_prob(prob, rh, rw):
            crop = postproc.crop_quad(bgr, box)
            if crop is None:
                continue
            logits = self._run("rec", rec_preprocess(crop))
            texts.append("".join(ch for ch, _ in postproc.ctc_decode(logits, self.chars)))
        return "".join(texts)

    @staticmethod
    def segment_lines(src):
        """Yield line crops (RGB uint8) following gif_to_text.

        ``src`` may be a path or an already-open PIL image (e.g. from GIF bytes)."""
        img = src if isinstance(src, Image.Image) else Image.open(src)
        arr = np.array(ImageOps.invert(img.convert("RGB")))
        H = arr.shape[0]
        for i in range((H - LINE_TOP) // LINE_GAP):
            y0 = LINE_TOP + i * LINE_GAP
            y1, y2 = y0 + CUT_PINYIN, y0 + LINE_H
            if y2 > H:
                break
            crop = Image.fromarray(arr[y1:y2, :, :])
            carr = np.array(crop.convert("L"))
            cols = np.where((carr > 128).sum(0) > 0)[0]
            if len(cols):
                x1, x2 = int(cols[0]), int(cols[-1]) + 1
                crop = crop.crop((max(0, x1 - 4), 0, min(crop.width, x2 + 4), crop.height))
            a = np.array(crop)
            if a.shape[1] < 200:
                h, w = a.shape[:2]
                a = np.array(Image.fromarray(a).resize((w * 2, h * 2), Image.NEAREST))
            yield a

    def read_page(self, src, det=False):
        return "\n".join(self.read_line(a, det=det) for a in self.segment_lines(src))
