"""PP-OCRv6 recognition / detection on vulkan-torch.

    ocr = Ocr()
    text = ocr.read_line(rgb)              # one already-cropped line -> text
    boxes = ocr.read_line(rgb, det=True)   # full det+rec on a region

Models load from `<repo>/model/ppocrv6/gguf/` unless overridden — pass
`Ocr(model_dir=...)` (or `rec_path=`/`det_path=`/`dict_path=`), or set the
`OCR_MODEL_DIR`/`OCR_PRECISION`/`OCR_REC_GGUF`/`OCR_DET_GGUF`/`OCR_DICT` env vars
(see ocr_vt/_paths.py). Only the rec GGUF + dict are needed for `det=False`.

Page segmentation (how a whole chapter image is cut into lines) is app-specific and
lives with the caller, not here.
"""
import os

import numpy as np
import cv2

import vulkantorch as mt

from ocr_vt import det, rec, weights, postproc, _paths

# det preprocess — paddlex/configs/pipelines/OCR.yaml (SubModules.TextDetection)
# overrides the model's own inference.yml: limit_side_len 64 (not 736), min.
DET_LIMIT, DET_MAX_SIDE = 64, 4000
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)      # BGR order (matches inference.yml)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)
REC_H, REC_W = 48, 320


def _pick(model_dir, name, precision):
    """<model_dir>/<name>.<precision>.gguf, or the .f32 build if that isn't there."""
    p = os.path.join(model_dir, f"{name}.{precision}.gguf")
    return p if os.path.isfile(p) else os.path.join(model_dir, f"{name}.f32.gguf")


def det_preprocess(bgr):
    """bgr uint8 HWC -> ([3,H,W], ratio_h, ratio_w); DetResizeForTest(64, min, 4000)."""
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
    def __init__(self, device=None, *, model_dir=None, precision=None,
                 rec_path=None, det_path=None, dict_path=None):
        """Paths resolve in order: explicit arg > env var > repo default.

        ``model_dir`` points at any directory holding the standard filenames
        (``ppocrv6_rec.<precision>.gguf``, ``ppocrv6_det.<precision>.gguf``,
        ``ppocrv6_dict.txt``) — see ocr_vt/_paths.py. Missing f16 builds fall back to
        f32. The detection model is loaded lazily, so a rec-only install needs only the
        rec GGUF + dict."""
        self.rt = mt.Runtime()
        self.dev = device or self.rt.gpu()
        prec = precision or _paths.PRECISION
        if model_dir:
            rec_path = rec_path or _pick(model_dir, "ppocrv6_rec", prec)
            det_path = det_path or _pick(model_dir, "ppocrv6_det", prec)
            dict_path = dict_path or os.path.join(model_dir, "ppocrv6_dict.txt")
        self._rec_path = rec_path or _paths.REC_GGUF
        self._det_path = det_path or _paths.DET_GGUF
        self._dict_path = dict_path or _paths.DICT_TXT
        for p in (self._rec_path, self._dict_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"OCR model not found: {p}\n"
                    "Put the GGUF under model/ppocrv6/gguf/, or set OCR_MODEL_DIR, or "
                    "pass model_dir=/rec_path= to Ocr().")
        self.wrec = weights.Weights(self._rec_path, self.dev)
        with open(self._dict_path, encoding="utf-8") as f:
            self.chars = f.read().split("\n")
        self._wdet = None
        self._cache = {}                    # (net, shape) -> (graph, x, out, mem), bounded

    @property
    def wdet(self):
        """Loaded on first det use (optional — the production path is rec-only)."""
        if self._wdet is None:
            self._wdet = weights.Weights(self._det_path, self.dev, arena_bytes=2 << 30)
        return self._wdet

    _CACHE_MAX = 24

    def _run(self, net, arr, dtype=np.float32):
        """Device-resident input (not g.input(), which pins to CPU) + a small shape
        cache. Capturing a graph is cheap (~2-3 ms) so this only avoids the CPU
        round-trip and the repeated scheduler alloc; the cache is bounded so many
        distinct rec widths can't blow GPU memory.

        ``net`` picks the graph: "det", "rec" (softmax probs) or "rec_ids" (CTC
        argmax ids — the production decode; ~18710x less data pulled off the GPU)."""
        arr = np.ascontiguousarray(arr, np.float32)
        key = (net, arr.shape)
        if net == "det":
            fn, w = det.forward, self.wdet
        elif net == "rec_ids":
            fn, w = rec.forward_ids, self.wrec
        else:
            fn, w = rec.forward, self.wrec

        ent = self._cache.get(key)
        if ent is not None:
            g, x, out, mem = ent
            g.set_input(x, arr.tobytes())
            g.compute_static()
            return np.frombuffer(out.to_bytes(), dtype).reshape(list(out.shape))

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
                return np.frombuffer(out.to_bytes(), dtype).reshape(list(out.shape))

        return np.frombuffer(out.to_bytes(), dtype).reshape(list(out.shape))

    def read_line(self, rgb, det=False):
        """rgb uint8 HWC (as passed to PaddleOCR, i.e. BGR-interpreted) -> text.

        det=False (default): the line is already cropped, so run recognition only —
        this is ~50x faster than the full det+rec pipeline and matches the user's
        workflow (lines are pre-cut by the segmenter)."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if not det:
            ids = self._run("rec_ids", rec_preprocess(bgr), np.int32)
            return postproc.ctc_decode_ids(ids, self.chars)
        det_in, rh, rw = det_preprocess(bgr)
        prob = self._run("det", det_in)
        texts = []
        for box in postproc.boxes_from_prob(prob, rh, rw):
            crop = postproc.crop_quad(bgr, box)
            if crop is None:
                continue
            ids = self._run("rec_ids", rec_preprocess(crop), np.int32)
            texts.append(postproc.ctc_decode_ids(ids, self.chars))
        return "\n".join(texts)
