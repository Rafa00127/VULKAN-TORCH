#!/usr/bin/env python
"""Dump PP-OCRv6 reference data for the vulkan-torch port.

Mirrors `ocr_server_gui.py:gif_to_text` (the user's real workload): invert the
frame, slice fixed-height line bands, trim horizontal whitespace, 2x-upscale
narrow lines, then OCR each line crop.

Run with the external venv python:
    "<venv>/python.exe" tools/dump_ref_ocr.py --image data/ocr/test_frame.png [--lines 6]

Writes to data/ocr/ref/:
    line{i}_in.npy      [1,3,H,W]   the det input for segmented line i
    line{i}_prob.npy    [1,1,H,W]   det sigmoid map
    line{i}_rec.npy     [1,3,48,320] the rec input (whole line resized to h48)
    line{i}_logits.npy  [1,T,C]     rec softmax logits
    ref.json            {image, lines:[{i, ocr_predict:[...], rec_text}]}

The *_in/*_rec arrays are *shared bytes*: the port runs the identical input, so
these validate the network weights independently of any preprocessing.
`ocr_predict` (the real PaddleOCR CPU pipeline on the same crop) is the
end-to-end ground truth.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from PIL import Image, ImageOps
import paddle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADDLEX = r"<PADDLEX_HOME>"
OUT = os.path.join(ROOT, "data", "ocr", "ref")

from paddlex.inference.models.text_detection.modeling.pp_ocrv6_medium_det import (
    PPOCRV6MediumDet, PPOCRV6MediumDetConfig,
)
from paddlex.inference.models.text_recognition.modeling.pp_ocrv6_small_rec import (
    PPOCRV6SmallRec, PPOCRV6SmallRecConfig,
)

# gif_to_text segmentation constants (verbatim)
LINE_TOP, LINE_GAP, LINE_H, CUT_PINYIN = 10, 38, 28, 10

# det preprocess (from PP-OCRv6_medium_det/inference.yml, BGR order)
DET_LIMIT, DET_MAX_SIDE = 736, 4000
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)
# rec preprocess (from PP-OCRv6_medium_rec/inference.yml)
REC_IMG_H, REC_IMG_W = 48, 320


def load(model_cls, cfg_cls, st_dir):
    with open(os.path.join(st_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = model_cls(cfg_cls(**cfg))
    model.eval()
    from safetensors.numpy import load_file
    raw = load_file(os.path.join(st_dir, "model.safetensors"))
    sd = {k: paddle.to_tensor(np.ascontiguousarray(v.T) if v.ndim == 2 else v)
          for k, v in raw.items()}
    model.set_hf_state_dict(sd)
    return model


def segment_lines(png_path):
    """Yield (index, rgb_uint8_crop) following gif_to_text exactly."""
    img = Image.open(png_path).convert("RGB")
    img = ImageOps.invert(img)
    arr = np.array(img)
    H = arr.shape[0]
    total = (H - LINE_TOP) // LINE_GAP
    out = []
    for i in range(total):
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
        out.append((i, a))
    return out


def det_preprocess(rgb):
    """rgb uint8 HWC -> [1,3,H,W]; mirrors DetResizeForTest.resize_image_type0
    (min-side 736, then max_side_limit 4000, THEN /32 snap), BGR, ImageNet norm."""
    import cv2
    bgr = rgb[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    if min(h, w) < DET_LIMIT:
        ratio = float(DET_LIMIT) / (h if h < w else w)
    else:
        ratio = 1.0
    rh, rw = int(h * ratio), int(w * ratio)
    if max(rh, rw) > DET_MAX_SIDE:                      # clamp BEFORE /32 snap
        r = float(DET_MAX_SIDE) / max(rh, rw)
        rh, rw = int(rh * r), int(rw * r)
    rh = max(int(round(rh / 32) * 32), 32)
    rw = max(int(round(rw / 32) * 32), 32)
    x = cv2.resize(bgr, (rw, rh)).astype(np.float32) * (1.0 / 255.0)
    x = (x - DET_MEAN) / DET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def rec_preprocess(rgb):
    """rgb uint8 -> [1,3,48,W]; mirrors OCRReisizeNormImg (dynamic width)."""
    import cv2, math
    bgr = rgb[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    max_wh = max(REC_IMG_W / REC_IMG_H, w / h)
    imgw = int(REC_IMG_H * max_wh)
    ratio = w / h
    rw = imgw if math.ceil(REC_IMG_H * ratio) > imgw else int(math.ceil(REC_IMG_H * ratio))
    x = cv2.resize(bgr, (rw, REC_IMG_H)).astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    pad = np.zeros((3, REC_IMG_H, imgw), np.float32)
    pad[:, :, :rw] = x
    return pad[None]


def greedy_ctc(logits, chars):
    idx = logits[0].argmax(-1).tolist()
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0 and i - 1 < len(chars):
            out.append(chars[i - 1])
        prev = i
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--lines", type=int, default=6, help="how many line crops to dump tensors for")
    ap.add_argument("--all-text", action="store_true", help="run ocr.predict on every line (slow)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    det = load(PPOCRV6MediumDet, PPOCRV6MediumDetConfig,
               os.path.join(PADDLEX, "PP-OCRv6_medium_det_safetensors"))
    rec = load(PPOCRV6SmallRec, PPOCRV6SmallRecConfig,
               os.path.join(PADDLEX, "PP-OCRv6_medium_rec_safetensors"))
    chars = [l.rstrip("\n") for l in open(os.path.join(ROOT, "data/ocr/ppocrv6_dict.txt"), encoding="utf-8")]
    segs = segment_lines(a.image)
    print(f"segmented {len(segs)} lines from {a.image}")

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, engine="paddle")

    n = min(a.lines, len(segs))
    manifest = {"image": os.path.abspath(a.image), "n_lines": len(segs), "lines": []}
    for idx, rgb in segs[: (len(segs) if a.all_text else n)]:
        det_in = det_preprocess(rgb)
        prob = det([det_in])[0]
        rec_in = rec_preprocess(rgb)
        logits = rec([rec_in])[0]
        res = ocr.predict(rgb)
        texts = []
        for r in res:
            d = r if isinstance(r, dict) else vars(r)
            texts.extend(d.get("rec_texts", []))
        entry = {"i": idx, "crop_hw": list(rgb.shape[:2]),
                 "ocr_predict": texts, "rec_text": greedy_ctc(logits, chars)}
        manifest["lines"].append(entry)
        print(f"  line {idx}: det_prob{prob.shape} rec_logits{logits.shape} "
              f"paddle={texts} mine={entry['rec_text']!r}")
        if idx < n:
            np.save(os.path.join(OUT, f"line{idx}_in.npy"), det_in)
            np.save(os.path.join(OUT, f"line{idx}_prob.npy"), prob)
            np.save(os.path.join(OUT, f"line{idx}_rec.npy"), rec_in)
            np.save(os.path.join(OUT, f"line{idx}_logits.npy"), logits)

    with open(os.path.join(OUT, "ref.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
