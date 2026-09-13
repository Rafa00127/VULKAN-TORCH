"""Full-page OCR with the ORIGINAL PaddleOCR — mirrors ocr_server_gui.py:gif_to_text
(no GUI, no HTTP). Lines are pre-cropped, so recognition-only by default; pass
--det to run the full det+rec pipeline instead (as gif_to_text originally did).

    "<venv>/python.exe" tools/ocr_paddle_page.py --image <path> --out <txt> [--det]

Prints: ``LOAD <s> RUN <s> chars <n>``.
"""
import argparse
import io
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR, TextRecognition

LINE_TOP, LINE_GAP, LINE_H, CUT_PINYIN = 10, 38, 28, 10


def _texts(result):
    out = []
    for r in result:
        d = r if isinstance(r, dict) else vars(r)
        out.extend(d.get("rec_texts", []) or ([d["rec_text"]] if d.get("rec_text") else []))
    return out


def page_text(model, data, use_det):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = ImageOps.invert(img)
    arr = np.array(img)
    H = arr.shape[0]
    lines = []
    for i in range((H - LINE_TOP) // LINE_GAP):
        y0 = LINE_TOP + i * LINE_GAP
        y1, y2 = y0 + CUT_PINYIN, y0 + LINE_H
        if y2 > H:
            break
        crop = Image.fromarray(arr[y1:y2, :, :])
        carr = np.array(crop.convert("L"))
        cols = np.where((carr > 128).sum(0) > 0)[0]
        if len(cols):
            x1, x2 = cols[0], cols[-1] + 1
            crop = crop.crop((max(0, x1 - 4), 0, min(crop.width, x2 + 4), crop.height))
        a = np.array(crop)
        if a.shape[1] < 200:
            h, w = a.shape[:2]
            a = np.array(Image.fromarray(a).resize((w * 2, h * 2), Image.NEAREST))
        if use_det:
            lines.append("".join(_texts(model.predict(a))))
        else:
            res = model.predict(a)
            txt = ""
            for r in res:
                d = r if isinstance(r, dict) else vars(r)
                txt += d.get("rec_text", "") or "".join(d.get("rec_texts", []))
            lines.append(txt)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det", action="store_true", help="run the full det+rec pipeline")
    a = ap.parse_args()

    t0 = time.perf_counter()
    if a.det:
        model = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                          use_textline_orientation=False, engine="paddle")
    else:
        model = TextRecognition(model_name="PP-OCRv6_medium_rec", engine="paddle")
    t_load = time.perf_counter() - t0

    with open(a.image, "rb") as f:
        data = f.read()
    t0 = time.perf_counter()
    text = page_text(model, data, a.det)
    t_run = time.perf_counter() - t0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"LOAD {t_load:.2f} RUN {t_run:.2f} chars {len(text)}")


if __name__ == "__main__":
    main()
