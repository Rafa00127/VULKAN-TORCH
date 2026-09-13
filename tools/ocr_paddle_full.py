"""Full-image OCR with the ORIGINAL PaddleOCR — no line segmentation.

    "<venv>/python.exe" tools/ocr_paddle_full.py --image X --out Y

Writes the rec_texts joined by newline. Prints ``LOAD <s> RUN <s> chars <n>``.
"""
import argparse
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    model = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                      use_textline_orientation=False, engine="paddle")
    t_load = time.perf_counter() - t0

    rgb = np.array(Image.open(a.image).convert("RGB"))
    t0 = time.perf_counter()
    res = model.predict(rgb)
    d = res[0] if isinstance(res[0], dict) else vars(res[0])
    text = "\n".join(d.get("rec_texts", []))
    t_run = time.perf_counter() - t0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"LOAD {t_load:.2f} RUN {t_run:.2f} chars {len(text)}")


if __name__ == "__main__":
    main()
