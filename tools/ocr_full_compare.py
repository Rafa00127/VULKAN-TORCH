"""Head-to-head on a FULL image (no segmentation): our port vs the original PaddleOCR.

    python tools/ocr_full_compare.py <image> [--out data/ocr/out]

Runs the port in-process (det+rec on the whole image), the original via the
external venv, writes both transcripts, reports char-accuracy and speed-up.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "example", "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from ocr_compare import levenshtein  # noqa: E402
from ocr_py.ocr import Ocr  # noqa: E402

VENV_PY = r"<external>/venv/Scripts/python.exe"
PADDLE_SCRIPT = os.path.join(ROOT, "tools", "ocr_paddle_full.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ocr", "out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.image))[0]

    ocr = Ocr()
    rgb = np.array(Image.open(a.image).convert("RGB"))
    t0 = time.perf_counter()
    port_text = ocr.read_line(rgb, det=True)
    t_port = time.perf_counter() - t0
    port_txt = os.path.join(a.out, f"{name}.port.det.txt")
    with open(port_txt, "w", encoding="utf-8") as f:
        f.write(port_text)

    pad_txt = os.path.join(a.out, f"{name}.paddle.det.txt")
    t_pad = None
    if os.path.isfile(pad_txt):
        print(f"paddle: cached ({pad_txt})")
    else:
        out = subprocess.run([VENV_PY, PADDLE_SCRIPT, "--image", a.image, "--out", pad_txt],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = (out.stdout or "").strip().splitlines()
        if out.returncode != 0 or not line:
            print("paddle failed:", out.stdout, out.stderr[-500:])
            return
        t_pad = float(dict(zip(line[-1].split()[0::2], line[-1].split()[1::2]))["RUN"])
    with open(pad_txt, encoding="utf-8") as f:
        pad_text = f.read()

    dist = levenshtein(pad_text, port_text)
    acc = 1.0 - dist / max(len(pad_text), 1)
    print(f"port    {t_port:7.2f}s  {len(port_text):5d} chars -> {port_txt}")
    print(f"paddle  {'(cached)' if t_pad is None else f'{t_pad:7.2f}s'}  {len(pad_text):5d} chars -> {pad_txt}")
    if t_pad:
        print(f"speed-up {t_pad / t_port:.2f}x")
    print(f"char-accuracy {acc * 100:.2f}%  (dist {dist})")


if __name__ == "__main__":
    main()
