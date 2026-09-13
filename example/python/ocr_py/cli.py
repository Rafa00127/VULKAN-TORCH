"""CLI: recognition (or det+rec) on a single already-cropped line image.

    python example/python/ocr_py/cli.py --line line.png
    python example/python/ocr_py/cli.py --line region.png --det

Page segmentation is app-specific and lives with the caller, not in this
library: the caller hands this entry point its line crops.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)                                     # vulkantorch (repo root)
sys.path.insert(0, os.path.join(ROOT, "example", "python"))  # ocr_py

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ocr_py.ocr import Ocr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True, help="a single cropped line (or region)")
    ap.add_argument("--det", action="store_true", help="run det+rec instead of rec-only")
    a = ap.parse_args()
    ocr = Ocr()
    rgb = np.array(Image.open(a.line).convert("RGB"))
    print(ocr.read_line(rgb, det=a.det))


if __name__ == "__main__":
    main()
