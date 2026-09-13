"""CLI: OCR a chapter frame (or a single line image) with vulkan-torch.

    python example/python/ocr_py/cli.py --image data/ocr/test_frame.png [--lines N]
    python example/python/ocr_py/cli.py --line path/to/line.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ocr_py.ocr import Ocr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="a chapter frame; run the line segmentation")
    ap.add_argument("--line", help="a single already-cropped line image")
    ap.add_argument("--lines", type=int, default=0, help="limit to the first N lines")
    a = ap.parse_args()
    ocr = Ocr()
    if a.line:
        rgb = np.array(Image.open(a.line).convert("RGB"))
        print(ocr.read_line(rgb))
        return
    if not a.image:
        ap.error("need --image or --line")
    segs = list(Ocr.segment_lines(a.image))
    if a.lines:
        segs = segs[: a.lines]
    for i, s in enumerate(segs):
        print(ocr.read_line(s))


if __name__ == "__main__":
    main()
