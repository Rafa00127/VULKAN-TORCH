"""App-specific page segmentation (the reader app's "chapter frame -> lines" rule).

Not part of the ocr_py library: it encodes a particular layout (fixed 38px line
pitch, a 10px pinyin strip above each 18px text line) used by the VIP chapter
frames. Mirrors ocr_server_gui.py:gif_to_text.
"""
import numpy as np
from PIL import Image, ImageOps

LINE_TOP, LINE_GAP, LINE_H, CUT_PINYIN = 10, 38, 28, 10


def segment_lines(src):
    """Yield RGBA->RGB line crops (uint8 HWC) from a path / PIL image / bytes."""
    if isinstance(src, Image.Image):
        img = src
    else:
        img = Image.open(src)
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


def page_text(ocr, src, det=False):
    return "\n".join(ocr.read_line(a, det=det) for a in segment_lines(src))
