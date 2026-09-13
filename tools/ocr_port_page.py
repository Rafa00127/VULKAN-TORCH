"""Run the vulkan-torch OCR port on one image and print ``RUN <s> chars <n>``.

Backend-agnostic: whatever ``vulkantorch/_vulkantorch*.pyd`` is installed wins.
The ROCm dll dir is added so the HIP-built pyd imports (harmless otherwise).

    python tools/ocr_port_page.py --image X --out Y [--det]
"""
import argparse
import os
import sys
import time

for d in (r"<rocm>/bin",):
    if os.path.isdir(d):
        try:
            os.add_dll_directory(d)
        except OSError:
            pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "example", "python"))

from ocr_py.ocr import Ocr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det", action="store_true")
    a = ap.parse_args()
    ocr = Ocr()
    back = ocr.dev.name()
    t0 = time.perf_counter()
    text = ocr.read_page(a.image, det=a.det)
    dt = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"RUN {dt:.2f} chars {len(text)} backend {back}")


if __name__ == "__main__":
    main()
