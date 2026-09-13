"""Head-to-head: our vulkan-torch PP-OCRv6 port vs the original PaddleOCR.

Runs the port in this (repo) env, the original via the external venv, over the
same full-page images. Writes both transcripts to <out>/, times each, and reports
transcription accuracy (1 - CER) and speed-up.

    python tools/ocr_compare.py                 # the two _test_*.gif frames
    python tools/ocr_compare.py img1.png img2.gif --out data/ocr/out
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "example", "python"))

VENV_PY = r"<external>/venv/Scripts/python.exe"
PADDLE_SCRIPT = os.path.join(ROOT, "tools", "ocr_paddle_page.py")
DEFAULT_IMAGES = [
    r"<external>/_test_1.gif",
    r"<external>/_test_2.gif",
]


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ocr", "out"))
    a = ap.parse_args()
    images = a.images or DEFAULT_IMAGES
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import segment
    from ocr_py.ocr import Ocr
    print("loading port models...")
    t0 = time.perf_counter()
    ocr = Ocr()
    print(f"  port LOAD {time.perf_counter() - t0:.2f}s")

    report = []
    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        print(f"\n=== {name} ===")

        t0 = time.perf_counter()
        port_text = segment.page_text(ocr, img)
        t_port = time.perf_counter() - t0
        port_txt = os.path.join(a.out, f"{name}.port.txt")
        with open(port_txt, "w", encoding="utf-8") as f:
            f.write(port_text)

        pad_txt = os.path.join(a.out, f"{name}.paddle.txt")
        t_pad = None
        if os.path.isfile(pad_txt):
            print(f"  paddle: cached ({pad_txt}), skipping CPU run")
        else:
            out = subprocess.run([VENV_PY, PADDLE_SCRIPT, "--image", img, "--out", pad_txt],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
            line = (out.stdout or "").strip().splitlines()
            if out.returncode != 0 or not line:
                print("  paddle failed:", out.stdout, out.stderr[-400:])
                continue
            t_pad = float(dict(zip(line[-1].split()[0::2], line[-1].split()[1::2]))["RUN"])
        with open(pad_txt, encoding="utf-8") as f:
            pad_text = f.read()

        dist = levenshtein(pad_text, port_text)
        acc = 1.0 - dist / max(len(pad_text), 1)
        speedup = (t_pad / t_port) if (t_pad and t_port) else 0.0
        report.append((name, t_port, t_pad, speedup, len(pad_text), len(port_text), acc))
        print(f"  port   {t_port:7.2f}s  {len(port_text):5d} chars -> {port_txt}")
        print(f"  paddle {"(cached)" if t_pad is None else f"{t_pad:7.2f}s"}  {len(pad_text):5d} chars -> {pad_txt}")
        print(f"  speed-up {speedup:.2f}x   char-accuracy {acc * 100:.2f}% (CER {100 * (1 - acc):.2f}%)")

    print("\n================ SUMMARY ================")
    print(f"{'image':10s} {'port(s)':>8s} {'paddle(s)':>10s} {'speedup':>8s} {'acc%':>7s}")
    for name, tp, tpd, su, cp, cq, acc in report:
        padt = f"{tpd:10.2f}" if tpd else "   (cached)"
        print(f"{name:10s} {tp:8.2f} {padt} {su:7.2f}x {acc * 100:6.2f}")
    print(f"\nport cached graphs: {len(ocr._cache)} "
          f"(det {sum(1 for k in ocr._cache if k[0]=='det')}, rec {sum(1 for k in ocr._cache if k[0]=='rec')})")
    print(f"\ntranscripts in {a.out}")


if __name__ == "__main__":
    main()
