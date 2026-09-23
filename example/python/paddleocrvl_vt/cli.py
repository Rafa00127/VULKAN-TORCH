"""PaddleOCR-VL on vulkantorch, greedy decoding.

    python example/python/paddleocrvl_vt/cli.py page.png -t ocr
    python example/python/paddleocrvl_vt/cli.py page.png -p "Table Recognition:" -o out.md
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)                                     # vulkantorch (repo root)
sys.path.insert(0, os.path.join(ROOT, "example", "python"))  # paddleocrvl_vt

# tasks, same prefixes llama-server is driven with (tools/vl_read_page.py, tests/paddlevl.py)
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "seal": "Seal Recognition:",
    "spotting": "Spotting:",
}


def resolve(path, default):
    """Accept an explicit path, else fall back to the default; a bare name is looked
    up in the model dir (see _paths)."""
    if path:
        return path if os.path.isabs(path) else os.path.join(
            os.path.dirname(default), path)
    return default


def main(argv=None):
    from paddleocrvl_vt import _paths
    from paddleocrvl_vt.vl import PaddleOCRVL

    p = argparse.ArgumentParser(description="PaddleOCR-VL (vulkantorch)")
    p.add_argument("image", nargs="?", help="image to read")
    p.add_argument("--model", default=None, help="language-model GGUF")
    p.add_argument("--mmproj", default=None, help="vision-tower GGUF")
    p.add_argument("--text-image", action="store_true",
                   help="treat the positional argument as raw prompt *text* and run "
                        "the prompt with no image (use to compare the LLM path alone "
                        "against llama-server)")
    p.add_argument("-t", "--task", default="ocr", choices=sorted(TASKS))
    p.add_argument("-p", "--prompt", help="raw prompt, overrides --task")
    p.add_argument("-o", "--output", help="write the result to this file")
    p.add_argument("-n", "--max-tokens", type=int, default=4096)
    a = p.parse_args(argv)

    prompt = a.prompt if a.prompt else TASKS[a.task]
    model_path = resolve(a.model, _paths.DEFAULT_GGUF)
    mmproj_path = resolve(a.mmproj, _paths.DEFAULT_MMPROJ)
    for path in (model_path, mmproj_path):
        if not os.path.isfile(path):
            raise SystemExit(f"not found: {path}\n"
                             f"set PADDLEOCRVL_MODEL_DIR, or pass --model/--mmproj")

    if a.image is None and not a.text_image:
        raise SystemExit("an image (or --text-image with a prompt) is required")

    t0 = time.time()
    vl = PaddleOCRVL(model_path, mmproj_path)
    print(f"loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    if a.text_image:
        text, n_prompt, n_gen = vl.generate(a.image, None, a.max_tokens)
    else:
        text, n_prompt, n_gen = vl.generate_image(a.image, prompt, a.max_tokens)

    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(text)

    dt = time.time() - t0
    print(f"--- prompt {n_prompt} tok | gen {n_gen} tok | {dt:.1f}s "
          f"({n_gen / max(dt, 1e-9):.1f} tok/s) ---", file=sys.stderr)


if __name__ == "__main__":
    main()
