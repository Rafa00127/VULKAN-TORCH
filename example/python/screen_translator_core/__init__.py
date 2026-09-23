"""划词翻译器外壳 — the model-agnostic screen translator.

Draw a box over any on-screen text, run a model on the crop, optionally translate the
result with an LLM (OpenAI-compatible endpoint), and show it either in the window or as
a pinned subtitle.

The shell here owns the UI, the config file, the hotkey and the LLM call; each example
contributes a small :class:`Backend` that says how to load its model and read one crop:

    from screen_translator_core import Backend, run

    class MyBackend(Backend):
        name = "MyModel"
        ...

    if __name__ == "__main__":
        raise SystemExit(run(MyBackend()))

See ``ocr_vt/screen_translator.py`` (PP-OCRv6) and ``paddleocrvl_vt/screen_translator.py``
(PaddleOCR-VL).
"""
from .shell import Backend, run, t

__all__ = ["Backend", "run", "t"]
