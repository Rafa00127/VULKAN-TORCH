"""Model paths for paddleocrvl_vt.

Both GGUF files are needed: the language model and the vision tower (the mmproj).
They are not in the repo, so the default is ``<repo>/model/paddleocrvl/`` and every
path can be overridden:

    PADDLEOCRVL_MODEL_DIR   dir holding the two GGUFs
    PADDLEOCRVL_GGUF        explicit language-model GGUF
    PADDLEOCRVL_MMPROJ      explicit mmproj GGUF
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

MODEL_DIR = os.environ.get("PADDLEOCRVL_MODEL_DIR") or os.path.join(ROOT, "model", "paddleocrvl")

# A relative --model/--mmproj on the command line is resolved against this dir.
DEFAULT_GGUF = os.environ.get("PADDLEOCRVL_GGUF",
                              os.path.join(MODEL_DIR, "PaddleOCR-VL-1.6-GGUF.gguf"))
DEFAULT_MMPROJ = os.environ.get("PADDLEOCRVL_MMPROJ",
                                os.path.join(MODEL_DIR, "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"))
