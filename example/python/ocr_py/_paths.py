"""Repo/data paths for ocr_py.

Weights live in ``<repo>/model/ppocrv6/gguf/`` by default, but every path can be
overridden so the library works when the models are downloaded elsewhere:

    OCR_MODEL_DIR    dir holding the GGUF files (default <repo>/model/ppocrv6/gguf/)
    OCR_PRECISION    f32 (default) | f16  -> ppocrv6_{rec,det}.<p>.gguf
    OCR_REC_GGUF     explicit recognition GGUF path
    OCR_DET_GGUF     explicit detection GGUF path
    OCR_DICT         explicit CTC dictionary path

``Ocr()`` also takes ``model_dir``/``precision``/``rec_path``/``det_path``/``dict_path``
for per-call overrides (which win over the env vars).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
MODEL = os.path.join(ROOT, "model")                  # converted weights (gitignored)
TORCH = os.path.join(MODEL, "ppocrv6", "torch")      # PaddleOCR2Pytorch reference
DATA = os.path.join(ROOT, "data", "ocr")             # OCR scratch (refs, transcripts)

GGUF = os.environ.get("OCR_MODEL_DIR") or os.path.join(MODEL, "ppocrv6", "gguf")
PRECISION = os.environ.get("OCR_PRECISION", "f32")   # applies to both rec and det


def _at(precision, name):
    """Resolve ``<name>.<precision>.gguf`` in the model dir, falling back to ``.f32``
    when the requested precision wasn't converted (e.g. no f16 det build)."""
    p = os.path.join(GGUF, f"{name}.{precision}.gguf")
    return p if os.path.isfile(p) else os.path.join(GGUF, f"{name}.f32.gguf")


DET_GGUF = os.environ.get("OCR_DET_GGUF", _at(PRECISION, "ppocrv6_det"))
REC_GGUF = os.environ.get("OCR_REC_GGUF", _at(PRECISION, "ppocrv6_rec"))
DICT_TXT = os.environ.get("OCR_DICT", os.path.join(GGUF, "ppocrv6_dict.txt"))
REF = os.path.join(DATA, "ref")
