"""Repo/data paths for ocr_py."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
MODEL = os.path.join(ROOT, "model")          # converted GGUF weights (gitignored)
DATA = os.path.join(ROOT, "data", "ocr")     # OCR scratch (refs, transcripts)
DET_GGUF = os.environ.get("OCR_DET_GGUF", os.path.join(MODEL, "ppocrv6_det.f32.gguf"))
REC_GGUF = os.environ.get("OCR_REC_GGUF", os.path.join(MODEL, "ppocrv6_rec.f32.gguf"))
DICT_TXT = os.path.join(MODEL, "ppocrv6_dict.txt")
REF = os.path.join(DATA, "ref")
