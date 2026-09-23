"""PaddleOCR-VL on vulkantorch (Python).

The port references llama.cpp's own implementation — ``src/models/paddleocr.cpp`` for
the ERNIE-4.5 backbone with M-RoPE and ``tools/mtmd/models/paddleocr.cpp`` for the
SigLIP vision tower and its mlp_AR projector — because the two consume the same GGUF.
There is no Python reference for this model.

    from paddleocrvl_vt import PaddleOCRVL
    vl = PaddleOCRVL("PaddleOCR-VL-1.6-GGUF.gguf", "...-mmproj.gguf")
    print(vl.generate_image("page.png")[0])
"""
from .vl import PaddleOCRVL

__all__ = ["PaddleOCRVL"]
