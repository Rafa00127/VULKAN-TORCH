"""PP-OCRv6 (det + rec) inference on vulkan-torch, segmentation-based.

Ports the user's `ocr_server_gui.py` workload: invert → slice fixed-height line
bands → trim → OCR each line. Weights come from the GGUF written by
tools/convert_ocr_to_gguf.py; reference text from the CPU PaddleOCR pipeline.
"""
