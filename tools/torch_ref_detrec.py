"""End-to-end det+rec with the PaddleOCR2Pytorch models on the ROCm GPU.

Self-contained (venv side). Reuses the port's CPU postprocess (box extraction, crop,
CTC decode are framework-agnostic numpy); only inference runs on torch. Same
preprocessing as the port. Mirrors ocr.py:read_line(det=True).

    "<venv>/python.exe" tools/torch_ref_detrec.py --image X --out Y
"""
import argparse
import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "移植参考", "PaddleOCR2Pytorch-main")
for p in (ROOT, REF):
    sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pytorchocr.modeling.architectures.base_model import BaseModel  # noqa: E402
from pytorchocr.modeling.backbones import rec_lcnetv4 as _lcv4  # noqa: E402

# reuse the port's postprocess without importing ocr_py/__init__ (which pulls vulkantorch)
_spec = importlib.util.spec_from_file_location(
    "oc_postproc", os.path.join(ROOT, "example", "python", "ocr_py", "postproc.py"))
postproc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(postproc)

_orig_rep = _lcv4.ConvBNAct.rep
_lcv4.ConvBNAct.rep = lambda s: None if isinstance(s.conv.padding, str) else _orig_rep(s)

DET_LIMIT, DET_MAX_SIDE = 64, 4000
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)
REC_H, REC_W = 48, 320
# drop the dict's trailing newline artifact so len(CHARS) == 18708 (the CTC head's
# space token is at index len+1, so an extra "" entry would break space decoding).
CHARS = open(os.path.join(REF, "pytorchocr/utils/dict/ppocrv6_dict.txt"),
             encoding="utf-8").read().split("\n")
if CHARS and CHARS[-1] == "":
    CHARS.pop()


def det_preprocess(bgr):
    h, w = bgr.shape[:2]
    ratio = DET_LIMIT / min(h, w) if min(h, w) < DET_LIMIT else 1.0
    rh, rw = int(h * ratio), int(w * ratio)
    if max(rh, rw) > DET_MAX_SIDE:
        r = DET_MAX_SIDE / max(rh, rw)
        rh, rw = int(rh * r), int(rw * r)
    rh = max(int(round(rh / 32) * 32), 32)
    rw = max(int(round(rw / 32) * 32), 32)
    x = cv2.resize(bgr, (rw, rh)).astype(np.float32) / 255.0
    x = (x - DET_MEAN) / DET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)), rh / h, rw / w


def rec_preprocess(bgr):
    h, w = bgr.shape[:2]
    imgw = int(REC_H * max(REC_W / REC_H, w / h))
    rw = min(int(np.ceil(REC_H * (w / h))), imgw)
    x = cv2.resize(bgr, (rw, REC_H)).astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    pad = np.zeros((3, REC_H, imgw), np.float32)
    pad[:, :, :rw] = x
    return np.ascontiguousarray(pad)


def build(cfgname, pth, rep=True, **kw):
    cfg = yaml.safe_load(open(os.path.join(REF, "configs", cfgname), encoding="utf-8"))["Architecture"]
    net = BaseModel(cfg, **kw)
    net.load_state_dict(torch.load(os.path.join(ROOT, "model/ppocrv6/torch", pth),
                                   map_location="cpu"), strict=False)
    net.eval()
    if rep:
        net.backbone.rep()
        neck = getattr(net, "neck", None)
        if neck is not None and hasattr(neck, "rep"):
            neck.rep()
    return net.to("cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segmented", action="store_true",
                    help="run det+rec per pre-cut line (mirrors segment.page_text)")
    a = ap.parse_args()

    det = build("det/PP-OCRv6/PP-OCRv6_medium_det.yml",
                "ptocr_v6_det_PP-OCRv6_medium_det_pretrained.pth")
    rec = build("rec/PP-OCRv6/PP-OCRv6_medium_rec.yml",
                "ptocr_v6_rec_PP-OCRv6_medium_rec_pretrained.pth",
                out_channels_list={"CTCLabelDecode": 18710, "NRTRLabelDecode": 18714,
                                   "SARLabelDecode": 18712}, out_channels=18710)

    img = np.array(Image.open(a.image).convert("RGB"))[:, :, ::-1]
    if a.segmented:
        import importlib.util as _iu
        _s = _iu.spec_from_file_location("seg", os.path.join(ROOT, "tools", "segment.py"))
        seg = _iu.module_from_spec(_s)
        _s.loader.exec_module(seg)
        crops = list(seg.segment_lines(a.image))
    else:
        crops = [img]

    def run_one(bgr):
        x, rh, rw = det_preprocess(bgr)
        with torch.no_grad():
            out = det(torch.from_numpy(x[None]).cuda())
            prob = (out["maps"] if isinstance(out, dict) else out).cpu().numpy()[0, 0]
            texts = []
            for b in postproc.boxes_from_prob(prob, rh, rw):
                crop = postproc.crop_quad(bgr, b)
                if crop is None:
                    continue
                lg = rec(torch.from_numpy(rec_preprocess(crop)[None]).cuda())
                texts.append("".join(ch for ch, _ in
                                     postproc.ctc_decode(lg.cpu().numpy()[0], CHARS)))
        return "\n".join(texts)

    for c in crops:                      # warm every det/rec shape MIOpen will need
        run_one(c)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    texts = [run_one(c) for c in crops]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    txt = "\n".join(texts)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"RUN {dt:.3f} chars {len(txt)} crops {len(crops)}")


if __name__ == "__main__":
    main()
