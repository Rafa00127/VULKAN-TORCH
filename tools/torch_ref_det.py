"""Reference speed/prob-map of the PaddleOCR2Pytorch PP-OCRv6 det model on the ROCm GPU.

Self-contained (venv side): no vulkantorch import. Uses the same det preprocessing as
the port (limit_side_len=64/min/4000). Writes the prob map (npy) and prints timing.

    "<venv>/python.exe" tools/torch_ref_det.py --image X [--out prob.npy]
                                               [--no-rep] [--iters N]
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "移植参考", "PaddleOCR2Pytorch-main")
for p in (ROOT, os.path.join(ROOT, "tools"), REF):
    sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pytorchocr.modeling.architectures.base_model import BaseModel  # noqa: E402
from pytorchocr.modeling.backbones import rec_lcnetv4 as _lcv4  # noqa: E402

# same fix as tools/torch_ref_rec.py: the repo's rep() mishandles padding="same"
_orig_rep = _lcv4.ConvBNAct.rep
_lcv4.ConvBNAct.rep = lambda s: None if isinstance(s.conv.padding, str) else _orig_rep(s)

DET_LIMIT, DET_MAX_SIDE = 64, 4000
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)


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


def build_net(rep=True):
    cfg = yaml.safe_load(open(os.path.join(REF, "configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml"),
                              encoding="utf-8"))["Architecture"]
    net = BaseModel(cfg)
    pth = os.path.join(ROOT, "model", "ppocrv6", "torch",
                       "ptocr_v6_det_PP-OCRv6_medium_det_pretrained.pth")
    net.load_state_dict(torch.load(pth, map_location="cpu"), strict=False)
    net.eval()
    if rep:
        net.backbone.rep()
        if hasattr(net.neck, "rep"):
            net.neck.rep()
    net.to("cuda")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None, help="write the prob map here (.npy)")
    ap.add_argument("--no-rep", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    print(f"torch {torch.__version__} hip {torch.version.hip} dev {torch.cuda.get_device_name(0)}")
    net = build_net(rep=not a.no_rep)

    bgr = np.array(Image.open(a.image).convert("RGB"))[:, :, ::-1]
    x, rh, rw = det_preprocess(bgr)
    t = torch.from_numpy(x[None]).cuda()
    print(f"image {bgr.shape[1]}x{bgr.shape[0]} -> det input {tuple(x.shape)}")

    with torch.no_grad():
        for _ in range(3):
            out = net(t)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            out = net(t)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.iters * 1000
        prob = (out["maps"] if isinstance(out, dict) else out).cpu().numpy()[0, 0]

    if a.out:
        np.save(a.out, prob.astype(np.float32))
    print(f"RUN {dt:.2f} ms   prob {prob.shape}  rh={rh:.4f} rw={rw:.4f}"
          f"{'' if a.no_rep else '  repped'}")


if __name__ == "__main__":
    main()
