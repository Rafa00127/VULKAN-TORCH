"""Reference speed of the PaddleOCR2Pytorch PP-OCRv6 rec model on the ROCm GPU.

Self-contained (venv side): no vulkantorch import. Segments an image the same way
the port does, runs the torch rec model per line, writes the transcript and prints
``RUN <s> chars <n>`` so it can be diffed against the port's output.

    "<venv>/python.exe" tools/torch_ref_rec.py --image X --out Y [--no-rep] [--batch]
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
from pytorchocr.modeling.architectures.base_model import BaseModel  # noqa: E402
from pytorchocr.modeling.backbones import rec_lcnetv4 as _lcv4  # noqa: E402
import segment  # noqa: E402

# The repo's ConvBNAct.rep() mishandles padding="same" (even kernel): it fuses to a
# symmetric (k-1)//2=0 pad, shrinking the stem branches by 1 and breaking the cat.
# Skip fusing those layers (only the stem's k=2 'same' convs); the rest fuse fine.
_orig_rep = _lcv4.ConvBNAct.rep


def _safe_rep(self):
    if isinstance(self.conv.padding, str):
        return
    _orig_rep(self)


_lcv4.ConvBNAct.rep = _safe_rep

REC_H, REC_W = 48, 320
# the dict file ends with a newline -> split() would yield a trailing "" that shifts
# the space index (len+1); drop it so len(CHARS) == 18708 as the CTC head expects.
CHARS = open(os.path.join(REF, "pytorchocr/utils/dict/ppocrv6_dict.txt"),
             encoding="utf-8").read().split("\n")
if CHARS and CHARS[-1] == "":
    CHARS.pop()


def rec_preprocess(bgr):
    h, w = bgr.shape[:2]
    imgw = int(REC_H * max(REC_W / REC_H, w / h))
    rw = min(int(np.ceil(REC_H * (w / h))), imgw)
    x = cv2.resize(bgr, (rw, REC_H)).astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    pad = np.zeros((3, REC_H, imgw), np.float32)
    pad[:, :, :rw] = x
    return np.ascontiguousarray(pad)


def ctc_decode(logits):
    idx = logits.argmax(-1)
    out, prev = [], -1
    for i in idx.tolist():
        if i != prev and i != 0:
            if 1 <= i <= len(CHARS):
                out.append(CHARS[i - 1])
            elif i == len(CHARS) + 1:
                out.append(" ")
        prev = i
    return "".join(out)


def build_net(rep=True):
    cfg = yaml.safe_load(open(os.path.join(REF, "configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml"),
                              encoding="utf-8"))["Architecture"]
    ocl = {"CTCLabelDecode": 18710, "NRTRLabelDecode": 18714, "SARLabelDecode": 18712}
    net = BaseModel(cfg, out_channels_list=ocl, out_channels=18710)
    w = os.path.join(ROOT, "model", "ppocrv6", "torch",
                     "ptocr_v6_rec_PP-OCRv6_medium_rec_pretrained.pth")
    net.load_state_dict(torch.load(w, map_location="cpu"), strict=False)
    net.eval()
    if rep:
        net.backbone.rep()
    net.to("cuda")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-rep", action="store_true")
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} hip {torch.version.hip} dev {gpu}")
    net = build_net(rep=not a.no_rep)

    crops = [c for c in segment.segment_lines(a.image)]
    xs = [rec_preprocess(np.ascontiguousarray(c[:, :, ::-1])) for c in crops]

    with torch.no_grad():
        # warm every distinct width so MIOpen kernel compilation happens before timing
        for x in xs:
            net(torch.from_numpy(x[None]).cuda())
        for _ in range(3):
            net(torch.from_numpy(xs[0][None]).cuda())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if a.batch:
            W = max(x.shape[2] for x in xs)
            buf = np.zeros((len(xs), 3, REC_H, W), np.float32)
            for i, x in enumerate(xs):
                buf[i, :, :, :x.shape[2]] = x
            out = net(torch.from_numpy(buf).cuda()).cpu().numpy()
            texts = [ctc_decode(out[i]) for i in range(len(xs))]
        else:
            texts = []
            for x in xs:
                out = net(torch.from_numpy(x[None]).cuda())
                texts.append(ctc_decode(out.cpu().numpy()[0]))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    txt = "\n".join(texts)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"RUN {dt:.3f} chars {len(txt)} lines {len(crops)} "
          f"({'batched' if a.batch else 'batch=1'}{'' if a.no_rep else ', repped'})")


if __name__ == "__main__":
    main()
