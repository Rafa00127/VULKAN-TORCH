"""DB box postprocess + CTC decode (PaddleOCR-compatible)."""
import cv2
import numpy as np

# DBPostProcess params from PP-OCRv6_medium_det/inference.yml
THRESH, BOX_THRESH, MAX_CAND, UNCLIP = 0.2, 0.45, 3000, 1.4


def _order_box(box):
    """PaddleOCR get_mini_boxes: sort 4 points into [TL, TR, BR, BL]."""
    pts = sorted(box.tolist(), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return np.array([pts[i1], pts[i2], pts[i3], pts[i4]], np.float32)


def _box_score(prob, box):
    """PaddleOCR box_score_fast: mean prob under the polygon."""
    h, w = prob.shape
    b = box.copy()
    xmin = int(np.clip(np.floor(b[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(b[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(b[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(b[:, 1].max()), 0, h - 1))
    b -= [xmin, ymin]
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    cv2.fillPoly(mask, [b.astype(np.int32)], 1)
    return float((prob[ymin:ymax + 1, xmin:xmax + 1] * mask).mean())


def boxes_from_prob(prob, ratio_h, ratio_w):
    """prob: [H,W] float -> list of (4,2) [TL,TR,BR,BL] boxes in ORIGINAL coords."""
    bitmap = (prob > THRESH).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours[:MAX_CAND]:
        if len(cnt) < 4:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
        if min(w, h) < 3:
            continue
        box = _order_box(cv2.boxPoints(((cx, cy), (w, h), ang)))
        if _box_score(prob, box) < BOX_THRESH:
            continue
        # unclip: offset every side outward by area*ratio/perimeter (pyclipper eq. for a rect)
        d = w * h * UNCLIP / (2.0 * (w + h) + 1e-6)
        box = _order_box(cv2.boxPoints(((cx, cy), (w + 2 * d, h + 2 * d), ang)))
        box[:, 0] = np.clip(box[:, 0] / ratio_w, 0, None)
        box[:, 1] = np.clip(box[:, 1] / ratio_h, 0, None)
        boxes.append(box.astype(np.float32))
    return boxes


def crop_quad(img, box):
    """Perspective-crop a 4-point box from HWC uint8 (PaddleOCR get_rotate_crop_image)."""
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if w < 1 or h < 1:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def ctc_decode(logits, chars):
    """logits: [T,C] softmax -> (text, per-step conf). Index 0 = blank, last = space."""
    idx = logits.argmax(-1)
    conf = logits.max(-1)
    out, prev = [], -1
    for t, i in enumerate(idx.tolist()):
        if i != prev and i != 0:
            if 1 <= i <= len(chars):
                out.append((chars[i - 1], float(conf[t])))
            elif i == len(chars) + 1:
                out.append((" ", float(conf[t])))
        prev = i
    return out
