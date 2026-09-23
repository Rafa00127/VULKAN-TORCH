"""DB box postprocess + CTC decode.

The DB part is a line-for-line port of PaddleOCR's
``paddlex...text_detection.processors.DBPostProcess``. The thresholds below are the
ones the OCR *pipeline* actually uses, from
``paddlex/configs/pipelines/OCR.yaml`` (SubModules.TextDetection) — note these
override the model's own ``inference.yml`` (which says 0.2/0.45/1.4).
"""
import cv2
import numpy as np

# paddlex/configs/pipelines/OCR.yaml -> SubModules.TextDetection
THRESH, BOX_THRESH, UNCLIP = 0.3, 0.6, 1.5
MAX_CAND = 1000        # DBPostProcess default (not overridden by the pipeline)
MIN_SIZE = 3           # DBPostProcess.min_size


def _get_mini_boxes(contour):
    """PaddleOCR get_mini_boxes: minAreaRect + sort points into [TL, TR, BR, BL]."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(contour.astype(np.float32))
    pts = sorted(cv2.boxPoints(((cx, cy), (w, h), ang)).tolist(), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return [pts[i1], pts[i2], pts[i3], pts[i4]], min(w, h)


def _unclip(box):
    """PaddleOCR unclip: grow the polygon by area*ratio/perimeter on every side.
    pyclipper's round offset of a rectangle == scaling it to (w+2d, h+2d)."""
    b = np.asarray(box, np.float32).reshape(-1, 2)
    d = cv2.contourArea(b) * UNCLIP / (cv2.arcLength(b, True) + 1e-6)
    (cx, cy), (w, h), ang = cv2.minAreaRect(b)
    return cv2.boxPoints(((cx, cy), (w + 2 * d, h + 2 * d), ang))


def _box_score(prob, box):
    """PaddleOCR box_score_fast: mean prob under the polygon."""
    h, w = prob.shape
    b = np.asarray(box, np.float32).copy()
    xmin = max(0, min(int(np.floor(b[:, 0].min())), w - 1))
    xmax = max(0, min(int(np.ceil(b[:, 0].max())), w - 1))
    ymin = max(0, min(int(np.floor(b[:, 1].min())), h - 1))
    ymax = max(0, min(int(np.ceil(b[:, 1].max())), h - 1))
    b[:, 0] -= xmin
    b[:, 1] -= ymin
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    cv2.fillPoly(mask, [b.astype(np.int32).reshape(-1, 1, 2)], 1)
    return cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def sort_quad_boxes(boxes):
    """PaddleOCR SortQuadBoxes: order top->bottom then left->right (in scaled px).

    Sort by the top-left point's (y, x), then a bubble pass that swaps neighbours
    whose y differ by <10 px and whose x is out of order."""
    if len(boxes) == 0:
        return boxes
    boxes = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    for i in range(len(boxes) - 1):
        for j in range(i, -1, -1):
            if (abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10
                    and boxes[j + 1][0][0] < boxes[j][0][0]):
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes


def boxes_from_prob(prob, ratio_h, ratio_w):
    """prob [H,W] -> list of (4,2) [TL,TR,BR,BL] boxes in ORIGINAL image coords.

    ``ratio_*`` are resized/original scales from det_preprocess."""
    h, w = prob.shape
    dest_w, dest_h = w / ratio_w, h / ratio_h
    ws, hs = dest_w / w, dest_h / h

    bitmap = (prob > THRESH).astype(np.uint8)
    contours = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    boxes = []
    for cnt in contours[:MAX_CAND]:
        pts, sside = _get_mini_boxes(cnt)
        if sside < MIN_SIZE:
            continue
        if _box_score(prob, pts) < BOX_THRESH:
            continue
        box, sside = _get_mini_boxes(_unclip(pts))
        if sside < MIN_SIZE + 2:
            continue
        box = np.asarray(box, np.float32)
        box[:, 0] = np.clip(np.round(box[:, 0] * ws), 0, dest_w)
        box[:, 1] = np.clip(np.round(box[:, 1] * hs), 0, dest_h)
        boxes.append(box)
    return sort_quad_boxes(boxes)


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
    """logits: [T,C] softmax -> [(char, conf)]. Index 0 = blank, last = space."""
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


def ctc_decode_ids(idx, chars):
    """CTC greedy decode from argmax ids -> str. Same result as joining
    ``ctc_decode``, but the ids come straight off the GPU (no [T,C] map transferred).
    Index 0 = blank, len(chars)+1 = space. Accepts any shape (flattened)."""
    out, prev = [], -1
    for i in np.asarray(idx).reshape(-1).tolist():
        i = int(i)
        if i != prev and i != 0:
            if 1 <= i <= len(chars):
                out.append(chars[i - 1])
            elif i == len(chars) + 1:
                out.append(" ")
        prev = i
    return "".join(out)
