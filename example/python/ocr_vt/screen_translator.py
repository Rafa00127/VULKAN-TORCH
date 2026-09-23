"""划词翻译器 — PP-OCRv6 后端。Draw a box over any on-screen text, OCR it (PP-OCRv6),
optionally translate it with an LLM (OpenAI-compatible endpoint).

    python example/python/ocr_vt/screen_translator.py [--config PATH]

The window, the selection overlay, the subtitle, the settings dialog, the hotkey and the
config file all live in the shared shell (`../screen_translator_core/`); this file is only the
part that is specific to PP-OCRv6. Everything — LLM, hotkey, model dir, options — is in
the config file (see ``DEFAULT_CONFIG`` below for the extra keys on top of the shell's):

    "det":       false,                # false = the box is one line, true = det+rec
    "orient":    "auto",               # "auto" | "h" | "v" (vertical text, manga)
    "model_dir": "",                   # "" = repo default (model/ppocrv6/gguf/)
    "precision": "f32",                # "f32" | "f16"

It resolves to an existing <repo>/data/ocr/screen_translator.json, else to
%APPDATA%/screen-translator/config.json; ``--config PATH`` / ``$OCR_TRANSLATOR_CONFIG``
override.
"""
import os
import sys

_PYDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../example/python
for _p in (os.path.dirname(os.path.dirname(_PYDIR)), _PYDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)      # so `screen_translator` and `ocr_vt` both resolve

import numpy as np                                      # noqa: E402

from screen_translator_core import Backend, run, t           # noqa: E402


def _reading_order(boxes):
    """-> list of visual lines, each a left-to-right list of boxes.

    ``ocr_vt`` follows PaddleOCR's SortQuadBoxes verbatim: order by the top-left corner,
    then only swap boxes whose tops are <10 px apart (a hard pixel threshold, so it does
    not scale with resolution or font size). An inline code span is a *short* box whose
    top sits lower than the text around it, so it ends up after the rest of the line.
    The library stays faithful to the reference; the tool regroups here by how much boxes
    overlap vertically -- a resolution-independent signal for "same line".
    """
    rows = []                                    # [y_top, y_bottom, [boxes]]
    for b in sorted(boxes, key=lambda b: b[:, 1].min()):
        y0, y1 = float(b[:, 1].min()), float(b[:, 1].max())
        for row in rows:
            if min(row[1], y1) - max(row[0], y0) > 0.5 * min(row[1] - row[0], y1 - y0):
                row[0], row[1] = min(row[0], y0), max(row[1], y1)
                row[2].append(b)
                break
        else:
            rows.append([y0, y1, [b]])
    rows.sort(key=lambda r: r[0])
    return [sorted(row[2], key=lambda b: b[:, 0].min()) for row in rows]


def _row_text(row, texts):
    """Join one line's pieces: a real visual gap between two boxes means the source had
    a space there; boxes that abut or overlap are one continuous run (det routinely cuts
    a CJK line mid-sentence, gap <= 0), so those are concatenated. Threshold is relative
    to the box height, so it does not depend on resolution or font size."""
    out = [texts[0]]
    for prev, cur, txt in zip(row, row[1:], texts[1:]):
        h = min(prev[:, 1].max() - prev[:, 1].min(), cur[:, 1].max() - cur[:, 1].min())
        gap = float(cur[:, 0].min()) - float(prev[:, 0].max())
        out.append(" " if gap > 0.2 * h else "")
        out.append(txt)
    return "".join(out)


def read_region(ocr, rgb):
    """det+rec over a captured region, read in order (see _reading_order)."""
    from ocr_vt import postproc
    from ocr_vt.ocr import det_preprocess, rec_preprocess
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    det_in, rh, rw = det_preprocess(bgr)
    boxes = postproc.boxes_from_prob(ocr._run("det", det_in), rh, rw)
    lines = []
    for row in _reading_order(boxes):
        kept, texts = [], []
        for box in row:
            crop = postproc.crop_quad(bgr, box)
            if crop is None:
                continue
            ids = ocr._run("rec_ids", rec_preprocess(crop), np.int32)
            kept.append(box)
            texts.append(postproc.ctc_decode_ids(ids, ocr.chars))
        if kept:
            lines.append(_row_text(kept, texts))
    return "\n".join(lines)


class OcrBackend(Backend):
    name = "PP-OCRv6"
    config_tag = "screen-translator"            # keeps the existing APPDATA dir
    config_env_name = "OCR_TRANSLATOR_CONFIG"   # and the env var the docs already name
    legacy_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "ocr", "screen_translator.json")
    extra_config = {"det": False, "orient": "auto", "model_dir": "", "precision": "f32"}
    extra_strings = {
        "zh": {
            "chk_det": "多行/整块",
            "chk_det_tip": "勾选 = det+rec：框里可有多行。不勾 = 整框当一行（快 ~50×）。",
            "dir_auto": "方向:自动", "dir_h": "方向:横排", "dir_v": "方向:竖排",
            "dir_tip": "竖排（漫画/日文竖排）：裁框后逆时针转 90° 再识别。\n"
                       "自动 = 按裁剪框的高宽比判断（高>宽 视为竖排）。",
            "settings_model_dir": "识别模型目录",
            "settings_model_dir_tip": "留空 = 仓库默认（model/ppocrv6/gguf/），也可用环境变量 "
                                      "OCR_MODEL_DIR。\n改这里会立刻重新加载模型。",
            "settings_precision": "精度",
            "settings_precision_tip": "f16 需要该目录里存在对应的 f16 GGUF；缺了就回落 f32。",
            "ph_model_dir": "留空 = 仓库默认 model/ppocrv6/gguf",
        },
        "en": {
            "chk_det": "Multi-line",
            "chk_det_tip": "On = det+rec: the box may hold several lines. Off = treat the "
                           "whole box as one line (~50× faster).",
            "dir_auto": "Dir:auto", "dir_h": "Dir:horiz", "dir_v": "Dir:vert",
            "dir_tip": "Vertical (manga / vertical Japanese): rotate the crop 90° CCW before "
                       "OCR.\nAuto = decide by the box's aspect ratio (taller than wide → vertical).",
            "settings_model_dir": "OCR model dir",
            "settings_model_dir_tip": "Empty = the repo default (model/ppocrv6/gguf/); "
                                      "OCR_MODEL_DIR works too.\nChanging it reloads the model "
                                      "right away.",
            "settings_precision": "Precision",
            "settings_precision_tip": "f16 needs an f16 GGUF in that dir; falls back to f32 otherwise.",
            "ph_model_dir": "empty = repo default model/ppocrv6/gguf",
        },
    }

    # -- model ---------------------------------------------------------------------

    def reload_key(self, cfg):
        return (cfg.get("model_dir", ""), cfg.get("precision", "f32"))

    def load(self, cfg):
        from ocr_vt.ocr import Ocr
        kw = {"precision": cfg.get("precision") or None}
        if cfg.get("model_dir"):
            kw["model_dir"] = cfg["model_dir"]
        return Ocr(**kw)

    def recognize(self, ocr, arr, options):
        from PIL import Image
        img = Image.fromarray(arr)
        # vertical text (manga): make the tall column a horizontal strip — rotate 90° CCW
        # so the glyphs lie along a left-to-right line (what rec reads).
        orient = options.get("orient", "auto")
        if orient == "v" or (orient == "auto" and arr.shape[0] > arr.shape[1]):
            img = img.rotate(90, expand=True)
        arr = np.asarray(img)
        # det off = one pre-cut line, nothing to order; det on = a region that may hold
        # several boxes, so read it in reading order
        return read_region(ocr, arr) if options.get("det") else ocr.read_line(arr)

    # -- widgets -------------------------------------------------------------------

    def options(self, parent, cfg):
        from PyQt6.QtWidgets import QCheckBox, QComboBox, QWidget, QHBoxLayout
        from screen_translator_core import t         # the shell's i18n table
        w = QWidget(parent)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        chk = QCheckBox(t("chk_det"), w)
        chk.setToolTip(t("chk_det_tip"))
        chk.setChecked(bool(cfg.get("det")))
        cmb = QComboBox(w)
        for key, val in [("dir_auto", "auto"), ("dir_h", "h"), ("dir_v", "v")]:
            cmb.addItem(t(key), val)
        for i in range(cmb.count()):
            if cmb.itemData(i) == cfg.get("orient", "auto"):
                cmb.setCurrentIndex(i)
                break
        cmb.setToolTip(t("dir_tip"))
        lay.addWidget(chk)
        lay.addWidget(cmb)
        return w, (lambda: {"det": chk.isChecked(), "orient": cmb.currentData()})

    def settings_section(self, parent, cfg):
        from PyQt6.QtWidgets import (QWidget, QFormLayout, QHBoxLayout, QLineEdit,
                                     QPushButton, QComboBox, QFileDialog)
        from screen_translator_core import t
        w = QWidget(parent)
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)
        ed_dir = QLineEdit(cfg.get("model_dir", ""), w)
        ed_dir.setPlaceholderText(t("ph_model_dir"))
        ed_dir.setToolTip(t("settings_model_dir_tip"))
        b_browse = QPushButton(t("settings_browse"), w)
        row = QHBoxLayout()
        row.addWidget(ed_dir)
        row.addWidget(b_browse)
        form.addRow(t("settings_model_dir"), row)
        cmb_prec = QComboBox(w)
        for p_ in ("f32", "f16"):
            cmb_prec.addItem(p_, p_)
        cmb_prec.setCurrentIndex(max(0, cmb_prec.findData(cfg.get("precision", "f32"))))
        cmb_prec.setToolTip(t("settings_precision_tip"))
        form.addRow(t("settings_precision"), cmb_prec)

        def browse():
            start = ed_dir.text().strip() or os.path.expanduser("~")
            d = QFileDialog.getExistingDirectory(w, t("settings_model_dir"), start)
            if d:
                ed_dir.setText(d)

        b_browse.clicked.connect(browse)
        return w, (lambda: {"model_dir": ed_dir.text().strip(),
                            "precision": cmb_prec.currentData()})

    def selftest(self):
        import ocr_vt  # noqa: F401  (verifies the sys.path wiring; no model load)


if __name__ == "__main__":
    raise SystemExit(run(OcrBackend()))
