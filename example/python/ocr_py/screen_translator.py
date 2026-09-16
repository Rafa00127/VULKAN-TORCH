"""划词翻译器 — draw a box over any on-screen text, OCR it (PP-OCRv6), optionally
translate it with an LLM (OpenAI-compatible endpoint).

PyQt6 UI. Click 「截屏划词」(or the global hotkey) -> the tool hides, a full-screen
overlay lets you drag a box (screen stays as-is, only the box is outlined) -> the box is
recognised (rec-only -> text) and, if the translate box is ticked, sent to the LLM.

    python example/python/ocr_py/screen_translator.py [--config PATH]

The UI is bilingual (中文 / English); pick the language in 「设置」/ Settings (saved in
config; takes effect on restart). LLM + hotkey + options live in the config file
(default: <repo>/data/ocr/screen_translator.json, gitignored):

    "ui_lang":   "zh",                 # "zh" | "en"
    "api_base":  "https://api.deepseek.com",   # any OpenAI-compatible base URL
    "api_model": "",                   # must be set (no default)
    "api_key":   ""                    # fill in here or in Settings

No paths are hardcoded; the API key lives in the (gitignored) config.
The hotkey is a SYSTEM-WIDE hotkey (Windows RegisterHotKey); OCR + network run on one
persistent worker thread, so the UI never blocks.
"""
import argparse
import ctypes
import json
import os
import queue
import sys
import time

import numpy as np

PYDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../example/python
ROOT = os.path.dirname(os.path.dirname(PYDIR))                          # repo root
for p in (ROOT, PYDIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from PyQt6.QtCore import Qt, QRect, QThread, pyqtSignal, QAbstractNativeEventFilter
from PyQt6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QGuiApplication,
                         QKeySequence, QShortcut)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QPushButton, QCheckBox, QPlainTextEdit,
                             QLabel, QComboBox, QKeySequenceEdit, QDialogButtonBox, QLineEdit)

# (native name, English name) — the target language list; shown as "→ <native>"
LANGUAGES = [("中文", "Chinese"), ("English", "English"), ("日本語", "Japanese"),
             ("한국어", "Korean"), ("Español", "Spanish"), ("Français", "French")]

DEFAULT_CONFIG = {"hotkey": "Ctrl+Alt+Shift+O", "target": "Chinese", "translate": False,
                  "det": False, "orient": "auto", "reasoning": "off", "ui_lang": "zh",
                  "api_base": "https://api.deepseek.com", "api_model": "",
                  "api_key": ""}


# ── i18n ─────────────────────────────────────────────────────────────────────────

STRINGS = {
    "zh": {
        "app_title": "划词翻译 (PP-OCRv6 + LLM)",
        "btn_capture": "截屏划词", "btn_settings": "设置",
        "chk_det": "多行/整块",
        "chk_det_tip": "勾选 = det+rec：框里可有多行。不勾 = 整框当一行（快 ~50×）。",
        "chk_translate": "翻译",
        "reason_off": "思考:关", "reason_low": "思考:低", "reason_high": "思考:高", "reason_max": "思考:最高",
        "reason_tip": "仅翻译时生效：off = 关闭思考；low/high/max = 开启思考并设推理强度。",
        "dir_auto": "方向:自动", "dir_h": "方向:横排", "dir_v": "方向:竖排",
        "dir_tip": "竖排（漫画/日文竖排）：裁框后逆时针转 90° 再识别。\n自动 = 按裁剪框的高宽比判断（高>宽 视为竖排）。",
        "lbl_src": "识别文本", "lbl_dst": "翻译",
        "status_hotkey_global": "快捷键 {key}：全局可用 · 点「{btn}」或按快捷键开始",
        "status_hotkey_scope": "快捷键 {key}：仅窗口聚焦时有效（该组合可能被占用） · 点「{btn}」或按快捷键开始",
        "status_capture_failed": "截图失败: {err}",
        "status_capturing": "识别中… ({w}×{h})",
        "status_recognized": "识别完成 · {n} 字{tail}",
        "status_translating": "  （翻译中…）",
        "status_done": "完成 · {n} 字（已翻译）",
        "status_error": "出错: ",
        "status_settings_saved": "设置已保存",
        "status_bad_hotkey": "设置已保存；但快捷键 '{key}' 无效（需修饰键+键），未改。",
        "status_lang_changed": "界面语言已保存 —— 重启程序后生效。",
        "settings_title": "设置",
        "settings_hotkey": "截图快捷键",
        "settings_hotkey_tip": "点这里然后按组合键。需要至少一个修饰键（Ctrl/Alt/Shift/Win）。",
        "settings_uilang": "界面语言",
        "settings_model": "模型",
        "ph_base": "https://api.deepseek.com  (OpenAI 兼容 base url)",
        "ph_model": "填模型名，如 deepseek-chat / deepseek-reasoner",
        "ph_key": "在此填入 API key（保存进 config）",
        "err_no_key": "未设置 API key —— 在「设置」里填 api_key（存进 config）",
        "err_no_model": "未设置模型名 —— 在「设置」里填 api_model（如 deepseek-chat / deepseek-reasoner）",
    },
    "en": {
        "app_title": "Screen Translator (PP-OCRv6 + LLM)",
        "btn_capture": "Capture", "btn_settings": "Settings",
        "chk_det": "Multi-line",
        "chk_det_tip": "On = det+rec: the box may hold several lines. Off = treat the whole box as one line (~50× faster).",
        "chk_translate": "Translate",
        "reason_off": "Think:off", "reason_low": "Think:low", "reason_high": "Think:high", "reason_max": "Think:max",
        "reason_tip": "Translation only: off = no thinking; low/high/max = enable thinking at that reasoning effort.",
        "dir_auto": "Dir:auto", "dir_h": "Dir:horiz", "dir_v": "Dir:vert",
        "dir_tip": "Vertical (manga / vertical Japanese): rotate the crop 90° CCW before OCR.\nAuto = decide by the box's aspect ratio (taller than wide → vertical).",
        "lbl_src": "Recognized text", "lbl_dst": "Translation",
        "status_hotkey_global": "Hotkey {key}: system-wide · click \"{btn}\" or press the hotkey to start",
        "status_hotkey_scope": "Hotkey {key}: only while this window is focused (combo may be taken) · click \"{btn}\" or press the hotkey",
        "status_capture_failed": "Capture failed: {err}",
        "status_capturing": "Recognizing… ({w}×{h})",
        "status_recognized": "Recognized · {n} chars{tail}",
        "status_translating": "  (translating…)",
        "status_done": "Done · {n} chars (translated)",
        "status_error": "Error: ",
        "status_settings_saved": "Settings saved",
        "status_bad_hotkey": "Saved, but hotkey '{key}' is invalid (needs a modifier + a key) — left unchanged.",
        "status_lang_changed": "UI language saved — restart to apply.",
        "settings_title": "Settings",
        "settings_hotkey": "Capture hotkey",
        "settings_hotkey_tip": "Click, then press the combo. Needs at least one modifier (Ctrl/Alt/Shift/Win).",
        "settings_uilang": "UI language",
        "settings_model": "Model",
        "ph_base": "https://api.deepseek.com  (OpenAI-compatible base url)",
        "ph_model": "model name, e.g. deepseek-chat / deepseek-reasoner",
        "ph_key": "paste API key here (saved into config)",
        "err_no_key": "no API key set — fill api_key in Settings (stored in config)",
        "err_no_model": "no model set — fill api_model in Settings (e.g. deepseek-chat / deepseek-reasoner)",
    },
}

_LANG = "zh"


def t(_key, **kw):
    s = STRINGS.get(_LANG, {}).get(_key) or STRINGS["zh"].get(_key) or _key
    return s.format(**kw) if kw else s


# ── settings file ───────────────────────────────────────────────────────────────

def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:      # noqa: BLE001
        print(f"warn: bad config {path}: {e}", file=sys.stderr)
    return cfg


def save_config(path, cfg):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:      # noqa: BLE001
        print(f"warn: could not save {path}: {e}", file=sys.stderr)


# ── global hotkey (Windows) ──────────────────────────────────────────────────────

WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
_VK = {**{c: ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
       **{str(d): ord(str(d)) for d in range(10)},
       **{f"F{i}": 0x6F + i for i in range(1, 13)},
       "SPACE": 0x20, "TAB": 0x09, "ENTER": 0x0D, "RETURN": 0x0D, "ESC": 0x1B, "ESCAPE": 0x1B,
       "INSERT": 0x2D, "DELETE": 0x2E, "HOME": 0x24, "END": 0x23,
       "PAGEUP": 0x21, "PAGEDOWN": 0x22}


def parse_hotkey(spec):
    """'Ctrl+Alt+O' -> (mods, vk), or None. Needs >=1 modifier + 1 key."""
    mods, vk = 0, None
    for p in [x.strip() for x in spec.replace(" ", "").split("+") if x.strip()]:
        low = p.lower()
        if low in ("ctrl", "control"): mods |= MOD_CONTROL
        elif low == "alt": mods |= MOD_ALT
        elif low == "shift": mods |= MOD_SHIFT
        elif low in ("win", "meta", "super"): mods |= MOD_WIN
        elif p.upper() in _VK: vk = _VK[p.upper()]
        else: return None
    return (mods, vk) if vk is not None and mods else None


class GlobalHotkey:
    """System-wide hotkey via RegisterHotKey; a no-op (returns False) off Windows."""

    def __init__(self, app, callback):
        self._cb = callback
        self._id = 1
        self._ok = False
        self._user32 = None
        if sys.platform != "win32":
            return
        try:
            self._user32 = ctypes.windll.user32
            outer = self

            class _Filter(QAbstractNativeEventFilter):
                def nativeEventFilter(self, etype, message):    # noqa: N802
                    try:
                        from ctypes import wintypes
                        msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
                        if msg.message == WM_HOTKEY:
                            outer._cb()
                            return True, 0
                    except Exception:      # noqa: BLE001
                        pass
                    return False, 0

            self._filter = _Filter()
            app.installNativeEventFilter(self._filter)
        except Exception as e:      # noqa: BLE001
            print(f"warn: global hotkey unavailable: {e}", file=sys.stderr)
            self._user32 = None

    def set(self, spec):
        if self._user32 is None:
            return False
        if self._ok:
            self._user32.UnregisterHotKey(None, self._id)
            self._ok = False
        parsed = parse_hotkey(spec)
        if not parsed:
            return False
        mods, vk = parsed
        self._ok = bool(self._user32.RegisterHotKey(None, self._id, mods | MOD_NOREPEAT, vk))
        return self._ok

    def release(self):
        if self._user32 is not None and self._ok:
            self._user32.UnregisterHotKey(None, self._id)
            self._ok = False


# ── OCR + translation worker (one thread, one Ocr instance) ──────────────────────

def _reading_order(boxes):
    """-> list of visual lines, each a left-to-right list of boxes.

    ``ocr_py`` follows PaddleOCR's SortQuadBoxes verbatim: order by the top-left corner,
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
    for prev, cur, t in zip(row, row[1:], texts[1:]):
        h = min(prev[:, 1].max() - prev[:, 1].min(), cur[:, 1].max() - cur[:, 1].min())
        gap = float(cur[:, 0].min()) - float(prev[:, 0].max())
        out.append(" " if gap > 0.2 * h else "")
        out.append(t)
    return "".join(out)


def read_region(ocr, rgb):
    """det+rec over a captured region, read in order (see _reading_order)."""
    from ocr_py import postproc
    from ocr_py.ocr import det_preprocess, rec_preprocess
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


class Worker(QThread):
    recognized = pyqtSignal(str)   # OCR text, emitted immediately
    translated = pyqtSignal(str)   # translation ("" if none) — emitted after
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._q = queue.Queue()
        self._ocr = None
        self._llm = {"base": "", "model": "", "key": ""}

    def set_llm(self, base, model, key):
        self._llm = {"base": base or "", "model": model or "", "key": key or ""}

    def submit(self, arr, translate, target, det, orient, reasoning):
        self._q.put((arr, translate, target, det, orient, reasoning))

    def stop(self):
        self._q.put(None)
        self.wait(3000)

    def run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            arr, do_translate, target, det, orient, reasoning = item
            try:
                from PIL import Image
                img = Image.fromarray(arr)
                # vertical text (manga): make the tall column a horizontal strip — rotate
                # 90° CCW so the glyphs lie along a left-to-right line (what rec reads).
                if orient == "v" or (orient == "auto" and arr.shape[0] > arr.shape[1]):
                    img = img.rotate(90, expand=True)
                arr = np.asarray(img)
                if self._ocr is None:
                    from ocr_py.ocr import Ocr
                    self._ocr = Ocr()
                # det off = one pre-cut line, nothing to order; det on = a region that
                # may hold several boxes, so read it in reading order
                text = read_region(self._ocr, arr) if det else self._ocr.read_line(arr)
                self.recognized.emit(text)          # show source the moment OCR is done
                trans = ""
                if do_translate and text.strip():
                    trans = translate_text(text, target, self._llm, reasoning)
                self.translated.emit(trans)         # then fill in the translation
            except Exception as e:      # noqa: BLE001
                self.error.emit(f"{type(e).__name__}: {e}")


def translate_text(text, target, llm, reasoning="off"):
    """Chat completion against an OpenAI-compatible endpoint (base/model/key from config).

    reasoning ∈ {off, low, high, max}: off disables thinking
    (thinking.type=disabled), the rest enable it with reasoning_effort."""
    import requests
    key = (llm.get("key") or "").strip()
    if not key:
        raise RuntimeError(t("err_no_key"))
    model = (llm.get("model") or "").strip()
    if not model:
        raise RuntimeError(t("err_no_model"))
    base = (llm.get("base") or "https://api.deepseek.com").strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": f"Translate the user's text into {target}. Output only the "
                        f"translation — no quotes, no explanation, no original."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2, "stream": False,
    }
    if reasoning in ("low", "high", "max"):
        payload["reasoning_effort"] = reasoning
        payload["thinking"] = {"type": "enabled"}
    else:                                   # off / anything else → no thinking
        payload["thinking"] = {"type": "disabled"}

    r = requests.post(
        f"{base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ── full-screen selection overlay (no dimming — screen stays as-is) ───────────────

class CaptureOverlay(QWidget):
    picked = pyqtSignal(object)   # PIL.Image crop, or None if cancelled

    def __init__(self, pil_img, disp_pixmap, geo, scale):
        super().__init__(None, Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self._img = pil_img          # physical-res PIL image (for the OCR crop)
        self._disp = disp_pixmap     # logical-size QPixmap (for painting)
        self._scale = scale          # (sx, sy) physical / logical
        self.setGeometry(geo)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._o = None
        self._c = None

    def paintEvent(self, _):
        p = QPainter(self)
        p.drawPixmap(0, 0, self._disp)          # exact screen copy, no tint
        if self._o and self._c:
            r = QRect(self._o, self._c).normalized()
            p.setPen(QPen(QColor(90, 170, 255), 2))
            p.drawRect(r)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._o = e.position().toPoint()
            self._c = self._o
            self.update()

    def mouseMoveEvent(self, e):
        if self._o:
            self._c = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton or not self._o:
            return
        r = QRect(self._o, self._c).normalized()
        self.close()
        if r.width() < 4 or r.height() < 4:
            self.picked.emit(None)
            return
        sx, sy = self._scale
        box = (int(r.x() * sx), int(r.y() * sy), int(r.right() * sx), int(r.bottom() * sy))
        self.picked.emit(self._img.crop(box))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
            self.picked.emit(None)


# ── settings dialog ──────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, parent, cfg):
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        form = QFormLayout(self)
        self.keyedit = QKeySequenceEdit(QKeySequence(cfg.get("hotkey", DEFAULT_CONFIG["hotkey"])))
        self.keyedit.setToolTip(t("settings_hotkey_tip"))
        form.addRow(t("settings_hotkey"), self.keyedit)

        self.cmb_ui = QComboBox()
        for label, val in [("中文", "zh"), ("English", "en")]:
            self.cmb_ui.addItem(label, val)
        for i in range(self.cmb_ui.count()):
            if self.cmb_ui.itemData(i) == cfg.get("ui_lang", "zh"):
                self.cmb_ui.setCurrentIndex(i)
                break
        form.addRow(t("settings_uilang"), self.cmb_ui)

        self.ed_base = QLineEdit(cfg.get("api_base", ""))
        self.ed_base.setPlaceholderText(t("ph_base"))
        self.ed_model = QLineEdit(cfg.get("api_model", ""))
        self.ed_model.setPlaceholderText(t("ph_model"))
        self.ed_key = QLineEdit(cfg.get("api_key", ""))
        self.ed_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_key.setPlaceholderText(t("ph_key"))
        form.addRow("API base", self.ed_base)
        form.addRow(t("settings_model"), self.ed_model)
        form.addRow("API key", self.ed_key)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def hotkey(self):
        return self.keyedit.keySequence().toString()

    def ui_lang(self):
        return self.cmb_ui.currentData()

    def llm(self):
        return {"api_base": self.ed_base.text().strip(),
                "api_model": self.ed_model.text().strip(),
                "api_key": self.ed_key.text().strip()}


# ── main window ──────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, args, cfg, cfg_path):
        super().__init__()
        self._args = args
        self._cfg = cfg
        self._cfg_path = cfg_path
        self._overlay = None
        self.setWindowTitle(t("app_title"))
        self.resize(460, 380)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self._worker = Worker()
        self._worker.recognized.connect(self._on_recognized)
        self._worker.translated.connect(self._on_translated)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._worker.set_llm(cfg.get("api_base"), cfg.get("api_model"), cfg.get("api_key"))

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        top = QHBoxLayout()
        self.btn = QPushButton(t("btn_capture"))
        self.btn.setStyleSheet("QPushButton{padding:8px;font-weight:bold}")
        self.btn.clicked.connect(self.capture)
        self.chk_det = QCheckBox(t("chk_det"))
        self.chk_det.setToolTip(t("chk_det_tip"))
        self.chk = QCheckBox(t("chk_translate"))
        self.cmb = QComboBox()
        for label, name in LANGUAGES:
            self.cmb.addItem(f"→ {label}", name)
        self.cmb_reason = QComboBox()
        for key, val in [("reason_off", "off"), ("reason_low", "low"),
                         ("reason_high", "high"), ("reason_max", "max")]:
            self.cmb_reason.addItem(t(key), val)
        self.cmb_reason.setToolTip(t("reason_tip"))
        self.btn_set = QPushButton(t("btn_settings"))
        self.btn_set.clicked.connect(self.open_settings)
        self.cmb_dir = QComboBox()
        for key, val in [("dir_auto", "auto"), ("dir_h", "h"), ("dir_v", "v")]:
            self.cmb_dir.addItem(t(key), val)
        self.cmb_dir.setToolTip(t("dir_tip"))
        top.addWidget(self.btn)
        top.addWidget(self.chk_det)
        top.addWidget(self.cmb_dir)
        top.addWidget(self.chk)
        top.addWidget(self.cmb)
        top.addWidget(self.cmb_reason)
        top.addWidget(self.btn_set)
        top.addStretch(1)
        lay.addLayout(top)

        lay.addWidget(QLabel(t("lbl_src")))
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        lay.addWidget(self.txt, 2)
        lay.addWidget(QLabel(t("lbl_dst")))
        self.trans = QPlainTextEdit()
        self.trans.setReadOnly(True)
        lay.addWidget(self.trans, 3)

        # restore saved settings
        self.chk_det.setChecked(bool(cfg.get("det")))
        self.chk.setChecked(bool(cfg.get("translate")))
        for i in range(self.cmb.count()):
            if self.cmb.itemData(i) == cfg.get("target"):
                self.cmb.setCurrentIndex(i)
                break
        for i in range(self.cmb_dir.count()):
            if self.cmb_dir.itemData(i) == cfg.get("orient"):
                self.cmb_dir.setCurrentIndex(i)
                break
        for i in range(self.cmb_reason.count()):
            if self.cmb_reason.itemData(i) == cfg.get("reasoning"):
                self.cmb_reason.setCurrentIndex(i)
                break
        self.chk_det.toggled.connect(self._persist)
        self.chk.toggled.connect(self._persist)
        self.cmb.currentIndexChanged.connect(self._persist)
        self.cmb_dir.currentIndexChanged.connect(self._persist)
        self.cmb_reason.currentIndexChanged.connect(self._persist)

        # hotkey: system-wide first, window-scoped fallback
        self._hotkey = GlobalHotkey(QApplication.instance(), self.capture)
        self._shortcut = QShortcut(QKeySequence(cfg.get("hotkey", DEFAULT_CONFIG["hotkey"])), self)
        self._shortcut.activated.connect(self.capture)
        self._apply_hotkey(cfg.get("hotkey", DEFAULT_CONFIG["hotkey"]), announce=False)
        self.statusBar().showMessage(t(
            "status_hotkey_global" if self._hotkey_global else "status_hotkey_scope",
            key=self._cfg["hotkey"], btn=t("btn_capture")))

    def _apply_hotkey(self, spec, announce=True):
        gok = self._hotkey.set(spec)
        try:
            self._shortcut.setKey(QKeySequence(spec))
        except Exception:      # noqa: BLE001
            pass
        if announce:
            self.btn.setToolTip(f"{spec} ({'global' if gok else 'window'})")
        self._hotkey_global = gok

    # -- capture ------------------------------------------------------------------

    def capture(self):
        if self._overlay is not None:
            return
        self.hide()
        QApplication.processEvents()
        time.sleep(0.15)   # let this window actually disappear
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab(all_screens=True)
        except Exception as e:      # noqa: BLE001
            self.show()
            self.statusBar().showMessage(t("status_capture_failed", err=e))
            return

        geo = QGuiApplication.screens()[0].geometry()
        for s in QGuiApplication.screens()[1:]:
            geo = geo.united(s.geometry())
        # Physical-resolution grab; DON'T downscale — tag it with the devicePixelRatio so
        # Qt paints it 1:1 into the (logical-sized, high-DPI) overlay. Scaling it down and
        # letting Qt scale back up is what made it look blurry.
        disp = QPixmap.fromImage(_pil_to_qimage(img))
        sx = img.width / geo.width() if geo.width() else 1.0
        sy = img.height / geo.height() if geo.height() else 1.0
        disp.setDevicePixelRatio(sx)
        scale = (sx, sy)

        self._overlay = CaptureOverlay(img, disp, geo, scale)
        self._overlay.picked.connect(self._on_picked)
        self._overlay.show()

    def _on_picked(self, crop):
        self._overlay = None
        self.show()
        self.raise_()
        if crop is None:
            return
        arr = np.asarray(crop.convert("RGB"))
        self.statusBar().showMessage(t("status_capturing", w=arr.shape[1], h=arr.shape[0]))
        self._worker.submit(arr, self.chk.isChecked(), self.cmb.currentData(),
                            self.chk_det.isChecked(), self.cmb_dir.currentData(),
                            self.cmb_reason.currentData())

    def _on_recognized(self, text):
        self.txt.setPlainText(text)     # source shows as soon as OCR finishes
        tail = t("status_translating") if self.chk.isChecked() else ""
        self.statusBar().showMessage(t("status_recognized", n=len(text), tail=tail))

    def _on_translated(self, trans):
        self.trans.setPlainText(trans)
        if trans:
            self.statusBar().showMessage(t("status_done", n=len(self.txt.toPlainText())))

    def _on_error(self, msg):
        self.statusBar().showMessage(t("status_error") + msg)

    # -- settings ------------------------------------------------------------------

    def open_settings(self):
        dlg = SettingsDialog(self, self._cfg)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            llm = dlg.llm()
            self._cfg.update(llm)
            self._worker.set_llm(llm["api_base"], llm["api_model"], llm["api_key"])
            lang_changed = dlg.ui_lang() != self._cfg.get("ui_lang")
            self._cfg["ui_lang"] = dlg.ui_lang()
            spec = dlg.hotkey()
            if parse_hotkey(spec) is None:
                self.statusBar().showMessage(t("status_bad_hotkey", key=spec))
            else:
                self._cfg["hotkey"] = spec
                self._apply_hotkey(spec)
                self.statusBar().showMessage(t("status_lang_changed") if lang_changed
                                             else t("status_settings_saved"))
            self._persist()
        return

    def _persist(self, *_):
        self._cfg["det"] = self.chk_det.isChecked()
        self._cfg["translate"] = self.chk.isChecked()
        self._cfg["target"] = self.cmb.currentData()
        self._cfg["orient"] = self.cmb_dir.currentData()
        self._cfg["reasoning"] = self.cmb_reason.currentData()
        save_config(self._cfg_path, self._cfg)

    def closeEvent(self, e):
        self._hotkey.release()
        self._worker.stop()
        super().closeEvent(e)


def _pil_to_qimage(img):
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    return QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888).copy()


def main(argv=None):
    global _LANG
    ap = argparse.ArgumentParser(description="Screen translator (PP-OCRv6 + LLM)")
    ap.add_argument("--config", default=os.environ.get(
        "OCR_TRANSLATOR_CONFIG", os.path.join(ROOT, "data", "ocr", "screen_translator.json")),
        help="settings file (default: <repo>/data/ocr/screen_translator.json, gitignored)")
    ap.add_argument("--lang", choices=["zh", "en"], help="UI language (overrides config)")
    ap.add_argument("--selftest", action="store_true", help="construct + smoke-test, no GUI loop")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    save_config(args.config, cfg)   # write back the full schema (so all keys are visible)
    _LANG = args.lang or cfg.get("ui_lang", "zh")

    app = QApplication(sys.argv[:1])
    win = MainWindow(args, cfg, args.config)

    if args.selftest:
        from ocr_py.ocr import Ocr   # verifies the sys.path wiring (no model load)
        dlg = SettingsDialog(win, cfg)   # catch field/import errors without opening it
        safe = {k: ("<set>" if k == "api_key" and v else v) for k, v in win._cfg.items()}
        print(f"OK: ocr_py import ok; ui_lang={_LANG}; config={args.config}")
        print(f"    cfg={safe}")   # api_key redacted
        print(f"    hotkey_global={getattr(win, '_hotkey_global', None)}; "
              f"api_key={'set' if cfg.get('api_key') else 'empty'}")
        win._hotkey.release()
        win._worker.stop()
        return 0

    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
