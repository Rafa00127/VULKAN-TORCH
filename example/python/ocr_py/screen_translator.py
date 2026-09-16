"""划词翻译器 — draw a box over any on-screen text, OCR it (PP-OCRv6), optionally
translate it with an LLM (OpenAI-compatible endpoint).

PyQt6 UI. Click 「截屏划词」(or the global hotkey) -> the tool hides, a full-screen
overlay lets you drag a box (screen stays as-is, only the box is outlined) -> the box is
recognised (rec-only -> text) and, if the translate box is ticked, sent to the LLM.

    python example/python/ocr_py/screen_translator.py [--config PATH]

The UI is bilingual (中文 / English); pick the language in 「设置」/ Settings (saved in
config; takes effect on restart). Everything — LLM, hotkey, OCR model, options — lives in
the config file:

    "ui_lang":   "zh",                 # "zh" | "en"
    "api_base":  "https://api.deepseek.com",   # any OpenAI-compatible base URL
    "api_model": "",                   # must be set (no default)
    "api_key":   "",                   # fill in here or in Settings
    "model_dir": "",                   # "" = repo default (model/ppocrv6/gguf/)
    "precision": "f32",                # "f32" | "f16"

It resolves to an existing <repo>/data/ocr/screen_translator.json, else to
%APPDATA%/screen-translator/config.json (outside the repo, so a copy of the tool next to
another project keeps its own settings); --config PATH / $OCR_TRANSLATOR_CONFIG override.
The hotkey is a SYSTEM-WIDE hotkey (Windows RegisterHotKey); OCR + network run on one
persistent worker thread, so the UI never blocks.
"""
import argparse
import ctypes
import json
import os
import queue
import subprocess
import sys
import time

import numpy as np

PYDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../example/python
ROOT = os.path.dirname(os.path.dirname(PYDIR))                          # repo root
for p in (ROOT, PYDIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from PyQt6.QtCore import (Qt, QRect, QRectF, QPointF, QThread, pyqtSignal,
                          QAbstractNativeEventFilter)
from PyQt6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QFont, QPainterPath,
                         QTextLayout, QTextOption, QGuiApplication, QKeySequence,
                         QShortcut, QAction, QIcon)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QPushButton, QCheckBox, QPlainTextEdit,
                             QLabel, QComboBox, QKeySequenceEdit, QDialogButtonBox, QLineEdit,
                             QFileDialog, QSpinBox, QMenu, QSystemTrayIcon)

# (native name, English name) — the target language list; shown as "→ <native>"
LANGUAGES = [("中文", "Chinese"), ("English", "English"), ("日本語", "Japanese"),
             ("한국어", "Korean"), ("Español", "Spanish"), ("Français", "French")]

DEFAULT_CONFIG = {"hotkey": "Ctrl+Alt+Shift+O", "target": "Chinese", "translate": False,
                  "det": False, "orient": "auto", "reasoning": "off", "ui_lang": "zh",
                  "api_base": "https://api.deepseek.com", "api_model": "",
                  "api_key": "", "model_dir": "", "precision": "f32",
                  "overlay": True, "overlay_pos": None, "overlay_size": 28,
                  "overlay_alpha": 150, "overlay_src": True, "overlay_locked": True,
                  "overlay_opacity": 100}


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
        "chk_overlay": "字幕",
        "chk_overlay_tip": "把识别结果显示成置顶悬浮字幕（像音乐播放器的歌词），默认鼠标穿透、不挡操作。",
        "btn_move": "移动",
        "btn_move_tip": "勾上后可拖动字幕层定位（再取消勾选就锁定并记住位置）。",
        "settings_overlay": "字幕显示原文",
        "settings_overlay_opacity": "整体不透明度 (%)",
        "settings_overlay_lock": "锁定位置（鼠标穿透）",
        "settings_overlay_lock_tip": "勾上 = 字幕层不收鼠标（点击落到下面的窗口），也不能拖；取消勾选就能拖到任意位置。",
        "btn_quit": "退出",
        "tray_hint": "已缩到托盘 · 双击图标恢复窗口，右键退出",
        "settings_overlay_size": "译文字号",
        "settings_overlay_alpha": "字幕背景浓度",
        "settings_uilang": "界面语言",
        "settings_model_dir": "识别模型目录",
        "settings_model_dir_tip": "留空 = 仓库默认（model/ppocrv6/gguf/），也可用环境变量 OCR_MODEL_DIR。\n改这里会在下次识别时重新加载模型。",
        "settings_precision": "精度",
        "settings_precision_tip": "f16 需要该目录里存在对应的 f16 GGUF；缺了就回落 f32。",
        "ph_model_dir": "留空 = 仓库默认 model/ppocrv6/gguf",
        "settings_browse": "浏览…",
        "settings_config": "配置文件",
        "settings_config_tip": "换位置：启动时加 --config <路径>，或设环境变量 OCR_TRANSLATOR_CONFIG。",
        "btn_open_config": "打开所在文件夹",
        "status_model_changed": "设置已保存 —— 模型将在下次识别时重新加载",
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
        "chk_overlay": "Subtitle",
        "chk_overlay_tip": "Show the result as a pinned on-screen subtitle (music-player-lyrics style); click-through by default so it never blocks your work.",
        "btn_move": "Move",
        "btn_move_tip": "Tick to drag the subtitle where you want it (untick to lock the position in).",
        "settings_overlay": "Subtitle: show source",
        "settings_overlay_opacity": "Overall opacity (%)",
        "settings_overlay_lock": "Lock position (click-through)",
        "settings_overlay_lock_tip": "Ticked = the subtitle ignores the mouse (clicks fall through to what is below) and cannot be dragged. Untick to drag it anywhere.",
        "btn_quit": "Quit",
        "tray_hint": "Minimized to the tray · double-click the icon to restore, right-click to quit",
        "settings_overlay_size": "Subtitle font size",
        "settings_overlay_alpha": "Subtitle background",
        "settings_uilang": "UI language",
        "settings_model_dir": "OCR model dir",
        "settings_model_dir_tip": "Empty = the repo default (model/ppocrv6/gguf/); OCR_MODEL_DIR works too.\nChanging it reloads the model on the next capture.",
        "settings_precision": "Precision",
        "settings_precision_tip": "f16 needs an f16 GGUF in that dir; falls back to f32 otherwise.",
        "ph_model_dir": "empty = repo default model/ppocrv6/gguf",
        "settings_browse": "Browse…",
        "settings_config": "Config file",
        "settings_config_tip": "To move it: pass --config <path>, or set OCR_TRANSLATOR_CONFIG.",
        "btn_open_config": "Open folder",
        "status_model_changed": "Saved — the model reloads on the next capture",
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


def default_config_path():
    """Settings live outside the repo by default.

    The old default (<repo>/data/ocr/screen_translator.json) travelled with the checkout,
    so the settings effectively vanished as soon as the tool was run from a copy."""
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "screen-translator", "config.json")


def resolve_config_path(override=None):
    """Which settings file this run uses.

    ``--config`` / OCR_TRANSLATOR_CONFIG wins outright. Otherwise an **existing in-repo
    <repo>/data/ocr/screen_translator.json** wins (that is where the settings have always
    lived, and nothing is moved behind your back); only when there is no such file — e.g.
    the tool copied next to another project — does it fall back to the user profile.
    """
    if override:
        return override
    default = default_config_path()
    legacy = os.path.join(ROOT, "data", "ocr", "screen_translator.json")
    return legacy if (not os.path.exists(default) and os.path.exists(legacy)) else default


def open_config_folder(path):
    """Reveal the config file in the OS file manager (best effort)."""
    d = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(d, exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", d])
    except Exception as e:      # noqa: BLE001
        print(f"warn: could not open {d}: {e}", file=sys.stderr)


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
        self._model = ("", "")       # (model_dir, precision) the live _ocr was built with
        self._llm = {"base": "", "model": "", "key": ""}

    def set_model(self, model_dir, precision):
        """Repoint OCR at another model dir. Lazy: dropping the instance frees the old
        weights, so the next job loads the new ones (costs a few seconds, once)."""
        if (model_dir, precision) != self._model:
            self._model = (model_dir, precision)
            self._ocr = None

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
                    kw = {"precision": self._model[1] or None}
                    if self._model[0]:
                        kw["model_dir"] = self._model[0]
                    self._ocr = Ocr(**kw)
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


# ── pinned subtitle (lyrics-style) ───────────────────────────────────────────────

class SubtitleOverlay(QWidget):
    """置顶字幕窗：原文一行（小/淡）+ 译文一行（大/描边），像播放器歌词钉在屏幕上。

    **默认鼠标穿透**（``WindowTransparentForInput``）：点不到它，也就不挡下面任何操作；
    主窗口的「移动」按钮解锁之后才能拖动（这时画一圈虚线边框提示）。
    锁定/解锁靠切换窗口旗标实现——Qt 改旗标会把窗口隐藏掉，所以改完必须重新 ``show()``。"""

    moved = pyqtSignal(int, int)

    OUTLINE = QColor(0, 20, 70)       # 描边色（深蓝，贴截图那种观感）
    PAD, GAP, RADIUS = 18, 6, 12      # 内边距 / 两行间距 / 圆角
    SRC_SCALE = 0.55                  # 原文行相对译文的字号比

    def __init__(self, cfg):
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self._src = self._dst = ""
        self._drag = None
        self._locked = True            # 锁定 == 穿透
        self._placed = False
        self._saved_pos = None
        self.apply_cfg(cfg)

    # -- config / content --------------------------------------------------------

    def apply_cfg(self, cfg):
        self._size = int(cfg.get("overlay_size") or 28)
        alpha = cfg.get("overlay_alpha")
        self._alpha = 150 if alpha is None else int(alpha)
        self._show_src = bool(cfg.get("overlay_src", True))
        self._saved_pos = cfg.get("overlay_pos")
        op = cfg.get("overlay_opacity")
        self.setWindowOpacity(max(0.2, min(1.0, (100 if op is None else int(op)) / 100.0)))
        self._relayout()
        self.update()

    def set_text(self, src, dst=""):
        """src = 识别出的原文；dst = 译文（空 = 没翻译 / 没开翻译 → 原文当主行显示）。"""
        self._src, self._dst = (src or "").strip(), (dst or "").strip()
        self._relayout()
        if not self._placed and (self._src or self._dst):
            self.place(self._saved_pos)      # 首次有内容时才定尺寸，定了再摆位置
            self._placed = True
        self.update()

    def set_visible(self, on):
        if on:
            self.show()
            self.raise_()
        else:
            self.hide()

    def set_locked(self, locked):
        """锁 = 穿透；解锁后可拖。"""
        if locked == self._locked:
            return
        self._locked = locked
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, locked)
        if visible:
            self.show()                      # 改旗标会隐藏窗口
        self.update()

    def place(self, pos=None):
        """pos = [x, y]；没有就摆到主屏底部居中（歌词那个位置）。"""
        if pos and len(pos) == 2:
            self.move(int(pos[0]), int(pos[1]))
            return
        scr = QGuiApplication.primaryScreen().availableGeometry()
        self.move(scr.center().x() - self.width() // 2,
                  scr.bottom() - self.height() - 140)

    # -- painting ----------------------------------------------------------------

    def _fonts(self):
        """-> (主行字体, 原文行字体)。**两个 QFont 要一直被持有**：PyQt6 的
        QFontMetrics 没有 ``font()``，取字体只能自己在手边留着传。"""
        main = QFont()
        main.setPointSizeF(max(6.0, self._size))
        main.setBold(True)
        sub = QFont()
        sub.setPointSizeF(max(5.0, self._size * self.SRC_SCALE))
        return main, sub

    def _lines(self, text, font, width):
        """-> (QTextLayout, [QTextLine])，按宽度排好版。

        两个必须记住的点：
        * **layout 要跟着 lines 一起活着**——QTextLine 只是 layout 内部数据的视图，layout 一
          被回收，line 就成了悬空指针（拿它去 measure/draw 会直接崩，而且未必立刻崩）。
        * **行高取 line.height()，别用 QFontMetrics 的 lineSpacing()**：字体回退时（拿默认拉丁
          字体渲染日文/中文）实际字形比 metrics 高，照 metrics 的行距往下排会行行重叠。"""
        layout = QTextLayout(text, font)
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(opt)
        layout.beginLayout()
        lines = []
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            lines.append(line)
        layout.endLayout()
        return layout, lines

    def _relayout(self):
        main, sub = self._dst or self._src, self._src if self._dst else ""
        if not self._show_src:
            sub = ""
        font_main, font_sub = self._fonts()
        scr = QGuiApplication.primaryScreen().availableGeometry()
        maxw = max(200, int(scr.width() * 0.8) - 2 * self.PAD)

        def natural(text, font):        # 不限宽先排一遍，量真实宽度（含字体回退）
            _lay, lines = self._lines(text, font, 1 << 20)
            return max((ln.naturalTextWidth() for ln in lines), default=0.0)

        tw = min(int(max(natural(main, font_main), natural(sub, font_sub), 80.0)), maxw)
        _lay1, main_lines = self._lines(main, font_main, tw)
        h = sum(ln.height() for ln in main_lines)
        if sub:
            _lay2, sub_lines = self._lines(sub, font_sub, tw)
            h += sum(ln.height() for ln in sub_lines) + self.GAP
        self.resize(tw + 2 * self.PAD, int(h) + 2 * self.PAD)

    def paintEvent(self, _):
        main, sub = self._dst or self._src, self._src if (self._dst and self._show_src) else ""
        font_main, font_sub = self._fonts()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if self._alpha:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, self._alpha))
            p.drawRoundedRect(r, self.RADIUS, self.RADIUS)
        if not self._locked:                 # 解锁提示：现在能拖
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(120, 190, 255, 200), 1, Qt.PenStyle.DashLine))
            p.drawRoundedRect(r, self.RADIUS, self.RADIUS)

        tw = self.width() - 2 * self.PAD
        y = float(self.PAD)
        if sub:
            p.setPen(QColor(255, 255, 255, 190))
            sub_lay, sub_lines = self._lines(sub, font_sub, tw)   # 用 line.draw：字体回退也画得对
            for ln in sub_lines:
                ln.draw(p, QPointF((self.width() - ln.naturalTextWidth()) / 2.0, y))
                y += ln.height()
            y += self.GAP
        main_lay, main_lines = self._lines(main, font_main, tw)
        for ln in main_lines:
            seg = main[ln.textStart():ln.textStart() + ln.textLength()]
            path = QPainterPath()
            path.addText(QPointF((self.width() - ln.naturalTextWidth()) / 2.0, y + ln.ascent()),
                         font_main, seg)
            p.strokePath(path, QPen(self.OUTLINE, max(1.5, self._size / 7.0)))
            p.fillPath(path, QColor(255, 255, 255))
            y += ln.height()

    # -- dragging (only while unlocked) ------------------------------------------

    def mousePressEvent(self, e):
        if not self._locked and e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return
        self._drag = None
        self.moved.emit(self.x(), self.y())   # 位置交给主窗口落盘


# ── settings dialog ──────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, parent, cfg, cfg_path=""):
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

        self.ed_modeldir = QLineEdit(cfg.get("model_dir", ""))
        self.ed_modeldir.setPlaceholderText(t("ph_model_dir"))
        self.ed_modeldir.setToolTip(t("settings_model_dir_tip"))
        b_browse = QPushButton(t("settings_browse"))
        b_browse.clicked.connect(self._browse_model_dir)
        row_model = QHBoxLayout()
        row_model.addWidget(self.ed_modeldir)
        row_model.addWidget(b_browse)
        form.addRow(t("settings_model_dir"), row_model)

        self.cmb_prec = QComboBox()
        for p in ("f32", "f16"):
            self.cmb_prec.addItem(p, p)
        self.cmb_prec.setCurrentIndex(max(0, self.cmb_prec.findData(cfg.get("precision", "f32"))))
        self.cmb_prec.setToolTip(t("settings_precision_tip"))
        form.addRow(t("settings_precision"), self.cmb_prec)

        self.chk_ovsrc = QCheckBox(t("settings_overlay"))
        self.chk_ovsrc.setChecked(bool(cfg.get("overlay_src", True)))
        form.addRow("", self.chk_ovsrc)
        self.sp_ovsize = QSpinBox()
        self.sp_ovsize.setRange(16, 72)
        self.sp_ovsize.setValue(int(cfg.get("overlay_size") or 28))
        form.addRow(t("settings_overlay_size"), self.sp_ovsize)
        self.sp_ovalpha = QSpinBox()
        self.sp_ovalpha.setRange(0, 255)
        alpha = cfg.get("overlay_alpha")
        self.sp_ovalpha.setValue(150 if alpha is None else int(alpha))
        form.addRow(t("settings_overlay_alpha"), self.sp_ovalpha)
        self.sp_ovop = QSpinBox()
        self.sp_ovop.setRange(20, 100)
        self.sp_ovop.setSuffix(" %")
        op = cfg.get("overlay_opacity")
        self.sp_ovop.setValue(100 if op is None else int(op))
        form.addRow(t("settings_overlay_opacity"), self.sp_ovop)
        self.chk_ovlock = QCheckBox(t("settings_overlay_lock"))
        self.chk_ovlock.setChecked(bool(cfg.get("overlay_locked", True)))
        self.chk_ovlock.setToolTip(t("settings_overlay_lock_tip"))
        form.addRow("", self.chk_ovlock)

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

        self.ed_cfgpath = QLineEdit(cfg_path)
        self.ed_cfgpath.setReadOnly(True)
        self.ed_cfgpath.setToolTip(t("settings_config_tip"))
        b_cfg = QPushButton(t("btn_open_config"))
        b_cfg.clicked.connect(lambda: open_config_folder(cfg_path))
        row_cfg = QHBoxLayout()
        row_cfg.addWidget(self.ed_cfgpath)
        row_cfg.addWidget(b_cfg)
        form.addRow(t("settings_config"), row_cfg)

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

    def model(self):
        return {"model_dir": self.ed_modeldir.text().strip(),
                "precision": self.cmb_prec.currentData()}

    def overlay(self):
        return {"overlay_src": self.chk_ovsrc.isChecked(),
                "overlay_size": self.sp_ovsize.value(),
                "overlay_alpha": self.sp_ovalpha.value(),
                "overlay_opacity": self.sp_ovop.value(),
                "overlay_locked": self.chk_ovlock.isChecked()}

    def _browse_model_dir(self):
        start = self.ed_modeldir.text().strip() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, t("settings_model_dir"), start)
        if d:
            self.ed_modeldir.setText(d)


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
        self._worker.set_model(cfg.get("model_dir", ""), cfg.get("precision", "f32"))

        self._sub = SubtitleOverlay(cfg)          # 歌词式置顶字幕层（穿透，不挡操作）
        self._sub.moved.connect(self._on_sub_moved)
        self._sub.set_locked(bool(cfg.get("overlay_locked", True)))
        self._quitting = False
        self._tray_hinted = False

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
        self.chk_overlay = QCheckBox(t("chk_overlay"))
        self.chk_overlay.setToolTip(t("chk_overlay_tip"))
        self.btn_move = QPushButton(t("btn_move"))
        self.btn_move.setCheckable(True)
        self.btn_move.setToolTip(t("btn_move_tip"))
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
        top.addWidget(self.chk_overlay)
        top.addWidget(self.btn_move)
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
        self.chk_overlay.setChecked(bool(cfg.get("overlay", True)))
        self._sub.set_visible(self.chk_overlay.isChecked())
        self.btn_move.setChecked(not bool(cfg.get("overlay_locked", True)))

        self.chk_det.toggled.connect(self._persist)
        self.chk.toggled.connect(self._persist)
        self.cmb.currentIndexChanged.connect(self._persist)
        self.cmb_dir.currentIndexChanged.connect(self._persist)
        self.cmb_reason.currentIndexChanged.connect(self._persist)
        self.chk_overlay.toggled.connect(self._on_overlay_toggled)
        self.btn_move.toggled.connect(self._on_move_toggled)

        self._setup_tray()

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
        self._was_visible = self.isVisible()   # 从托盘划词时不弹窗口，结果看字幕
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
        if self._was_visible:
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
        self._sub.set_text(text)        # 译文留空 → 原文先顶上，等翻译回来再换
        tail = t("status_translating") if self.chk.isChecked() else ""
        self.statusBar().showMessage(t("status_recognized", n=len(text), tail=tail))

    def _on_translated(self, trans):
        self.trans.setPlainText(trans)
        self._sub.set_text(self.txt.toPlainText(), trans)
        if trans:
            self.statusBar().showMessage(t("status_done", n=len(self.txt.toPlainText())))

    def _on_error(self, msg):
        self.statusBar().showMessage(t("status_error") + msg)

    # -- subtitle ------------------------------------------------------------------

    def _on_overlay_toggled(self, on):
        self._sub.set_visible(on)
        if not on:
            self.btn_move.setChecked(False)     # 不显示就无所谓移动
        self.act_overlay.blockSignals(True)     # 托盘菜单里的勾跟着走
        self.act_overlay.setChecked(on)
        self.act_overlay.blockSignals(False)
        self._persist()

    def _on_move_toggled(self, on):
        self._sub.set_locked(not on)            # 勾上 = 解锁，可拖
        self._cfg["overlay_locked"] = not on
        self._persist()                         # 落盘（连拖动后的位置一起）

    def _on_sub_moved(self, x, y):
        self._cfg["overlay_pos"] = [x, y]
        self._persist()

    # -- tray ----------------------------------------------------------------------

    def _tray_icon(self):
        """托盘/窗口图标：画一个圆角方块 +「译」，省得带图标资源。"""
        pm = QPixmap(64, 64)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(60, 120, 220))
        p.drawRoundedRect(QRectF(3, 3, 58, 58), 14, 14)
        font = QFont()
        font.setPointSizeF(30)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(0, 0, 64, 64), Qt.AlignmentFlag.AlignCenter, "译")
        p.end()
        return QIcon(pm)

    def _setup_tray(self):
        icon = self._tray_icon()
        self.setWindowIcon(icon)
        self.tray = QSystemTrayIcon(icon, self)
        menu = QMenu(self)
        for label, slot in ((t("btn_capture"), self.capture),
                            (t("btn_settings"), self.open_settings)):
            act = QAction(label, self)
            act.triggered.connect(slot)
            menu.addAction(act)
        self.act_overlay = QAction(t("chk_overlay"), self)
        self.act_overlay.setCheckable(True)
        self.act_overlay.setChecked(self.chk_overlay.isChecked())
        self.act_overlay.toggled.connect(self.chk_overlay.setChecked)
        menu.addAction(self.act_overlay)
        menu.addSeparator()
        quit_act = QAction(t("btn_quit"), self)
        quit_act.triggered.connect(self._quit)
        menu.addAction(quit_act)
        self._tray_menu = menu          # 持有引用，否则菜单会被回收
        self.tray.setContextMenu(menu)
        self.tray.setToolTip(t("app_title"))
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show()
            self.raise_()
            self.activateWindow()

    def _quit(self):
        """真退出：字幕是**独立顶层窗口**，close 主窗口带不走它；再加上
        setQuitOnLastWindowClosed(False)，不显式收尾进程会一直挂着。"""
        self._quitting = True
        self._sub.close()
        self.tray.hide()
        self.close()
        QApplication.quit()

    # -- settings ------------------------------------------------------------------

    def open_settings(self):
        dlg = SettingsDialog(self, self._cfg, self._cfg_path)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            llm = dlg.llm()
            self._cfg.update(llm)
            self._worker.set_llm(llm["api_base"], llm["api_model"], llm["api_key"])
            model = dlg.model()
            model_changed = (model["model_dir"], model["precision"]) != (
                self._cfg.get("model_dir", ""), self._cfg.get("precision", "f32"))
            self._cfg.update(model)
            self._worker.set_model(model["model_dir"], model["precision"])
            self._cfg.update(dlg.overlay())
            self._sub.apply_cfg(self._cfg)
            self._sub.set_locked(bool(self._cfg.get("overlay_locked", True)))
            self.btn_move.setChecked(not bool(self._cfg.get("overlay_locked", True)))
            lang_changed = dlg.ui_lang() != self._cfg.get("ui_lang")
            self._cfg["ui_lang"] = dlg.ui_lang()
            spec = dlg.hotkey()
            if parse_hotkey(spec) is None:
                self.statusBar().showMessage(t("status_bad_hotkey", key=spec))
            else:
                self._cfg["hotkey"] = spec
                self._apply_hotkey(spec)
                if model_changed:
                    self.statusBar().showMessage(t("status_model_changed"))
                else:
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
        self._cfg["overlay"] = self.chk_overlay.isChecked()
        save_config(self._cfg_path, self._cfg)

    def closeEvent(self, e):
        if not self._quitting:              # 叉叉 = 缩到托盘，不等于退出
            e.ignore()
            self.hide()
            if not self._tray_hinted:
                self._tray_hinted = True
                self.tray.showMessage(t("app_title"), t("tray_hint"),
                                      QSystemTrayIcon.MessageIcon.Information, 3000)
            return
        self._hotkey.release()
        self._worker.stop()
        self.tray.hide()
        super().closeEvent(e)


def _pil_to_qimage(img):
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    return QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888).copy()


def main(argv=None):
    global _LANG
    ap = argparse.ArgumentParser(description="Screen translator (PP-OCRv6 + LLM)")
    ap.add_argument("--config", default=resolve_config_path(os.environ.get("OCR_TRANSLATOR_CONFIG")),
                    help="settings file (default: an existing <repo>/data/ocr/screen_translator.json, "
                         "else %APPDATA%/screen-translator/config.json)")
    ap.add_argument("--lang", choices=["zh", "en"], help="UI language (overrides config)")
    ap.add_argument("--selftest", action="store_true", help="construct + smoke-test, no GUI loop")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    save_config(args.config, cfg)   # write back the full schema (so all keys are visible)
    _LANG = args.lang or cfg.get("ui_lang", "zh")

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)   # 叉叉只缩到托盘，退出走托盘菜单
    win = MainWindow(args, cfg, args.config)

    if args.selftest:
        from ocr_py.ocr import Ocr   # verifies the sys.path wiring (no model load)
        dlg = SettingsDialog(win, cfg, args.config)   # catch field/import errors without opening it
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
