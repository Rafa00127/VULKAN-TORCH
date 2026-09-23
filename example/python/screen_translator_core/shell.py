"""划词翻译器外壳 — the model-agnostic half of the screen translator.

Everything that is not "how do I recognize a cropped region" lives here: the PyQt6
window, the full-screen selection overlay, the pinned subtitle, the settings dialog, the
system-wide hotkey, the tray icon, the i18n table, the config file, and the LLM call.

An example plugs in a :class:`Backend` (how to load a model and read one crop) via
``run(backend)``; see ``ocr_vt/screen_translator.py`` and
``paddleocrvl_vt/screen_translator.py``. The two entry points keep their own paths, so
``python example/python/ocr_vt/screen_translator.py`` still works.

Design notes worth keeping (learned the hard way, see the comments at each site):

* the OCR/LLM work runs on one persistent ``QThread``, so the UI never blocks;
* the subtitle is an independent top-level window, so ``close()`` on the main window
  does not take it down — quitting has to be explicit;
* ``QTextLine`` views die with their ``QTextLayout``, and line height must come from
  ``QTextLine.height()`` rather than the font metrics (font fallback).
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

from PyQt6.QtCore import (Qt, QRect, QRectF, QPointF, QThread, pyqtSignal,   # noqa: E402
                          QAbstractNativeEventFilter)
from PyQt6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QFont,     # noqa: E402
                         QPainterPath, QTextLayout, QTextOption, QGuiApplication,
                         QKeySequence, QShortcut, QAction, QIcon)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QDialog,     # noqa: E402
                             QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton,
                             QCheckBox, QPlainTextEdit, QLabel, QComboBox,
                             QKeySequenceEdit, QDialogButtonBox, QLineEdit,
                             QFileDialog, QSpinBox, QMenu, QSystemTrayIcon)

# (native name, English name) — the target language list; shown as "→ <native>"
LANGUAGES = [("中文", "Chinese"), ("English", "English"), ("日本語", "Japanese"),
             ("한국어", "Korean"), ("Español", "Spanish"), ("Français", "French")]

# the keys the shell itself owns; a backend adds its own via Backend.extra_config
DEFAULT_CONFIG = {"hotkey": "Ctrl+Alt+Shift+O", "target": "Chinese", "translate": False,
                  "reasoning": "off", "ui_lang": "zh",
                  "api_base": "https://api.deepseek.com", "api_model": "",
                  "api_key": "",
                  "overlay": True, "overlay_pos": None, "overlay_size": 28,
                  "overlay_alpha": 150, "overlay_src": True, "overlay_locked": True,
                  "overlay_opacity": 100}


# ── i18n ─────────────────────────────────────────────────────────────────────────

STRINGS = {
    "zh": {
        "btn_capture": "截屏划词", "btn_settings": "设置",
        "chk_translate": "翻译",
        "reason_off": "思考:关", "reason_low": "思考:低", "reason_high": "思考:高", "reason_max": "思考:最高",
        "reason_tip": "仅翻译时生效：off = 关闭思考；low/high/max = 开启思考并设推理强度。",
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
        "status_model_loading": "模型加载中…",
        "status_model_ready": "模型已就绪 · 点「{btn}」或按快捷键 {key} 开始",
        "status_bad_hotkey": "设置已保存；但快捷键 '{key}' 无效（需修饰键+键），未改。",
        "status_lang_changed": "界面语言已保存 —— 重启程序后生效。",
        "status_model_changed": "设置已保存 —— 正在重新加载模型…",
        "settings_title": "设置",
        "settings_hotkey": "截图快捷键",
        "settings_hotkey_tip": "点这里然后按组合键。需要至少一个修饰键（Ctrl/Alt/Shift/Win）。",
        "settings_model_section": "模型",
        "settings_browse": "浏览…",
        "settings_config": "配置文件",
        "settings_config_tip": "换位置：启动时加 --config <路径>，或设环境变量 {env}。",
        "btn_open_config": "打开所在文件夹",
        "settings_uilang": "界面语言",
        "chk_overlay": "字幕",
        "chk_overlay_tip": "把识别结果显示成置顶悬浮字幕（像音乐播放器的歌词），默认鼠标穿透、不挡操作。",
        "btn_move": "移动",
        "btn_move_tip": "勾上后可拖动字幕层定位（再取消勾选就锁定并记住位置）。",
        "settings_overlay": "字幕显示原文",
        "settings_overlay_opacity": "整体不透明度 (%)",
        "settings_overlay_lock": "锁定位置（鼠标穿透）",
        "settings_overlay_lock_tip": "勾上 = 字幕层不收鼠标（点击落到下面的窗口），也不能拖；取消勾选就能拖到任意位置。",
        "settings_overlay_size": "译文字号",
        "settings_overlay_alpha": "字幕背景浓度",
        "btn_quit": "退出",
        "tray_hint": "已缩到托盘 · 双击图标恢复窗口，右键退出",
        "ph_base": "https://api.deepseek.com  (OpenAI 兼容 base url)",
        "ph_model": "填模型名，如 deepseek-chat / deepseek-reasoner",
        "ph_key": "在此填入 API key（保存进 config）",
        "err_no_key": "未设置 API key —— 在「设置」里填 api_key（存进 config）",
        "err_no_model": "未设置模型名 —— 在「设置」里填 api_model（如 deepseek-chat / deepseek-reasoner）",
        "lbl_api_model": "模型名",
    },
    "en": {
        "btn_capture": "Capture", "btn_settings": "Settings",
        "chk_translate": "Translate",
        "reason_off": "Think:off", "reason_low": "Think:low", "reason_high": "Think:high", "reason_max": "Think:max",
        "reason_tip": "Translation only: off = no thinking; low/high/max = enable thinking at that reasoning effort.",
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
        "status_model_loading": "Loading model…",
        "status_model_ready": "Model ready · click \"{btn}\" or press {key} to start",
        "status_bad_hotkey": "Saved, but hotkey '{key}' is invalid (needs a modifier + a key) — left unchanged.",
        "status_lang_changed": "UI language saved — restart to apply.",
        "status_model_changed": "Saved — reloading the model…",
        "settings_title": "Settings",
        "settings_hotkey": "Capture hotkey",
        "settings_hotkey_tip": "Click, then press the combo. Needs at least one modifier (Ctrl/Alt/Shift/Win).",
        "settings_model_section": "Model",
        "settings_browse": "Browse…",
        "settings_config": "Config file",
        "settings_config_tip": "To move it: pass --config <path>, or set {env}.",
        "btn_open_config": "Open folder",
        "settings_uilang": "UI language",
        "chk_overlay": "Subtitle",
        "chk_overlay_tip": "Show the result as a pinned on-screen subtitle (music-player-lyrics style); click-through by default so it never blocks your work.",
        "btn_move": "Move",
        "btn_move_tip": "Tick to drag the subtitle where you want it (untick to lock the position in).",
        "settings_overlay": "Subtitle: show source",
        "settings_overlay_opacity": "Overall opacity (%)",
        "settings_overlay_lock": "Lock position (click-through)",
        "settings_overlay_lock_tip": "Ticked = the subtitle ignores the mouse (clicks fall through to what is below) and cannot be dragged. Untick to drag it anywhere.",
        "settings_overlay_size": "Subtitle font size",
        "settings_overlay_alpha": "Subtitle background",
        "btn_quit": "Quit",
        "tray_hint": "Minimized to the tray · double-click the icon to restore, right-click to quit",
        "ph_base": "https://api.deepseek.com  (OpenAI-compatible base url)",
        "ph_model": "model name, e.g. deepseek-chat / deepseek-reasoner",
        "ph_key": "paste API key here (saved into config)",
        "err_no_key": "no API key set — fill api_key in Settings (stored in config)",
        "err_no_model": "no model set — fill api_model in Settings (e.g. deepseek-chat / deepseek-reasoner)",
        "lbl_api_model": "Model",
    },
}

_LANG = "zh"


def t(_key, **kw):
    s = STRINGS.get(_LANG, {}).get(_key) or STRINGS["zh"].get(_key) or _key
    return s.format(**kw) if kw else s


# ── backend interface ────────────────────────────────────────────────────────────

class Backend:
    """What the shell needs from a model. Subclass one per example.

    The two hooks that actually do anything are :meth:`load` and :meth:`recognize`;
    ``reload_key`` says when a loaded model has to be dropped, ``options`` /
    ``settings_section`` return the widgets whose state is stored in the config, and
    ``extra_*`` extend the config schema and the i18n table.
    """

    #: shown in the window title, the tray tooltip and the settings section
    name = "Backend"
    #: config file location: %APPDATA%/<config_tag>/config.json
    config_tag = "screen-translator"
    #: env var overriding the config path. "" derives SCREEN_TRANSLATOR_<TAG>_CONFIG;
    #: set it explicitly to keep an already-documented name working.
    config_env_name = ""
    #: keep using this in-repo config if it exists (legacy default location)
    legacy_config = None
    #: extra config keys + their defaults (also the backend's own settings state)
    extra_config = {}
    #: extra i18n entries: {"zh": {...}, "en": {...}}
    extra_strings = {}

    def load(self, cfg):
        """Build/return the model. Called on the worker thread; cache it here."""
        raise NotImplementedError

    def recognize(self, model, arr, options):
        """RGB uint8 HWC crop + the option state -> text. Worker thread."""
        raise NotImplementedError

    def reload_key(self, cfg):
        """Hashable: when this changes the cached model is dropped and reloaded."""
        raise NotImplementedError

    def options(self, parent, cfg):
        """-> (widget for the toolbar, callable -> dict of config keys)."""
        return None, (lambda: {})

    def settings_section(self, parent, cfg):
        """-> (widget for the settings dialog, callable -> dict of config keys)."""
        return None, (lambda: {})

    def selftest(self):
        """Import the example package, to prove the sys.path wiring works."""


# ── settings file ───────────────────────────────────────────────────────────────

def config_env(backend):
    return backend.config_env_name or \
        backend.config_tag.upper().replace("-", "_") + "_CONFIG"


def default_config_path(backend):
    """Settings live outside the repo by default.

    The old default (<repo>/data/ocr/screen_translator.json) travelled with the checkout,
    so the settings effectively vanished as soon as the tool was run from a copy."""
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, backend.config_tag, "config.json")


def resolve_config_path(backend, override=None):
    """Which settings file this run uses.

    ``--config`` / <CONFIG_ENV> wins outright. Otherwise an **existing in-repo legacy
    path** wins (that is where the settings have always lived, and nothing is moved
    behind your back); only when there is no such file — e.g. the tool copied next to
    another project — does it fall back to the user profile.
    """
    if override:
        return override
    default = default_config_path(backend)
    legacy = backend.legacy_config
    return legacy if (legacy and not os.path.exists(default) and os.path.exists(legacy)) \
        else default


def load_config(path, backend):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(backend.extra_config)
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


# ── recognition + translation worker (one thread, one model instance) ────────────

class Worker(QThread):
    recognized = pyqtSignal(str)   # text, emitted immediately
    translated = pyqtSignal(str)   # translation ("" if none) — emitted after
    error = pyqtSignal(str)
    ready = pyqtSignal()           # the model is loaded (warmup finished)

    def __init__(self, backend):
        super().__init__()
        self._backend = backend
        self._q = queue.Queue()
        self._cfg = {}
        self._model = None
        self._key = None             # reload_key the live model was built with
        self._llm = {"base": "", "model": "", "key": ""}

    def set_model(self, cfg):
        """Repoint the backend at another model. Dropping the instance frees the old
        weights; the new ones are loaded by the next warmup/capture (a few seconds,
        once). Called from the GUI thread, so the next job is what actually loads."""
        self._cfg = dict(cfg)
        key = self._backend.reload_key(cfg)
        if key != self._key:
            self._key = key
            self._model = None

    def set_llm(self, base, model, key):
        self._llm = {"base": base or "", "model": model or "", "key": key or ""}

    def warmup(self):
        """Load the model now, on the worker thread, and keep it. The window opens
        either way; this just moves the one-off cost off the first capture."""
        self._q.put(("warmup",))

    def submit(self, arr, translate, target, options, reasoning):
        self._q.put(("run", arr, translate, target, options, reasoning))

    def stop(self):
        self._q.put(None)
        self.wait(3000)

    def run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                if item[0] == "warmup":      # queued jobs are tagged; see submit()
                    self._ensure_model()
                    self.ready.emit()
                    continue
                _, arr, do_translate, target, options, reasoning = item
                model = self._ensure_model()
                text = self._backend.recognize(model, arr, options)
                self.recognized.emit(text)          # show source the moment it is done
                trans = ""
                if do_translate and text.strip():
                    trans = translate_text(text, target, self._llm, reasoning)
                self.translated.emit(trans)         # then fill in the translation
            except Exception as e:      # noqa: BLE001
                self.error.emit(f"{type(e).__name__}: {e}")

    def _ensure_model(self):
        if self._model is None:
            self._model = self._backend.load(self._cfg)
        return self._model


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
        self._img = pil_img          # physical-res PIL image (for the crop)
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
    def __init__(self, parent, cfg, cfg_path, backend):
        super().__init__(parent)
        self._backend = backend
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

        # the model section belongs to the backend
        head = QLabel(t("settings_model_section") + " · " + backend.name)
        head.setStyleSheet("font-weight:bold")
        form.addRow(head)
        sec_w, self._model_read = backend.settings_section(self, cfg)
        if sec_w is not None:
            form.addRow(sec_w)

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
        form.addRow(t("lbl_api_model"), self.ed_model)
        form.addRow("API key", self.ed_key)

        self.ed_cfgpath = QLineEdit(cfg_path)
        self.ed_cfgpath.setReadOnly(True)
        self.ed_cfgpath.setToolTip(t("settings_config_tip", env=config_env(backend)))
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
        return self._model_read()

    def overlay(self):
        return {"overlay_src": self.chk_ovsrc.isChecked(),
                "overlay_size": self.sp_ovsize.value(),
                "overlay_alpha": self.sp_ovalpha.value(),
                "overlay_opacity": self.sp_ovop.value(),
                "overlay_locked": self.chk_ovlock.isChecked()}


# ── main window ──────────────────────────────────────────────────────────────────

def app_title():
    return f"划词翻译 ({_BACKEND.name} + LLM)"


class MainWindow(QMainWindow):
    def __init__(self, args, cfg, cfg_path, backend):
        super().__init__()
        self._args = args
        self._backend = backend
        self._cfg = cfg
        self._cfg_path = cfg_path
        self._overlay = None
        self.setWindowTitle(app_title())
        self.resize(460, 380)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self._worker = Worker(backend)
        self._worker.recognized.connect(self._on_recognized)
        self._worker.translated.connect(self._on_translated)
        self._worker.error.connect(self._on_error)
        self._worker.ready.connect(self._on_model_ready)
        self._worker.start()
        self._worker.set_llm(cfg.get("api_base"), cfg.get("api_model"), cfg.get("api_key"))
        self._worker.set_model(cfg)

        self._sub = SubtitleOverlay(cfg)          # 歌词式置顶字幕层（穿透，不挡操作）
        self._sub.moved.connect(self._on_sub_moved)
        self._sub.set_locked(bool(cfg.get("overlay_locked", True)))
        self._quitting = False

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        top = QHBoxLayout()
        self.btn = QPushButton(t("btn_capture"))
        self.btn.setStyleSheet("QPushButton{padding:8px;font-weight:bold}")
        self.btn.clicked.connect(self.capture)
        top.addWidget(self.btn)
        # the model's own knobs go right next to the capture button
        self._opt_w, self._opt_read = backend.options(self, cfg)
        if self._opt_w is not None:
            top.addWidget(self._opt_w)
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
        self.chk.setChecked(bool(cfg.get("translate")))
        for i in range(self.cmb.count()):
            if self.cmb.itemData(i) == cfg.get("target"):
                self.cmb.setCurrentIndex(i)
                break
        for i in range(self.cmb_reason.count()):
            if self.cmb_reason.itemData(i) == cfg.get("reasoning"):
                self.cmb_reason.setCurrentIndex(i)
                break
        self.chk_overlay.setChecked(bool(cfg.get("overlay", True)))
        self._sub.set_visible(self.chk_overlay.isChecked())
        self.btn_move.setChecked(not bool(cfg.get("overlay_locked", True)))

        self.chk.toggled.connect(self._persist)
        self.cmb.currentIndexChanged.connect(self._persist)
        self.cmb_reason.currentIndexChanged.connect(self._persist)
        self.chk_overlay.toggled.connect(self._on_overlay_toggled)
        self.btn_move.toggled.connect(self._on_move_toggled)

        self._setup_tray()

        # hotkey: system-wide first, window-scoped fallback
        self._hotkey = GlobalHotkey(QApplication.instance(), self.capture)
        self._shortcut = QShortcut(QKeySequence(cfg.get("hotkey", DEFAULT_CONFIG["hotkey"])), self)
        self._shortcut.activated.connect(self.capture)
        self._apply_hotkey(cfg.get("hotkey", DEFAULT_CONFIG["hotkey"]), announce=False)

        # Load the model now (on the worker) and keep it resident, so the first capture
        # is a capture and nothing else. --selftest stays model-free on purpose.
        if not getattr(args, "selftest", False):
            self.statusBar().showMessage(t("status_model_loading"))
            self._worker.warmup()
        else:
            self.statusBar().showMessage(t(
                "status_hotkey_global" if self._hotkey_global else "status_hotkey_scope",
                key=self._cfg["hotkey"], btn=t("btn_capture")))

    def _on_model_ready(self):
        self.statusBar().showMessage(t("status_model_ready", btn=t("btn_capture"),
                                        key=self._cfg["hotkey"]))

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
                            self._opt_read(), self.cmb_reason.currentData())

    def _on_recognized(self, text):
        self.txt.setPlainText(text)     # source shows as soon as recognition finishes
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
        # 只开关显示。**不碰锁定状态**——以前隐藏时会顺手把「移动」取消勾选，
        # 而那会把 overlay_locked 落成 true，于是下次再勾上字幕就默认是锁定的。
        # 锁定与否只由「移动」按钮和设置对话框决定，跟显示与否无关。
        self._sub.set_visible(on)
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
        self.tray.setToolTip(app_title() + "\n" + t("tray_hint"))
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
        dlg = SettingsDialog(self, self._cfg, self._cfg_path, self._backend)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            llm = dlg.llm()
            self._cfg.update(llm)
            self._worker.set_llm(llm["api_base"], llm["api_model"], llm["api_key"])
            model = dlg.model()
            model_changed = any(model[k] != self._cfg.get(k) for k in model)
            self._cfg.update(model)
            self._worker.set_model(self._cfg)
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
                    self._worker.warmup()       # reload now; "ready" replaces the message
                else:
                    self.statusBar().showMessage(t("status_lang_changed") if lang_changed
                                                 else t("status_settings_saved"))
            self._persist()
        return

    def _persist(self, *_):
        self._cfg["translate"] = self.chk.isChecked()
        self._cfg["target"] = self.cmb.currentData()
        self._cfg["reasoning"] = self.cmb_reason.currentData()
        self._cfg["overlay"] = self.chk_overlay.isChecked()
        self._cfg.update(self._opt_read())
        save_config(self._cfg_path, self._cfg)

    def closeEvent(self, e):
        if not self._quitting:              # 叉叉 = 缩到托盘，不等于退出
            e.ignore()
            self.hide()
            # No tray balloon: on Windows it plays the notification sound whatever icon
            # we pass (Qt has no NIIF_NOSOUND, and Win11 turns the balloon into a toast
            # whose sound comes from the toast settings). The hint lives in the tray
            # tooltip instead -- hover the icon to see it.
            self.statusBar().showMessage(t("tray_hint"))
            return
        self._hotkey.release()
        self._worker.stop()
        self.tray.hide()
        super().closeEvent(e)


def _pil_to_qimage(img):
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    return QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888).copy()


def run(backend, argv=None):
    """Entry point for an example: ``run(MyBackend(), argv)``."""
    global _LANG, _BACKEND
    _BACKEND = backend
    for lang, extra in (backend.extra_strings or {}).items():
        STRINGS.setdefault(lang, {}).update(extra)

    ap = argparse.ArgumentParser(description=f"Screen translator ({backend.name} + LLM)")
    ap.add_argument("--config", default=resolve_config_path(backend, os.environ.get(config_env(backend))),
                    help="settings file (default: an existing in-repo legacy path, else "
                         f"%APPDATA%/{backend.config_tag}/config.json)")
    ap.add_argument("--lang", choices=["zh", "en"], help="UI language (overrides config)")
    ap.add_argument("--selftest", action="store_true", help="construct + smoke-test, no GUI loop")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, backend)
    save_config(args.config, cfg)   # write back the full schema (so all keys are visible)
    _LANG = args.lang or cfg.get("ui_lang", "zh")

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)   # 叉叉只缩到托盘，退出走托盘菜单
    win = MainWindow(args, cfg, args.config, backend)

    if args.selftest:
        backend.selftest()                    # verifies the sys.path wiring (no model load)
        dlg = SettingsDialog(win, cfg, args.config, backend)   # catch field errors
        safe = {k: ("<set>" if k == "api_key" and v else v) for k, v in win._cfg.items()}
        print(f"OK: selftest ok; ui_lang={_LANG}; config={args.config}")
        print(f"    cfg={safe}")   # api_key redacted
        print(f"    hotkey_global={getattr(win, '_hotkey_global', None)}; "
              f"api_key={'set' if cfg.get('api_key') else 'empty'}")
        win._hotkey.release()
        win._worker.stop()
        return 0

    win.show()
    return app.exec()


_BACKEND = Backend()   # replaced by run()
