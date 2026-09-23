"""划词翻译器 — PaddleOCR-VL 后端。Draw a box over any on-screen text, read it with
PaddleOCR-VL, optionally translate it with an LLM (OpenAI-compatible endpoint).

    python example/python/paddleocrvl_vt/screen_translator.py [--config PATH]

The window, the selection overlay, the subtitle, the settings dialog, the hotkey and the
config file all live in the shared shell (`../screen_translator_core/`); this file is only the
part that is specific to PaddleOCR-VL. Everything — LLM, hotkey, model files, the task
prefix — is in the config file (extra keys on top of the shell's):

    "task":      "OCR:",               # the task prefix (see TASKS)
    "model_dir": "",                   # dir holding both GGUFs; "" = repo default
    "model":     "",                   # "" = <model_dir>/PaddleOCR-VL-1.6-GGUF.gguf
    "mmproj":    "",                   # "" = <model_dir>/PaddleOCR-VL-1.6-GGUF-mmproj.gguf
    "max_tokens": 1024,                # decode budget per capture

It resolves to an existing <repo>/data/paddleocrvl/screen_translator.json, else to
%APPDATA%/screen-translator-vl/config.json; ``--config PATH`` / ``$VL_TRANSLATOR_CONFIG``
override.

The box should be a line, a block, or a table — whole pages degenerate (see
sharing_docs/paddleocrvl.md), so select a region rather than the entire document.
"""
import os
import sys

_PYDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../example/python
for _p in (os.path.dirname(os.path.dirname(_PYDIR)), _PYDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)      # so `screen_translator` and `paddleocrvl_vt` resolve

from screen_translator_core import Backend, run, t           # noqa: E402

# the task prefixes the model was trained with — same list as cli.py / llama-server
TASKS = [("OCR", "OCR:"), ("Table Recognition", "Table Recognition:"),
         ("Formula Recognition", "Formula Recognition:"),
         ("Chart Recognition", "Chart Recognition:"),
         ("Seal Recognition", "Seal Recognition:"), ("Spotting", "Spotting:")]


class VlBackend(Backend):
    name = "PaddleOCR-VL"
    config_tag = "screen-translator-vl"
    config_env_name = "VL_TRANSLATOR_CONFIG"
    legacy_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "paddleocrvl", "screen_translator.json")
    extra_config = {"task": "OCR:", "model_dir": "", "model": "", "mmproj": "",
                    "max_tokens": 1024}
    extra_strings = {
        "zh": {
            "lbl_task": "任务",
            "task_tip": "喂给模型的字面前缀。OCR = 整块文字；表格/公式/图表/印章/定位各有前缀。",
            "settings_model_dir": "模型目录",
            "settings_model_dir_tip": "放两个 GGUF 的目录；留空 = 仓库默认 model/paddleocrvl/。\n"
                                      "改这里会立刻重新加载模型。",
            "settings_model_file": "主模型",
            "settings_mmproj_file": "视觉塔 (mmproj)",
            "ph_model_dir": "留空 = 仓库默认 model/paddleocrvl",
            "ph_model_file": "留空 = 目录里的 PaddleOCR-VL-1.6-GGUF.gguf",
            "ph_mmproj_file": "留空 = 目录里的 PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
            "settings_max_tokens": "每帧最多输出 token",
        },
        "en": {
            "lbl_task": "Task",
            "task_tip": "The literal prefix fed to the model. OCR = plain text; table / "
                        "formula / chart / seal / spotting have their own.",
            "settings_model_dir": "Model dir",
            "settings_model_dir_tip": "The directory holding both GGUFs; empty = the repo "
                                      "default model/paddleocrvl/.\nChanging it reloads the "
                                      "model right away.",
            "settings_model_file": "Model",
            "settings_mmproj_file": "Vision tower (mmproj)",
            "ph_model_dir": "empty = repo default model/paddleocrvl",
            "ph_model_file": "empty = PaddleOCR-VL-1.6-GGUF.gguf in that dir",
            "ph_mmproj_file": "empty = PaddleOCR-VL-1.6-GGUF-mmproj.gguf in that dir",
            "settings_max_tokens": "Max output tokens per capture",
        },
    }

    # -- model ---------------------------------------------------------------------

    def _paths(self, cfg):
        """The two GGUFs. An explicit file wins; else a model dir (with the stock
        filenames); else the example's own default (which honours the env vars)."""
        from paddleocrvl_vt import _paths
        d = cfg.get("model_dir")
        if not d:
            return (cfg.get("model") or _paths.DEFAULT_GGUF,
                    cfg.get("mmproj") or _paths.DEFAULT_MMPROJ)
        return (cfg.get("model") or os.path.join(d, "PaddleOCR-VL-1.6-GGUF.gguf"),
                cfg.get("mmproj") or os.path.join(d, "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"))

    def reload_key(self, cfg):
        return self._paths(cfg)

    def load(self, cfg):
        from paddleocrvl_vt import PaddleOCRVL
        model, mmproj = self._paths(cfg)
        for path in (model, mmproj):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"not found: {path} — set the model dir in 设置 / Settings, or the "
                    f"PADDLEOCRVL_MODEL_DIR environment variable")
        return PaddleOCRVL(model, mmproj)

    def recognize(self, vl, arr, options):
        from PIL import Image
        n = int(options.get("max_tokens") or 1024)
        text, _n_prompt, _n_gen = vl.read(Image.fromarray(arr),
                                         options.get("task") or "OCR:", n)
        return text

    # -- widgets -------------------------------------------------------------------

    def options(self, parent, cfg):
        from PyQt6.QtWidgets import QComboBox, QLabel, QWidget, QHBoxLayout
        w = QWidget(parent)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        cmb = QComboBox(w)
        for label, prefix in TASKS:
            cmb.addItem(label, prefix)
        for i in range(cmb.count()):
            if cmb.itemData(i) == cfg.get("task", "OCR:"):
                cmb.setCurrentIndex(i)
                break
        cmb.setToolTip(t("task_tip"))
        lay.addWidget(QLabel(t("lbl_task"), w))
        lay.addWidget(cmb)
        return w, (lambda: {"task": cmb.currentData()})

    def settings_section(self, parent, cfg):
        from PyQt6.QtWidgets import (QWidget, QFormLayout, QHBoxLayout, QLineEdit,
                                     QPushButton, QSpinBox, QFileDialog)
        w = QWidget(parent)
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)

        def browse_dir(edit, title):
            start = edit.text().strip() or os.path.expanduser("~")
            d = QFileDialog.getExistingDirectory(w, title, start)
            if d:
                edit.setText(d)

        def browse_file(edit, title):
            start = edit.text().strip() or os.path.expanduser("~")
            path, _ = QFileDialog.getOpenFileName(w, title, start, "GGUF (*.gguf)")
            if path:
                edit.setText(path)

        def row(edit, placeholder, tip, on_browse):
            edit.setPlaceholderText(placeholder)
            if tip:
                edit.setToolTip(tip)
            b = QPushButton(t("settings_browse"), w)
            b.clicked.connect(on_browse)
            lay = QHBoxLayout()
            lay.addWidget(edit)
            lay.addWidget(b)
            return lay

        ed_dir = QLineEdit(cfg.get("model_dir", ""), w)
        form.addRow(t("settings_model_dir"),
                    row(ed_dir, t("ph_model_dir"), t("settings_model_dir_tip"),
                        lambda: browse_dir(ed_dir, t("settings_model_dir"))))

        ed_model = QLineEdit(cfg.get("model", ""), w)
        form.addRow(t("settings_model_file"),
                    row(ed_model, t("ph_model_file"), "",
                        lambda: browse_file(ed_model, t("settings_model_file"))))
        ed_mm = QLineEdit(cfg.get("mmproj", ""), w)
        form.addRow(t("settings_mmproj_file"),
                    row(ed_mm, t("ph_mmproj_file"), "",
                        lambda: browse_file(ed_mm, t("settings_mmproj_file"))))

        sp_max = QSpinBox(w)
        sp_max.setRange(64, 8192)
        sp_max.setSingleStep(64)
        sp_max.setValue(int(cfg.get("max_tokens") or 1024))
        form.addRow(t("settings_max_tokens"), sp_max)

        return w, (lambda: {"model_dir": ed_dir.text().strip(),
                            "model": ed_model.text().strip(),
                            "mmproj": ed_mm.text().strip(),
                            "max_tokens": sp_max.value()})

    def selftest(self):
        import paddleocrvl_vt  # noqa: F401  (verifies sys.path wiring; no model load)


if __name__ == "__main__":
    raise SystemExit(run(VlBackend()))
