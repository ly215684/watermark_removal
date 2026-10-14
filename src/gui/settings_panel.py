"""Right-hand settings dock: detection/propagation/inpaint parameters and tools."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.gui.mask_canvas import Tool
from src.utils.config import Settings


class SettingsPanel(QWidget):
    toolChanged = Signal(object)          # Tool
    brushSizeChanged = Signal(int)
    requestPreview = Signal()
    requestProcess = Signal()
    requestCancel = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._output_dir = ""
        self._preserve_audio = True
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_tools())
        layout.addWidget(self._build_detection())
        layout.addWidget(self._build_propagation())
        layout.addWidget(self._build_inpaint())
        layout.addWidget(self._build_actions())
        layout.addStretch(1)

    # --- widget groups -----------------------------------------------------
    def _build_tools(self) -> QWidget:
        box = QGroupBox("遮罩工具")
        form = QFormLayout(box)
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("矩形", Tool.RECT)
        self.tool_combo.addItem("画笔", Tool.BRUSH)
        self.tool_combo.addItem("橡皮擦", Tool.ERASER)
        self.tool_combo.currentIndexChanged.connect(
            lambda: self.toolChanged.emit(self.tool_combo.currentData()))
        form.addRow("工具", self.tool_combo)

        self.brush_slider = QSlider()
        self.brush_slider.setOrientation(Qt.Horizontal)
        self.brush_slider.setRange(2, 200)
        self.brush_slider.setValue(25)
        self.brush_slider.valueChanged.connect(self.brushSizeChanged.emit)
        form.addRow("画笔大小", self.brush_slider)
        return box

    def _build_detection(self) -> QWidget:
        box = QGroupBox("检测")
        form = QFormLayout(box)
        self.method_combo = QComboBox()
        self.method_combo.addItem("手动（绘制遮罩）", "manual")
        self.method_combo.addItem("自动 - Florence-2", "florence2")
        self.method_combo.addItem("自动 - PaddleOCR（文字）", "paddleocr")
        form.addRow("方式", self.method_combo)

        self.prompt_edit = _line("watermark")
        form.addRow("提示词", self.prompt_edit)

        self.skip_spin = _spin(1, 10, 1)
        form.addRow("每 N 帧检测一次", self.skip_spin)

        self.maxbbox_spin = _dspin(1.0, 100.0, 25.0)
        form.addRow("框最大占画面百分比", self.maxbbox_spin)
        return box

    def _build_propagation(self) -> QWidget:
        box = QGroupBox("动态水印（视频）")
        form = QFormLayout(box)
        self.prop_combo = QComboBox()
        for label, val in [("自动", "auto"), ("静态（不处理）", "none"),
                           ("光流", "flow"), ("跟踪", "tracking")]:
            self.prop_combo.addItem(label, val)
        form.addRow("传播方式", self.prop_combo)

        self.fade_in = _dspin(0.0, 30.0, 0.0)
        self.fade_out = _dspin(0.0, 30.0, 0.0)
        form.addRow("淡入（秒）", self.fade_in)
        form.addRow("淡出（秒）", self.fade_out)
        return box

    def _build_inpaint(self) -> QWidget:
        box = QGroupBox("修复引擎")
        form = QFormLayout(box)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("LaMa（图像 + 逐帧）", "lama")
        self.engine_combo.addItem("STTN（视频时序）", "sttn")
        self.engine_combo.addItem("ProPainter（强运动）", "propainter")
        self.engine_combo.addItem("OpenCV（快速回退）", "opencv")
        form.addRow("引擎", self.engine_combo)

        self.lama_fast = QCheckBox("LaMa 极速（降采样）")
        self.double_pass = QCheckBox("双重修复（减少残留）")
        form.addRow(self.lama_fast)
        form.addRow(self.double_pass)

        self.sttn_neighbor = _spin(1, 50, 10)
        self.sttn_ref = _spin(1, 50, 10)
        self.sttn_load = _spin(5, 100, 30)
        form.addRow("STTN 邻近帧步长", self.sttn_neighbor)
        form.addRow("STTN 参考帧长度", self.sttn_ref)
        form.addRow("STTN 最大加载数", self.sttn_load)
        return box

    def _build_actions(self) -> QWidget:
        box = QGroupBox("操作")
        v = QVBoxLayout(box)
        self.preview_btn = QPushButton("预览检测")
        self.preview_btn.clicked.connect(self.requestPreview.emit)
        self.process_btn = QPushButton("开始去除")
        self.process_btn.clicked.connect(self.requestProcess.emit)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.requestCancel.emit)
        self.cancel_btn.setEnabled(False)
        v.addWidget(self.preview_btn)
        v.addWidget(self.process_btn)
        v.addWidget(self.cancel_btn)
        return box

    # --- state -------------------------------------------------------------
    def set_processing(self, running: bool) -> None:
        self.process_btn.setEnabled(not running)
        self.preview_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def to_settings(self) -> Settings:
        return Settings(
            detection_method=self.method_combo.currentData(),
            detection_prompt=self.prompt_edit.text().strip() or "watermark",
            detection_skip=self.skip_spin.value(),
            max_bbox_percent=self.maxbbox_spin.value(),
            propagate=self.prop_combo.currentData(),
            fade_in=self.fade_in.value(),
            fade_out=self.fade_out.value(),
            inpaint_engine=self.engine_combo.currentData(),
            lama_super_fast=self.lama_fast.isChecked(),
            double_pass=self.double_pass.isChecked(),
            sttn_neighbor_stride=self.sttn_neighbor.value(),
            sttn_reference_length=self.sttn_ref.value(),
            sttn_max_load_num=self.sttn_load.value(),
            output_dir=self._output_dir,
            preserve_audio=self._preserve_audio,
        )

    def apply_settings(self, s: Settings) -> None:
        _set_combo_by_data(self.method_combo, s.detection_method)
        self.prompt_edit.setText(s.detection_prompt)
        self.skip_spin.setValue(s.detection_skip)
        self.maxbbox_spin.setValue(s.max_bbox_percent)
        _set_combo_by_data(self.prop_combo, s.propagate)
        self.fade_in.setValue(s.fade_in)
        self.fade_out.setValue(s.fade_out)
        _set_combo_by_data(self.engine_combo, s.inpaint_engine)
        self.lama_fast.setChecked(s.lama_super_fast)
        self.double_pass.setChecked(s.double_pass)
        self.sttn_neighbor.setValue(s.sttn_neighbor_stride)
        self.sttn_ref.setValue(s.sttn_reference_length)
        self.sttn_load.setValue(s.sttn_max_load_num)
        self._output_dir = s.output_dir
        self._preserve_audio = s.preserve_audio

    def set_output_dir(self, path: str) -> None:
        self._output_dir = path


# --- small factory helpers ---------------------------------------------------
def _line(text: str) -> QLineEdit:
    e = QLineEdit()
    e.setText(text)
    return e


def _spin(minv: int, maxv: int, val: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(minv, maxv)
    s.setValue(val)
    return s


def _dspin(minv: float, maxv: float, val: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(minv, maxv)
    s.setSingleStep(0.1)
    s.setValue(val)
    return s


def _set_combo_by_data(combo: QComboBox, data) -> None:  # noqa: ANN001
    idx = combo.findData(data)
    if idx >= 0:
        combo.setCurrentIndex(idx)
