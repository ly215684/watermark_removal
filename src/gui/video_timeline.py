"""Video transport controls: play/pause, scrub slider, frame stepper.

Only owns the frame index and playback timer; loading/decoding of actual frames
is delegated to the parent through the ``frameChanged`` signal so this widget
stays independent of the video backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class VideoTimeline(QWidget):
    frameChanged = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._count = 0
        self._index = 0
        self._fps = 25.0
        self._path: Optional[Path] = None

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider)

        self.play_btn = QPushButton("▶")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._toggle_play)

        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(lambda: self.step(1))

        self.label = QLabel("0 / 0")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label)

    # --- public API --------------------------------------------------------
    def load(self, path: str | Path, count: int, fps: float) -> None:
        self._path = Path(path)
        self._count = max(0, count)
        self._fps = fps if fps > 0 else 25.0
        self._index = 0
        self.slider.setMaximum(max(0, self._count - 1))
        self.slider.setValue(0)
        self.stop()
        self._update_label()
        self.frameChanged.emit(0)

    def clear(self) -> None:
        self._path = None
        self._count = 0
        self.slider.setMaximum(0)
        self.stop()

    def current_index(self) -> int:
        return self._index

    def set_index(self, index: int) -> None:
        index = max(0, min(self._count - 1, index))
        self.slider.setValue(index)

    def step(self, delta: int) -> None:
        self.set_index(self._index + delta)

    def stop(self) -> None:
        self.play_btn.setChecked(False)
        self._timer.stop()

    # --- internals ---------------------------------------------------------
    def _toggle_play(self, checked: bool) -> None:
        if checked:
            interval = max(10, int(1000 / self._fps))
            self._timer.start(interval)
        else:
            self._timer.stop()

    def _on_tick(self) -> None:
        if self._count == 0:
            self.stop()
            return
        nxt = self._index + 1
        if nxt >= self._count:
            self.stop()
            nxt = 0
        self.set_index(nxt)

    def _on_slider(self, value: int) -> None:
        self._index = value
        self._update_label()
        self.frameChanged.emit(value)

    def _update_label(self) -> None:
        self.label.setText(f"{self._index} / {max(0, self._count - 1)}")
