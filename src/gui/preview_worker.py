"""One-shot background worker that runs auto-detection on a single frame."""
from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.types import DetectionMethod, JobRequest, MediaKind
from src.detection import factory as det_factory
from src.utils.config import Settings
from src.utils.device import resolve_device
from src.utils.logger import get_logger

_logger = get_logger("gui.preview")


class PreviewWorker(QThread):
    done = Signal(object)   # emits np.ndarray mask (may be all-zero)
    failed = Signal(str)

    def __init__(self, frame: np.ndarray, method: str, settings: Settings) -> None:
        super().__init__()
        self._frame = frame
        self._method = method
        self._settings = settings

    def run(self) -> None:  # noqa: D102
        try:
            device = resolve_device(self._settings.device_prefer)
            req = JobRequest(
                input_path="", output_path="", kind=MediaKind.IMAGE,
                method=DetectionMethod(self._method),
                engine=None, manual_masks={},
                detection_prompt=self._settings.detection_prompt,
                detection_skip=self._settings.detection_skip,
                max_bbox_percent=self._settings.max_bbox_percent,
            )
            detector = det_factory.build(DetectionMethod(self._method), req, device)
            detector.load()
            result = detector.detect(self._frame)
            detector.close()
            merged = result.merged()
            mask = merged.data if merged else np.zeros(self._frame.shape[:2], np.uint8)
            self.done.emit(mask)
        except Exception as exc:  # noqa: BLE001
            _logger.error("Preview failed: %s", exc)
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
