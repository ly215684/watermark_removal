"""Manual detector: wraps masks drawn by the user on the canvas.

The pipeline already receives explicit masks via :class:`JobRequest`; this class
exists so manual mode shares the same :class:`Detector` interface, mainly for
"preview" and single-image flows.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.core.types import DetectionMethod, DetectionResult, Mask2D
from src.detection.base import Detector


class ManualDetector(Detector):
    method = DetectionMethod.MANUAL
    requires_model = False

    def __init__(self, masks: Optional[Dict[int, np.ndarray]] = None) -> None:
        # frame_index -> mask (H, W) uint8
        self._masks: Dict[int, np.ndarray] = dict(masks or {})

    def load(self) -> None:
        return None

    def set_mask(self, frame_index: int, mask: np.ndarray) -> None:
        self._masks[frame_index] = mask

    def detect(self, image: np.ndarray, frame_index: int = 0) -> DetectionResult:  # type: ignore[override]
        mask = self._masks.get(frame_index)
        if mask is None:
            return DetectionResult(masks=[], method=self.method)
        h, w = image.shape[:2]
        if mask.shape[:2] != (h, w):
            import cv2

            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return DetectionResult(masks=[Mask2D(data=mask, keyframe_index=frame_index)],
                               method=self.method)
