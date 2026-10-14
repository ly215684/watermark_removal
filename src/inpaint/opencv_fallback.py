"""OpenCV ``cv2.inpaint`` fallback.

No deep-learning dependency; useful for quick previews or when model weights are
unavailable. Quality is far below LaMa/STTN but it never fails to load.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.inpaint.base import Inpainter


class OpenCVInpainter(Inpainter):
    name = "opencv"
    supports_temporal = False
    weight_key = None

    def __init__(self, device, radius: int = 5, method: int = cv2.INPAINT_TELEA) -> None:  # noqa: ANN001
        super().__init__(device)
        self.radius = radius
        self.method = method

    def load(self) -> None:
        return None

    def inpaint_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or not np.any(mask):
            return image
        m = (mask > 0).astype(np.uint8) * 255
        return cv2.inpaint(image, m, self.radius, self.method)
