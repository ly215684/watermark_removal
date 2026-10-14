"""Common interface for inpainting engines.

Two flavours exist:
* per-frame (image) engines - LaMa, OpenCV: implement :meth:`inpaint_image`
* temporal (video) engines  - STTN, ProPainter: implement :meth:`inpaint_video`

``supports_temporal`` lets the video pipeline choose the right call. Per-frame
engines are automatically applied frame-by-frame by the pipeline.
"""
from __future__ import annotations

import abc
from typing import List

import numpy as np

from src.core.types import CancelCheck, Progress
from src.utils.device import DeviceInfo


class Inpainter(abc.ABC):
    name: str = "base"
    supports_temporal: bool = False
    #: weight key managed by :mod:`src.models.downloader` (None if none needed)
    weight_key: str | None = None

    def __init__(self, device: DeviceInfo) -> None:
        self.device = device

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights onto the target device (idempotent)."""

    @abc.abstractmethod
    def inpaint_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return a repaired BGR image for a single 0/255 mask."""

    def inpaint_video(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        progress=None,          # ProgressCb
        cancel: CancelCheck | None = None,
    ) -> List[np.ndarray]:
        """Temporal repair. Default: apply :meth:`inpaint_image` per frame."""
        outputs: List[np.ndarray] = []
        total = len(frames)
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            if cancel and cancel():
                break
            outputs.append(self.inpaint_image(frame, mask))
            if progress:
                progress(Progress(value=int(100 * (i + 1) / max(1, total)),
                                  current=i + 1, total=total,
                                  message=f"{self.name} {i + 1}/{total}"))
        return outputs

    def close(self) -> None:  # pragma: no cover - optional override
        pass
