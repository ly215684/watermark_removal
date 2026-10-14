"""Shared data types used across detection, propagation and inpainting.

Numpy conventions:
* images/frames are BGR uint8 arrays of shape (H, W, 3)
* masks are single channel uint8 arrays of shape (H, W) with values 0 or 255
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class DetectionMethod(str, Enum):
    MANUAL = "manual"
    FLORENCE2 = "florence2"
    PADDLEOCR = "paddleocr"


class InpaintEngine(str, Enum):
    LAMA = "lama"
    STTN = "sttn"
    PROPAINTER = "propainter"
    OPENCV = "opencv"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DONE = "done"


@dataclass
class BBox:
    """Axis-aligned box in pixel coordinates (x0, y0, x1, y1)."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Mask2D:
    """A binary mask for one frame/image plus optional provenance."""
    data: np.ndarray            # (H, W) uint8, 0/255
    boxes: List[BBox] = field(default_factory=list)
    keyframe_index: Optional[int] = None

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def any(self) -> bool:
        return bool(np.any(self.data > 0))


@dataclass
class DetectionResult:
    """Output of a detector for a single image/frame."""
    masks: List[Mask2D] = field(default_factory=list)
    method: DetectionMethod = DetectionMethod.MANUAL

    def merged(self) -> Optional[Mask2D]:
        if not self.masks:
            return None
        base = np.zeros_like(self.masks[0].data, dtype=np.uint8)
        for m in self.masks:
            base = np.maximum(base, m.data)
        boxes: List[BBox] = []
        for m in self.masks:
            boxes.extend(m.boxes)
        return Mask2D(data=base, boxes=boxes)


@dataclass
class JobRequest:
    """Everything needed to run one image/video job."""
    input_path: str
    output_path: str
    kind: MediaKind
    method: DetectionMethod
    engine: InpaintEngine
    # manual masks supplied from the canvas, keyed by frame index (0 for image)
    manual_masks: dict = field(default_factory=dict)
    # auto-detection settings
    detection_prompt: str = "watermark"
    detection_skip: int = 1
    max_bbox_percent: float = 25.0
    # propagation
    propagate: str = "auto"
    fade_in: float = 0.0
    fade_out: float = 0.0
    # inpaint options
    lama_super_fast: bool = False
    double_pass: bool = False
    mask_mode: str = "box"
    sttn_neighbor_stride: int = 10
    sttn_reference_length: int = 10
    sttn_max_load_num: int = 30
    # io
    preserve_audio: bool = True
    device_prefer: str = "auto"


@dataclass
class Progress:
    value: int          # 0..100
    message: str = ""
    current: int = 0
    total: int = 0


# A cancel-check callable returns True when the job should stop early.
CancelCheck = Callable[[], bool]
ProgressCb = Callable[[Progress], None]
