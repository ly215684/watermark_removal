"""Construct a detector for the requested method."""
from __future__ import annotations

from src.core.types import DetectionMethod, JobRequest
from src.detection.base import Detector
from src.detection.manual import ManualDetector


def build(method: DetectionMethod, req: JobRequest | None = None, device=None) -> Detector:  # noqa: ANN001
    if method == DetectionMethod.MANUAL:
        masks = {(int(k) if isinstance(k, str) else k): v for k, v in (req.manual_masks.items())} if req else {}
        return ManualDetector(masks)

    if method == DetectionMethod.FLORENCE2:
        from src.detection.florence2 import Florence2Detector

        return Florence2Detector(
            device,
            prompt=req.detection_prompt if req else "watermark",
            max_bbox_percent=req.max_bbox_percent if req else 25.0,
        )

    if method == DetectionMethod.PADDLEOCR:
        from src.detection.paddleocr_det import PaddleOCRDetector

        return PaddleOCRDetector(max_bbox_percent=req.max_bbox_percent if req else 60.0)

    raise ValueError(f"Unknown detection method: {method}")
