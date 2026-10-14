"""PaddleOCR text-region detection for text-like watermarks / subtitles.

Aligns with the VSR approach: PP-OCR's text detector localizes text boxes, which
are turned into masks for inpainting. Best for textual watermarks, not logos.

Heavy / optional: requires ``paddlepaddle`` + ``paddleocr``.
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.core.types import BBox, DetectionMethod, DetectionResult, Mask2D
from src.detection.base import Detector, polygon_to_mask, quad_to_box
from src.utils.logger import get_logger

_logger = get_logger("paddleocr")


class PaddleOCRDetector(Detector):
    method = DetectionMethod.PADDLEOCR
    requires_model = True

    def __init__(self, lang: str = "en", det_score_thresh: float = 0.6,
                 max_bbox_percent: float = 60.0) -> None:
        self.lang = lang
        self.det_score_thresh = det_score_thresh
        self.max_bbox_percent = max_bbox_percent
        self._ocr = None

    def load(self) -> None:
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR

        _logger.info("Initializing PaddleOCR (lang=%s)", self.lang)
        try:
            self._ocr = PaddleOCR(use_angle_cls=False, lang=self.lang,
                                  show_log=False, det=True, rec=False, cls=False)
        except TypeError:
            # Newer paddleocr removed show_log / changed args
            self._ocr = PaddleOCR(lang=self.lang)

    def detect(self, image: np.ndarray) -> DetectionResult:
        if self._ocr is None:
            self.load()
        h, w = image.shape[:2]
        frame_area = float(h * w)
        raw = self._ocr.ocr(image, det=True, rec=False, cls=False) if self._has_kwargs() \
            else self._ocr.ocr(image)

        polys: List[np.ndarray] = []
        # paddleocr returns [[ [box, (text, score)], ... ]] (one page) or [box,...]
        page = raw[0] if raw and isinstance(raw[0], list) else raw
        for entry in page or []:
            box, score = self._unpack(entry)
            if box is None:
                continue
            if score is not None and float(score) < self.det_score_thresh:
                continue
            pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            bbox = quad_to_box(pts)
            if frame_area > 0 and (bbox.area / frame_area) * 100.0 > self.max_bbox_percent:
                continue
            polys.append(pts)

        if not polys:
            return DetectionResult(masks=[], method=self.method)
        mask = np.zeros((h, w), np.uint8)
        boxes: List[BBox] = []
        for pts in polys:
            mask = np.maximum(mask, polygon_to_mask((h, w), pts))
            boxes.append(quad_to_box(pts))
        # dilate a bit to cover anti-aliased glyph edges
        import cv2

        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        _logger.info("PaddleOCR found %d text region(s)", len(polys))
        return DetectionResult(masks=[Mask2D(data=mask, boxes=boxes)], method=self.method)

    def _has_kwargs(self) -> bool:  # pragma: no cover - version adapter
        import inspect

        try:
            sig = inspect.signature(self._ocr.ocr)
            return "det" in sig.parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _unpack(entry):  # noqa: ANN001
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[1], (list, tuple)):
            return entry[0], (entry[1][1] if len(entry[1]) >= 2 else None)
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return entry[0], entry[1]
        return entry, None
