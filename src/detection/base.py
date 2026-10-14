"""Detector interface and mask helpers.

A detector turns an image/frame into a :class:`DetectionResult` (list of
:class:`Mask2D`). Manual mode simply forwards canvas masks and needs no model.
"""
from __future__ import annotations

import abc
from typing import List

import cv2
import numpy as np

from src.core.types import BBox, DetectionMethod, DetectionResult, Mask2D


class Detector(abc.ABC):
    method: DetectionMethod
    #: whether this detector needs a downloadable/AI model
    requires_model: bool = False

    @abc.abstractmethod
    def load(self) -> None:
        """Initialize the backing model (no-op for manual)."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Return masks covering watermarks/text in a BGR image."""

    def close(self) -> None:  # pragma: no cover - optional
        pass


# --- mask utilities ----------------------------------------------------------
def bbox_to_mask(shape_hw: tuple[int, int], box: BBox, fill: int = 255) -> np.ndarray:
    """Create a filled-rectangle mask from a bbox, clipped to the frame."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    x0 = max(0, min(w - 1, box.x0))
    x1 = max(0, min(w, box.x1))
    y0 = max(0, min(h - 1, box.y0))
    y1 = max(0, min(h, box.y1))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = fill
    return mask


def boxes_to_mask(shape_hw, boxes: List[BBox], dilate: int = 2) -> np.ndarray:  # noqa: ANN001
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for b in boxes:
        mask = np.maximum(mask, bbox_to_mask(shape_hw, b))
    if dilate > 0 and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=dilate)
    return mask


def polygon_to_mask(shape_hw, points: np.ndarray, fill: int = 255) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    pts = points.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, pts, fill)
    return mask


def to_box(points: np.ndarray) -> BBox:
    """Bounding box around a set of (x, y) points."""
    xs, ys = points[:, 0], points[:, 1]
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def quad_to_box(quad: np.ndarray) -> BBox:
    """Convert a 4-point polygon (PaddleOCR/Florence quad) to an axis-aligned box."""
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    return to_box(pts)
