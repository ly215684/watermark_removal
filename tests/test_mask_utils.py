"""Dependency-light smoke tests for core mask helpers and propagation."""
import numpy as np

from src.core.types import BBox
from src.detection.base import bbox_to_mask, boxes_to_mask, quad_to_box, polygon_to_mask


def test_bbox_to_mask_fills_region():
    mask = bbox_to_mask((100, 200), BBox(10, 20, 50, 60))
    assert mask.shape == (100, 200)
    assert mask[30, 30] == 255
    assert mask[0, 0] == 0
    assert mask.dtype == np.uint8


def test_boxes_to_mask_dilates():
    mask = boxes_to_mask((50, 50), [BBox(20, 20, 25, 25)], dilate=2)
    # dilation should grow the 5x5 box beyond its borders
    assert mask[18, 18] == 255
    assert mask.any()


def test_quad_to_box():
    quad = np.array([[1, 2], [10, 2], [10, 8], [1, 8]], dtype=np.float32)
    box = quad_to_box(quad)
    assert (box.x0, box.y0, box.x1, box.y1) == (1, 2, 10, 8)


def test_polygon_to_mask():
    pts = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.int32)
    mask = polygon_to_mask((40, 40), pts)
    assert mask[20, 20] == 255
    assert mask[0, 0] == 0
