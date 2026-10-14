"""Smoke tests for video mask propagation (constant mode + fade)."""
import numpy as np

from src.core.mask_propagation import build_video_masks


def _frames(n, h=20, w=30):
    return [np.zeros((h, w, 3), np.uint8) for _ in range(n)]


def test_constant_propagation_reuses_union():
    frames = _frames(5)
    m0 = np.zeros((20, 30), np.uint8)
    m0[0:5, 0:5] = 255
    m2 = np.zeros((20, 30), np.uint8)
    m2[10:15, 10:15] = 255
    masks = build_video_masks(frames, {0: m0, 2: m2}, mode="none", fps=25.0)
    assert len(masks) == 5
    # every frame should carry the union of both keyframe masks
    for m in masks:
        assert m[2, 2] == 255
        assert m[12, 12] == 255


def test_empty_keyframes_yields_empty_masks():
    frames = _frames(3)
    masks = build_video_masks(frames, {}, mode="auto")
    assert len(masks) == 3
    assert all(not m.any() for m in masks)
