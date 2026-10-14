"""Turn sparse keyframe masks into a per-frame mask sequence for a video.

Dynamic (moving / fading) watermarks are only annotated on a few keyframes; this
module propagates them across the clip so every frame gets an accurate mask.

Strategies
----------
* ``none``     - static watermark: union of keyframe masks reused on all frames.
* ``flow``     - dense optical-flow (Farneback) warping of keyframe masks, so a
                 moving region follows the motion.
* ``tracking`` - OpenCV object tracking (CSRT/KCF) seeded from the mask bbox.
* ``auto``     - ``flow`` when more than one keyframe, otherwise ``none``.

Fade in/out simply extends a keyframe's influence forward/backward in time.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("propagation")


def _union(masks: List[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(masks[0])
    for m in masks:
        out = np.maximum(out, m)
    return out


def _warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp a binary mask using a dense flow field (HxWx2)."""
    h, w = mask.shape
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    map_x = map_x + flow[..., 0]
    map_y = map_y + flow[..., 1]
    warped = cv2.remap(mask.astype(np.uint8), map_x, map_y,
                       interpolation=cv2.INTER_NEAREST, borderValue=0)
    return warped


def _to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def propagate_constant(masks: List[np.ndarray], total: int) -> List[np.ndarray]:
    base = _union(masks)
    return [base.copy() for _ in range(total)]


def propagate_flow(frames: List[np.ndarray], keyframes: Dict[int, np.ndarray],
                   total: int) -> List[np.ndarray]:
    """Warp each keyframe mask forwards and backwards through neighbouring flow."""
    result = [np.zeros(frames[0].shape[:2], np.uint8) for _ in range(total)]
    for kf_idx, kf_mask in keyframes.items():
        # forward
        cur = kf_mask.copy()
        result[kf_idx] = np.maximum(result[kf_idx], cur)
        for i in range(kf_idx + 1, total):
            flow = cv2.calcOpticalFlowFarneback(
                _to_gray(frames[i - 1]), _to_gray(frames[i]), None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            cur = _warp_mask(cur, flow)
            result[i] = np.maximum(result[i], cur)
        # backward
        cur = kf_mask.copy()
        for i in range(kf_idx - 1, -1, -1):
            flow = cv2.calcOpticalFlowFarneback(
                _to_gray(frames[i + 1]), _to_gray(frames[i]), None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            cur = _warp_mask(cur, flow)
            result[i] = np.maximum(result[i], cur)
    # ensure the original keyframe pixels stay masked
    for kf_idx, kf_mask in keyframes.items():
        result[kf_idx] = np.maximum(result[kf_idx], kf_mask)
    return result


def propagate_tracking(frames: List[np.ndarray], keyframes: Dict[int, np.ndarray],
                       total: int) -> List[np.ndarray]:
    """Track each keyframe's mask bbox through the clip with a CSRT/KCF tracker."""
    result = [np.zeros(frames[0].shape[:2], np.uint8) for _ in range(total)]
    for kf_idx, kf_mask in keyframes.items():
        ys, xs = np.where(kf_mask > 0)
        if len(xs) == 0:
            continue
        box = (int(xs.min()), int(ys.min()),
               int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        tracker = _make_tracker()
        tracker = _init_tracker(tracker, frames[kf_idx], box)
        if tracker is None:
            result[kf_idx] = np.maximum(result[kf_idx], kf_mask)
            continue
        # forward
        for i in range(kf_idx, total):
            ok, b = tracker.update(frames[i])
            if ok:
                x, y, bw, bh = [int(v) for v in b]
                result[i][max(0, y):min(result[i].shape[0], y + bh),
                          max(0, x):min(result[i].shape[1], x + bw)] = 255
        # backward (re-init and walk in reverse)
        tracker = _make_tracker()
        tracker = _init_tracker(tracker, frames[kf_idx], box)
        if tracker is None:
            continue
        for i in range(kf_idx, -1, -1):
            ok, b = tracker.update(frames[i])
            if ok:
                x, y, bw, bh = [int(v) for v in b]
                result[i][max(0, y):min(result[i].shape[0], y + bh),
                          max(0, x):min(result[i].shape[1], x + bw)] = 255
    return result


def _make_tracker():
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        try:
            return cv2.legacy.TrackerCSRT_create()
        except Exception:  # noqa: BLE001
            return None


def _init_tracker(tracker, frame: np.ndarray, box):
    if tracker is None:
        return None
    try:
        tracker.init(frame, tuple(float(v) for v in box))
        return tracker
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Tracker init failed: %s", exc)
        return None


def apply_fade(masks: List[np.ndarray], keyframes: Dict[int, np.ndarray],
               fps: float, fade_in: float, fade_out: float) -> List[np.ndarray]:
    """Extend each keyframe's mask backwards (fade-in) / forwards (fade-out)."""
    if (fade_in <= 0 and fade_out <= 0) or fps <= 0:
        return masks
    out = list(masks)
    span_in = int(round(fade_in * fps))
    span_out = int(round(fade_out * fps))
    for kf_idx, kf_mask in keyframes.items():
        for i in range(max(0, kf_idx - span_in), min(len(out), kf_idx + span_out + 1)):
            out[i] = np.maximum(out[i], kf_mask)
    return out


def build_video_masks(frames: List[np.ndarray], keyframes: Dict[int, np.ndarray],
                      mode: str = "auto", fps: float = 25.0,
                      fade_in: float = 0.0, fade_out: float = 0.0) -> List[np.ndarray]:
    total = len(frames)
    if total == 0 or not keyframes:
        return [np.zeros(frames[0].shape[:2], np.uint8) for _ in range(total)] if frames else []

    resolved = mode
    if resolved == "auto":
        resolved = "flow" if len(keyframes) > 1 else "none"

    if resolved == "none":
        masks = propagate_constant(list(keyframes.values()), total)
    elif resolved == "tracking":
        masks = propagate_tracking(frames, keyframes, total)
    elif resolved == "flow":
        try:
            masks = propagate_flow(frames, keyframes, total)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Flow propagation failed (%s); using constant mask.", exc)
            masks = propagate_constant(list(keyframes.values()), total)
    else:
        masks = propagate_constant(list(keyframes.values()), total)

    masks = apply_fade(masks, keyframes, fps, fade_in, fade_out)
    return masks
