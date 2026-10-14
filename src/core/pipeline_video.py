"""End-to-end pipeline for a video job (supports dynamic watermarks).

Steps:
  decode frames -> build keyframe masks (manual or per-frame auto detection) ->
  propagate masks across the clip -> inpaint (per-frame or temporal) -> encode +
  mux original audio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from src.core import mask_propagation
from src.core.types import (
    CancelCheck,
    DetectionMethod,
    JobRequest,
    MediaKind,
    Progress,
    ProgressCb,
)
from src.detection import factory as det_factory
from src.inpaint import factory as inpaint_factory
from src.utils import video_io
from src.utils.device import resolve_device
from src.utils.logger import get_logger

_logger = get_logger("pipeline.video")


def _collect_keyframes(req: JobRequest, frames: List[np.ndarray], device) -> Dict[int, np.ndarray]:
    keyframes: Dict[int, np.ndarray] = {}
    if req.method == DetectionMethod.MANUAL:
        for k, v in req.manual_masks.items():
            idx = int(k)
            if 0 <= idx < len(frames):
                keyframes[idx] = v
        return keyframes

    # auto detection sampled every ``detection_skip`` frames
    detector = det_factory.build(req.method, req, device)
    detector.load()
    try:
        step = max(1, int(req.detection_skip))
        for idx in range(0, len(frames), step):
            result = detector.detect(frames[idx])
            merged = result.merged()
            if merged is not None and merged.any():
                keyframes[idx] = merged.data
    finally:
        detector.close()
    return keyframes


def run(req: JobRequest, progress: ProgressCb | None = None,
        cancel: CancelCheck | None = None) -> str:
    assert req.kind == MediaKind.VIDEO
    device = resolve_device(req.device_prefer)
    _logger.info("Video job: %s -> %s (%s)", req.input_path, req.output_path, device.label)

    def _p(value: int, msg: str = "", cur: int = 0, total: int = 0) -> None:
        if progress:
            progress(Progress(value=value, message=msg, current=cur, total=total))

    _p(3, "decoding frames")
    meta = video_io.probe(req.input_path)
    frames: List[np.ndarray] = list(video_io.read_frames(req.input_path))
    total = len(frames)
    if total == 0:
        raise RuntimeError("video contains no readable frames")
    _logger.info("Decoded %d frames @ %.2f fps, %dx%d", total, meta.fps, meta.width, meta.height)

    if cancel and cancel():
        raise InterruptedError("cancelled")

    _p(12, "building masks")
    keyframes = _collect_keyframes(req, frames, device)
    if not keyframes:
        _logger.warning("No watermark regions detected; copying video through.")
        masks = [np.zeros((meta.height, meta.width), np.uint8) for _ in range(total)]
    else:
        masks = mask_propagation.build_video_masks(
            frames, keyframes, mode=req.propagate, fps=meta.fps,
            fade_in=req.fade_in, fade_out=req.fade_out,
        )
    _logger.info("Masked %d/%d frames", sum(1 for m in masks if m.any()), total)

    Path(req.output_path).parent.mkdir(parents=True, exist_ok=True)
    inpainter = inpaint_factory.build_with_fallback(req.engine, device, req)

    if inpainter.supports_temporal:
        _p(25, f"{inpainter.name} temporal inpaint")
        filled = inpainter.inpaint_video(
            frames, masks,
            progress=_adapt_temporal_progress(_p),
            cancel=cancel,
        )
        _p(85, "encoding")
        writer = video_io.FrameWriter(
            req.output_path, meta.width, meta.height, meta.fps,
            source_video=req.input_path, preserve_audio=req.preserve_audio,
        )
        try:
            for i, f in enumerate(filled):
                if cancel and cancel():
                    break
                writer.write(f)
        finally:
            writer.close()
    else:
        _p(25, f"{inpainter.name} per-frame inpaint")
        writer = video_io.FrameWriter(
            req.output_path, meta.width, meta.height, meta.fps,
            source_video=req.input_path, preserve_audio=req.preserve_audio,
        )
        try:
            for i, (frame, mask) in enumerate(zip(frames, masks)):
                if cancel and cancel():
                    break
                out = inpainter.inpaint_image(frame, mask)
                if req.double_pass and np.any(mask):
                    out = inpainter.inpaint_image(out, mask)
                writer.write(out)
                _p(25 + int(60 * (i + 1) / total), "inpainting", i + 1, total)
        finally:
            writer.close()

    inpainter.close()
    if cancel and cancel():
        _p(100, "cancelled")
        raise InterruptedError("cancelled")
    _p(100, "done")
    return req.output_path


def _adapt_temporal_progress(p):
    def _cb(progress: Progress) -> None:
        # map engine 0..100 into 25..85 window
        mapped = 25 + int(0.6 * progress.value)
        p(mapped, progress.message, progress.current, progress.total)
    return _cb
