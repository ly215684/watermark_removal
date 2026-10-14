"""End-to-end pipeline for a single image job."""
from __future__ import annotations

from pathlib import Path

import numpy as np

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

_logger = get_logger("pipeline.image")


def _get_mask(req: JobRequest, image: np.ndarray, device):
    if req.method == DetectionMethod.MANUAL:
        mask = req.manual_masks.get(0) or req.manual_masks.get("0")
        if mask is None:
            return None
        if mask.shape[:2] != image.shape[:2]:
            import cv2

            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    detector = det_factory.build(req.method, req, device)
    detector.load()
    result = detector.detect(image)
    merged = result.merged()
    detector.close()
    return merged.data if merged else None


def run(req: JobRequest, progress: ProgressCb | None = None,
        cancel: CancelCheck | None = None) -> str:
    assert req.kind == MediaKind.IMAGE
    device = resolve_device(req.device_prefer)
    _logger.info("Image job: %s -> %s (%s)", req.input_path, req.output_path, device.label)

    def _p(value: int, msg: str = "") -> None:
        if progress:
            progress(Progress(value=value, message=msg))

    _p(5, "loading image")
    image = video_io.load_image(req.input_path)

    if cancel and cancel():
        raise InterruptedError("cancelled")

    _p(20, "detecting")
    mask = _get_mask(req, image, device)
    if mask is None or not np.any(mask):
        _logger.warning("No mask produced; nothing to remove.")
        # Still write an output copy so batch flows remain predictable.
        video_io.save_image(req.output_path, image)
        _p(100, "no watermark found")
        return req.output_path

    _p(40, f"loading engine: {req.engine.value}")
    inpainter = inpaint_factory.build_with_fallback(req.engine, device, req)

    _p(60, "inpainting")
    out = inpainter.inpaint_image(image, mask)
    if req.double_pass:
        _p(80, "inpainting (2nd pass)")
        out = inpainter.inpaint_image(out, mask)
    inpainter.close()

    _p(92, "saving")
    Path(req.output_path).parent.mkdir(parents=True, exist_ok=True)
    video_io.save_image(req.output_path, out)
    _p(100, "done")
    return req.output_path
