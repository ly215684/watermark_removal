"""Build an inpainting engine for a request, with graceful degradation.

If a temporal engine (STTN / ProPainter) is unavailable on this machine we log a
warning and fall back to per-frame LaMa, which always works once its weights are
downloaded. As a last resort we use OpenCV inpaint.
"""
from __future__ import annotations

from src.core.types import InpaintEngine, JobRequest
from src.inpaint.base import Inpainter
from src.inpaint.errors import EngineUnavailable
from src.inpaint.lama import LamaInpainter
from src.inpaint.opencv_fallback import OpenCVInpainter
from src.inpaint.propainter import ProPainterInpainter
from src.inpaint.sttn import STTNInpainter
from src.utils.device import DeviceInfo
from src.utils.logger import get_logger

_logger = get_logger("inpaint.factory")


def build(engine: InpaintEngine, device: DeviceInfo, req: JobRequest | None = None) -> Inpainter:
    if engine == InpaintEngine.OPENCV:
        return OpenCVInpainter(device)

    if engine == InpaintEngine.LAMA:
        return LamaInpainter(device, super_fast=bool(req and req.lama_super_fast))

    if engine == InpaintEngine.STTN:
        return STTNInpainter(
            device,
            neighbor_stride=req.sttn_neighbor_stride if req else 10,
            reference_length=req.sttn_reference_length if req else 10,
            max_load_num=req.sttn_max_load_num if req else 30,
        )

    if engine == InpaintEngine.PROPAINTER:
        return ProPainterInpainter(device)

    return LamaInpainter(device)


def build_with_fallback(engine: InpaintEngine, device: DeviceInfo, req: JobRequest) -> Inpainter:
    """Build the requested engine and ensure it loads; degrade if it cannot."""
    candidate = build(engine, device, req)
    try:
        candidate.load()
        return candidate
    except EngineUnavailable as exc:
        _logger.warning("Engine '%s' unavailable: %s. Falling back to LaMa.", engine.value, exc)
        fallback = LamaInpainter(device, super_fast=req.lama_super_fast)
        try:
            fallback.load()
            return fallback
        except Exception as exc2:  # noqa: BLE001
            _logger.warning("LaMa failed to load: %s. Falling back to OpenCV.", exc2)
            opencv = OpenCVInpainter(device)
            opencv.load()
            return opencv
