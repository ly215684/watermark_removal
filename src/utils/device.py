"""Compute device selection with graceful CUDA -> CPU fallback.

Importing this module never fails even if torch is not installed; in that case
:func:`resolve_device` returns ``"cpu"`` and :data:`TORCH_AVAILABLE` is False so
callers can raise informative errors only when a model is actually requested.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.logger import get_logger

_logger = get_logger("device")

try:  # pragma: no cover - depends on environment
    import torch

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001 - torch optional at import time
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class DeviceInfo:
    name: str          # "cuda" | "cpu"
    label: str         # human readable, e.g. "CUDA (NVIDIA GeForce RTX 3080)"
    torch_device: str  # value usable by ``torch.device(...)``
    available: bool

    @property
    def is_cuda(self) -> bool:
        return self.name == "cuda"


def detect() -> DeviceInfo:
    """Return the best available device."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(idx)
        return DeviceInfo(
            name="cuda",
            label=f"CUDA ({gpu_name})",
            torch_device=f"cuda:{idx}",
            available=True,
        )
    label = "CPU"
    if not TORCH_AVAILABLE:
        label = "CPU (torch not installed)"
    return DeviceInfo(name="cpu", label=label, torch_device="cpu", available=True)


def resolve_device(prefer: str = "auto") -> DeviceInfo:
    """Resolve a device, honoring an explicit ``prefer`` of ``cuda``/``cpu``/``auto``."""
    info = detect()
    if prefer == "cpu":
        return DeviceInfo("cpu", "CPU", "cpu", True)
    if prefer == "cuda" and not info.is_cuda:
        _logger.warning("CUDA requested but unavailable; falling back to CPU.")
    return info
