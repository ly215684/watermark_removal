"""User configuration persistence (JSON) with sensible defaults.

Stores engine/parameter choices so they survive restarts. Kept dependency-free
(no Qt) so pipelines and CLI usage can read the same config.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from src.models import paths
from src.utils.logger import get_logger

_logger = get_logger("config")


@dataclass
class Settings:
    # device
    device_prefer: str = "auto"          # auto | cuda | cpu
    # detection
    detection_method: str = "manual"     # manual | florence2 | paddleocr
    detection_prompt: str = "watermark"
    detection_skip: int = 1              # detect every N frames (video)
    max_bbox_percent: float = 25.0       # ignore auto boxes larger than this % of frame
    # propagation
    propagate: str = "auto"              # auto | tracking | flow | none
    fade_in: float = 0.0                 # extend mask backwards (seconds)
    fade_out: float = 0.0                # extend mask forwards (seconds)
    # inpaint
    inpaint_engine: str = "lama"         # lama | sttn | propainter | opencv
    lama_super_fast: bool = False
    double_pass: bool = False
    mask_mode: str = "box"               # box | stroke
    sttn_neighbor_stride: int = 10
    sttn_reference_length: int = 10
    sttn_max_load_num: int = 30
    # io
    output_dir: str = ""
    preserve_audio: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load() -> Settings:
    path = paths.config_path()
    if not path.exists():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in Settings.__dict__ or k in Settings.__dataclass_fields__}
        return Settings(**known)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Failed to read config (%s); using defaults.", exc)
        return Settings()


def save(settings: Settings) -> None:
    path = paths.config_path()
    try:
        path.write_text(json.dumps(settings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Failed to write config: %s", exc)
