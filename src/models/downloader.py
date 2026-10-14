"""Model weight downloader with progress reporting and optional SHA256 checks.

Only the weights we load directly are managed here (LaMa / STTN / ProPainter).
Florence-2 is fetched by ``transformers`` and PaddleOCR by ``paddleocr`` from
their own hubs, so they are configured elsewhere rather than downloaded here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import requests

from src.models import paths
from src.utils.logger import get_logger

_logger = get_logger("downloader")

ProgressCb = Callable[[int, int], None]  # (downloaded_bytes, total_bytes)


@dataclass
class ModelSpec:
    key: str
    filename: str
    url: str
    sub_dir: str
    sha256: Optional[str] = None

    def local_path(self) -> Path:
        return paths.sub_dir(self.sub_dir) / self.filename


# Registry of directly-managed checkpoints.
# NOTE: URLs track upstream releases; if a link changes, update it here.
REGISTRY: Dict[str, ModelSpec] = {
    "lama": ModelSpec(
        key="lama",
        filename="big-lama.pt",
        url="https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
        sub_dir="lama",
    ),
    "sttn": ModelSpec(
        key="sttn",
        filename="sttn_rgb_500k.pth",
        url="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/backend/models/sttn/sttn_rgb_500k.pth",
        sub_dir="sttn",
    ),
    "propainter": ModelSpec(
        key="propainter",
        filename="ProPainter.pth",
        url="https://github.com/sczhou/ProPainter/releases/download/v1.0.0/ProPainter.pth",
        sub_dir="propainter",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_ready(key: str) -> bool:
    spec = REGISTRY[key]
    return spec.local_path().exists()


def ensure(key: str, progress: Optional[ProgressCb] = None, force: bool = False) -> Path:
    """Download *key* if missing and return its local path.

    Raises ``RuntimeError`` on network or checksum failure.
    """
    spec = REGISTRY[key]
    dest = spec.local_path()
    if dest.exists() and not force:
        if spec.sha256 and _sha256(dest) != spec.sha256:
            _logger.warning("Cached %s failed checksum; re-downloading.", key)
        else:
            return dest

    _logger.info("Downloading %s -> %s", spec.url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(spec.url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if spec.sha256:
        actual = _sha256(tmp)
        if actual != spec.sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {key}: expected {spec.sha256}, got {actual}")
    tmp.replace(dest)
    return dest
