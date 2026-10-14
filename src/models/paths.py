"""Local cache locations for downloaded model weights.

All weights live under ``~/.watermarkremoval/models`` so repeated runs do not
re-download large checkpoints. The directory can be overridden with the
``WMR_MODEL_DIR`` environment variable (useful for portable / CI setups).
"""
from __future__ import annotations

import os
from pathlib import Path


def model_dir() -> Path:
    """Return (and create) the root directory for cached model weights."""
    override = os.environ.get("WMR_MODEL_DIR")
    base = Path(override) if override else Path.home() / ".watermarkremoval" / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def sub_dir(name: str) -> Path:
    """Return a named sub-directory inside the model cache, creating it."""
    path = model_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# Well-known weight locations -------------------------------------------------
def lama_dir() -> Path:
    return sub_dir("lama")


def sttn_dir() -> Path:
    return sub_dir("sttn")


def propainter_dir() -> Path:
    return sub_dir("propainter")


def florence_dir() -> Path:
    return sub_dir("florence2")


def paddleocr_dir() -> Path:
    return sub_dir("paddleocr")


def config_path() -> Path:
    """User configuration file (JSON) alongside the model cache."""
    root = model_dir().parent
    root.mkdir(parents=True, exist_ok=True)
    return root / "config.json"
