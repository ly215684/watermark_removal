"""Resource path helper that works both from source and a PyInstaller bundle."""
from __future__ import annotations

import sys
from pathlib import Path


def resource_path(rel: str) -> Path:
    """Resolve *rel* against the bundle temp dir when frozen, else project root."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / rel
    # running from source: project root is two levels above src/utils
    return Path(__file__).resolve().parents[2] / rel
