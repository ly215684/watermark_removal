"""Project-wide logging setup.

Import ``get_logger`` and use module-scoped loggers. A Qt-friendly sink can be
attached at runtime (see :mod:`src.gui.workers`) so log lines surface in the UI.
"""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("wmr")
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"wmr.{name}")


class QtLogSink(logging.Handler):
    """Logging handler that forwards records to a callable (e.g. a Qt signal)."""

    def __init__(self, callback) -> None:  # noqa: ANN001 - callable[str]
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            self._callback(self.format(record))
        except Exception:  # noqa: BLE001 - never break logging
            pass
