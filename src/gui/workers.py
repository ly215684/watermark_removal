"""Background QThread workers so the GUI never blocks during inference."""
from __future__ import annotations

import logging
import traceback

from PySide6.QtCore import QThread, Signal

from src.core import pipeline_image, pipeline_video
from src.core.types import JobRequest, MediaKind, Progress
from src.models import downloader
from src.utils.logger import QtLogSink, get_logger

_logger = get_logger("workers")


class TaskWorker(QThread):
    """Runs one or more jobs (batch) on a worker thread."""

    progress = Signal(int, str)          # percent, message
    log = Signal(str)
    job_done = Signal(str)               # output path of a finished item
    finished_all = Signal()
    failed = Signal(str)

    def __init__(self, jobs: list[JobRequest]) -> None:
        super().__init__()
        self._jobs = jobs
        self._cancelled = False
        self._attach_log_sink()

    def _attach_log_sink(self) -> None:
        sink = QtLogSink(self.log.emit)
        sink.setLevel(logging.INFO)
        sink.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logging.getLogger("wmr").addHandler(sink)
        self._sink = sink

    def cancel(self) -> None:
        self._cancelled = True

    def _cancel_check(self):
        return lambda: self._cancelled

    def run(self) -> None:  # noqa: D102 - QThread entry
        try:
            n = len(self._jobs)
            for i, job in enumerate(self._jobs):
                if self._cancelled:
                    break
                base = int(100 * i / n)
                span = int(100 / n)

                def _emit(p: Progress, base=base, span=span) -> None:
                    self.progress.emit(base + int(p.value * span / 100), p.message)

                run = pipeline_image.run if job.kind == MediaKind.IMAGE else pipeline_video.run
                out = run(job, progress=_emit, cancel=self._cancel_check())
                self.job_done.emit(out)
            if self._cancelled:
                self.log.emit("Task cancelled by user.")
            self.finished_all.emit()
        except Exception as exc:  # noqa: BLE001
            _logger.error("Job failed: %s", exc)
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            logging.getLogger("wmr").removeHandler(self._sink)


class DownloadWorker(QThread):
    """Downloads a managed model checkpoint with progress."""

    progress = Signal(int, str)          # percent, message
    finished_ok = Signal(str)            # local path
    failed = Signal(str)

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key

    def run(self) -> None:  # noqa: D102 - QThread entry
        try:
            def _cb(done: int, total: int) -> None:
                pct = int(100 * done / total) if total else 0
                self.progress.emit(pct, f"downloading {self._key} ({done // 1024}KB)")

            path = downloader.ensure(self._key, progress=_cb)
            self.finished_ok.emit(str(path))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
