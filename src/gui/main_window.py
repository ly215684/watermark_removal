"""Main application window wiring the canvas, timeline, settings and workers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QProgressBar,
    QWidget,
    QPlainTextEdit,
    QMessageBox,
)

from src.core.types import (
    DetectionMethod,
    InpaintEngine,
    JobRequest,
    MediaKind,
)
from src.gui.mask_canvas import MaskCanvas
from src.gui.preview_worker import PreviewWorker
from src.gui.settings_panel import SettingsPanel
from src.gui.video_timeline import VideoTimeline
from src.gui.workers import DownloadWorker, TaskWorker
from src.utils import config, video_io
from src.utils.device import detect
from src.utils.logger import get_logger
from src.utils.resource import resource_path

_logger = get_logger("gui.main")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Watermark Remover")
        self._apply_window_icon()
        self.resize(1200, 760)

        self._input_path: Optional[Path] = None
        self._kind: MediaKind = MediaKind.IMAGE
        self._current_frame: Optional[np.ndarray] = None
        self._manual_masks: Dict[int, np.ndarray] = {}
        self._batch_paths: List[Path] = []
        self._worker: Optional[TaskWorker] = None
        self._preview_worker: Optional[PreviewWorker] = None

        self._build_ui()
        self._build_menu()
        self._load_settings()
        self._update_device_label()

    # --- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        hbox = QHBoxLayout(central)

        left = QWidget()
        left_layout = QHBoxLayout(left)
        left_layout.addWidget(self._build_canvas_area())
        hbox.addWidget(left, 3)

        self.settings_panel = SettingsPanel()
        dock = QDockWidget("Settings", self)
        dock.setWidget(self.settings_panel)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setMinimumWidth(320)

        self.setCentralWidget(central)

        # bottom status bar
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        bar = self.statusBar()
        bar.addWidget(self.status_label)
        bar.addPermanentWidget(self.progress)

        # log dock
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_dock = QDockWidget("Log", self)
        log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

        # connect settings signals
        sp = self.settings_panel
        sp.toolChanged.connect(self.canvas.set_tool)
        sp.brushSizeChanged.connect(self.canvas.set_brush_size)
        sp.requestProcess.connect(self.on_start)
        sp.requestCancel.connect(self.on_cancel)
        sp.requestPreview.connect(self.on_preview)

    def _build_canvas_area(self) -> QWidget:
        wrap = QWidget()
        from PySide6.QtWidgets import QVBoxLayout

        v = QVBoxLayout(wrap)
        self.canvas = MaskCanvas()
        v.addWidget(self.canvas, 1)
        self.timeline = VideoTimeline()
        self.timeline.setVisible(False)
        self.timeline.frameChanged.connect(self.on_frame_changed)
        v.addWidget(self.timeline)
        return wrap

    def _build_menu(self) -> None:
        bar: QMenuBar = self.menuBar()
        file_menu = bar.addMenu("File")
        open_file = QAction("Open file…", self)
        open_file.setShortcut(QKeySequence("Ctrl+O"))
        open_file.triggered.connect(self.on_open_file)
        open_dir = QAction("Open folder (batch)…", self)
        open_dir.triggered.connect(self.on_open_folder)
        out_dir = QAction("Set output folder…", self)
        out_dir.triggered.connect(self.on_set_output_dir)
        file_menu.addAction(open_file)
        file_menu.addAction(open_dir)
        file_menu.addAction(out_dir)

        models_menu = bar.addMenu("Models")
        for key, label in [("lama", "Download LaMa"), ("sttn", "Download STTN"),
                           ("propainter", "Download ProPainter")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, k=key: self.on_download_model(k))
            models_menu.addAction(act)

        edit_menu = bar.addMenu("Edit")
        undo = QAction("Undo mask", self)
        undo.setShortcut(QKeySequence("Ctrl+Z"))
        undo.triggered.connect(self.canvas.undo)
        clear = QAction("Clear mask", self)
        clear.setShortcut(QKeySequence("Ctrl+Del"))
        clear.triggered.connect(self.canvas.clear_mask)
        edit_menu.addAction(undo)
        edit_menu.addAction(clear)

        self.device_label = QLabel()
        bar.setCornerWidget(self.device_label, Qt.TopRightCorner)

    # --- device / status ---------------------------------------------------
    def _apply_window_icon(self) -> None:
        try:
            from PySide6.QtGui import QIcon

            for rel in ("assets/app.ico", "assets/app.png"):
                path = resource_path(rel)
                if path.exists():
                    self.setWindowIcon(QIcon(str(path)))
                    return
        except Exception:  # noqa: BLE001 - icon is best-effort
            pass

    def _update_device_label(self) -> None:
        info = detect()
        self.device_label.setText(f"Device: {info.label}")

    def _log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)
        _logger.info(msg)

    # --- open / load -------------------------------------------------------
    def on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image or video", "",
            "Media (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)")
        if not path:
            return
        self.load_media(Path(path))

    def on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder to batch process")
        if not folder:
            return
        files = video_io.collect_media(folder)
        if not files:
            QMessageBox.information(self, "Batch", "No supported media found in folder.")
            return
        self._batch_paths = files
        self._log(f"Batch: {len(files)} files queued.")
        # show first file for reference
        self.load_media(files[0])

    def on_set_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.settings_panel.set_output_dir(folder)
            self._log(f"Output folder: {folder}")

    def load_media(self, path: Path) -> None:
        self._input_path = path
        self._manual_masks.clear()
        self.timeline.stop()
        if video_io.is_image(path):
            self._kind = MediaKind.IMAGE
            self.timeline.setVisible(False)
            img = video_io.load_image(path)
            self._current_frame = img
            self.canvas.set_frame(img)
            self._log(f"Loaded image: {path.name} ({img.shape[1]}x{img.shape[0]})")
        elif video_io.is_video(path):
            self._kind = MediaKind.VIDEO
            meta = video_io.probe(path)
            self.timeline.setVisible(True)
            self.timeline.load(path, meta.frame_count, meta.fps)
            frame = video_io.read_frame_at(path, 0)
            self._current_frame = frame
            if frame is not None:
                self.canvas.set_frame(frame)
            self._log(f"Loaded video: {path.name} ({meta.width}x{meta.height}, "
                      f"{meta.frame_count} frames @ {meta.fps:.1f}fps)")
        else:
            QMessageBox.warning(self, "Unsupported", f"Not a supported file: {path.name}")

    def on_frame_changed(self, index: int) -> None:
        if self._input_path is None or self._kind != MediaKind.VIDEO:
            return
        # stash current frame's mask
        self._commit_current_mask()
        frame = video_io.read_frame_at(self._input_path, index)
        if frame is None:
            return
        self._current_frame = frame
        self.canvas.set_frame(frame)
        mask = self._manual_masks.get(index)
        if mask is not None:
            self.canvas.set_mask(mask)

    def _commit_current_mask(self) -> None:
        if self._current_frame is None:
            return
        mask = self.canvas.get_mask()
        if mask is None:
            return
        idx = 0 if self._kind == MediaKind.IMAGE else self.timeline.current_index()
        if mask.any():
            self._manual_masks[idx] = mask.copy()
        elif idx in self._manual_masks:
            del self._manual_masks[idx]

    # --- preview detection -------------------------------------------------
    def on_preview(self) -> None:
        method = self.settings_panel.method_combo.currentData()
        if method == "manual" or self._current_frame is None:
            self._log("Preview needs an auto detection method and a loaded frame.")
            return
        s = self.settings_panel.to_settings()
        self._log(f"Running {method} preview on current frame…")
        self._preview_worker = PreviewWorker(self._current_frame.copy(), method, s)
        self._preview_worker.done.connect(self._on_preview_done)
        self._preview_worker.failed.connect(lambda m: self._log(f"Preview failed: {m}"))
        self.settings_panel.preview_btn.setEnabled(False)
        self._preview_worker.start()

    def _on_preview_done(self, mask: np.ndarray) -> None:
        self.settings_panel.preview_btn.setEnabled(True)
        if mask is None or not mask.any():
            self._log("Preview: nothing detected.")
            return
        self.canvas.set_mask(mask)
        self._log("Preview: detection overlay applied (also saved as mask).")

    # --- process -----------------------------------------------------------
    def on_start(self) -> None:
        if self._input_path is None:
            QMessageBox.information(self, "Start", "Open a file first.")
            return
        self._commit_current_mask()
        if self._batch_paths:
            jobs = [self._build_job(p) for p in self._batch_paths]
        else:
            jobs = [self._build_job(self._input_path)]
        self._run_jobs(jobs)

    def _build_job(self, path: Path) -> JobRequest:
        s = self.settings_panel.to_settings()
        out = self._output_path_for(path)
        kind = MediaKind.IMAGE if video_io.is_image(path) else MediaKind.VIDEO
        masks = dict(self._manual_masks) if path == self._input_path and kind == MediaKind.VIDEO else {}
        if kind == MediaKind.IMAGE and path == self._input_path:
            m = self.canvas.get_mask()
            if m is not None and m.any():
                masks = {0: m.copy()}
        req = JobRequest(
            input_path=str(path),
            output_path=str(out),
            kind=kind,
            method=DetectionMethod(s.detection_method),
            engine=InpaintEngine(s.inpaint_engine),
            manual_masks=masks,
            detection_prompt=s.detection_prompt,
            detection_skip=s.detection_skip,
            max_bbox_percent=s.max_bbox_percent,
            propagate=s.propagate,
            fade_in=s.fade_in,
            fade_out=s.fade_out,
            lama_super_fast=s.lama_super_fast,
            double_pass=s.double_pass,
            mask_mode=s.mask_mode,
            sttn_neighbor_stride=s.sttn_neighbor_stride,
            sttn_reference_length=s.sttn_reference_length,
            sttn_max_load_num=s.sttn_max_load_num,
            preserve_audio=s.preserve_audio,
            device_prefer=s.device_prefer,
        )
        return req

    def _output_path_for(self, path: Path) -> Path:
        out_dir = Path(self.settings_panel.to_settings().output_dir) if self.settings_panel.to_settings().output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{path.stem}_nowm{path.suffix}"

    def _run_jobs(self, jobs: List[JobRequest]) -> None:
        self.settings_panel.set_processing(True)
        self.progress.setValue(0)
        self._log(f"Starting {len(jobs)} job(s), engine={jobs[0].engine.value}, "
                  f"detection={jobs[0].method.value}")
        self._worker = TaskWorker(jobs)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._log)
        self._worker.job_done.connect(lambda out: self._log(f"Output: {out}"))
        self._worker.finished_all.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        if message:
            self.status_label.setText(message)

    def _on_finished(self) -> None:
        self.settings_panel.set_processing(False)
        self.status_label.setText("Done")
        self._batch_paths = []
        self._save_settings()

    def _on_failed(self, message: str) -> None:
        self.settings_panel.set_processing(False)
        self.status_label.setText("Failed")
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "Processing failed", message.splitlines()[0])

    def on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._log("Cancelling…")

    # --- model download ----------------------------------------------------
    def on_download_model(self, key: str) -> None:
        self._log(f"Downloading model: {key}")
        w = DownloadWorker(key)
        w.progress.connect(lambda v, m: self._on_progress(v, m))
        w.finished_ok.connect(lambda p: self._log(f"Model ready: {p}"))
        w.failed.connect(lambda m: self._log(f"Download failed: {m}"))
        w.start()
        self._download_worker = w  # keep ref

    # --- settings persistence ----------------------------------------------
    def _load_settings(self) -> None:
        self._settings = config.load()
        self.settings_panel.apply_settings(self._settings)

    def _save_settings(self) -> None:
        config.save(self.settings_panel.to_settings())

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        super().closeEvent(event)
