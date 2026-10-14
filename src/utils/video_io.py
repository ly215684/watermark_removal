"""Video read/write helpers.

* Reading uses OpenCV (robust, no external dependency).
* Writing prefers FFmpeg (to preserve the source audio track) and falls back to
  OpenCV's ``VideoWriter`` when FFmpeg is not on PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("video_io")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMeta(str(path), width, height, float(fps), count)
    finally:
        cap.release()


def read_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield frames as BGR uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_frame_at(path: str | Path, index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # Handle unicode paths on Windows via numpy buffer decode.
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def save_image(path: str | Path, img: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", img)
    if ok:
        buf.tofile(str(p))
    else:  # pragma: no cover - fallback
        cv2.imwrite(str(p), img)


class FrameWriter:
    """Buffered frame writer.

    Uses a raw-pipe into FFmpeg when available (fast, high quality, allows audio
    mux afterwards); otherwise falls back to OpenCV ``VideoWriter``.
    """

    def __init__(
        self,
        out_path: str | Path,
        width: int,
        height: int,
        fps: float,
        source_video: Optional[str | Path] = None,
        preserve_audio: bool = True,
    ) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.source_video = str(source_video) if source_video else None
        self._use_ffmpeg = preserve_audio and has_ffmpeg() and self.source_video is not None
        # When we must mux audio from source, write to a temp silent video first.
        self._temp_no_audio = self.out_path.with_suffix(".__noaudio__.mp4")
        self._proc: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self._open()

    def _target_for_writer(self) -> Path:
        return self._temp_no_audio if self._use_ffmpeg else self.out_path

    def _open(self) -> None:
        target = self._target_for_writer()
        if has_ffmpeg():
            cmd = [
                "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p", str(target),
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return
            except Exception as exc:  # noqa: BLE001
                _logger.warning("FFmpeg pipe unavailable (%s); using OpenCV writer.", exc)
        self._cv_writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
            (self.width, self.height),
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        elif self._cv_writer is not None:
            self._cv_writer.write(frame)

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
        if self._use_ffmpeg and self.source_video:
            self._mux_audio()

    def _mux_audio(self) -> None:
        """Copy the source audio track onto the newly written (silent) video."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self._temp_no_audio),
            "-i", str(self.source_video),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(self.out_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode == 0:
            self._temp_no_audio.unlink(missing_ok=True)
        else:  # pragma: no cover - depends on environment
            _logger.warning("Audio mux failed; keeping silent video.")
            shutil.move(str(self._temp_no_audio), str(self.out_path))


def ffprobe_streams(path: str | Path) -> dict:
    """Return JSON stream metadata (used to detect an existing audio track)."""
    if not has_ffmpeg():
        return {}
    cmd = ["ffprobe", "-v", "quiet", "-print_json", "-show_streams", str(path)]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        return json.loads(out.stdout.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def has_audio_track(path: str | Path) -> bool:
    meta = ffprobe_streams(path)
    return any(s.get("codec_type") == "audio" for s in meta.get("streams", []))


def collect_media(directory: str | Path) -> List[Path]:
    """List image/video files directly inside *directory*."""
    directory = Path(directory)
    files: List[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and (is_image(p) or is_video(p)):
            files.append(p)
    return files
