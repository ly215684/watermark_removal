"""Interactive mask-editing canvas built on QGraphicsView.

Displays an image/video frame and lets the user paint removal masks with three
tools: rectangle, brush and eraser. The mask is kept as a single-channel numpy
array (0/255) matching the displayed frame resolution.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)


class Tool(str, Enum):
    RECT = "rect"
    BRUSH = "brush"
    ERASER = "eraser"


def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _mask_to_overlay(mask: np.ndarray) -> QPixmap:
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # red
    rgba[..., 3] = np.where(mask > 0, 120, 0)
    qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class MaskCanvas(QGraphicsView):
    """A QGraphicsView with mask painting overlays."""

    maskChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self._base_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setZValue(10)
        self.scene.addItem(self._base_item)
        self.scene.addItem(self._overlay_item)

        self._mask: Optional[np.ndarray] = None
        self._tool = Tool.RECT
        self._brush_size = 25
        self._drawing = False
        self._start_pt: Optional[QPointF] = None
        self._last_pt: Optional[QPointF] = None
        self._preview: Optional[QGraphicsRectItem] = None
        self._undo_stack: list[np.ndarray] = []

    # --- public API --------------------------------------------------------
    def set_frame(self, img_bgr: np.ndarray, keep_mask: bool = False) -> None:
        h, w = img_bgr.shape[:2]
        self._base_item.setPixmap(_bgr_to_qpixmap(img_bgr))
        self.scene.setRect(0, 0, w, h)
        if not keep_mask or self._mask is None or self._mask.shape[:2] != (h, w):
            self._mask = np.zeros((h, w), dtype=np.uint8)
            self._undo_stack.clear()
        self._refresh_overlay()
        self.fit_view()

    def clear_mask(self) -> None:
        if self._mask is not None:
            self._push_undo()
            self._mask[:] = 0
            self._refresh_overlay()
            self.maskChanged.emit()

    def get_mask(self) -> Optional[np.ndarray]:
        return self._mask

    def set_mask(self, mask: np.ndarray) -> None:
        if self._mask is None or self._mask.shape[:2] != mask.shape[:2]:
            return
        self._push_undo()
        self._mask = np.maximum(self._mask, mask)
        self._refresh_overlay()
        self.maskChanged.emit()

    def set_tool(self, tool: Tool) -> None:
        self._tool = tool

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, int(size))

    def undo(self) -> None:
        if self._undo_stack:
            self._mask = self._undo_stack.pop()
            self._refresh_overlay()
            self.maskChanged.emit()

    def fit_view(self) -> None:
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # --- internals ---------------------------------------------------------
    def _push_undo(self) -> None:
        if self._mask is not None:
            self._undo_stack.append(self._mask.copy())
            if len(self._undo_stack) > 20:
                self._undo_stack.pop(0)

    def _refresh_overlay(self) -> None:
        if self._mask is None:
            self._overlay_item.setPixmap(QPixmap())
            return
        self._overlay_item.setPixmap(_mask_to_overlay(self._mask))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self.fit_view()

    def _to_img(self, event: QMouseEvent) -> QPointF:
        return self.mapToScene(event.pos())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mask is None or event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        pt = self._to_img(event)
        self._drawing = True
        self._push_undo()
        if self._tool == Tool.RECT:
            self._start_pt = pt
            pen = QPen(QColor(255, 0, 0))
            pen.setStyle(Qt.DashLine)
            self._preview = self.scene.addRect(QRectF(pt, pt), pen)
        else:
            self._last_pt = pt
            self._paint_dot(pt)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._drawing or self._mask is None:
            return super().mouseMoveEvent(event)
        pt = self._to_img(event)
        if self._tool == Tool.RECT and self._preview is not None and self._start_pt is not None:
            self._preview.setRect(QRectF(self._start_pt, pt).normalized())
        else:
            self._paint_segment(self._last_pt, pt)
            self._last_pt = pt
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._drawing or self._mask is None:
            return super().mouseReleaseEvent(event)
        pt = self._to_img(event)
        if self._tool == Tool.RECT and self._start_pt is not None:
            self._commit_rect(self._start_pt, pt)
        if self._preview is not None:
            self.scene.removeItem(self._preview)
            self._preview = None
        self._drawing = False
        self._start_pt = None
        self._last_pt = None
        self.maskChanged.emit()
        super().mouseReleaseEvent(event)

    def _commit_rect(self, a: QPointF, b: QPointF) -> None:
        x0, x1 = sorted((int(a.x()), int(b.x())))
        y0, y1 = sorted((int(a.y()), int(b.y())))
        self._mask[y0:y1, x0:x1] = 255
        self._refresh_overlay()

    def _paint_dot(self, pt: QPointF) -> None:
        self._paint_segment(pt, pt)

    def _paint_segment(self, a: Optional[QPointF], b: Optional[QPointF]) -> None:
        if a is None or b is None:
            return
        value = 255 if self._tool == Tool.BRUSH else 0
        cv2.line(self._mask, (int(a.x()), int(a.y())), (int(b.x()), int(b.y())),
                 int(value), thickness=self._brush_size)
        self._refresh_overlay()
