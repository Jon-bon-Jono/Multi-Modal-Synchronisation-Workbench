"""RGB video display widget with lightweight experimental overlays."""
from __future__ import annotations

import numpy as np

from sync_workbench.experimental.anchoring_gui.visualization_utils import (
    BODY8_LIMB_CONNECTIONS,
    VIDEO_FRAME_DIMS,
    point_rgba,
)


def _imports():
    from PySide6.QtCore import QPointF  # type: ignore
    from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap  # type: ignore
    from PySide6.QtWidgets import QLabel  # type: ignore

    return QLabel, QImage, QPixmap, QPainter, QPen, QBrush, QColor, QPointF


class VideoPanel:  # wrapper to avoid hard PySide dependency at import time
    def __new__(cls):
        QLabel, QImage, QPixmap, QPainter, QPen, QBrush, QColor, QPointF = _imports()

        class _VideoPanel(QLabel):
            def __init__(self):
                super().__init__()
                self.setMinimumSize(480, 270)
                self.setScaledContents(True)
                self.setText("RGB frame")
                self.show_video = True
                self.show_pose2d = False
                self.show_projected_pc = False
                self.projected_pc_color_mode = "constant"
                self.default_width, self.default_height = VIDEO_FRAME_DIMS
                self._last_shape: tuple[int, int] | None = None

            def set_options(
                self,
                *,
                show_video: bool | None = None,
                show_pose2d: bool | None = None,
                show_projected_pc: bool | None = None,
                projected_pc_color_mode: str | None = None,
            ) -> None:
                if show_video is not None:
                    self.show_video = bool(show_video)
                if show_pose2d is not None:
                    self.show_pose2d = bool(show_pose2d)
                if show_projected_pc is not None:
                    self.show_projected_pc = bool(show_projected_pc)
                if projected_pc_color_mode is not None:
                    self.projected_pc_color_mode = str(projected_pc_color_mode or "constant")

            def set_frame(self, frame_rgb: np.ndarray) -> None:
                self.set_scene(frame_rgb=frame_rgb)

            def set_scene(
                self,
                *,
                frame_rgb: np.ndarray | None = None,
                pose2d: np.ndarray | None = None,
                projected_points: np.ndarray | None = None,
            ) -> None:
                arr = self._base_frame(frame_rgb)
                h, w, c = arr.shape
                if c != 3:
                    raise ValueError("Expected RGB frame with shape (H, W, 3).")
                image = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image.copy())
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.Antialiasing, True)
                if self.show_projected_pc and projected_points is not None:
                    self._draw_projected_points(painter, projected_points, w, h)
                if self.show_pose2d and pose2d is not None:
                    self._draw_pose2d(painter, pose2d, w, h)
                painter.end()
                self.setPixmap(pixmap)

            def _base_frame(self, frame_rgb: np.ndarray | None) -> np.ndarray:
                if self.show_video and frame_rgb is not None:
                    arr = np.ascontiguousarray(frame_rgb)
                    if arr.ndim != 3 or arr.shape[2] != 3:
                        raise ValueError("Expected RGB frame with shape (H, W, 3).")
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    self._last_shape = (int(arr.shape[0]), int(arr.shape[1]))
                    return arr
                if self._last_shape is not None:
                    h, w = self._last_shape
                else:
                    w, h = self.default_width, self.default_height
                return np.zeros((h, w, 3), dtype=np.uint8)

            def _draw_projected_points(self, painter, projected_points: np.ndarray, width: int, height: int) -> None:
                pts = np.asarray(projected_points, dtype=float)

                if pts.size == 0:
                    return

                if pts.ndim != 2 or pts.shape[1] < 2:
                    return

                mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
                mask &= (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
                pts = pts[mask]

                if pts.size == 0:
                    return

                if pts.shape[0] > 8000:
                    step = max(1, int(np.ceil(pts.shape[0] / 8000)))
                    pts = pts[::step]

                rgba = point_rgba(pts, self.projected_pc_color_mode)

                if isinstance(rgba, tuple):
                    rgba_arr = np.tile(np.asarray(rgba, dtype=float), (pts.shape[0], 1))
                else:
                    rgba_arr = np.asarray(rgba, dtype=float)

                radius = 3

                for (x, y), colour in zip(pts[:, :2], rgba_arr):
                    qcolour = QColor(
                        int(np.clip(colour[0], 0.0, 1.0) * 255),
                        int(np.clip(colour[1], 0.0, 1.0) * 255),
                        int(np.clip(colour[2], 0.0, 1.0) * 255),
                        int(np.clip(colour[3], 0.0, 1.0) * 255),
                    )
                    painter.setPen(QPen(qcolour, 1))
                    painter.setBrush(QBrush(qcolour))
                    painter.drawEllipse(QPointF(float(x), float(y)), radius, radius)

            def _draw_pose2d(self, painter, pose2d: np.ndarray, width: int, height: int) -> None:
                pose = np.asarray(pose2d, dtype=float)
                if pose.size == 0 or pose.ndim != 3 or pose.shape[-1] < 2:
                    return
                line_pen = QPen(QColor(80, 180, 255, 230), 3)
                joint_pen = QPen(QColor(255, 255, 255, 240), 1)
                joint_brush = QBrush(QColor(80, 180, 255, 230))
                painter.setPen(line_pen)
                for person in range(pose.shape[0]):
                    for a, b in BODY8_LIMB_CONNECTIONS:
                        if a >= pose.shape[1] or b >= pose.shape[1]:
                            continue
                        p0 = pose[person, a]
                        p1 = pose[person, b]
                        if not self._valid_uv(p0, width, height) or not self._valid_uv(p1, width, height):
                            continue
                        if pose.shape[-1] >= 3 and (p0[2] <= 0.05 or p1[2] <= 0.05):
                            continue
                        painter.drawLine(int(round(p0[0])), int(round(p0[1])), int(round(p1[0])), int(round(p1[1])))
                painter.setPen(joint_pen)
                painter.setBrush(joint_brush)
                for person in range(pose.shape[0]):
                    for joint in range(pose.shape[1]):
                        p = pose[person, joint]
                        if not self._valid_uv(p, width, height):
                            continue
                        if pose.shape[-1] >= 3 and p[2] <= 0.05:
                            continue
                        painter.drawEllipse(int(round(p[0])) - 3, int(round(p[1])) - 3, 6, 6)

            @staticmethod
            def _valid_uv(p: np.ndarray, width: int, height: int) -> bool:
                return bool(np.isfinite(p[0]) and np.isfinite(p[1]) and 0 <= p[0] < width and 0 <= p[1] < height)

        return _VideoPanel()
