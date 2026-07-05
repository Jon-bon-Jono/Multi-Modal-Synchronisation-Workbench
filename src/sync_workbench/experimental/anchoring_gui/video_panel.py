"""RGB video display widget."""
from __future__ import annotations

import numpy as np


def _imports():
    from PySide6.QtGui import QImage, QPixmap  # type: ignore
    from PySide6.QtWidgets import QLabel  # type: ignore
    return QLabel, QImage, QPixmap


class VideoPanel:  # wrapper to avoid hard PySide dependency at import time
    def __new__(cls):
        QLabel, QImage, QPixmap = _imports()

        class _VideoPanel(QLabel):
            def __init__(self):
                super().__init__()
                self.setMinimumSize(480, 270)
                self.setScaledContents(True)
                self.setText("RGB frame")

            def set_frame(self, frame_rgb: np.ndarray) -> None:
                arr = np.ascontiguousarray(frame_rgb)
                h, w, c = arr.shape
                if c != 3:
                    raise ValueError("Expected RGB frame with shape (H, W, 3).")
                image = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
                self.setPixmap(QPixmap.fromImage(image.copy()))

        return _VideoPanel()
