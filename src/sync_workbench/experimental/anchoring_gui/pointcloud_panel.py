"""3D radar point-cloud display widget."""
from __future__ import annotations

import numpy as np


def _imports():
    import pyqtgraph.opengl as gl  # type: ignore
    return gl


class PointCloudPanel:
    def __new__(cls):
        gl = _imports()

        class _PointCloudPanel(gl.GLViewWidget):
            def __init__(self):
                super().__init__()
                self.setMinimumSize(480, 360)
                self.scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3)), size=5)
                self.addItem(self.scatter)

            def set_points(self, points: np.ndarray) -> None:
                pts = np.asarray(points)
                if pts.size == 0:
                    pos = np.empty((0, 3))
                else:
                    pos = pts[:, :3].astype(float)
                self.scatter.setData(pos=pos, size=5)

        return _PointCloudPanel()
