"""3D radar point-cloud display widget."""
from __future__ import annotations

import numpy as np

from sync_workbench.experimental.anchoring_gui.visualization_utils import (
    KINECT_LIMB_INDEX_CONNECTIONS,
    finite_pose_limbs,
    point_rgba,
    pose3d_to_pc,
    valid_points,
)


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
                self.color_mode = "constant"
                self.filter_noise = False
                self.show_pose3d = False
                self.scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3)), size=5)
                self.pose_scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3)), size=6, color=(0.1, 0.5, 1.0, 1.0))
                self.pose_lines: list[object] = []
                self.addItem(self.scatter)
                self.addItem(self.pose_scatter)

            def set_options(
                self,
                *,
                color_mode: str | None = None,
                filter_noise: bool | None = None,
                show_pose3d: bool | None = None,
            ) -> None:
                if color_mode is not None:
                    self.color_mode = str(color_mode)
                if filter_noise is not None:
                    self.filter_noise = bool(filter_noise)
                if show_pose3d is not None:
                    self.show_pose3d = bool(show_pose3d)

            def set_points(self, points: np.ndarray) -> None:
                self.set_scene(points, pose3d=None)

            def set_scene(self, points: np.ndarray | None, *, pose3d: np.ndarray | None = None) -> None:
                pts = valid_points(points, filter_noise=self.filter_noise)
                if pts.size == 0:
                    pos = np.empty((0, 3), dtype=float)
                    colors = np.empty((0, 4), dtype=float)
                else:
                    pos = pts[:, :3].astype(float)
                    colors = point_rgba(pts, self.color_mode)
                self.scatter.setData(pos=pos, color=colors, size=5)
                self._set_pose3d(pose3d if self.show_pose3d else None)

            def _clear_pose_lines(self) -> None:
                for item in self.pose_lines:
                    try:
                        self.removeItem(item)
                    except Exception:
                        pass
                self.pose_lines.clear()

            def _set_pose3d(self, pose3d: np.ndarray | None) -> None:
                self._clear_pose_lines()
                pose_pc = pose3d_to_pc(pose3d)
                if pose_pc.size == 0:
                    self.pose_scatter.setData(pos=np.empty((0, 3), dtype=float))
                    return
                joints = pose_pc.reshape(-1, 3)
                finite = np.all(np.isfinite(joints), axis=1)
                self.pose_scatter.setData(pos=joints[finite], color=(0.1, 0.5, 1.0, 1.0), size=7)
                for p0, p1 in finite_pose_limbs(pose_pc, KINECT_LIMB_INDEX_CONNECTIONS):
                    item = gl.GLLinePlotItem(pos=np.vstack([p0, p1]), color=(0.1, 0.5, 1.0, 1.0), width=2, antialias=True)
                    self.addItem(item)
                    self.pose_lines.append(item)

        return _PointCloudPanel()
