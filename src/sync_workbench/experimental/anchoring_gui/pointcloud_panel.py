"""3D radar point-cloud display widget."""
from __future__ import annotations

import numpy as np

from sync_workbench.experimental.anchoring_gui.visualization_utils import (
    KINECT_LIMB_INDEX_CONNECTIONS,
    SENSOR_HEIGHT_M,
    finite_pose_limbs,
    point_rgba,
    points_sensor_to_world,
    pose3d_to_world,
    valid_points,
)


def _imports():
    from PySide6.QtCore import Qt  # type: ignore
    import pyqtgraph.opengl as gl  # type: ignore

    return gl, Qt

class PointCloudPanel:
    def __new__(cls):
        gl, Qt = _imports()

        class _PointCloudPanel(gl.GLViewWidget):
            def __init__(self):
                super().__init__()
                self.setMinimumSize(480, 360)

                self.color_mode = "constant"
                self.filter_noise = False
                self.show_pose3d = False
                self.pan_mode = False

                self.scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3)), size=5)
                self.pose_scatter = gl.GLScatterPlotItem(
                    pos=np.empty((0, 3)),
                    size=6,
                    color=(0.1, 0.5, 1.0, 1.0),
                )
                self.pose_lines: list[object] = []

                self._add_world_reference_items()

                self.addItem(self.scatter)
                self.addItem(self.pose_scatter)

                # Reasonable initial view for floor/world coordinates.
                try:
                    self.setCameraPosition(distance=6.0, elevation=18.0, azimuth=-90.0)
                except Exception:
                    pass

            def _add_world_reference_items(self) -> None:
                """Add static world-frame reference items.

                Floor is z=0. The sensor is at approximately (0, 0, SENSOR_HEIGHT_M).
                """
                # Semi-transparent floor plane.
                floor_vertices = np.array(
                    [
                        [-3.0, -1.0, 0.0],
                        [3.0, -1.0, 0.0],
                        [3.0, 7.0, 0.0],
                        [-3.0, 7.0, 0.0],
                    ],
                    dtype=float,
                )
                floor_faces = np.array(
                    [
                        [0, 1, 2],
                        [0, 2, 3],
                    ],
                    dtype=int,
                )

                self.floor = gl.GLMeshItem(
                    vertexes=floor_vertices,
                    faces=floor_faces,
                    smooth=False,
                    drawEdges=False,
                    color=(0.45, 0.45, 0.45, 0.18),
                )
                self.addItem(self.floor)

                # Grid on the floor. It spans approximately x=[-3, 3], y=[-1, 7].
                self.floor_grid = gl.GLGridItem()
                self.floor_grid.setSize(x=6.0, y=8.0, z=0.0)
                self.floor_grid.setSpacing(x=0.5, y=0.5, z=0.0)
                self.floor_grid.translate(0.0, 3.0, 0.0)
                self.addItem(self.floor_grid)

                # World-frame axes from the floor origin.
                self.axis = gl.GLAxisItem()
                self.axis.setSize(x=1.0, y=1.0, z=1.0)
                self.addItem(self.axis)

                # Optional marker for the assumed sensor position.
                sensor_pos = np.array([[0.0, 0.0, SENSOR_HEIGHT_M]], dtype=float)
                self.sensor_marker = gl.GLScatterPlotItem(
                    pos=sensor_pos,
                    size=9,
                    color=(1.0, 1.0, 1.0, 1.0),
                )
                self.addItem(self.sensor_marker)

            def set_pan_mode(self, enabled: bool) -> None:
                """Switch mouse interaction between orbit/rotate mode and pan mode.

                In pan mode, left-drag translates the 3D view instead of rotating it.
                """
                self.pan_mode = bool(enabled)
                try:
                    self.setMouseMode("pan" if self.pan_mode else "orbit")
                except Exception:
                    pass
            
            def keyPressEvent(self, event) -> None:
                step = 1.0

                key = event.key()

                if key == Qt.Key_Left:
                    self.pan(-step, 0.0, 0.0, relative="view-upright")
                    event.accept()
                    return

                if key == Qt.Key_Right:
                    self.pan(step, 0.0, 0.0, relative="view-upright")
                    event.accept()
                    return

                if key == Qt.Key_Up:
                    self.pan(0.0, step, 0.0, relative="view-upright")
                    event.accept()
                    return

                if key == Qt.Key_Down:
                    self.pan(0.0, -step, 0.0, relative="view-upright")
                    event.accept()
                    return

                if key == Qt.Key_PageUp:
                    self.pan(0.0, 0.0, step, relative="global")
                    event.accept()
                    return

                if key == Qt.Key_PageDown:
                    self.pan(0.0, 0.0, -step, relative="global")
                    event.accept()
                    return

                super().keyPressEvent(event)
            
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
                    # Transform xyz into the world frame once, using a vectorised operation.
                    # Doppler/SNR/target-ID columns are preserved for colouring/filtering.
                    pts_world = points_sensor_to_world(pts)
                    pos = pts_world[:, :3].astype(float)
                    colors = point_rgba(pts_world, self.color_mode)

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

                pose_world = pose3d_to_world(pose3d)

                if pose_world.size == 0:
                    self.pose_scatter.setData(pos=np.empty((0, 3), dtype=float))
                    return

                joints = pose_world.reshape(-1, 3)
                finite = np.all(np.isfinite(joints), axis=1)

                self.pose_scatter.setData(
                    pos=joints[finite],
                    color=(0.1, 0.5, 1.0, 1.0),
                    size=7,
                )

                for p0, p1 in finite_pose_limbs(pose_world, KINECT_LIMB_INDEX_CONNECTIONS):
                    item = gl.GLLinePlotItem(
                        pos=np.vstack([p0, p1]),
                        color=(0.1, 0.5, 1.0, 1.0),
                        width=2,
                        antialias=True,
                    )
                    self.addItem(item)
                    self.pose_lines.append(item)

        return _PointCloudPanel()
