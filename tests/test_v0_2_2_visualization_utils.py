from __future__ import annotations

import numpy as np

from sync_workbench.experimental.anchoring_gui.visualization_utils import (
    filter_noise_points,
    point_rgba,
    pose3d_to_pc,
    project_pc_to_digital,
)


def test_filter_noise_points_removes_nan_and_gtrack_noise_ids():
    points = np.array(
        [
            [0, 1, 2, 0.1, 12, 0],
            [0, 1, 2, 0.1, 12, 253],
            [0, 1, 2, 0.1, 12, 254],
            [0, 1, 2, 0.1, 12, 255],
            [0, 1, 2, 0.1, 12, np.nan],
            [0, 1, 2, 0.1, 12, 7],
        ],
        dtype=float,
    )
    filtered = filter_noise_points(points)
    assert filtered.shape[0] == 2
    assert filtered[:, 5].tolist() == [0, 7]


def test_pose3d_to_pc_uses_legacy_axis_transform():
    pose = np.array([[[1000.0, 2000.0, 3000.0, 0.9]]])
    pc_pose = pose3d_to_pc(pose)
    assert np.allclose(pc_pose[0, 0], [1.0, 3.0, -2.0])


def test_point_rgba_and_projection_smoke():
    points = np.array([[0.0, 2.0, 1.0, -0.5, 10.0, 1.0], [0.2, 2.2, 1.1, 0.5, 20.0, 2.0]])
    rgba = point_rgba(points, "snr")
    assert rgba.shape == (2, 4)
    projected = project_pc_to_digital(points, filter_noise=False)
    assert projected.ndim == 2
    assert projected.shape[1] == 2
