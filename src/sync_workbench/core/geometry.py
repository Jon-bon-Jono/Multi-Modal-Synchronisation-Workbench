"""Versioned approximate display geometry shared by GUI and dataset exports.

This is the historical GUI alignment, not a calibrated sensor extrinsic model.
"""
from __future__ import annotations

import math
import numpy as np

GEOMETRY_PROFILE_ID = "syncwb.gui_world.v1"

SENSOR_HEIGHT_M = 1.76
SENSOR_PITCH_DOWN_DEG = 30.0


def sensor_to_world_rotation_matrix(*, pitch_down_deg: float = SENSOR_PITCH_DOWN_DEG) -> np.ndarray:
    """Rotation from current 3D sensor frame to world frame.

    Assumes sensor-frame axes:
        x = right
        y = forward/range
        z = up relative to sensor

    A positive pitch_down_deg means the sensor forward axis points downward
    relative to the horizontal floor plane.
    """
    theta = math.radians(float(pitch_down_deg))
    c = math.cos(theta)
    s = math.sin(theta)

    # Equivalent to R_x(-theta). Sensor +y maps to world +y and -z.
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=float,
    )


def sensor_xyz_to_world_xyz(
    xyz: np.ndarray,
    *,
    height_m: float = SENSOR_HEIGHT_M,
    pitch_down_deg: float = SENSOR_PITCH_DOWN_DEG,
) -> np.ndarray:
    """Vectorised sensor-frame xyz -> world-frame xyz transform."""
    arr = np.asarray(xyz, dtype=float)

    if arr.size == 0:
        return arr.reshape((-1, 3)) if arr.ndim == 2 else arr.copy()

    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be xyz with size 3, got shape {arr.shape}.")

    rot = sensor_to_world_rotation_matrix(pitch_down_deg=pitch_down_deg)
    out = arr @ rot.T
    out[..., 2] += float(height_m)
    return out


def points_sensor_to_world(points: np.ndarray | None) -> np.ndarray:
    """Return point-cloud array with xyz columns transformed to world frame.

    Non-spatial columns such as Doppler, SNR and target ID are preserved.
    """
    arr = as_points_array(points)

    if arr.size == 0:
        return arr

    out = arr.astype(float, copy=True)
    out[:, :3] = sensor_xyz_to_world_xyz(out[:, :3])
    return out


def pose3d_to_world(poses: np.ndarray | None) -> np.ndarray:
    """Transform Kinect 3D pose payload to the common world frame.

    This keeps the existing Kinect->point-cloud axis convention from
    pose3d_to_pc(...), then applies the sensor->world pitch/height transform.
    """
    pose_sensor = pose3d_to_pc(poses)

    if pose_sensor.size == 0:
        return pose_sensor

    return sensor_xyz_to_world_xyz(pose_sensor)

KINECT_JOINT_NAMES = [
    "pelvis",
    "spine - navel",
    "spine - chest",
    "neck",
    "left clavicle",
    "left shoulder",
    "left elbow",
    "left wrist",
    "left hand",
    "left handtip",
    "left thumb",
    "right clavicle",
    "right shoulder",
    "right elbow",
    "right wrist",
    "right hand",
    "right handtip",
    "right thumb",
    "left hip",
    "left knee",
    "left ankle",
    "left foot",
    "right hip",
    "right knee",
    "right ankle",
    "right foot",
    "head",
    "nose",
    "left eye",
    "left ear",
    "right eye",
    "right ear",
]

def as_points_array(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.empty((0, 6), dtype=float)
    arr = np.asarray(points)
    if arr.size == 0:
        return np.empty((0, 6), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 6), dtype=float)
    return arr


def pose3d_to_pc(poses: np.ndarray | None, *, scale: float = 1e-3) -> np.ndarray:
    """Transform Kinect 3D pose coordinates into radar point-cloud coordinates.

    Input is Kinect coordinates in millimetres, shape (people, joints, 3/4).
    Output is radar point-cloud coordinates in metres, shape (people, joints, 3).
    """
    if poses is None:
        return np.empty((0, 0, 3), dtype=float)
    arr = np.asarray(poses, dtype=float)
    if arr.size == 0 or arr.ndim != 3 or arr.shape[-1] < 3:
        return np.empty((0, 0, 3), dtype=float)
    out = arr[..., [0, 2, 1]].copy()
    out[..., 2] *= -1.0
    out *= float(scale)
    return out[..., :3]


