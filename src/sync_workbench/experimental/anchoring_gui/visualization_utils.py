"""Small visualisation helpers for the experimental anchoring GUI.

These functions deliberately live under the experimental GUI namespace. They
encode the current UNSW-PANOPTES visualisation assumptions used for the rough
anchoring tool, not a final calibration subsystem.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

NOISE_TARGET_IDS = {253, 254, 255}
VIDEO_FRAME_DIMS = (1280, 720)  # width, height for Kinect RGB LQ frames
DOPPLER_COLOUR_LIMIT_MPS = 3.0
SNR_COLOUR_LIMIT = np.log2(100)
SENSOR_HEIGHT_M = 1.76
SENSOR_PITCH_DOWN_DEG = 30.0

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

KINECT_LIMB_CONNECTIONS = [
    ("pelvis", "spine - navel"),
    ("spine - navel", "spine - chest"),
    ("spine - chest", "neck"),
    ("neck", "head"),
    ("spine - chest", "left clavicle"),
    ("left clavicle", "left shoulder"),
    ("left shoulder", "left elbow"),
    ("left elbow", "left wrist"),
    ("left wrist", "left hand"),
    ("left hand", "left handtip"),
    ("left hand", "left thumb"),
    ("spine - chest", "right clavicle"),
    ("right clavicle", "right shoulder"),
    ("right shoulder", "right elbow"),
    ("right elbow", "right wrist"),
    ("right wrist", "right hand"),
    ("right hand", "right handtip"),
    ("right hand", "right thumb"),
    ("pelvis", "left hip"),
    ("left hip", "left knee"),
    ("left knee", "left ankle"),
    ("left ankle", "left foot"),
    ("pelvis", "right hip"),
    ("right hip", "right knee"),
    ("right knee", "right ankle"),
    ("right ankle", "right foot"),
]
_KINECT_NAME_TO_IDX = {name: idx for idx, name in enumerate(KINECT_JOINT_NAMES)}
KINECT_LIMB_INDEX_CONNECTIONS = [(_KINECT_NAME_TO_IDX[a], _KINECT_NAME_TO_IDX[b]) for a, b in KINECT_LIMB_CONNECTIONS]

BODY8_LIMB_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (2, 1),
    (2, 4),
    (1, 3),
    (4, 6),
    (3, 5),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (19, 11),
    (19, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (15, 24),
    (16, 25),
    (15, 20),
    (15, 22),
    (16, 21),
    (16, 23),
    (19, 18),
    (18, 17),
    (18, 5),
    (18, 6),
]

# Kinect colour calibration constants from the old exploratory visualisation
# code. These are only used for the experimental GUI overlay of radar points in
# the digital RGB frame.
COLOR_CALIBRATION_PARAMS = {
    "cx": 638.5570678710938,
    "cy": 368.1064147949219,
    "fx": 611.3757934570312,
    "fy": 611.5018920898438,
    "k1": 0.49009519815444946,
    "k2": -2.669199228286743,
    "k3": 1.5499054193496704,
    "k4": 0.36899223923683167,
    "k5": -2.490204334259033,
    "k6": 1.47518789768219,
    "codx": 0.0,
    "cody": 0.0,
    "p1": 0.0005917315138503909,
    "p2": -0.0002122735750162974,
    "metric_radius": 1.7000000476837158,
}

TRANSLATION = np.array([-32.0264778137207, -1.8919668197631836, 3.9770002365112305], dtype=float)
ROTATION = np.array(
    [
        [0.9999954104423523, 0.0030102587770670652, -0.00032541988184675574],
        [-0.0029597715474665165, 0.9945205450057983, 0.10449939966201782],
        [0.0006382070132531226, -0.10449796169996262, 0.9945248961448669],
    ],
    dtype=float,
)


def as_points_array(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.empty((0, 6), dtype=float)
    arr = np.asarray(points)
    if arr.size == 0:
        return np.empty((0, 6), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 6), dtype=float)
    return arr


def filter_noise_points(points: np.ndarray | None) -> np.ndarray:
    """Remove GTRACK noise points.

    Points are noisy when target id is NaN, 253, 254, or 255. The target id is
    expected in column 5 of the radar point array.
    """
    arr = as_points_array(points)
    if arr.size == 0 or arr.shape[1] < 6:
        return arr
    try:
        target_ids = arr[:, 5].astype(float)
    except Exception:
        target_ids = np.array([float(x) if str(x).lower() != "nan" else np.nan for x in arr[:, 5]], dtype=float)
    mask = np.isfinite(target_ids)
    for bad in NOISE_TARGET_IDS:
        mask &= target_ids != float(bad)
    return arr[mask]


def valid_points(points: np.ndarray | None, *, filter_noise: bool = False) -> np.ndarray:
    arr = filter_noise_points(points) if filter_noise else as_points_array(points)
    if arr.size == 0:
        return np.empty((0, max(6, arr.shape[1] if arr.ndim == 2 else 6)), dtype=float)
    xyz = arr[:, :3].astype(float)
    mask = np.all(np.isfinite(xyz), axis=1)
    return arr[mask]


def _normalise(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros(vals.shape, dtype=float)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if math.isclose(vmin, vmax):
        return np.zeros(vals.shape, dtype=float) + 0.5
    return np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)


def point_rgba(points: np.ndarray, color_mode: str = "constant") -> np.ndarray | tuple[float, float, float, float]:
    """Return pyqtgraph-compatible RGBA values for point cloud display."""
    arr = as_points_array(points)
    mode = str(color_mode or "constant").lower()
    if arr.size == 0:
        return np.empty((0, 4), dtype=float)
    if mode not in {"snr", "doppler"}:
        return (1.0, 0.2, 0.1, 0.85)
    col = 4 if mode == "snr" else 3
    if arr.shape[1] <= col:
        return (1.0, 0.2, 0.1, 0.85)
    values = arr[:, col].astype(float)
    if mode == "doppler":
        limit = float(DOPPLER_COLOUR_LIMIT_MPS)
        clipped = np.clip(values, -limit, limit)
        norm = (clipped + limit) / (2.0 * limit)
    elif mode == "snr":
        logged = np.log2(np.clip(values, 0, None))
        limit = float(SNR_COLOUR_LIMIT)
        clipped = np.clip(values, 0, limit)
        norm = logged / limit
    else:
        norm = _normalise(values)
    norm = np.nan_to_num(norm, nan=0.5, posinf=1.0, neginf=0.0)
    rgba = np.zeros((arr.shape[0], 4), dtype=float)
    # Simple blue -> green -> red ramp.
    rgba[:, 0] = np.clip(2.0 * norm - 0.2, 0.0, 1.0)
    rgba[:, 1] = np.clip(1.0 - np.abs(norm - 0.5) * 2.0, 0.0, 1.0)
    rgba[:, 2] = np.clip(1.2 - 2.0 * norm, 0.0, 1.0)
    rgba[:, 3] = 0.9
    return rgba


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


def _transform_extrinsics_point(xyz: np.ndarray) -> np.ndarray:
    return ROTATION @ np.asarray(xyz, dtype=float) + TRANSLATION


def _project_kinect_xyz_to_digital(xyz: np.ndarray) -> np.ndarray:
    p = COLOR_CALIBRATION_PARAMS
    xyz = _transform_extrinsics_point(xyz)
    if not np.all(np.isfinite(xyz)) or abs(float(xyz[2])) < 1e-9:
        return np.array([np.nan, np.nan], dtype=float)

    uv = np.array([xyz[0] / xyz[2], xyz[1] / xyz[2]], dtype=float)
    xp = uv[0] - p["codx"]
    yp = uv[1] - p["cody"]
    xp2 = xp * xp
    yp2 = yp * yp
    xyp = xp * yp
    rs = xp2 + yp2
    rss = rs * rs
    rsc = rss * rs
    a = 1.0 + p["k1"] * rs + p["k2"] * rss + p["k3"] * rsc
    b = 1.0 + p["k4"] * rs + p["k5"] * rss + p["k6"] * rsc
    d = a / b if b != 0 else a
    xp_d = xp * d
    yp_d = yp * d
    rs_2xp2 = rs + 2.0 * xp2
    rs_2yp2 = rs + 2.0 * yp2
    xp_d += rs_2xp2 * p["p2"] + 2.0 * xyp * p["p1"]
    yp_d += rs_2yp2 * p["p1"] + 2.0 * xyp * p["p2"]
    uv[0] = (xp_d + p["codx"]) * p["fx"] + p["cx"]
    uv[1] = (yp_d + p["cody"]) * p["fy"] + p["cy"]
    return uv


def project_pc_to_digital(points: np.ndarray | None, *, filter_noise: bool = False, max_points: int | None = 8000) -> np.ndarray:
    """Project radar point-cloud points to Kinect RGB digital image coordinates.

    This follows the old exploratory code path: radar PC metres -> millimetres,
    then approximate Kinect coordinates [x, z, y], then Kinect colour digital
    projection.
    """
    arr = valid_points(points, filter_noise=filter_noise)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if max_points is not None and arr.shape[0] > int(max_points):
        step = max(1, int(math.ceil(arr.shape[0] / int(max_points))))
        arr = arr[::step]
    pts3d = arr[:, :3].astype(float) * 1e3
    cam_coords = np.column_stack([pts3d[:, 0], -pts3d[:, 2], pts3d[:, 1]])
    out = np.empty((cam_coords.shape[0], 2), dtype=float)
    for idx, xyz in enumerate(cam_coords):
        out[idx] = _project_kinect_xyz_to_digital(xyz)

    mask = np.all(np.isfinite(out), axis=1)

    # Preserve the radar attributes after projection so the 2D overlay can use
    # the same colouring convention as the 3D point cloud:
    #
    # output columns:
    #   0 = projected u pixel
    #   1 = projected v pixel
    #   2 = original radar z, kept only as a dummy third coordinate
    #   3 = Doppler velocity
    #   4 = SNR
    #   5 = GTRACK target ID
    #
    # This keeps Doppler/SNR/target-ID in the same column positions expected by
    # point_rgba(...).
    projected = np.full((arr.shape[0], max(6, arr.shape[1])), np.nan, dtype=float)
    projected[:, 0:2] = out
    projected[:, 2:] = arr[:, 2:]

    return projected[mask]


def finite_pose_limbs(pose_xyz: np.ndarray, connections: Iterable[tuple[int, int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    arr = np.asarray(pose_xyz, dtype=float)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    if arr.size == 0 or arr.ndim != 3:
        return lines
    for person in range(arr.shape[0]):
        for a, b in connections:
            if a >= arr.shape[1] or b >= arr.shape[1]:
                continue
            p0 = arr[person, a, :3]
            p1 = arr[person, b, :3]
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                lines.append((p0, p1))
    return lines
