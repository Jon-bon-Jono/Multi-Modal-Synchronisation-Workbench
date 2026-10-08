"""Display geometry shared by GUI and dataset exports.

Optional calibrated Kinect/radar extrinsics precede the historical floor transform.
"""
from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

GEOMETRY_PROFILE_ID = "syncwb.gui_world.v1"

SENSOR_HEIGHT_M = 1.76
SENSOR_PITCH_DOWN_DEG = 30.0


@dataclass(frozen=True)
class SpatialCalibration:
    """Immutable snapshot of a desk-v2 calibration; matrices operate in metres."""

    source_name: str
    sha256: str
    document_json: str
    native_kinect_to_radar: tuple
    radar_to_native_kinect: tuple

    def kinect_mm_to_radar_m(self, xyz):
        matrix = np.asarray(self.native_kinect_to_radar)
        return (np.asarray(xyz, dtype=float) * 0.001) @ matrix[:3, :3].T + matrix[:3, 3]

    def radar_m_to_kinect_mm(self, xyz):
        matrix = np.asarray(self.radar_to_native_kinect)
        return (np.asarray(xyz, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]) * 1000

    def provenance(self):
        return {"source_name": self.source_name, "sha256": self.sha256,
                "document": json.loads(self.document_json)}


def load_spatial_calibration(path: str | Path | None) -> SpatialCalibration | None:
    """Validate direction, units, rigid matrices and redundant native-axis forms.

    Audit JSONs and older ambiguous schemas are rejected, never guessed. Read once
    so an edited file cannot silently change an open session or a running export.
    """
    if path is None:
        return None
    path = Path(path)
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
        if not isinstance(document, dict) or document.get("schema") != "radar-kinect-desk-calibration-v2" or document.get("units") != "m":
            raise ValueError("Expected radar-kinect-desk-calibration-v2 with units 'm'")
        axis = np.asarray(document["kinect_native_to_radar_axes_3x3"], dtype=float)
        expected_axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
        if axis.shape != (3, 3) or not np.allclose(axis, expected_axis, rtol=0, atol=1e-8):
            raise ValueError("Unsupported Kinect axis convention")
        matrices = {}
        for name in ("radar_to_kinect_4x4", "kinect_to_radar_4x4",
                     "native_kinect_to_radar_4x4", "radar_to_native_kinect_4x4"):
            matrix = np.asarray(document[name], dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"{name} must be a finite 4x4 matrix")
            rotation = matrix[:3, :3]
            if (not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
                    or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
                    or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6)):
                raise ValueError(f"{name} must be a proper rigid transform")
            matrices[name] = matrix
        forward, inverse = matrices["radar_to_kinect_4x4"], matrices["kinect_to_radar_4x4"]
        basis = np.eye(4)
        basis[:3, :3] = axis
        if (not np.allclose(forward @ inverse, np.eye(4), rtol=0, atol=1e-6)
                or not np.allclose(inverse @ basis, matrices["native_kinect_to_radar_4x4"], rtol=0, atol=1e-6)
                or not np.allclose(basis.T @ forward, matrices["radar_to_native_kinect_4x4"], rtol=0, atol=1e-6)):
            raise ValueError("Calibration directions/native-axis transforms are inconsistent")
        return SpatialCalibration(path.name, hashlib.sha256(raw).hexdigest(),
                                  json.dumps(document, sort_keys=True, allow_nan=False),
                                  tuple(map(tuple, matrices["native_kinect_to_radar_4x4"])),
                                  tuple(map(tuple, matrices["radar_to_native_kinect_4x4"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid spatial calibration {path.name}: {exc}") from exc


def geometry_metadata(calibration: SpatialCalibration | None = None):
    rotation = sensor_to_world_rotation_matrix()
    axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    metadata = dict(profile_id=GEOMETRY_PROFILE_ID, coordinate_frame="approximate_floor_world",
                    units="m", axes=["right", "forward", "up"], handedness="right",
                    sensor_pitch_down_deg=SENSOR_PITCH_DOWN_DEG, sensor_height_m=SENSOR_HEIGHT_M,
                    radar_rotation=rotation.tolist(), translation_m=[0, 0, SENSOR_HEIGHT_M],
                    kinect_mm_to_sensor_m=(axis * 0.001).tolist(),
                    kinect_mm_to_world_linear=(rotation @ axis * 0.001).tolist(),
                    kinect_world_translation_m=[0, 0, SENSOR_HEIGHT_M],
                    convention="column vectors: world = linear @ input + translation",
                    calibration_status="historical GUI assumption; no calibrated Kinect-radar extrinsics",
                    point_filter="none", temporal_accumulation="none")
    if calibration is not None:
        matrix = np.asarray(calibration.native_kinect_to_radar)
        metadata.update(profile_id="syncwb.calibrated_gui_world.v1_" + calibration.sha256,
                        calibration_status="selected desk-v2 extrinsics; floor pitch/height remain GUI assumptions",
                        spatial_calibration=calibration.provenance(),
                        kinect_mm_to_sensor_m=(matrix[:3, :3] * 0.001).tolist(),
                        kinect_to_sensor_translation_m=matrix[:3, 3].tolist(),
                        kinect_mm_to_world_linear=(rotation @ matrix[:3, :3] * 0.001).tolist())
        metadata["kinect_world_translation_m"] = (rotation @ matrix[:3, 3] + np.array(metadata["translation_m"])).tolist()
    return metadata


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


def pose3d_to_world(poses: np.ndarray | None, *, calibration: SpatialCalibration | None = None) -> np.ndarray:
    """Transform Kinect 3D pose payload to the common world frame.

    Optional calibration maps native Kinect millimetres into radar sensor metres.
    Otherwise use the historical axis convention. Both paths then apply the
    existing sensor->world pitch/height transform exactly once.
    """
    pose_sensor = pose3d_to_pc(poses, calibration=calibration)

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


def pose3d_to_pc(poses: np.ndarray | None, *, scale: float = 1e-3,
                 calibration: SpatialCalibration | None = None) -> np.ndarray:
    """Transform Kinect 3D pose coordinates into radar point-cloud coordinates.

    Input is Kinect coordinates in millimetres, shape (people, joints, 3/4).
    Output is radar point-cloud coordinates in metres, shape (people, joints, 3).
    """
    if poses is None:
        return np.empty((0, 0, 3), dtype=float)
    arr = np.asarray(poses, dtype=float)
    if arr.size == 0 or arr.ndim != 3 or arr.shape[-1] < 3:
        return np.empty((0, 0, 3), dtype=float)
    if calibration is not None:
        if scale != 1e-3:
            raise ValueError("Calibrated Kinect poses must use native millimetres")
        return calibration.kinect_mm_to_radar_m(arr[..., :3])
    out = arr[..., [0, 2, 1]].copy()
    out[..., 2] *= -1.0
    out *= float(scale)
    return out[..., :3]


