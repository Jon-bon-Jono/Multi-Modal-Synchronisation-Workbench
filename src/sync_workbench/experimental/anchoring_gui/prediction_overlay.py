"""Optional MM-Fi 3D pose-prediction overlay for the anchoring GUI.

The inference script writes poses in the radar sensor coordinate frame when the
``pred_globally_aligned`` array is selected.  The GUI indexes those predictions
by radar ``target_sample_index`` and applies its existing sensor-to-world
transform only at render time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MMFI17_NAMES = (
    "pelvis_center",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
    "thorax",
    "neck",
    "face_center",
    "head",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
)

MMFI17_EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (7, 11), (11, 12), (12, 13),
    (7, 14), (14, 15), (15, 16),
)


@dataclass(frozen=True)
class PosePredictionFrame:
    target_sample_index: int
    pose_sensor_xyz_m: np.ndarray
    source_sample_index: int | None = None
    num_people: float | None = None
    metric_eligible_single_person: bool | None = None


class PosePredictionOverlay:
    """Load one inference ``predictions.npz`` file and provide O(1) lookup.

    Duplicate target indices can occur when more than one RGB frame maps to the
    same radar frame.  The model input is keyed only by the radar window, so
    duplicate predictions should be equivalent; the first occurrence is kept.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pose_array: str = "pred_globally_aligned",
    ) -> None:
        self.path = Path(path)
        self.pose_array = str(pose_array)
        if not self.path.is_file():
            raise FileNotFoundError(f"Pose prediction file not found: {self.path}")

        with np.load(self.path, allow_pickle=False) as payload:
            available = set(payload.files)
            required = {self.pose_array, "target_sample_index"}
            missing = sorted(required - available)
            if missing:
                raise ValueError(
                    f"Prediction file {self.path} is missing arrays {missing}; "
                    f"available arrays are {sorted(available)}"
                )

            poses = np.asarray(payload[self.pose_array], dtype=np.float32)
            target_indices = np.asarray(payload["target_sample_index"]).reshape(-1)
            source_indices = (
                np.asarray(payload["source_sample_index"]).reshape(-1)
                if "source_sample_index" in payload
                else None
            )
            num_people = (
                np.asarray(payload["num_people"], dtype=float).reshape(-1)
                if "num_people" in payload
                else None
            )
            eligible = (
                np.asarray(payload["metric_eligible_single_person"], dtype=bool).reshape(-1)
                if "metric_eligible_single_person" in payload
                else None
            )
            joint_names = (
                tuple(str(value) for value in np.asarray(payload["joint_names"]).reshape(-1))
                if "joint_names" in payload
                else None
            )

        if poses.ndim != 3 or poses.shape[1:] != (17, 3):
            raise ValueError(
                f"Expected {self.pose_array!r} with shape [N, 17, 3], got {poses.shape}"
            )
        if len(target_indices) != len(poses):
            raise ValueError(
                "target_sample_index length does not match the prediction array: "
                f"{len(target_indices)} != {len(poses)}"
            )
        for name, values in (
            ("source_sample_index", source_indices),
            ("num_people", num_people),
            ("metric_eligible_single_person", eligible),
        ):
            if values is not None and len(values) != len(poses):
                raise ValueError(f"{name} length does not match predictions")
        if joint_names is not None and joint_names != MMFI17_NAMES:
            raise ValueError(
                "Prediction topology does not match the native MM-Fi 17-joint order. "
                f"Found {joint_names}; expected {MMFI17_NAMES}."
            )

        self.total_rows = int(len(poses))
        self._frames: dict[int, PosePredictionFrame] = {}
        duplicate_rows = 0
        for row_index, raw_target_index in enumerate(target_indices):
            target_index = int(raw_target_index)
            if target_index in self._frames:
                duplicate_rows += 1
                continue

            source_index = None if source_indices is None else int(source_indices[row_index])
            people = None
            if num_people is not None and np.isfinite(num_people[row_index]):
                people = float(num_people[row_index])
            is_eligible = None if eligible is None else bool(eligible[row_index])

            self._frames[target_index] = PosePredictionFrame(
                target_sample_index=target_index,
                pose_sensor_xyz_m=np.asarray(poses[row_index], dtype=np.float32).copy(),
                source_sample_index=source_index,
                num_people=people,
                metric_eligible_single_person=is_eligible,
            )

        self.duplicate_rows = int(duplicate_rows)

    def get(self, target_sample_index: int) -> PosePredictionFrame | None:
        return self._frames.get(int(target_sample_index))

    def __len__(self) -> int:
        return len(self._frames)

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "pose_array": self.pose_array,
            "rows": self.total_rows,
            "unique_target_samples": len(self),
            "duplicate_target_rows_ignored": self.duplicate_rows,
            "coordinate_frame": "radar_sensor_xyz_metres",
            "topology": "MMFI17",
        }
