from __future__ import annotations

import numpy as np
import pytest

from sync_workbench.cli.main import build_parser
from sync_workbench.experimental.anchoring_gui.prediction_overlay import (
    MMFI17_NAMES,
    PosePredictionOverlay,
)


def test_prediction_overlay_indexes_by_target_sample(tmp_path):
    path = tmp_path / "predictions.npz"
    poses = np.arange(3 * 17 * 3, dtype=np.float32).reshape(3, 17, 3)
    np.savez_compressed(
        path,
        pred_globally_aligned=poses,
        target_sample_index=np.asarray([10, 12, 12]),
        source_sample_index=np.asarray([7, 8, 9]),
        num_people=np.asarray([1.0, 2.0, 2.0]),
        metric_eligible_single_person=np.asarray([True, False, False]),
        joint_names=np.asarray(MMFI17_NAMES),
    )

    overlay = PosePredictionOverlay(path)

    assert len(overlay) == 2
    assert overlay.total_rows == 3
    assert overlay.duplicate_rows == 1
    assert overlay.get(11) is None
    frame = overlay.get(10)
    assert frame is not None
    assert frame.source_sample_index == 7
    assert frame.metric_eligible_single_person is True
    np.testing.assert_array_equal(frame.pose_sensor_xyz_m, poses[0])


def test_prediction_overlay_rejects_wrong_topology(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez_compressed(
        path,
        pred_globally_aligned=np.zeros((1, 17, 3), dtype=np.float32),
        target_sample_index=np.asarray([1]),
        joint_names=np.asarray([f"joint_{i}" for i in range(17)]),
    )

    with pytest.raises(ValueError, match="native MM-Fi"):
        PosePredictionOverlay(path)


def test_cli_accepts_optional_pose_prediction_overlay():
    args = build_parser().parse_args([
        "anchoring-gui",
        "--sqlite", "workbench.sqlite",
        "--artifact-root", "artifact_store",
        "--rgb-root", "rgb",
        "--subject", "19_MM",
        "--mapping-version", "piecewise_map",
        "--pose-predictions", "predictions.npz",
        "--pose-prediction-array", "pred_globally_aligned",
    ])

    assert args.pose_predictions == "predictions.npz"
    assert args.pose_prediction_array == "pred_globally_aligned"
