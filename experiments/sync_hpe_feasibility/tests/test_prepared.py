"""Integration checks on the prepared pilot, skipped on fresh checkouts."""
import json

import numpy as np
import pytest

from hpe_sync.common import load_config, safe_output


@pytest.fixture(scope="module")
def prepared():
    cfg = load_config()
    folder = safe_output(cfg["output"])
    if not (folder / "prepared.npz").exists():
        pytest.skip("Prepare 19_MM to run data integration checks")
    wanted = ["label_rows", "map_valid", "centers", "split", "block", "pose_time",
              "radar_time", "radar_sample_index", "pose_sample_index", "expected_people"]
    wanted += [f"{name}_{task}" for name in ("mask", "common_mask", "cohort")
               for task in ("image", "shape")]
    with np.load(folder / "prepared.npz") as a:
        data = {key: a[key] for key in wanted}
    manifest = json.loads((folder / "manifest.json").read_text())
    return data, cfg, manifest


@pytest.mark.parametrize("task", ["image", "shape"])
def test_same_people_and_target_components_across_conditions(prepared, task):
    d, cfg, _ = prepared
    selected = np.flatnonzero(d[f"cohort_{task}"])
    assert d["map_valid"][:, selected].all()
    label_rows = d["label_rows"][:, selected]
    people = d["expected_people"][label_rows]
    assert np.all(people == people[0])
    assert np.all(people > 0)
    intersection = np.logical_and.reduce(d[f"mask_{task}"][label_rows])
    np.testing.assert_array_equal(intersection, d[f"common_mask_{task}"][selected])
    assert (intersection.sum(axis=1) >= 2 * cfg["minimum_valid_joints"]).all()


@pytest.mark.parametrize("task", ["image", "shape"])
def test_no_shared_input_or_label_frames_between_splits(prepared, task):
    d, cfg, _ = prepared
    inputs, labels = [], []
    for split in range(3):
        selected = np.flatnonzero(d[f"cohort_{task}"] & (d["split"] == split))
        windows = d["centers"][selected, None] + np.arange(1 - cfg["max_context_frames"], 1)
        inputs.append(set(windows.ravel()))
        labels.append(set(d["label_rows"][:, selected].ravel()))
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert not inputs[a] & inputs[b]
        assert not labels[a] & labels[b]


def test_timestamps_preserve_anchor_units_and_frame_identity(prepared):
    d, cfg, manifest = prepared
    radar_lookup = dict(zip(d["radar_sample_index"], d["radar_time"]))
    pose_lookup = dict(zip(d["pose_sample_index"], d["pose_time"]))
    for anchor in manifest["anchors"]:
        assert abs(pose_lookup[anchor["source_sample_index"]] - anchor["source_time"]) < 2e-6
        assert abs(radar_lookup[anchor["target_sample_index"]] - anchor["target_time"]) < 2e-6
    assert .04 < np.median(np.diff(d["radar_time"])) < .06
    assert .06 < np.median(np.diff(d["pose_time"])) < .08
