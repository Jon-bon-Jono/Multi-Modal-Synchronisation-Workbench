import json
from pathlib import Path
import sqlite3

import h5py
import numpy as np
import pandas as pd
import pytest

from sync_workbench.core.geometry import points_sensor_to_world, pose3d_to_world
from sync_workbench.ingestion.raw_point_cloud_package import file_hash, json_hash
from sync_workbench.services.training_export_service import export_training_data
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.storage.ragged_npz import RaggedNpzWriter


def make_source(tmp_path, *, restart=False, partial=True):
    db, root = tmp_path / "source.sqlite", tmp_path / "artifacts"
    root.mkdir()
    store = SQLiteCoreStore(db)
    store.initialise_empty()
    rows = {name: [] for name in ("SUBJECT", "DEVICE_RUN", "RUN_SAMPLE", "RUN_TIMELINE_MODEL",
            "SAMPLE_TIME_ESTIMATE", "SAMPLE_SUMMARY", "SAMPLE_ARTIFACT", "POINT_CLOUD_VERSION",
            "MAPPING_VERSION", "SYNC_MODEL")}
    rows["SUBJECT"].append(dict(subject_id="S"))
    rk = dict(subject_id="S", run_id="R", device_type="radar_raw")
    binding = dict(hdf5_sha256="a" * 64, **rk, source_frame_number_base=0)
    vid = "raw_" + json_hash(binding)
    points = np.array([[1, 2, 3, 4, 5, 255]], dtype="float32")
    raw = {i: (np.empty((0, 6), np.float32) if i == 0 else points) for i in range(9) if i != 1 or not partial}
    RaggedNpzWriter.write(root / "raw.npz", raw, tail_shape=(6,), dtype="float32")
    manifest = {**binding, "tracking_enabled": True, "point_columns": ["x", "y", "z", "radial_velocity", "snr", "association_id"],
        "point_dtype": "float32", "processing_cfg": "cfg", "generator_metadata": {
            "tracking_enabled": True, "source_config_sha256": "b" * 64, "native_executable_sha256": "c" * 64,
            "native_version": "test", "calibration_sources": [], "tracking_state_start_sequence_index": 0,
            "point_coordinate_frame": "sensor", "point_units": ["m", "m", "m", "m/s", "linear_power_ratio"]}}
    rows["POINT_CLOUD_VERSION"].append({**rk, "point_cloud_version_id": vid, "readable_label": "selected cloud",
        "artifact_ref": "raw.npz", "artifact_sha256": file_hash(root / "raw.npz"), "provenance_json": json.dumps({"version": manifest})})

    def run(key, times, timeline):
        rows["DEVICE_RUN"].append({**key, "nominal_fps": 1})
        rows["RUN_TIMELINE_MODEL"].append({**key, "timeline_model_id": timeline, "timeline_model_type": "test"})
        for i, t in enumerate(times):
            rows["RUN_SAMPLE"].append({**key, "sample_index": i, "sample_kind": "frame", "notes": f"source_frame_number={i+1}"})
            rows["SAMPLE_TIME_ESTIMATE"].append({**key, "sample_index": i, "timeline_model_id": timeline,
                "time_value_sec": t, "time_kind": "numeric"})

    run(rk, list(range(9)), "raw_time")
    for i in range(9):
        rows["SAMPLE_SUMMARY"].append({**rk, "sample_index": i, "point_cloud_version_id": vid,
            "point_status": "available" if i in raw else "unprocessed", "point_count": len(raw[i]) if i in raw else None})
        if i in raw:
            rows["SAMPLE_ARTIFACT"].append({**rk, "sample_index": i, "point_cloud_version_id": vid,
                "artifact_role": "radar_points", "artifact_ref": "raw.npz", "storage_key": "artifact_store", "artifact_format": "ragged_npz"})
    selections = []
    sources = [("K1", [0, 1, 2, 3])] if restart else [("K1", list(range(9)))]
    if restart:
        sources.append(("K2", [5, 6, 7, 8]))
    for name, times in sources:
        pk = dict(subject_id="S", run_id=name, device_type="kinect_rgb")
        run(pk, times, "pose_time")
        poses = {i: np.full((2 if i == 2 else 1, 32, 4), [1000, 2000, 3000, 2], dtype="float32") for i in range(len(times))}
        RaggedNpzWriter.write(root / f"{name}.npz", poses)
        for i, pose in poses.items():
            rows["SAMPLE_SUMMARY"].append({**pk, "sample_index": i, "point_cloud_version_id": "", "num_people": len(pose), "num_3d": len(pose)})
            rows["SAMPLE_ARTIFACT"].append({**pk, "sample_index": i, "point_cloud_version_id": "", "artifact_role": "pose3d",
                "artifact_ref": f"{name}.npz", "storage_key": "artifact_store", "artifact_format": "ragged_npz"})
        identity = dict(subject_id="S", source_run_id=name, source_device_type="kinect_rgb", target_run_id="R", target_device_type="radar_raw")
        params = dict(source_timeline_model_id="pose_time", target_timeline_model_id="raw_time",
                      weak_support_threshold_ms=75, max_allowed_delta_ms=200, primary_policy="supported-only", allow_numeric_identity=True)
        rows["MAPPING_VERSION"].append({**identity, "mapping_version_id": name, "source_sync_model_id": name,
            "mapping_method": "initial_nearest_for_anchoring", "parameters_json": json.dumps(params)})
        rows["SYNC_MODEL"].append({**identity, "sync_model_id": name, "model_type": "identity_time", "extrapolation_policy": "disallow",
            "source_timeline_model_id": "pose_time", "target_timeline_model_id": "raw_time", "parameters_json": "{}"})
        selections.append(dict(subject_id="S", mapping_version_id=name, point_cloud_version_id=vid))
    for name, values in rows.items():
        store.write_table(name, pd.DataFrame(values))
    return dict(sqlite_path=db, artifact_root=root, output=tmp_path / "export", selections=selections)


def test_restart_separate_modalities_geometry_and_source_immutability(tmp_path):
    args = make_source(tmp_path, restart=True)
    before = {p: file_hash(p) for p in [args["sqlite_path"], *args["artifact_root"].glob("*")]}
    result = export_training_data(**args)
    assert result["payload_file_count"] == 4
    out = args["output"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["segments"]) == 2
    all_radar = []
    for s in manifest["segments"]:
        with h5py.File(out / s["files"]["radar"]["path"], "r") as r, h5py.File(out / s["files"]["pose"]["path"], "r") as p:
            assert "poses" not in r and "points" not in p
            assert r.attrs["aligned_origin"] == p.attrs["aligned_origin"]
            all_radar.extend(r["frames/sample_index"][:])
            np.testing.assert_allclose(r["points/values"][0], points_sensor_to_world(np.array([[1, 2, 3, 4, 5, 255]]))[0])
            np.testing.assert_allclose(p["poses/xyz"][0], pose3d_to_world(np.full((1, 32, 4), [1000, 2000, 3000, 2]))[0])
            assert p["frames/num_people"][2] == 2
            assert p["frames/pose_person_count"][2] == 2
            np.testing.assert_array_equal(p["poses/confidence"][:], 2)
            assert p["poses/finite_xyz"][:].all()
            assert r["correspondence/matched"][:].all()
            np.testing.assert_array_equal(r["frames/aligned_time_s"][:], p["frames/aligned_time_s"][:])
            if r["frames/sample_index"][0] == 0:
                assert r["frames/point_count"][0] == 0
                assert r["frames/payload_available"][0]
                assert r["frames/point_count"][1] == -1
                assert not r["frames/payload_available"][1]
    assert all_radar == [0, 1, 2, 3, 5, 6, 7, 8]
    assert all(file_hash(p) == digest for p, digest in before.items())
    assert manifest["geometry"]["profile_id"] == "syncwb.gui_world.v1"


def test_dry_run_and_label_resolution(tmp_path):
    args = make_source(tmp_path)
    args.pop("selections")
    result = export_training_data(**args, subjects=["S"], mapping_version_id="K1", point_cloud_label="selected cloud", dry_run=True)
    assert result["payload_file_count"] == 2
    assert not args["output"].exists()


def test_piecewise_predicted_times_and_no_extrapolation(tmp_path):
    args = make_source(tmp_path)
    with sqlite3.connect(args["sqlite_path"]) as c:
        c.execute("UPDATE SYNC_MODEL SET model_type='piecewise_affine', parameters_json=?", (json.dumps({"segments": [
            dict(source_left=1, source_right=5, slope=1, intercept=2)]}),))
        c.execute("UPDATE MAPPING_VERSION SET mapping_method='nearest_predicted_time'")
    export_training_data(**args)
    with h5py.File(next(args["output"].glob("*.radar.h5")), "r") as r, h5py.File(next(args["output"].glob("*.pose.h5")), "r") as p:
        np.testing.assert_array_equal(r["frames/sample_index"][:], [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(p["frames/sample_index"][:], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(p["frames/aligned_time_s"][:], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(p["frames/timeline_time_s"][:], [1, 2, 3, 4, 5])


def test_nearest_tolerance_and_tie_break():
    from sync_workbench.services.training_export_service import _nearest
    indices, residuals = _nearest(np.array([0.5, 1.9, 4.0]), np.array([0.0, 1.0, 2.0]), 600)
    np.testing.assert_array_equal(indices, [0, 2, -1])
    np.testing.assert_allclose(residuals, [-500, 100, -2000])


@pytest.mark.parametrize("change,match", [
    ("method", "Mixed mapping"), ("recipe", "Mixed processing"),
    ("duplicate", "Duplicate run pair"), ("version", "Multiple point-cloud"),
    ("missing_pair", "Missing mapping selection"), ("online", "offline-raw"),
    ("overlap", "Ambiguous overlapping"),
])
def test_reject_inconsistent_selection(tmp_path, change, match):
    args = make_source(tmp_path, restart=True)
    with sqlite3.connect(args["sqlite_path"]) as c:
        if change == "method":
            c.execute("UPDATE MAPPING_VERSION SET mapping_method='different' WHERE mapping_version_id='K2'")
        elif change == "duplicate":
            args["selections"].append(args["selections"][0].copy())
        elif change == "missing_pair":
            args["selections"].pop()
        elif change == "online":
            c.execute("UPDATE MAPPING_VERSION SET target_device_type='radar_pc'")
        elif change == "overlap":
            c.execute("UPDATE SAMPLE_TIME_ESTIMATE SET time_value_sec=time_value_sec-5 WHERE run_id='K2'")
        elif change == "version":
            args["selections"][1]["point_cloud_version_id"] = "raw_" + "d" * 64
        elif change == "recipe":
            # Second acquisition and run-specific result, but different processing cfg.
            c.row_factory = sqlite3.Row
            cloud = dict(c.execute("SELECT * FROM POINT_CLOUD_VERSION").fetchone())
            m = json.loads(cloud["provenance_json"])["version"]
            m["run_id"] = "R2"
            m["processing_cfg"] = "different config"
            binding = {k:m[k] for k in ("hdf5_sha256", "subject_id", "run_id", "device_type", "source_frame_number_base")}
            cloud.update(run_id="R2", point_cloud_version_id="raw_" + json_hash(binding), provenance_json=json.dumps({"version":m}))
            c.execute('INSERT INTO POINT_CLOUD_VERSION ('+','.join(cloud)+') VALUES ('+','.join('?' for _ in cloud)+')', tuple(cloud.values()))
            for table in ("DEVICE_RUN", "RUN_SAMPLE", "RUN_TIMELINE_MODEL", "SAMPLE_TIME_ESTIMATE"):
                for row in list(c.execute(f"SELECT * FROM {table} WHERE run_id='R'")):
                    d=dict(row);d["run_id"]="R2"
                    c.execute(f'INSERT INTO {table} ('+','.join(d)+') VALUES ('+','.join('?' for _ in d)+')',tuple(d.values()))
            c.execute("UPDATE MAPPING_VERSION SET target_run_id='R2' WHERE mapping_version_id='K2'")
            c.execute("UPDATE SYNC_MODEL SET target_run_id='R2' WHERE sync_model_id='K2'")
            args["selections"][1]["point_cloud_version_id"] = cloud["point_cloud_version_id"]
    with pytest.raises(ValueError, match=match):
        export_training_data(**args)
    assert not args["output"].exists()


def test_protected_roots_existing_output_and_corrupt_bundle(tmp_path):
    args = make_source(tmp_path)
    for output in (args["artifact_root"] / "bad", tmp_path / "recordings" / "bad"):
        with pytest.raises(ValueError, match="read-only"):
            export_training_data(**{**args, "output": output}, read_only_roots=[tmp_path / "recordings"])
    args["output"].mkdir()
    with pytest.raises(FileExistsError):
        export_training_data(**args)
    args["output"].rmdir()
    with sqlite3.connect(args["sqlite_path"]) as c:
        c.execute("UPDATE POINT_CLOUD_VERSION SET artifact_sha256=?", ("0" * 64,))
    with pytest.raises(ValueError, match="checksum"):
        export_training_data(**args)
    assert not args["output"].exists()


def test_timestamp_gaps_create_blocks_not_extra_files(tmp_path):
    args = make_source(tmp_path)
    with sqlite3.connect(args["sqlite_path"]) as c:
        c.execute("UPDATE SAMPLE_TIME_ESTIMATE SET time_value_sec=time_value_sec+4 WHERE run_id='K1' AND sample_index>=3")
    result = export_training_data(**args)
    assert result["payload_file_count"] == 2
    with h5py.File(next(args["output"].glob("*.radar.h5")), "r") as r, h5py.File(next(args["output"].glob("*.pose.h5")), "r") as p:
        np.testing.assert_array_equal(r["frames/sample_index"][:], list(range(9)))
        assert len(set(r["frames/sequence_block"][:])) == 2
        assert len(set(p["frames/sequence_block"][:])) == 2
        assert not r["correspondence/matched"][4]


def test_partial_write_failure_never_publishes(tmp_path, monkeypatch):
    import sync_workbench.services.training_export_service as exporter
    args = make_source(tmp_path, restart=True)
    write = exporter._write_segment
    calls = []

    def fail_second(*a, **kw):
        if calls:
            raise RuntimeError("simulated write failure")
        calls.append(True)
        return write(*a, **kw)

    monkeypatch.setattr(exporter, "_write_segment", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        exporter.export_training_data(**args)
    assert not args["output"].exists()
    assert not list(tmp_path.glob(".syncwb-training-*"))


def test_missing_pose_keeps_people_metadata(tmp_path):
    args = make_source(tmp_path)
    with sqlite3.connect(args["sqlite_path"]) as c:
        c.execute("DELETE FROM SAMPLE_ARTIFACT WHERE run_id='K1' AND sample_index=2")
    export_training_data(**args)
    with h5py.File(next(args["output"].glob("*.pose.h5")), "r") as p:
        assert p["frames/num_people"][2] == 2
        assert p["frames/pose_person_count"][2] == 0
        assert not p["frames/payload_available"][2]


def test_homogeneous_multi_subject_results_have_distinct_ids(tmp_path):
    args = make_source(tmp_path)
    with sqlite3.connect(args["sqlite_path"]) as c:
        c.row_factory = sqlite3.Row
        cloud = dict(c.execute("SELECT * FROM POINT_CLOUD_VERSION").fetchone())
        m = json.loads(cloud["provenance_json"])["version"]
        m["subject_id"] = "T"
        vid = "raw_" + json_hash({k:m[k] for k in ("hdf5_sha256", "subject_id", "run_id", "device_type", "source_frame_number_base")})
        for table in ("SUBJECT", "DEVICE_RUN", "RUN_SAMPLE", "RUN_TIMELINE_MODEL", "SAMPLE_TIME_ESTIMATE",
                      "SAMPLE_SUMMARY", "SAMPLE_ARTIFACT", "POINT_CLOUD_VERSION", "MAPPING_VERSION", "SYNC_MODEL"):
            for row in list(c.execute(f"SELECT * FROM {table} WHERE subject_id='S'")):
                d = dict(row); d["subject_id"] = "T"
                if d.get("point_cloud_version_id"):
                    d["point_cloud_version_id"] = vid
                if table == "POINT_CLOUD_VERSION":
                    d["provenance_json"] = json.dumps({"version": m})
                c.execute(f'INSERT INTO {table} ('+','.join(d)+') VALUES ('+','.join('?' for _ in d)+')', tuple(d.values()))
    args["selections"].append(dict(subject_id="T", mapping_version_id="K1", point_cloud_version_id=vid))
    result = export_training_data(**args)
    assert result["payload_file_count"] == 4
    manifest = json.loads((args["output"] / "manifest.json").read_text())
    assert manifest["subjects"] == ["S", "T"]


def test_cli_selection_mode_and_help(tmp_path, capsys):
    from sync_workbench.cli.main import main
    args = make_source(tmp_path)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(args["selections"]))
    assert main(["export-training-data", "--sqlite", str(args["sqlite_path"]), "--artifact-root", str(args["artifact_root"]),
                 "--output", str(args["output"]), "--selection", str(path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert not args["output"].exists()
