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


def test_2d_activity_pts_camera_calibration_and_independent_people(tmp_path):
    from sync_workbench.storage.jsonl_index import IndexedJsonlWriter

    args = make_source(tmp_path)
    root = args["artifact_root"]
    poses = {0: np.empty((0, 26, 3)), 2: np.full((3, 26, 3), [100.125, 200.25, 0.75])}
    poses[2][1, 4, 0] = np.nan
    RaggedNpzWriter.write(root / "2d.npz", poses, tail_shape=(26, 3), dtype="float64")
    RaggedNpzWriter.write(root / "conf.npz", {0: np.empty(0), 2: np.array([0.9, 0.8, 0.7])}, tail_shape=(), dtype="float64")
    infos = IndexedJsonlWriter.write(root / "activity.jsonl", [(0, {}), (2, {"01-Activity": ["飲む", "drink"]})])
    with sqlite3.connect(args["sqlite_path"]) as conn:
        def add(table, **fields):
            row = dict(subject_id="S", run_id="K1", device_type="kinect_rgb", **fields)
            conn.execute(f'INSERT INTO "{table}" (' + ','.join(row) + ') VALUES (' + ','.join('?' for _ in row) + ')', tuple(row.values()))
        for idx in poses:
            for role, ref in (("pose2d", "2d.npz"), ("conf2d", "conf.npz")):
                add("SAMPLE_ARTIFACT", sample_index=idx, point_cloud_version_id="", artifact_role=role,
                    artifact_ref=ref, storage_key="artifact_store", artifact_format="ragged_npz")
        for info in infos:
            add("SAMPLE_ARTIFACT", sample_index=info.sample_index, point_cloud_version_id="", artifact_role="activity",
                artifact_ref="activity.jsonl", storage_key="artifact_store", artifact_format="jsonl",
                artifact_member_key=json.dumps(dict(byte_offset=info.byte_offset, nbytes=info.nbytes)))
        add("RUN_TIMELINE_MODEL", timeline_model_id="rgb_pts_elapsed", timeline_model_type="identity_observed")
        add("SAMPLE_TIME_ESTIMATE", sample_index=2, timeline_model_id="rgb_pts_elapsed", time_value_sec=2.125, time_kind="pts_based")
        add("RUN_ASSET", asset_id="video", asset_role="rgb_video", storage_key="rgb", asset_ref="S/K1/rgb.mp4")
        conn.execute("UPDATE SAMPLE_SUMMARY SET num_2d=3, num_people=4 WHERE run_id='K1' AND sample_index=2")
    camera = tmp_path / "kinect" / "S" / "K1" / "kinect_camera_recording_calibration.json"
    camera.parent.mkdir(parents=True)
    document = {"CalibrationInformation": {"Cameras": [{"Purpose": "test", "Intrinsics": {"ModelParameters": [0.5]}}]}}
    camera.write_text(json.dumps(document))
    before = {p: file_hash(p) for p in [args["sqlite_path"], camera, *root.glob("*")]}
    export_training_data(**args, kinect_root=tmp_path / "kinect")
    with h5py.File(next(args["output"].glob("*.pose.h5")), "r") as h:
        assert h["frames/pose2d_payload_available"][:3].tolist() == [True, False, True]
        assert h["frames/pose2d_person_count"][:3].tolist() == [0, 0, 3]
        assert h["frames/pose_person_count"][2] == 2
        assert h["frames/num_people"][2] == 4
        assert h["frames/num_2d"][2] == 3
        assert h["frames/num_3d"][2] == 2
        np.testing.assert_allclose(h["poses2d/xy"][:], poses[2][..., :2], equal_nan=True)
        np.testing.assert_allclose(h["poses2d/confidence"][:], 0.75)
        np.testing.assert_allclose(h["poses2d/bbox_confidence"][:], [0.9, 0.8, 0.7])
        assert not h["poses2d/finite_xy"][1, 4]
        assert json.loads(h["poses2d"].attrs["joint_names_json"])[19] == "Hip_Center"
        assert json.loads(h["frames/activity_json"][2]) == {"01-Activity": ["飲む", "drink"]}
        assert json.loads(h["frames/activity_json"][1]) is None
        assert h["frames/activity_payload_available"][:3].tolist() == [True, False, True]
        assert h["frames/rgb_pts_s"][2] == 2.125
        assert np.isnan(h["frames/rgb_pts_s"][0])
    manifest = json.loads((args["output"] / "manifest.json").read_text())
    kinect = manifest["selections"][0]["kinect"]
    assert kinect["camera_calibration"]["document"] == document
    assert kinect["camera_calibration"]["sha256"] == file_hash(camera)
    assert kinect["video_assets"][0]["asset_ref"] == "S/K1/rgb.mp4"
    assert not kinect["video_included"]
    assert {r["artifact_ref"] for r in manifest["source_artifacts"]} >= {"2d.npz", "conf.npz", "activity.jsonl"}
    assert all(file_hash(p) == digest for p, digest in before.items())


def test_missing_2d_is_explicit_and_supplied_camera_root_is_required(tmp_path):
    args = make_source(tmp_path)
    with pytest.raises(ValueError, match="Missing or non-portable"):
        export_training_data(**args, kinect_root=tmp_path / "absent", dry_run=True)
    assert not args["output"].exists()
    export_training_data(**args)
    with h5py.File(next(args["output"].glob("*.pose.h5")), "r") as h:
        assert h["poses2d/xy"].shape == (0, 26, 2)
        assert not h["frames/pose2d_payload_available"][:].any()
        assert not h["frames/conf2d_payload_available"][:].any()
        assert np.all(h["frames/num_2d"][:] == -1)


def test_2d_bbox_count_mismatch_does_not_publish_export(tmp_path):
    args = make_source(tmp_path)
    root = args["artifact_root"]
    RaggedNpzWriter.write(root / "2d.npz", {0: np.zeros((1, 26, 3))})
    RaggedNpzWriter.write(root / "conf.npz", {0: np.zeros(2)}, tail_shape=())
    with sqlite3.connect(args["sqlite_path"]) as conn:
        for role, ref in (("pose2d", "2d.npz"), ("conf2d", "conf.npz")):
            conn.execute("INSERT INTO SAMPLE_ARTIFACT (subject_id,run_id,device_type,sample_index,point_cloud_version_id,artifact_role,artifact_ref,storage_key,artifact_format) VALUES ('S','K1','kinect_rgb',0,'',?,?,'artifact_store','ragged_npz')", (role, ref))
    with pytest.raises(ValueError, match="bounding-box scores"):
        export_training_data(**args)
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


def test_subject_calibration_inputs_may_differ_but_exact_provenance_is_retained(tmp_path):
    args = make_source(tmp_path)
    with sqlite3.connect(args["sqlite_path"]) as conn:
        conn.row_factory = sqlite3.Row
        first = dict(conn.execute("SELECT * FROM POINT_CLOUD_VERSION").fetchone())
        manifest = json.loads(first["provenance_json"])["version"]
        manifest["generator_metadata"]["calibration_sources"] = [{"sha256": "1" * 64}]
        conn.execute("UPDATE POINT_CLOUD_VERSION SET provenance_json=?", (json.dumps({"version": manifest}),))
        manifest["subject_id"] = "T"
        manifest["generator_metadata"]["calibration_sources"] = [{"sha256": "2" * 64}]
        binding = {k: manifest[k] for k in ("hdf5_sha256", "subject_id", "run_id", "device_type", "source_frame_number_base")}
        new_id = "raw_" + json_hash(binding)
        for table in SQLiteCoreStore(args["sqlite_path"]).list_tables():
            columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            if "subject_id" not in columns:
                continue
            for original in list(conn.execute(f'SELECT * FROM "{table}" WHERE subject_id=\'S\'')):
                row = dict(original)
                row["subject_id"] = "T"
                if row.get("point_cloud_version_id") == first["point_cloud_version_id"]:
                    row["point_cloud_version_id"] = new_id
                if table == "POINT_CLOUD_VERSION":
                    row["provenance_json"] = json.dumps({"version": manifest})
                conn.execute(f'INSERT INTO "{table}" (' + ','.join(row) + ') VALUES (' + ','.join('?' for _ in row) + ')', tuple(row.values()))
    args["selections"].append(dict(subject_id="T", mapping_version_id="K1", point_cloud_version_id=new_id))
    export_training_data(**args)
    result = json.loads((args["output"] / "manifest.json").read_text())
    assert result["subjects"] == ["S", "T"]
    assert result["schema"] == "syncwb.training_export.v3"
    assert result["processing_compatibility"]["schema"] == "syncwb.processing_compatibility.v2"
    assert "calibration_sha256" not in result["processing_compatibility"]
    assert [s["exact_processing_recipe"]["calibration_sha256"] for s in result["selections"]] == [["1" * 64], ["2" * 64]]
    assert len({s["exact_processing_recipe_id"] for s in result["selections"]}) == 2
    recipes_by_subject = {s["selection"]["subject_id"]: s["exact_processing_recipe_id"] for s in result["selections"]}
    for segment in result["segments"]:
        with h5py.File(args["output"] / segment["files"]["radar"]["path"], "r") as file:
            assert file.attrs["processing_recipe_id"] == recipes_by_subject[segment["subject_id"]]
            assert file.attrs["processing_compatibility_id"] == result["processing_compatibility_id"]


def test_calibration_procedure_and_enabled_state_remain_compatibility_constraints(tmp_path):
    from copy import deepcopy
    from sync_workbench.services.training_export_service import processing_compatibility
    args = make_source(tmp_path)
    with sqlite3.connect(args["sqlite_path"]) as conn:
        m = json.loads(conn.execute("SELECT provenance_json FROM POINT_CLOUD_VERSION").fetchone()[0])["version"]
    calibrated = deepcopy(m)
    calibrated["generator_metadata"]["calibration_sources"] = [{"sha256": "1" * 64}]
    assert processing_compatibility(m) != processing_compatibility(calibrated)
    other = deepcopy(calibrated)
    other["generator_metadata"]["calibration_procedure"] = {"algorithm": "different"}
    assert processing_compatibility(calibrated) != processing_compatibility(other)


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


def test_calibrated_export_matches_gui_geometry_and_preserves_radar(tmp_path):
    from test_spatial_calibration import write_calibration
    from sync_workbench.core.geometry import load_spatial_calibration
    args = make_source(tmp_path)
    path = write_calibration(tmp_path)
    calibration = load_spatial_calibration(path)
    before = file_hash(path)
    export_training_data(**args, spatial_calibration_path=path)
    manifest = json.loads((args["output"] / "manifest.json").read_text())
    assert manifest["geometry"]["spatial_calibration"]["sha256"] == before
    assert manifest["geometry"]["spatial_calibration"]["document"] == json.loads(path.read_text())
    with h5py.File(next(args["output"].glob("*.pose.h5")), "r") as p:
        expected = pose3d_to_world(np.full((1,32,4), [1000,2000,3000,2]), calibration=calibration)
        np.testing.assert_allclose(p["poses/xyz"][0], expected[0], rtol=1e-6)
        assert json.loads(p.attrs["geometry_json"]) == manifest["geometry"]
        np.testing.assert_array_equal(p["poses/confidence"][:], 2)
    with h5py.File(next(args["output"].glob("*.radar.h5")), "r") as r:
        np.testing.assert_allclose(r["points/values"][0], points_sensor_to_world(np.array([[1,2,3,4,5,255]]))[0])
    assert file_hash(path) == before


def test_cli_calibration_is_forwarded_to_preflight(tmp_path, capsys):
    from test_spatial_calibration import write_calibration
    from sync_workbench.cli.main import main
    args = make_source(tmp_path)
    path = write_calibration(tmp_path)
    assert main(["export-training-data", "--sqlite", str(args["sqlite_path"]), "--artifact-root", str(args["artifact_root"]),
                 "--output", str(args["output"]), "--subject", "S", "--mapping-version", "K1",
                 "--point-cloud-label", "selected cloud", "--spatial-calibration", str(path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["geometry"]["spatial_calibration"]["sha256"] == file_hash(path)
    assert not args["output"].exists()
