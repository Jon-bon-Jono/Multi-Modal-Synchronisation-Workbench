import json
import sqlite3
import gc

import numpy as np
import pandas as pd
import pytest

from sync_workbench.ingestion.raw_point_cloud_package import file_hash
from sync_workbench.services.package_import_service import PackageImportService
from sync_workbench.services.ingestion_service import IngestionService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.storage.point_cloud_migration import migrate_point_cloud_versions
from test_versioned_raw_point_clouds import write_package


@pytest.fixture
def case(tmp_path):
    package, db, artifacts, videos = (tmp_path / n for n in ("package", "master.sqlite", "artifacts", "videos"))
    vid, _ = write_package(package)
    runs = pd.read_pickle(package / "device_runs.zst")
    kinect = runs.iloc[0].to_dict()
    kinect.update(run_id="K", device_type="kinect_rgb")
    pd.concat([runs, pd.DataFrame([kinect])], ignore_index=True).to_pickle(package / "device_runs.zst")
    rows = []
    for i in range(3):
        rows.append(dict(subject_id="P", run_id="K", frame_number=i+1, sample_kind="frame", pts_sec=float(i),
            video_ref="P/K/rgb.mp4", wallclock_est=f"2024-01-01T12:00:0{i}.000000", num_people=1,
            kinect_internal_elapsed_sec=float(i), smartcup_os_time=f"2024-01-01T12:00:0{i}.000000",
            pose2d=np.ones((1,26,3)), conf2d=np.ones(1), pose3d=np.ones((1,32,4)), activity={"walk": ["yes"]}))
    pd.DataFrame(rows).to_pickle(package / "rgb_samples.zst")
    (videos / "P/K").mkdir(parents=True)
    (videos / "P/K/rgb.mp4").write_bytes(b"video fixture")
    store = SQLiteCoreStore(db)
    store.initialise_empty()
    store.write_table("SUBJECT", pd.DataFrame([dict(subject_id="existing", notes="keep")]))
    store.write_table("ANCHOR", pd.DataFrame([dict(subject_id="existing", anchor_id="saved", anchor_type="manual")]))
    migrate_point_cloud_versions(db)
    return dict(input_dir=package, sqlite_path=db, artifact_root=artifacts, rgb_root=videos), vid


def snapshot(args):
    return {str(p): file_hash(p) for p in [args["sqlite_path"], *args["artifact_root"].rglob("*")] if p.is_file()}


def test_add_subject_dry_run_noop_and_repack(case):
    args, _ = case
    service = PackageImportService()
    before = snapshot(args)
    report = service.import_package(**args, dry_run=True)
    assert report["rows_added"] > 0 and snapshot(args) == before
    assert not args["artifact_root"].exists()
    result = service.import_package(**args)
    assert result["status"] == "imported"
    assert result["tables"]["RUN_SAMPLE"]["added"] == 6
    before = snapshot(args)
    # Package-local frame_id/order and compression are not capture identity.
    path = args["input_dir"] / "rgb_samples.zst"
    rgb = pd.read_pickle(path).iloc[::-1].copy()
    rgb.index = [91, 92, 93]
    rgb.index.name = "frame_id"
    rgb.to_pickle(path, compression={"method": "zstd", "level": 2})
    repeat = service.import_package(**args)
    assert repeat["status"] == "already_imported" and repeat["rows_added"] == 0
    assert repeat["artifact_files_added"] == 0 and snapshot(args) == before
    assert SQLiteCoreStore(args["sqlite_path"]).read_table("ANCHOR").anchor_id.tolist() == ["saved"]


@pytest.mark.parametrize("activity_mode", ["absent", "null", "mixed"])
def test_optional_activity_import_and_student_export(case, tmp_path, activity_mode):
    from pathlib import Path
    from sync_workbench.services.mapping_service import MappingService
    from sync_workbench.sync.mapping import TimelineSelection
    from sync_workbench.services.payload_service import PayloadService
    from sync_workbench.deployment.student_package import export_student_package
    from sync_workbench.deployment.package_layout import verify_package

    args, vid = case
    path = args["input_dir"] / "rgb_samples.zst"
    rgb = pd.read_pickle(path)
    if activity_mode == "absent":
        rgb = rgb.drop(columns="activity")
    else:
        rgb["activity"] = [None, np.nan, {} if activity_mode == "mixed" else pd.NA]
    rgb.to_pickle(path)
    service = PackageImportService()
    before = snapshot(args)
    service.import_package(**args, dry_run=True)
    assert snapshot(args) == before
    service.import_package(**args)
    store = SQLiteCoreStore(args["sqlite_path"])
    summary = store.read_table("SAMPLE_SUMMARY")
    summary = summary[summary.device_type == "kinect_rgb"].sort_values("sample_index")
    assert summary.has_activity.astype(int).tolist() == [0, 0, int(activity_mode == "mixed")]
    payloads = PayloadService(args["sqlite_path"], args["artifact_root"])
    with pytest.raises(KeyError, match="No artifact role"):
        payloads.get_payload("P", "K", "kinect_rgb", 0, "activity")
    assert payloads.get_payload("P", "K", "kinect_rgb", 0, "pose3d").shape == (1, 32, 4)
    if activity_mode == "mixed":
        assert payloads.get_payload("P", "K", "kinect_rgb", 2, "activity") == {}
    else:
        assert not list(args["artifact_root"].rglob("activity.jsonl"))
    assert service.import_package(**args)["status"] == "already_imported"

    cloud = store.read_table("POINT_CLOUD_VERSION")
    raw_run = cloud.loc[cloud.point_cloud_version_id == vid, "run_id"].iloc[0]
    MappingService(args["sqlite_path"]).generate_nearest_mapping(
        TimelineSelection("P", "K", "kinect_rgb", "rgb_wallclock_from_pts"),
        TimelineSelection("P", raw_run, "radar_raw", "radar_raw_wallclock_from_start_end"),
        mapping_version_id="optional_activity_initial", top_k=1)
    output = tmp_path / "student"
    export_student_package(sqlite_path=args["sqlite_path"], artifact_root=args["artifact_root"],
        rgb_root=args["rgb_root"], output=output, subject_id="P",
        mapping_version_id="optional_activity_initial", point_cloud_version_id=vid,
        application_root=Path(__file__).resolve().parents[1])
    assert verify_package(output)["full_checksums"]
    student = PayloadService(output / "database/template.sqlite", output / "assets/artifacts")
    with pytest.raises(KeyError, match="No artifact role"):
        student.get_payload("P", "K", "kinect_rgb", 0, "activity")
    assert student.get_payload("P", "K", "kinect_rgb", 0, "pose3d").shape == (1, 32, 4)
    if activity_mode == "mixed":
        assert student.get_payload("P", "K", "kinect_rgb", 2, "activity") == {}


@pytest.mark.parametrize("change", ["pose", "timing", "coverage", "version", "video"])
def test_conflicts_leave_master_unchanged(case, change):
    args, _ = case
    service = PackageImportService()
    service.import_package(**args)
    before = snapshot(args)
    path = args["input_dir"] / "rgb_samples.zst"
    frame = pd.read_pickle(path)
    if change == "pose":
        frame.iloc[0].pose3d[0,0,0] = 99
    elif change == "timing":
        frame.loc[0, "wallclock_est"] = "2024-01-01T11:59:59.000000"
    elif change == "coverage":
        frame = frame.iloc[:2]
    elif change == "video":
        (args["rgb_root"] / "P/K/rgb.mp4").write_bytes(b"changed video")
    else:
        manifest_path = args["input_dir"] / "radar_raw_point_cloud_versions.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["versions"][0]["processing_cfg"] = "changed"
        manifest_path.write_text(json.dumps(manifest))
    frame.to_pickle(path)
    with pytest.raises(ValueError):
        service.import_package(**args)
    assert snapshot(args) == before


def test_raw_only_new_version_preserves_kinect(case, tmp_path):
    args, _ = case
    service = PackageImportService()
    service.import_package(**args)
    before = SQLiteCoreStore(args["sqlite_path"]).read_table("RUN_SAMPLE")
    other = tmp_path / "raw_only"
    write_package(other, seed="b")
    report = service.import_package(**{**args, "input_dir": other, "rgb_root": None})
    assert report["tables"]["RUN_SAMPLE"]["added"] == 0
    assert report["tables"]["POINT_CLOUD_VERSION"]["added"] == 1
    pd.testing.assert_frame_equal(before, SQLiteCoreStore(args["sqlite_path"]).read_table("RUN_SAMPLE"))


def test_existing_metadata_is_completed_and_unselected_catalogue_ignored(case):
    args, _ = case
    runs_path = args["input_dir"] / "device_runs.zst"
    runs = pd.read_pickle(runs_path)
    store = SQLiteCoreStore(args["sqlite_path"])
    store.write_table("DEVICE_RUN", runs)
    extra = runs.iloc[:1].copy()
    extra.subject_id = "unselected"
    pd.concat([runs,extra]).to_pickle(runs_path)
    result = PackageImportService().import_package(**args)
    assert result["tables"]["DEVICE_RUN"]["added"] == 0
    assert result["tables"]["RUN_SAMPLE"]["added"] == 6
    assert "unselected" not in store.read_table("SUBJECT").subject_id.tolist()


def test_file_publication_failure_rolls_back_sql_and_created_files(case, monkeypatch):
    import sync_workbench.services.package_import_service as module
    args, _ = case
    before = snapshot(args)
    real = module.shutil.copyfileobj
    calls = []
    def fail(src, dst, *rest, **kwargs):
        # Raw staging also uses shutil; fail only during publication to master artifacts.
        if str(args["artifact_root"]) in str(dst.name):
            calls.append(dst.name)
            if len(calls) == 2:
                raise OSError("publication failed")
        return real(src, dst, *rest, **kwargs)
    monkeypatch.setattr(module.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="publication failed"):
        PackageImportService().import_package(**args)
    assert snapshot(args) == before


def test_orphan_destination_and_subset_numbering_rejected(case):
    args, _ = case
    orphan = args["artifact_root"] / "subjects/P/kinect_rgb/K/pose3d.npz"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    before = snapshot(args)
    with pytest.raises(ValueError, match="Unregistered"):
        PackageImportService().import_package(**args)
    assert snapshot(args) == before
    orphan.unlink()
    p = args["input_dir"] / "rgb_samples.zst"
    rgb = pd.read_pickle(p)
    rgb.frame_number = [2,3,4]
    rgb.to_pickle(p)
    with pytest.raises(ValueError, match="native frame"):
        PackageImportService().import_package(**args)


def test_legacy_ingest_cannot_reset_master(case):
    args, _ = case
    before = snapshot(args)
    with pytest.raises(FileExistsError, match="NEW database"):
        IngestionService().ingest_temp_package(args["input_dir"], args["sqlite_path"])
    assert snapshot(args) == before


def test_import_closes_staging_connections_without_garbage_collection(case):
    args, _ = case
    enabled = gc.isenabled()
    gc.disable()
    try:
        assert PackageImportService().import_package(**args, dry_run=True)["status"] == "validated"
    finally:
        if enabled:
            gc.enable()


def test_kinect_recompression_is_not_a_payload_conflict(case):
    args, _ = case
    service = PackageImportService()
    service.import_package(**args)
    path = args["artifact_root"] / "subjects/P/kinect_rgb/K/pose3d.npz"
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    np.savez(path, **arrays)  # Same native arrays, different ZIP compression.
    before = snapshot(args)
    assert service.import_package(**args)["status"] == "already_imported"
    assert snapshot(args) == before


def test_cli_dry_run_uses_additive_importer(case, capsys):
    from sync_workbench.cli.main import main
    args, _ = case
    before = snapshot(args)
    assert main(["import-package", "--input", str(args["input_dir"]),
                 "--sqlite", str(args["sqlite_path"]), "--artifact-root", str(args["artifact_root"]),
                 "--rgb-root", str(args["rgb_root"]), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert snapshot(args) == before
