import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from sync_workbench.ingestion.raw_point_cloud_package import RawCloudPackage, file_hash, json_hash
from sync_workbench.services.raw_point_cloud_import_service import RawPointCloudImportService
from sync_workbench.services.payload_service import PayloadService
from sync_workbench.services.artifact_audit_service import ArtifactAuditService
from sync_workbench.storage.point_cloud_migration import migrate_database_copy, migrate_point_cloud_versions
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


def write_package(root, seed="a", partial=False, change=None):
    root.mkdir(exist_ok=True)
    timestamps = [f"2024-01-01T12:00:0{i}.000000" for i in range(3)]
    binding = dict(hdf5_sha256=seed * 64, subject_id="P", run_id="R", device_type="radar_raw", source_frame_number_base=0)
    vid = "raw_" + json_hash(binding)
    points = np.zeros((6 if seed == "a" else 7, 6), dtype=np.float32)
    points[:, 5] = [250, 251, 252, 253, 254, 255] + ([1] if seed != "a" else [])
    samples = pd.DataFrame({"subject_id": ["P"]*3, "run_id": ["R"]*3, "frame_number": [1,2,3], "sample_kind": ["frame"]*3,
        "estimated_wallclock_from_start_end": timestamps, "points": [points, np.empty((0,6), np.float32), points.copy()],
        "point_count": pd.array([len(points), None if partial else 0, len(points)], dtype="Int64"),
        "point_count_filtered": pd.array([len(points)-3, None if partial else 0, len(points)-3], dtype="Int64"),
        "point_cloud_version_id": [vid]*3, "point_status": ["available", "unprocessed" if partial else "available", "available"]})
    acquisition = dict(subject_id="P", run_id="R", device_type="radar_raw", frame_count=3,
        start_wallclock_est=timestamps[0], end_wallclock_est=timestamps[-1],
        acquisition_timeline_sha256=json_hash(dict(subject_id="P", run_id="R", frame_number=[1,2,3], timestamps=timestamps)))
    version = {**binding, "point_cloud_version_id": vid, "readable_label": seed, "tracking_enabled": True,
        "hdf5_schema": "iwr6843.raw_point_cloud_sequence.hdf5.v1", "point_dtype": "float32",
        "point_columns": ["x","y","z","radial_velocity","snr","association_id"],
        "visualization_excluded_association_ids": [253,254,255], "acquisition_frame_count": 3,
        "processed_frame_count": 2 if partial else 3, "point_count": len(points)*2,
        "source_frame_ids": [0,2] if partial else [0,1,2], "source_sequence_indices": [0,2] if partial else [0,1,2],
        "acquisition": acquisition, "processing_cfg": "config", "generator_metadata": {"tracking_enabled": True,
        "source_config_sha256": "c"*64, "native_executable_sha256": "d"*64, "native_version": "test",
        "created_utc": "2024-01-01", "calibration_sources": [], "tracking_state_start_sequence_index": 0}}
    if change:
        change(version, samples)
    samples.to_pickle(root / "radar_raw_samples.zst", compression="zstd")
    manifest = dict(schema="syncwb.raw_point_cloud_package.v1", sample_file="radar_raw_samples.zst",
        sample_row_count=len(samples), sample_file_sha256=file_hash(root / "radar_raw_samples.zst"), versions=[version])
    (root / "radar_raw_point_cloud_versions.json").write_text(json.dumps(manifest))
    pd.DataFrame([{**{k: acquisition[k] for k in ("subject_id","run_id","device_type","start_wallclock_est","end_wallclock_est")},
                   "nominal_fps": 1, "notes": ""}]).to_pickle(root / "device_runs.zst", compression="zstd")
    return vid, samples


@pytest.fixture
def backend(tmp_path):
    source = tmp_path / "original.sqlite"
    store = SQLiteCoreStore(source)
    store.initialise_empty()
    store.write_table("SUBJECT", pd.DataFrame([dict(subject_id="P")]))
    store.write_table("DEVICE_RUN", pd.DataFrame([dict(subject_id="P", run_id="R", device_type="radar_pc")]))
    store.write_table("ANCHOR", pd.DataFrame([dict(subject_id="P", anchor_id="existing", anchor_type="manual")]))
    store.write_table("SAMPLE_MAPPING", pd.DataFrame([dict(subject_id="P", mapping_version_id="M", source_run_id="R",
        source_device_type="radar_raw", source_sample_index=0, target_run_id="R", target_device_type="radar_raw",
        target_sample_index=2, is_primary=True, rank=1)]))
    # Exercise a genuinely old schema, not just a fresh version-aware schema.
    with sqlite3.connect(source) as conn:
        for table in ("RUN_ASSET", "SAMPLE_ARTIFACT", "SAMPLE_SUMMARY"):
            conn.execute(f'ALTER TABLE {table} DROP COLUMN point_cloud_version_id')
        conn.execute('ALTER TABLE SAMPLE_SUMMARY DROP COLUMN point_status')
        conn.execute('DROP TABLE POINT_CLOUD_VERSION')
    before = file_hash(source)
    db = tmp_path / "migrated.sqlite"
    result = migrate_database_copy(source, db)
    assert result["registered_versions"] == 1
    assert file_hash(source) == before
    root = tmp_path / "artifacts"
    return db, root


def test_copy_migration_is_idempotent_and_preserves_annotations(backend):
    db, root = backend
    assert migrate_point_cloud_versions(db)["registered_versions"] == 0
    versions = SQLiteCoreStore(db).read_table("POINT_CLOUD_VERSION")
    assert versions.point_cloud_version_id.tolist() == ["online_original"]
    assert SQLiteCoreStore(db).read_table("ANCHOR").anchor_id.tolist() == ["existing"]
    with pytest.raises(FileExistsError):
        migrate_database_copy(db, db)


def test_two_versions_share_acquisition_and_mapping(backend, tmp_path):
    db, root = backend
    importer = RawPointCloudImportService()
    ids = []
    snapshot = None
    for seed in ("a", "b"):
        package = tmp_path / seed
        vid, samples = write_package(package, seed)
        ids.append(vid)
        assert importer.import_package(package, db, root)["versions"][0]["status"] == "imported"
        if snapshot is None:
            snapshot = {name: SQLiteCoreStore(db).read_table(name) for name in ("RUN_SAMPLE", "SAMPLE_TIME_ESTIMATE", "ANCHOR", "SAMPLE_MAPPING")}
        else:
            for name, frame in snapshot.items():
                pd.testing.assert_frame_equal(frame, SQLiteCoreStore(db).read_table(name))
    payloads = PayloadService(db, root)
    assert len(payloads.list_point_cloud_versions("P", "R", "radar_raw")) == 2
    for vid, count in zip(ids, [6,7]):
        cloud = payloads.get_payload("P", "R", "radar_raw", 0, "radar_points", point_cloud_version_id=vid)
        assert cloud.dtype == np.float32 and cloud.shape == (count,6)
        summary = payloads.get_sample_summary("P","R","radar_raw",0,point_cloud_version_id=vid)
        assert int(summary["point_count_filtered"]) == count-3  # 250/251/252 are retained.
        empty = payloads.get_payload("P", "R", "radar_raw", 1, "radar_points", point_cloud_version_id=vid)
        assert empty.shape == (0, 6)
        pair = payloads.get_mapped_pair_payloads("P", "M", 0, source_point_cloud_version_id=vid, target_point_cloud_version_id=vid)
        np.testing.assert_array_equal(pair["source_payloads"]["radar_points"], pair["target_payloads"]["radar_points"])
    with pytest.raises(ValueError, match="explicitly"):
        payloads.get_payload("P", "R", "radar_raw", 0, "radar_points")
    with pytest.raises(KeyError, match="not registered"):
        payloads.get_sample_payloads("P", "R", "radar_raw", 0, point_cloud_version_id="wrong")
    before = file_hash(root / f"point_cloud_versions/{ids[-1]}/points.npz")
    assert importer.import_package(tmp_path / "b", db, root)["versions"][0]["status"] == "already_imported"
    assert before == file_hash(root / f"point_cloud_versions/{ids[-1]}/points.npz")
    assert ArtifactAuditService(db,root).audit_cloud_versions().empty


def test_partial_coverage_is_not_empty_cloud(backend, tmp_path):
    db, root = backend
    vid, _ = write_package(tmp_path / "p", partial=True)
    RawPointCloudImportService().import_package(tmp_path/"p", db, root)
    payloads = PayloadService(db,root)
    summary = payloads.get_sample_summary("P","R","radar_raw",1,point_cloud_version_id=vid)
    assert summary["point_status"] == "unprocessed" and pd.isna(summary["point_count"])
    with pytest.raises(KeyError):
        payloads.get_payload("P","R","radar_raw",1,"radar_points",point_cloud_version_id=vid)
    with pytest.raises(KeyError, match="unprocessed"):
        payloads.get_ragged_payload_window("P","R","radar_raw",0,"radar_points",radius=2,point_cloud_version_id=vid)
    assert ArtifactAuditService(db,root).audit_cloud_versions().empty


@pytest.mark.parametrize("change", [
    lambda m,s: m.update(tracking_enabled=False),
    lambda m,s: m.update(point_cloud_version_id="raw_" + "0"*64),
    lambda m,s: m.update(visualization_excluded_association_ids=[250,251,252,253,254,255]),
    lambda m,s: s.at.__setitem__((0,"points"), s.iloc[0].points.astype(np.float64)),
    lambda m,s: s.at.__setitem__((0,"point_count_filtered"), 0),
    lambda m,s: s.at.__setitem__((0,"frame_number"), 2),
    lambda m,s: m["acquisition"].update(acquisition_timeline_sha256="0"*64),
])
def test_bad_package_does_not_mutate_database(backend, tmp_path, change):
    db, root = backend
    write_package(tmp_path/"p", change=change)
    before = file_hash(db)
    with pytest.raises(ValueError):
        RawPointCloudImportService().import_package(tmp_path/"p", db, root)
    assert before == file_hash(db)
    assert not root.exists()


def test_checksum_conflict_and_legacy_routes(backend,tmp_path):
    from sync_workbench.ingestion.temp_package import TempPackage
    db,root=backend
    p=tmp_path/"p"
    write_package(p)
    with pytest.raises(ValueError,match="import-raw-point-clouds"):
        TempPackage.read(p)
    with (p/"radar_raw_samples.zst").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError,match="checksum"):
        RawCloudPackage.read(p)


def test_same_id_changed_payload_and_existing_timeline_rejected(backend,tmp_path):
    db,root=backend
    p=tmp_path/"p"
    write_package(p)
    importer=RawPointCloudImportService()
    importer.import_package(p,db,root)
    def changed(m,s):
        s.iloc[0].points[0,0] = 10
    write_package(p,change=changed)
    before=file_hash(db)
    with pytest.raises(ValueError,match="different payload"):
        importer.import_package(p,db,root)
    assert file_hash(db)==before
    write_package(p,"b")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE SAMPLE_TIME_ESTIMATE SET time_value_datetime='wrong' WHERE sample_index=0")
    with pytest.raises(ValueError,match="timeline differs"):
        importer.import_package(p,db,root)
    assert len(SQLiteCoreStore(db).read_table("POINT_CLOUD_VERSION"))==2


def test_import_rollback_removes_only_its_new_bundle(backend,tmp_path):
    db,root=backend
    write_package(tmp_path/"p")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TRIGGER fail_import BEFORE INSERT ON SAMPLE_SUMMARY BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    with pytest.raises(sqlite3.IntegrityError,match="test failure"):
        RawPointCloudImportService().import_package(tmp_path/"p",db,root)
    assert not list(root.rglob("*.npz"))
    assert SQLiteCoreStore(db).read_table("RUN_SAMPLE").empty
    assert len(SQLiteCoreStore(db).read_table("POINT_CLOUD_VERSION"))==1


def test_relabel_is_idempotent_but_provenance_conflict_fails(backend,tmp_path):
    db,root=backend
    p=tmp_path/"p"
    write_package(p)
    importer=RawPointCloudImportService()
    importer.import_package(p,db,root)
    write_package(p,change=lambda m,s: m.update(readable_label="new label", source_hdf5_path_at_packaging="new/path"))
    assert importer.import_package(p,db,root)["versions"][0]["status"] == "already_imported"
    write_package(p,change=lambda m,s: m.update(processing_cfg="conflicting settings"))
    with pytest.raises(ValueError,match="conflicting.*provenance"):
        importer.import_package(p,db,root)


def test_audit_detects_version_specific_summary_corruption(backend,tmp_path):
    db,root=backend
    p=tmp_path/"p"
    vid,_=write_package(p)
    RawPointCloudImportService().import_package(p,db,root)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE SAMPLE_SUMMARY SET point_count_filtered=0 WHERE point_cloud_version_id=? AND sample_index=0",(vid,))
    issues=ArtifactAuditService(db,root).audit_cloud_versions()
    assert len(issues)==1
    assert issues.iloc[0].point_cloud_version_id==vid
    assert "point counts/status" in issues.iloc[0].issue
