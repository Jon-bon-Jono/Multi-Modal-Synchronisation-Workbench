from pathlib import Path

import numpy as np
import pandas as pd

from sync_workbench.services.artifact_audit_service import ArtifactAuditService
from sync_workbench.services.artifact_build_service import ArtifactBuildService
from sync_workbench.services.ingestion_service import IngestionService
from sync_workbench.services.pair_inspection_service import PairInspectionService
from sync_workbench.services.payload_service import PayloadService
from sync_workbench.storage.ragged_npz import RaggedNpzReader, RaggedNpzWriter
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.services.mapping_service import MappingService
from sync_workbench.sync.mapping import TimelineSelection


def _write_temp_package(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    device_runs = pd.DataFrame(
        [
            {
                "subject_id": "P001",
                "run_id": "Session-A",
                "device_type": "kinect_rgb",
                "start_wallclock_est": "2024-01-01T12:00:00.000000",
                "end_wallclock_est": "2024-01-01T12:00:01.000000",
                "nominal_fps": 15,
                "notes": "",
                "source_host_id": "host1",
                "session_folder_ref": "P001/Session-A",
            },
            {
                "subject_id": "P001",
                "run_id": "Session-A",
                "device_type": "radar_pc",
                "start_wallclock_est": "2024-01-01T12:00:00.000000",
                "end_wallclock_est": "2024-01-01T12:00:01.000000",
                "nominal_fps": 20,
                "notes": "",
                "source_host_id": "host2",
                "session_folder_ref": "P001/Session-A",
            },
        ]
    )
    rgb = pd.DataFrame(
        {
            "frame_number": [0, 1, 2],
            "pts_sec": [0.0, 1 / 15, 2 / 15],
            "subject_id": ["P001"] * 3,
            "run_id": ["Session-A"] * 3,
            "sample_kind": ["frame"] * 3,
            "video_ref": ["P001/Session-A/kinect_camera_recording_rgb_lq.mp4"] * 3,
            "num_people": [1, 1, 0],
            "wallclock_est": [
                "2024-01-01T12:00:00.000000",
                "2024-01-01T12:00:00.066667",
                "2024-01-01T12:00:00.133333",
            ],
            "pose3d": [np.ones((1, 32, 4)), np.empty((0, 32, 4)), np.empty((0, 32, 4))],
            "num_3d": [1, 0, 0],
            "kinect_internal_elapsed_sec": [0.0, np.nan, np.nan],
            "smartcup_os_time": ["2024-01-01T12:00:00.010000", "", ""],
            "pose2d": [np.full((2, 26, 3), 2.0), np.full((1, 26, 3), 3.0), np.empty((0, 26, 3))],
            "conf2d": [np.asarray([0.9, 0.8]), np.asarray([0.7]), np.empty((0,))],
            "num_2d": [2, 1, 0],
            "activity": [
                {"01-Activity": ["walk"]},
                {"01-Activity": ["turn"]},
                {"01-Activity": []},
            ],
        }
    )
    radar = pd.DataFrame(
        {
            "subject_id": ["P001"] * 4,
            "run_id": ["Session-A"] * 4,
            "frame_number": [0, 1, 2, 3],
            "sample_kind": ["frame"] * 4,
            "observed_wallclock": [
                "2024-01-01T12:00:00.000000",
                "2024-01-01T12:00:00.050000",
                "2024-01-01T12:00:00.100000",
                "2024-01-01T12:00:00.150000",
            ],
            "points": [
                np.asarray([[1, 2, 3, 4, 5, 6]], dtype=float),
                np.asarray([[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]], dtype=float),
                np.empty((0, 6)),
                np.asarray([[19, 20, 21, 22, 23, 24]], dtype=float),
            ],
            "point_count": [1, 2, 0, 1],
            "point_count_filtered": [1, 2, 0, 1],
        }
    )
    device_runs.to_pickle(root / "device_runs.zst", compression=None)
    rgb.to_pickle(root / "rgb_samples.zst", compression=None)
    radar.to_pickle(root / "radar_pc_samples.zst", compression=None)


def test_ragged_npz_roundtrip(tmp_path: Path):
    path = tmp_path / "payload.npz"
    RaggedNpzWriter.write(
        path,
        [(2, np.ones((2, 6))), (0, np.empty((0, 6))), (1, np.full((1, 6), 3.0))],
        tail_shape=(6,),
        dtype="float64",
    )
    reader = RaggedNpzReader(path)
    assert reader.validate() == []
    assert reader.get(0).shape == (0, 6)
    assert reader.get(1).shape == (1, 6)
    assert reader.get(2).shape == (2, 6)
    assert np.all(reader.get(1) == 3.0)


def test_artifact_build_payload_roundtrip_and_audit(tmp_path: Path):
    pkg = tmp_path / "temp"
    sqlite_path = tmp_path / "workbench.sqlite"
    artifact_root = tmp_path / "artifacts"
    _write_temp_package(pkg)
    IngestionService().ingest_temp_package(pkg, sqlite_path)

    result = ArtifactBuildService().build_from_temp_package(pkg, sqlite_path, artifact_root)
    assert len(result.sample_artifacts) == (3 * 4) + 4  # RGB roles per sample + radar points per sample
    assert len(result.sample_summary) == 7

    store = SQLiteCoreStore(sqlite_path)
    assert "SAMPLE_SUMMARY" in store.list_tables()
    assert set(store.read_table("SAMPLE_ARTIFACT")["artifact_role"]) == {"pose2d", "conf2d", "pose3d", "activity", "radar_points"}

    payloads = PayloadService(sqlite_path, artifact_root)
    pose2d = payloads.get_payload("P001", "Session-A", "kinect_rgb", 0, "pose2d")
    assert pose2d.shape == (2, 26, 3)
    assert np.all(pose2d == 2.0)
    activity = payloads.get_payload("P001", "Session-A", "kinect_rgb", 1, "activity")
    assert activity["01-Activity"] == ["turn"]
    points = payloads.get_payload("P001", "Session-A", "radar_pc", 1, "radar_points")
    assert points.shape == (2, 6)

    issues = ArtifactAuditService(sqlite_path, artifact_root).audit()
    assert issues.empty


def test_mapped_pair_payload_inspection(tmp_path: Path):
    pkg = tmp_path / "temp"
    sqlite_path = tmp_path / "workbench.sqlite"
    artifact_root = tmp_path / "artifacts"
    _write_temp_package(pkg)
    IngestionService().ingest_temp_package(pkg, sqlite_path)
    ArtifactBuildService().build_from_temp_package(pkg, sqlite_path, artifact_root)

    source = TimelineSelection("P001", "Session-A", "kinect_rgb", "rgb_wallclock_from_pts")
    target = TimelineSelection("P001", "Session-A", "radar_pc", "radar_pc_linear_from_index")
    MappingService(sqlite_path).generate_nearest_mapping(
        source,
        target,
        mapping_version_id="rgb_to_pc_initial_v001",
        top_k=1,
        source_window_policy="all",
    )

    inspected = PairInspectionService(sqlite_path, artifact_root).inspect_pair(
        "rgb_to_pc_initial_v001",
        0,
        subject_id="P001",
    )
    assert inspected["source"]["payload_roles"] == ["activity", "conf2d", "pose2d", "pose3d"]
    assert inspected["target"]["payload_roles"] == ["radar_points"]
    assert inspected["payload_shapes"]["source"]["pose2d"]["shape"] == [2, 26, 3]
    assert inspected["payload_shapes"]["target"]["radar_points"]["shape"][1] == 6
