from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.export_pc_raw_sequence import export_sequence
from sync_workbench.services.artifact_build_service import ArtifactBuildService
from sync_workbench.services.ingestion_service import IngestionService
from sync_workbench.services.mapping_service import MappingService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import TimelineSelection


def _write_base_package(root: Path) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2024-01-01T12:00:00")
    device_runs = pd.DataFrame(
        [
            {
                "subject_id": "P001",
                "run_id": "RGB-A",
                "device_type": "kinect_rgb",
                "start_wallclock_est": start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "end_wallclock_est": (start + pd.Timedelta(seconds=0.2)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "nominal_fps": 15,
                "notes": "",
            },
            {
                "subject_id": "P001",
                "run_id": "RADAR-A",
                "device_type": "radar_pc",
                "start_wallclock_est": start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "end_wallclock_est": (start + pd.Timedelta(seconds=0.2)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "nominal_fps": 20,
                "notes": "",
            },
            {
                "subject_id": "P001",
                "run_id": "RADAR-A",
                "device_type": "radar_raw",
                "start_wallclock_est": start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "end_wallclock_est": (start + pd.Timedelta(seconds=0.2)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "nominal_fps": 20,
                "notes": "",
            },
        ]
    )
    rgb_seconds = np.asarray([0.0, 1 / 15, 2 / 15, 0.2])
    rgb = pd.DataFrame(
        {
            "frame_number": np.arange(1, 5),
            "pts_sec": rgb_seconds,
            "subject_id": "P001",
            "run_id": "RGB-A",
            "sample_kind": "frame",
            "video_ref": "P001/RGB-A/rgb.mp4",
            "wallclock_est": [
                (start + pd.Timedelta(seconds=float(value))).strftime("%Y-%m-%dT%H:%M:%S.%f")
                for value in rgb_seconds
            ],
            "pose2d": [
                np.full((1, 26, 3), index, dtype=float)
                for index in range(4)
            ],
            "pose3d": [
                np.full((1, 32, 4), index, dtype=float)
                for index in range(4)
            ],
        }
    )
    pc_seconds = np.arange(5) * 0.05
    pc = pd.DataFrame(
        {
            "subject_id": "P001",
            "run_id": "RADAR-A",
            "frame_number": np.arange(1, 6, dtype=float),
            "sample_kind": "frame",
            "observed_wallclock": [
                (start + pd.Timedelta(seconds=float(value))).strftime("%Y-%m-%dT%H:%M:%S.%f")
                for value in pc_seconds
            ],
            "points": [np.full((index + 1, 6), index, dtype=float) for index in range(5)],
            "point_count": np.arange(1, 6),
            "point_count_filtered": np.arange(1, 6),
        }
    )
    raw = pd.DataFrame(
        {
            "subject_id": "P001",
            "run_id": "RADAR-A",
            "frame_number": np.arange(1, 6, dtype=float),
            "sample_kind": "frame",
            "estimated_wallclock_from_start_end": [
                (start + pd.Timedelta(seconds=float(value))).strftime("%Y-%m-%dT%H:%M:%S.%f")
                for value in pc_seconds
            ],
            "points": [np.empty((0, 6)) for _ in range(5)],
            "point_count": np.zeros(5, dtype=np.int64),
            "point_count_filtered": np.zeros(5, dtype=np.int32),
        }
    )
    device_runs.to_pickle(root / "device_runs.zst", compression=None)
    rgb.to_pickle(root / "rgb_samples.zst", compression=None)
    pc.to_pickle(root / "radar_pc_samples.zst", compression=None)
    return raw


def test_additive_raw_ingestion_preserves_mapping_and_exports_sequence(tmp_path: Path):
    package = tmp_path / "temp"
    sqlite_path = tmp_path / "workbench.sqlite"
    artifact_root = tmp_path / "artifacts"
    raw = _write_base_package(package)

    IngestionService().ingest_temp_package(package, sqlite_path)
    ArtifactBuildService().build_from_temp_package(package, sqlite_path, artifact_root)
    MappingService(sqlite_path).generate_nearest_mapping(
        TimelineSelection("P001", "RGB-A", "kinect_rgb", "rgb_wallclock_from_pts"),
        TimelineSelection("P001", "RADAR-A", "radar_pc", "radar_pc_linear_from_index"),
        mapping_version_id="rgb_to_pc_test",
        top_k=1,
        source_window_policy="all",
        primary_policy="nearest-any",
    )

    raw.to_pickle(package / "radar_raw_samples.zst", compression=None)
    result = IngestionService().ingest_radar_raw_samples(package, sqlite_path)
    assert len(result.tables["RUN_SAMPLE"]) == 5
    IngestionService().ingest_radar_raw_samples(package, sqlite_path)

    store = SQLiteCoreStore(sqlite_path)
    assert len(store.read_table("MAPPING_VERSION")) == 1
    raw_samples = store.read_table("RUN_SAMPLE")
    raw_samples = raw_samples[raw_samples["device_type"] == "radar_raw"]
    assert len(raw_samples) == 5
    assert raw_samples.iloc[0]["notes"] == "source_frame_number=1.0"
    raw_models = store.read_table("RUN_TIMELINE_MODEL")
    assert "radar_raw_wallclock_from_start_end" in set(raw_models["timeline_model_id"])
    raw_times = store.read_table("SAMPLE_TIME_ESTIMATE")
    assert len(raw_times[raw_times["device_type"] == "radar_raw"]) == 5

    output = export_sequence(
        Namespace(
            sqlite=str(sqlite_path),
            artifact_root=str(artifact_root),
            output=str(tmp_path / "exports"),
            sequence_name="test_sequence",
            subject="P001",
            mapping_version="rgb_to_pc_test",
            rgb_start_frame=1,
            rgb_end_frame=4,
            pc_start_frame=1,
            pc_end_frame=5,
        )
    )
    links = pd.read_csv(output / "frame_links.csv")
    assert links["pc_frame_number"].tolist() == [1, 2, 3, 4, 5]
    assert links["raw_frame_number"].tolist() == [1, 2, 3, 4, 5]
    assert np.allclose(links["raw_minus_pc_ms"], 0.0, atol=1e-5)

    with np.load(output / "point_clouds.npz", allow_pickle=False) as bundle:
        assert bundle["sample_index"].tolist() == [0, 1, 2, 3, 4]
        assert bundle["offsets"].tolist() == [0, 1, 3, 6, 10, 15]
        assert bundle["values"].shape == (15, 6)
    with np.load(output / "rgb_pose2d.npz", allow_pickle=False) as bundle:
        assert bundle["sample_index"].tolist() == [0, 1, 2, 3]
        assert bundle["values"].shape == (4, 26, 3)
    with np.load(output / "rgb_pose3d.npz", allow_pickle=False) as bundle:
        assert bundle["sample_index"].tolist() == [0, 1, 2, 3]
        assert bundle["values"].shape == (4, 32, 4)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["pose2d_joint_names"]) == 26
    assert len(metadata["pose3d_joint_names"]) == 32
    assert metadata["pose3d_coordinate_units"] == "millimetres"


def test_full_ingestion_includes_raw_samples(tmp_path: Path):
    package = tmp_path / "temp"
    raw = _write_base_package(package)
    raw.to_pickle(package / "radar_raw_samples.zst", compression=None)
    result = IngestionService().ingest_temp_package(package, tmp_path / "workbench.sqlite")
    run_samples = result.tables["RUN_SAMPLE"]
    assert len(run_samples[run_samples["device_type"] == "radar_raw"]) == 5
    assert len(result.tables["RUN_SAMPLE"]) == 14


def test_raw_point_artifacts_can_be_built_explicitly(tmp_path: Path):
    package = tmp_path / "temp"
    raw = _write_base_package(package)
    raw.at[0, "points"] = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=float)
    raw.loc[0, "point_count"] = 1
    raw.to_pickle(package / "radar_raw_samples.zst", compression=None)
    sqlite_path = tmp_path / "workbench.sqlite"
    IngestionService().ingest_temp_package(package, sqlite_path)
    result = ArtifactBuildService().build_from_temp_package(
        package,
        sqlite_path,
        tmp_path / "artifacts",
        devices=["radar_raw"],
    )
    assert set(result.sample_artifacts["device_type"]) == {"radar_raw"}
    assert set(result.sample_artifacts["artifact_role"]) == {"radar_points"}
