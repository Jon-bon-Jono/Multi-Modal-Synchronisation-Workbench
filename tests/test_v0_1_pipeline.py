from pathlib import Path

import numpy as np
import pandas as pd

from sync_workbench.services.ingestion_service import IngestionService
from sync_workbench.services.mapping_service import MappingService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
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
            "frame_number": [0, 1, 2, 3],
            "pts_sec": [0.0, 1 / 15, 2 / 15, 3 / 15],
            "subject_id": ["P001"] * 4,
            "run_id": ["Session-A"] * 4,
            "sample_kind": ["frame"] * 4,
            "video_ref": ["P001/Session-A/kinect_camera_recording_rgb_lq.mp4"] * 4,
            "num_people": [1] * 4,
            "wallclock_est": [
                "2024-01-01T12:00:00.000000",
                "2024-01-01T12:00:00.066667",
                "2024-01-01T12:00:00.133333",
                "2024-01-01T12:00:00.200000",
            ],
            "pose3d": [np.empty((0, 32, 4))] * 4,
            "num_3d": [0] * 4,
            "kinect_internal_elapsed_sec": [0.0, np.nan, 2 / 15, np.nan],
            "smartcup_os_time": [
                "2024-01-01T12:00:00.010000",
                "",
                "2024-01-01T12:00:00.143333",
                "",
            ],
            "pose2d": [np.empty((0, 26, 3))] * 4,
            "conf2d": [np.empty((0,))] * 4,
            "num_2d": [0] * 4,
            "activity": [{}] * 4,
        }
    )
    radar = pd.DataFrame(
        {
            "subject_id": ["P001"] * 6,
            "run_id": ["Session-A"] * 6,
            "frame_number": [0, 1, 2, 3, 4, 5],
            "sample_kind": ["frame"] * 6,
            "observed_wallclock": [
                "2024-01-01T12:00:00.000000",
                "2024-01-01T12:00:00.052000",
                "2024-01-01T12:00:00.099000",
                "2024-01-01T12:00:00.151000",
                "2024-01-01T12:00:00.202000",
                "2024-01-01T12:00:00.251000",
            ],
            "points": [np.empty((0, 6))] * 6,
            "point_count": [0] * 6,
            "point_count_filtered": [0] * 6,
        }
    )
    # These are uncompressed pickles with .zst names. The reader intentionally
    # supports this fallback so tests don't require optional zstandard.
    device_runs.to_pickle(root / "device_runs.zst", compression=None)
    rgb.to_pickle(root / "rgb_samples.zst", compression=None)
    radar.to_pickle(root / "radar_pc_samples.zst", compression=None)


def test_ingest_and_map(tmp_path: Path):
    pkg = tmp_path / "temp"
    _write_temp_package(pkg)
    sqlite_path = tmp_path / "workbench.sqlite"
    reports = tmp_path / "reports"

    result = IngestionService().ingest_temp_package(pkg, sqlite_path, reports_dir=reports)
    assert len(result.tables["SUBJECT"]) == 1
    assert len(result.tables["RUN_SAMPLE"]) == 10
    assert "SAMPLE_TIME_ESTIMATE" in result.tables
    assert reports.joinpath("ingestion_report.md").exists()

    store = SQLiteCoreStore(sqlite_path)
    models = store.read_table("RUN_TIMELINE_MODEL")
    assert "rgb_wallclock_from_pts" in set(models["timeline_model_id"])
    assert "radar_pc_linear_from_index" in set(models["timeline_model_id"])
    assert "rgb_smartcup_os_from_pts" in set(models["timeline_model_id"])

    source = TimelineSelection("P001", "Session-A", "kinect_rgb", "rgb_wallclock_from_pts")
    target = TimelineSelection("P001", "Session-A", "radar_pc", "radar_pc_linear_from_index")
    mapping = MappingService(sqlite_path).generate_nearest_mapping(
        source,
        target,
        mapping_version_id="rgb_to_pc_initial_v001",
        top_k=3,
    )
    assert len(mapping.sync_model) == 1
    assert mapping.sync_model.iloc[0]["model_type"] == "identity_time"
    assert len(mapping.mapping_version) == 1
    assert len(mapping.sample_mapping) == 12  # 4 RGB samples * top-k 3
    assert mapping.sample_mapping["source_sample_index"].nunique() == 4
    assert mapping.sample_mapping["is_primary"].sum() >= 3

    stored = store.read_table("SAMPLE_MAPPING")
    assert "rgb_to_pc_initial_v001" in set(stored["mapping_version_id"])

    # Reusing the same mapping_version_id without overwrite should fail.
    try:
        MappingService(sqlite_path).generate_nearest_mapping(
            source,
            target,
            mapping_version_id="rgb_to_pc_initial_v001",
            top_k=1,
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected duplicate mapping_version_id to fail without overwrite.")

    # With overwrite=True, old top-k=3 rows should be replaced by top-k=1 rows.
    MappingService(sqlite_path).generate_nearest_mapping(
        source,
        target,
        mapping_version_id="rgb_to_pc_initial_v001",
        top_k=1,
        overwrite=True,
    )

    stored_after_overwrite = store.read_table("SAMPLE_MAPPING")
    stored_after_overwrite = stored_after_overwrite[
        stored_after_overwrite["mapping_version_id"] == "rgb_to_pc_initial_v001"
    ]

    assert len(stored_after_overwrite) == 4  # 4 RGB samples * top-k 1
    assert set(stored_after_overwrite["rank"].astype(int)) == {1}


