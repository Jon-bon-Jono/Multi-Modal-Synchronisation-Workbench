from pathlib import Path

import numpy as np
import pandas as pd

from sync_workbench.core.tables import align_to_spec
from sync_workbench.services.anchor_service import AnchorEndpoint, AnchorService
from sync_workbench.services.mapping_lookup_service import MappingLookupService
from sync_workbench.services.piecewise_sync_service import PiecewiseSyncService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import TimelineSelection
from sync_workbench.sync.piecewise_affine import AnchorPair, fit_piecewise_affine


def test_piecewise_affine_identity_and_extrapolation_labels():
    pairs = [
        AnchorPair("a0", 0, 0, 0.0, 1.0),
        AnchorPair("a1", 10, 10, 10.0, 11.0),
        AnchorPair("a2", 20, 20, 20.0, 21.0),
    ]
    model = fit_piecewise_affine(pairs, extrapolation_policy="allow_linear")
    assert model.predict_one(5.0).target_time == 6.0
    assert model.predict_one(5.0).region == "interpolation"
    assert model.predict_one(-1.0).target_time == 0.0
    assert model.predict_one(-1.0).region == "left_extrapolation"
    assert model.predict_one(21.0).target_time == 22.0
    assert model.anchor_residuals()["residual_ms"].abs().max() == 0.0


def test_piecewise_rejects_duplicate_source_times():
    pairs = [AnchorPair("a0", 0, 0, 0.0, 0.0), AnchorPair("a1", 1, 1, 0.0, 1.0)]
    try:
        fit_piecewise_affine(pairs)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("Expected duplicate source anchor times to fail")


def _write_toy_sqlite(sqlite_path: Path):
    store = SQLiteCoreStore(sqlite_path)
    store.initialise_empty()
    subject = align_to_spec("SUBJECT", pd.DataFrame([{"subject_id": "P001", "experiment_date": "", "notes": ""}]))
    runs = align_to_spec(
        "DEVICE_RUN",
        pd.DataFrame(
            [
                {"subject_id": "P001", "run_id": "RGB-A", "device_type": "kinect_rgb", "start_wallclock_est": "", "end_wallclock_est": "", "nominal_fps": 15, "notes": ""},
                {"subject_id": "P001", "run_id": "PC-A", "device_type": "radar_pc", "start_wallclock_est": "", "end_wallclock_est": "", "nominal_fps": 20, "notes": ""},
            ]
        ),
    )
    samples = []
    for i in range(5):
        samples.append({"subject_id": "P001", "run_id": "RGB-A", "device_type": "kinect_rgb", "sample_index": i, "sample_kind": "frame", "notes": ""})
    for i in range(6):
        samples.append({"subject_id": "P001", "run_id": "PC-A", "device_type": "radar_pc", "sample_index": i, "sample_kind": "frame", "notes": ""})
    timelines = align_to_spec(
        "RUN_TIMELINE_MODEL",
        pd.DataFrame(
            [
                {"subject_id": "P001", "run_id": "RGB-A", "device_type": "kinect_rgb", "timeline_model_id": "rgb_t", "timeline_model_name": "", "timeline_model_type": "identity_observed", "source_time_basis": "", "parameters_json": "", "created_at": "", "notes": ""},
                {"subject_id": "P001", "run_id": "PC-A", "device_type": "radar_pc", "timeline_model_id": "pc_t", "timeline_model_name": "", "timeline_model_type": "identity_observed", "source_time_basis": "", "parameters_json": "", "created_at": "", "notes": ""},
            ]
        ),
    )
    times = []
    for i in range(5):
        times.append({"subject_id": "P001", "run_id": "RGB-A", "device_type": "kinect_rgb", "timeline_model_id": "rgb_t", "sample_index": i, "time_value_datetime": "", "time_value_sec": float(i), "time_kind": "device_elapsed", "reference_time_datetime": "", "residual_ms": 0.0, "notes": ""})
    for i in range(6):
        times.append({"subject_id": "P001", "run_id": "PC-A", "device_type": "radar_pc", "timeline_model_id": "pc_t", "sample_index": i, "time_value_datetime": "", "time_value_sec": float(i + 1), "time_kind": "device_elapsed", "reference_time_datetime": "", "residual_ms": 0.0, "notes": ""})
    store.write_table("SUBJECT", subject)
    store.write_table("DEVICE_RUN", runs)
    store.write_table("RUN_SAMPLE", align_to_spec("RUN_SAMPLE", pd.DataFrame(samples)))
    store.write_table("RUN_TIMELINE_MODEL", timelines)
    store.write_table("SAMPLE_TIME_ESTIMATE", align_to_spec("SAMPLE_TIME_ESTIMATE", pd.DataFrame(times)))
    return store


def test_anchor_service_and_piecewise_sync_roundtrip(tmp_path: Path):
    sqlite_path = tmp_path / "workbench.sqlite"
    store = _write_toy_sqlite(sqlite_path)
    anchors = AnchorService(sqlite_path)
    anchors.create_pair_anchor(
        subject_id="P001",
        anchor_id="a_start",
        source=AnchorEndpoint("RGB-A", "kinect_rgb", 0, "source"),
        target=AnchorEndpoint("PC-A", "radar_pc", 0, "target"),
        label="start",
        provenance={"created_by": "test", "initial_mapping_version_id": "initial"},
    )
    anchors.create_pair_anchor(
        subject_id="P001",
        anchor_id="a_end",
        source=AnchorEndpoint("RGB-A", "kinect_rgb", 4, "source"),
        target=AnchorEndpoint("PC-A", "radar_pc", 4, "target"),
        label="end",
        provenance={"created_by": "test", "initial_mapping_version_id": "initial"},
    )
    pair = anchors.list_pair_anchors(
        subject_id="P001",
        source_run_id="RGB-A",
        source_device_type="kinect_rgb",
        target_run_id="PC-A",
        target_device_type="radar_pc",
    )
    assert len(pair) == 2

    result = PiecewiseSyncService(sqlite_path).fit_piecewise_and_generate_mapping(
        TimelineSelection("P001", "RGB-A", "kinect_rgb", "rgb_t"),
        TimelineSelection("P001", "PC-A", "radar_pc", "pc_t"),
        sync_model_id="sync_piecewise_v001",
        mapping_version_id="map_piecewise_v001",
        top_k=1,
    )
    assert len(result.sync_model) == 1
    assert result.sync_model.iloc[0]["model_type"] == "piecewise_affine"
    assert len(result.model_anchor) == 2
    assert result.sample_mapping["source_sample_index"].nunique() == 5
    assert set(result.sample_mapping["target_sample_index"].astype(int)) == {0, 1, 2, 3, 4}

    lookup = MappingLookupService(sqlite_path)
    forward = lookup.map_source_to_target("P001", "map_piecewise_v001", 2)
    assert int(forward["target_sample_index"]) == 2
    inverse = lookup.map_target_to_source("P001", "map_piecewise_v001", 3)
    assert int(inverse["source_sample_index"]) == 3

    export_path = tmp_path / "anchors.json"
    payload = anchors.export_pair_anchors_json(
        export_path,
        subject_id="P001",
        source_run_id="RGB-A",
        source_device_type="kinect_rgb",
        target_run_id="PC-A",
        target_device_type="radar_pc",
    )
    assert len(payload["ANCHOR"]) == 2
    assert export_path.exists()
