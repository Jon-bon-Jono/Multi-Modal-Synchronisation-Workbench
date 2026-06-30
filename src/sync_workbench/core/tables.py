"""Canonical table definitions for v0.1.

The logical schema follows the project documentation, with one practical v0.1
adjustment: SAMPLE_TIME_ESTIMATE separates datetime-valued and numeric-valued
time coordinates. This avoids pretending that PTS/device-elapsed seconds are
native datetimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: tuple[str, ...]
    required: tuple[str, ...]
    key: tuple[str, ...]

    def empty(self) -> pd.DataFrame:
        return pd.DataFrame(columns=list(self.columns))


TABLE_SPECS: dict[str, TableSpec] = {
    "SUBJECT": TableSpec(
        "SUBJECT",
        ("subject_id", "experiment_date", "notes"),
        ("subject_id",),
        ("subject_id",),
    ),
    "DEVICE_RUN": TableSpec(
        "DEVICE_RUN",
        (
            "subject_id",
            "run_id",
            "device_type",
            "start_wallclock_est",
            "end_wallclock_est",
            "nominal_fps",
            "notes",
        ),
        ("subject_id", "run_id", "device_type"),
        ("subject_id", "run_id", "device_type"),
    ),
    "RUN_SAMPLE": TableSpec(
        "RUN_SAMPLE",
        ("subject_id", "run_id", "device_type", "sample_index", "sample_kind", "notes"),
        ("subject_id", "run_id", "device_type", "sample_index", "sample_kind"),
        ("subject_id", "run_id", "device_type", "sample_index"),
    ),
    "RUN_ASSET": TableSpec(
        "RUN_ASSET",
        (
            "asset_id",
            "subject_id",
            "run_id",
            "device_type",
            "asset_role",
            "storage_key",
            "asset_ref",
            "notes",
        ),
        ("asset_id", "subject_id", "run_id", "device_type", "asset_role", "asset_ref"),
        ("asset_id",),
    ),
    "SAMPLE_ARTIFACT": TableSpec(
        "SAMPLE_ARTIFACT",
        (
            "subject_id",
            "run_id",
            "device_type",
            "sample_index",
            "artifact_role",
            "artifact_ref",
            "notes",
        ),
        ("subject_id", "run_id", "device_type", "sample_index", "artifact_role", "artifact_ref"),
        ("subject_id", "run_id", "device_type", "sample_index", "artifact_role"),
    ),
    "RUN_TIMELINE_MODEL": TableSpec(
        "RUN_TIMELINE_MODEL",
        (
            "subject_id",
            "run_id",
            "device_type",
            "timeline_model_id",
            "timeline_model_name",
            "timeline_model_type",
            "source_time_basis",
            "parameters_json",
            "created_at",
            "notes",
        ),
        ("subject_id", "run_id", "device_type", "timeline_model_id", "timeline_model_type"),
        ("subject_id", "run_id", "device_type", "timeline_model_id"),
    ),
    "SAMPLE_TIME_ESTIMATE": TableSpec(
        "SAMPLE_TIME_ESTIMATE",
        (
            "subject_id",
            "run_id",
            "device_type",
            "timeline_model_id",
            "sample_index",
            "time_value_datetime",
            "time_value_sec",
            "time_kind",
            "reference_time_datetime",
            "residual_ms",
            "notes",
        ),
        ("subject_id", "run_id", "device_type", "timeline_model_id", "sample_index", "time_kind"),
        ("subject_id", "run_id", "device_type", "timeline_model_id", "sample_index"),
    ),
    "ANCHOR": TableSpec(
        "ANCHOR",
        ("subject_id", "anchor_id", "anchor_type", "label", "confidence", "notes"),
        ("subject_id", "anchor_id", "anchor_type"),
        ("subject_id", "anchor_id"),
    ),
    "ANCHOR_MEMBER": TableSpec(
        "ANCHOR_MEMBER",
        (
            "subject_id",
            "anchor_id",
            "run_id",
            "device_type",
            "sample_index",
            "member_role",
            "confidence",
            "notes",
        ),
        ("subject_id", "anchor_id", "run_id", "device_type", "sample_index"),
        ("subject_id", "anchor_id", "run_id", "device_type", "sample_index"),
    ),
    "SYNC_MODEL": TableSpec(
        "SYNC_MODEL",
        (
            "subject_id",
            "sync_model_id",
            "source_run_id",
            "source_device_type",
            "target_run_id",
            "target_device_type",
            "source_timeline_model_id",
            "target_timeline_model_id",
            "model_name",
            "model_type",
            "extrapolation_policy",
            "parameters_json",
            "notes",
        ),
        (
            "subject_id",
            "sync_model_id",
            "source_run_id",
            "source_device_type",
            "target_run_id",
            "target_device_type",
            "model_type",
        ),
        ("subject_id", "sync_model_id"),
    ),
    "MODEL_ANCHOR": TableSpec(
        "MODEL_ANCHOR",
        ("subject_id", "sync_model_id", "anchor_id"),
        ("subject_id", "sync_model_id", "anchor_id"),
        ("subject_id", "sync_model_id", "anchor_id"),
    ),
    "MAPPING_VERSION": TableSpec(
        "MAPPING_VERSION",
        (
            "subject_id",
            "mapping_version_id",
            "source_run_id",
            "source_device_type",
            "target_run_id",
            "target_device_type",
            "mapping_name",
            "mapping_method",
            "created_at",
            "parent_mapping_version_id",
            "source_sync_model_id",
            "notes",
        ),
        (
            "subject_id",
            "mapping_version_id",
            "source_run_id",
            "source_device_type",
            "target_run_id",
            "target_device_type",
            "mapping_method",
        ),
        ("subject_id", "mapping_version_id"),
    ),
    "SAMPLE_MAPPING": TableSpec(
        "SAMPLE_MAPPING",
        (
            "subject_id",
            "mapping_version_id",
            "source_run_id",
            "source_device_type",
            "source_sample_index",
            "target_run_id",
            "target_device_type",
            "target_sample_index",
            "predicted_minus_estimated_ms",
            "rank",
            "is_primary",
            "mapping_region_type",
            "support_status",
            "confidence_score",
        ),
        (
            "subject_id",
            "mapping_version_id",
            "source_run_id",
            "source_device_type",
            "source_sample_index",
            "target_run_id",
            "target_device_type",
            "target_sample_index",
        ),
        (
            "subject_id",
            "mapping_version_id",
            "source_run_id",
            "source_device_type",
            "source_sample_index",
            "target_run_id",
            "target_device_type",
            "target_sample_index",
        ),
    ),
}


def empty_canonical_tables(names: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    selected = names or TABLE_SPECS.keys()
    return {name: TABLE_SPECS[name].empty() for name in selected}


def align_to_spec(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exactly the canonical columns for a table."""
    spec = TABLE_SPECS[name]
    out = df.copy()
    for col in spec.columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out.loc[:, list(spec.columns)]
