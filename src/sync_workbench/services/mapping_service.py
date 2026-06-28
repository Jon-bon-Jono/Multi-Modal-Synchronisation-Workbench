"""Application service for nearest-time initial mapping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sync_workbench.core.ids import identity_sync_model_id
from sync_workbench.core.tables import align_to_spec
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import (
    NearestMappingConfig,
    TimelineSelection,
    generate_identity_sync_model_row,
    mapping_version_row,
    nearest_mapping_rows,
)


@dataclass
class MappingResult:
    sync_model: pd.DataFrame
    mapping_version: pd.DataFrame
    sample_mapping: pd.DataFrame
    diagnostics: pd.DataFrame


class MappingService:
    def __init__(self, sqlite_path: str | Path):
        self.store = SQLiteCoreStore(sqlite_path)

    def generate_nearest_mapping(
        self,
        source: TimelineSelection,
        target: TimelineSelection,
        *,
        mapping_version_id: str,
        sync_model_id: str | None = None,
        top_k: int = 3,
        weak_support_threshold_ms: float = 75.0,
        max_allowed_delta_ms: float = 200.0,
        extrapolation_policy: str = "disallow",
        allow_numeric_identity: bool = False,
        mapping_name: str | None = None,
        overwrite: bool = False,
    ) -> MappingResult:
        if source.subject_id != target.subject_id:
            raise ValueError("v0.1 nearest mapping expects source and target to share subject_id.")
        
        version_key = {
            "subject_id": source.subject_id,
            "mapping_version_id": mapping_version_id,
        }

        if self.store.row_exists("MAPPING_VERSION", version_key) and not overwrite:
            raise ValueError(
                f"Mapping version already exists: subject_id={source.subject_id}, "
                f"mapping_version_id={mapping_version_id}. Use a new mapping_version_id "
                "or pass overwrite=True / --overwrite to replace it."
            )

        source_times = self._select_times(source)
        target_times = self._select_times(target)
        if source_times.empty:
            raise ValueError(f"No source timeline rows found for {source}")
        if target_times.empty:
            raise ValueError(f"No target timeline rows found for {target}")

        config = NearestMappingConfig(
            top_k=top_k,
            weak_support_threshold_ms=weak_support_threshold_ms,
            max_allowed_delta_ms=max_allowed_delta_ms,
            extrapolation_policy=extrapolation_policy,
            allow_numeric_identity=allow_numeric_identity,
        )

        sync_id = sync_model_id or identity_sync_model_id(
            source.run_id,
            source.device_type,
            source.timeline_model_id,
            target.run_id,
            target.device_type,
            target.timeline_model_id,
        )
        sample_mapping = nearest_mapping_rows(
            source_times,
            target_times,
            source,
            target,
            mapping_version_id=mapping_version_id,
            config=config,
        )
        sync_model = align_to_spec(
            "SYNC_MODEL",
            pd.DataFrame(
                [
                    generate_identity_sync_model_row(
                        source,
                        target,
                        sync_model_id=sync_id,
                        extrapolation_policy=extrapolation_policy,
                        coordinate_kind="datetime_or_numeric_checked_at_generation",
                    )
                ]
            ),
        )
        mapping_version = align_to_spec(
            "MAPPING_VERSION",
            pd.DataFrame(
                [
                    mapping_version_row(
                        source,
                        target,
                        mapping_version_id=mapping_version_id,
                        sync_model_id=sync_id,
                        mapping_name=mapping_name,
                    )
                ]
            ),
        )
        diagnostics = self._diagnostics(sample_mapping)

        if overwrite:
            self.store.delete_where("SAMPLE_MAPPING", version_key)

        self.store.replace_or_append_without_key_conflicts("SYNC_MODEL", sync_model)
        self.store.replace_or_append_without_key_conflicts("MAPPING_VERSION", mapping_version)
        self.store.replace_or_append_without_key_conflicts("SAMPLE_MAPPING", sample_mapping)
        return MappingResult(sync_model, mapping_version, sample_mapping, diagnostics)

    def _select_times(self, selection: TimelineSelection) -> pd.DataFrame:
        times = self.store.read_table("SAMPLE_TIME_ESTIMATE")
        mask = (
            (times["subject_id"] == selection.subject_id)
            & (times["run_id"] == selection.run_id)
            & (times["device_type"] == selection.device_type)
            & (times["timeline_model_id"] == selection.timeline_model_id)
        )
        return times.loc[mask].copy()

    @staticmethod
    def _diagnostics(sample_mapping: pd.DataFrame) -> pd.DataFrame:
        if sample_mapping.empty:
            return pd.DataFrame(
                [
                    {"metric": "rows", "value": 0},
                    {"metric": "primary_rows", "value": 0},
                ]
            )
        primary = sample_mapping[sample_mapping["is_primary"].astype(bool)]
        rows = [
            {"metric": "rows", "value": int(len(sample_mapping))},
            {"metric": "primary_rows", "value": int(len(primary))},
            {"metric": "unique_source_samples", "value": int(sample_mapping["source_sample_index"].nunique())},
            {"metric": "unique_primary_target_samples", "value": int(primary["target_sample_index"].nunique())},
        ]
        for status, count in sample_mapping["support_status"].value_counts(dropna=False).items():
            rows.append({"metric": f"support_status:{status}", "value": int(count)})
        if not primary.empty:
            abs_delta = primary["predicted_minus_estimated_ms"].astype(float).abs()
            rows.extend(
                [
                    {"metric": "primary_abs_delta_ms_median", "value": float(abs_delta.median())},
                    {"metric": "primary_abs_delta_ms_p95", "value": float(abs_delta.quantile(0.95))},
                    {"metric": "primary_abs_delta_ms_max", "value": float(abs_delta.max())},
                ]
            )
        return pd.DataFrame(rows)
