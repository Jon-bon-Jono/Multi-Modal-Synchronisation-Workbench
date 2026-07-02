"""Application service for nearest-time initial mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sync_workbench.core.ids import identity_sync_model_id, slugify
from sync_workbench.core.tables import align_to_spec
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import (
    NearestMappingConfig,
    TimelineSelection,
    generate_identity_sync_model_row,
    mapping_version_row,
    nearest_mapping_rows,
    timeline_numeric_values,
)


@dataclass
class MappingResult:
    sync_model: pd.DataFrame
    mapping_version: pd.DataFrame
    sample_mapping: pd.DataFrame
    diagnostics: pd.DataFrame

@dataclass
class BatchMappingResult:
    pair_report: pd.DataFrame

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
        source_window_policy: str = "target_overlap",
        source_margin_ms: float = 0.0,
        primary_policy: str = "supported_only",
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
            source_window_policy=source_window_policy,
            source_margin_ms=source_margin_ms,
            primary_policy=primary_policy,
        )

        sync_id = sync_model_id or identity_sync_model_id(
            source.run_id,
            source.device_type,
            source.timeline_model_id,
            target.run_id,
            target.device_type,
            target.timeline_model_id,
        )

        mapping_parameters = {
            "top_k": int(top_k),
            "weak_support_threshold_ms": float(weak_support_threshold_ms),
            "max_allowed_delta_ms": float(max_allowed_delta_ms),
            "extrapolation_policy": extrapolation_policy,
            "allow_numeric_identity": bool(allow_numeric_identity),
            "source_window_policy": source_window_policy,
            "source_margin_ms": float(source_margin_ms),
            "primary_policy": primary_policy,
            "source_timeline_model_id": source.timeline_model_id,
            "target_timeline_model_id": target.timeline_model_id,
        }

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
                        parameters_json=json.dumps(mapping_parameters, sort_keys=True),
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

    def generate_nearest_mappings_for_overlaps(
        self,
        *,
        source_device: str,
        source_timeline: str,
        target_device: str,
        target_timeline: str,
        mapping_version_prefix: str,
        top_k: int = 3,
        min_overlap_sec: float = 5.0,
        source_window_policy: str = "target_overlap",
        source_margin_ms: float = 0.0,
        weak_support_threshold_ms: float = 75.0,
        max_allowed_delta_ms: float = 200.0,
        extrapolation_policy: str = "disallow",
        allow_numeric_identity: bool = False,
        primary_policy: str = "supported_only",
        overwrite: bool = False,
    ) -> BatchMappingResult:
        """Generate nearest mappings for every overlapping source-target run pair."""
        source_runs = self._available_timeline_runs(source_device, source_timeline)
        target_runs = self._available_timeline_runs(target_device, target_timeline)

        report_rows: list[dict[str, object]] = []

        for source_row in source_runs.to_dict("records"):
            source = TimelineSelection(
                str(source_row["subject_id"]),
                str(source_row["run_id"]),
                source_device,
                source_timeline,
            )

            same_subject_targets = target_runs[
                target_runs["subject_id"].astype(str) == source.subject_id
            ]

            for target_row in same_subject_targets.to_dict("records"):
                target = TimelineSelection(
                    str(target_row["subject_id"]),
                    str(target_row["run_id"]),
                    target_device,
                    target_timeline,
                )

                base_report = {
                    "subject_id": source.subject_id,
                    "source_run_id": source.run_id,
                    "source_device_type": source.device_type,
                    "source_timeline_model_id": source.timeline_model_id,
                    "target_run_id": target.run_id,
                    "target_device_type": target.device_type,
                    "target_timeline_model_id": target.timeline_model_id,
                }

                try:
                    source_cov = self._timeline_coverage(source)
                    target_cov = self._timeline_coverage(target)
                except ValueError as exc:
                    report_rows.append({**base_report, "status": "skipped_bad_timeline", "reason": str(exc)})
                    continue

                if source_cov is None:
                    report_rows.append({**base_report, "status": "skipped_missing_source_timeline", "reason": ""})
                    continue
                if target_cov is None:
                    report_rows.append({**base_report, "status": "skipped_missing_target_timeline", "reason": ""})
                    continue

                source_min, source_max, source_kind = source_cov
                target_min, target_max, target_kind = target_cov

                if source_kind != target_kind:
                    report_rows.append(
                        {
                            **base_report,
                            "source_start": source_min,
                            "source_end": source_max,
                            "target_start": target_min,
                            "target_end": target_max,
                            "status": "skipped_coordinate_mismatch",
                            "reason": f"source={source_kind}, target={target_kind}",
                        }
                    )
                    continue

                if source_kind == "numeric" and not allow_numeric_identity:
                    report_rows.append(
                        {
                            **base_report,
                            "source_start": source_min,
                            "source_end": source_max,
                            "target_start": target_min,
                            "target_end": target_max,
                            "status": "skipped_numeric_identity_disabled",
                            "reason": "",
                        }
                    )
                    continue

                overlap_start = max(source_min, target_min)
                overlap_end = min(source_max, target_max)
                overlap_sec = float(overlap_end - overlap_start)

                if overlap_sec <= 0:
                    report_rows.append(
                        {
                            **base_report,
                            "source_start": source_min,
                            "source_end": source_max,
                            "target_start": target_min,
                            "target_end": target_max,
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_sec": overlap_sec,
                            "status": "skipped_no_overlap",
                            "reason": "",
                        }
                    )
                    continue

                if overlap_sec < float(min_overlap_sec):
                    report_rows.append(
                        {
                            **base_report,
                            "source_start": source_min,
                            "source_end": source_max,
                            "target_start": target_min,
                            "target_end": target_max,
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_sec": overlap_sec,
                            "status": "skipped_below_min_overlap",
                            "reason": "",
                        }
                    )
                    continue

                mapping_version_id = self._batch_mapping_version_id(
                    mapping_version_prefix,
                    source,
                    target,
                )

                try:
                    result = self.generate_nearest_mapping(
                        source,
                        target,
                        mapping_version_id=mapping_version_id,
                        top_k=top_k,
                        weak_support_threshold_ms=weak_support_threshold_ms,
                        max_allowed_delta_ms=max_allowed_delta_ms,
                        extrapolation_policy=extrapolation_policy,
                        allow_numeric_identity=allow_numeric_identity,
                        overwrite=overwrite,
                        source_window_policy=source_window_policy,
                        source_margin_ms=source_margin_ms,
                        primary_policy=primary_policy,
                    )
                except ValueError as exc:
                    report_rows.append(
                        {
                            **base_report,
                            "source_start": source_min,
                            "source_end": source_max,
                            "target_start": target_min,
                            "target_end": target_max,
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_sec": overlap_sec,
                            "mapping_version_id": mapping_version_id,
                            "status": "failed",
                            "reason": str(exc),
                        }
                    )
                    continue

                primary_rows = int(result.sample_mapping["is_primary"].astype(bool).sum()) if not result.sample_mapping.empty else 0

                report_rows.append(
                    {
                        **base_report,
                        "source_start": source_min,
                        "source_end": source_max,
                        "target_start": target_min,
                        "target_end": target_max,
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "overlap_sec": overlap_sec,
                        "mapping_version_id": mapping_version_id,
                        "sample_mapping_rows": int(len(result.sample_mapping)),
                        "primary_rows": primary_rows,
                        "status": "mapped",
                        "reason": "",
                    }
                )

        return BatchMappingResult(pd.DataFrame(report_rows))

    def _available_timeline_runs(self, device_type: str, timeline_model_id: str) -> pd.DataFrame:
        times = self.store.read_table("SAMPLE_TIME_ESTIMATE")
        if times.empty:
            return pd.DataFrame(columns=["subject_id", "run_id"])

        mask = (
            (times["device_type"].astype(str) == str(device_type))
            & (times["timeline_model_id"].astype(str) == str(timeline_model_id))
        )
        return (
            times.loc[mask, ["subject_id", "run_id"]]
            .drop_duplicates()
            .sort_values(["subject_id", "run_id"])
            .reset_index(drop=True)
        )

    def _timeline_coverage(self, selection: TimelineSelection) -> tuple[float, float, str] | None:
        times = self._select_times(selection)
        if times.empty:
            return None

        values, kind = timeline_numeric_values(times)
        values = pd.to_numeric(values, errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            return None

        return float(values.min()), float(values.max()), kind

    @staticmethod
    def _batch_mapping_version_id(
        prefix: str,
        source: TimelineSelection,
        target: TimelineSelection,
    ) -> str:
        return (
            f"{slugify(prefix, max_len=40)}__"
            f"{slugify(source.subject_id, max_len=30)}__"
            f"{slugify(source.run_id, max_len=45)}__"
            f"{slugify(target.run_id, max_len=45)}"
        )
    
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
