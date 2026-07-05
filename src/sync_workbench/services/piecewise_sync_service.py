"""Service for fitting piecewise-affine sync models from canonical anchors."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sync_workbench.core.tables import align_to_spec
from sync_workbench.core.time_utils import utc_now_str
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import NearestMappingConfig, TimelineSelection, mapping_version_row, timeline_numeric_values
from sync_workbench.sync.piecewise_affine import AnchorPair, PiecewiseAffineModel, fit_piecewise_affine


@dataclass
class PiecewiseSyncResult:
    sync_model: pd.DataFrame
    model_anchor: pd.DataFrame
    mapping_version: pd.DataFrame
    sample_mapping: pd.DataFrame
    anchor_pairs: pd.DataFrame
    diagnostics: pd.DataFrame


class PiecewiseSyncService:
    def __init__(self, sqlite_path: str | Path):
        self.store = SQLiteCoreStore(sqlite_path)

    def fit_piecewise_and_generate_mapping(
        self,
        source: TimelineSelection,
        target: TimelineSelection,
        *,
        sync_model_id: str,
        mapping_version_id: str,
        parent_mapping_version_id: str = "",
        top_k: int = 3,
        weak_support_threshold_ms: float = 75.0,
        max_allowed_delta_ms: float = 200.0,
        extrapolation_policy: str = "disallow",
        primary_policy: str = "supported_only",
        overwrite: bool = False,
        require_monotonic_target: bool = True,
    ) -> PiecewiseSyncResult:
        if source.subject_id != target.subject_id:
            raise ValueError("Piecewise sync expects source and target to share subject_id.")

        version_key = {"subject_id": source.subject_id, "mapping_version_id": mapping_version_id}
        sync_key = {"subject_id": source.subject_id, "sync_model_id": sync_model_id}
        if self.store.row_exists("MAPPING_VERSION", version_key) and not overwrite:
            raise ValueError(f"Mapping version already exists: {version_key}. Use overwrite=True or a new ID.")
        if self.store.row_exists("SYNC_MODEL", sync_key) and not overwrite:
            raise ValueError(f"Sync model already exists: {sync_key}. Use overwrite=True or a new ID.")

        source_times = self._select_times(source)
        target_times = self._select_times(target)
        if source_times.empty:
            raise ValueError(f"No source timeline rows found for {source}")
        if target_times.empty:
            raise ValueError(f"No target timeline rows found for {target}")

        anchor_pairs = self._anchor_pairs(source, target, source_times, target_times)
        model = fit_piecewise_affine(
            anchor_pairs,
            extrapolation_policy=extrapolation_policy,
            require_monotonic_target=require_monotonic_target,
        )

        config = NearestMappingConfig(
            top_k=top_k,
            weak_support_threshold_ms=weak_support_threshold_ms,
            max_allowed_delta_ms=max_allowed_delta_ms,
            extrapolation_policy=extrapolation_policy,
            allow_numeric_identity=True,
            primary_policy=primary_policy,
            source_window_policy="all",
        )
        sample_mapping = self._mapping_rows_from_model(source_times, target_times, source, target, model, mapping_version_id, config)

        parameters = model.to_parameters_dict()
        parameters["diagnostics"] = {
            "anchor_residuals": model.anchor_residuals().to_dict("records"),
            "leave_one_out_errors": model.leave_one_out_errors().to_dict("records"),
            "top_k": int(top_k),
            "weak_support_threshold_ms": float(weak_support_threshold_ms),
            "max_allowed_delta_ms": float(max_allowed_delta_ms),
            "primary_policy": primary_policy,
            "source_timeline_model_id": source.timeline_model_id,
            "target_timeline_model_id": target.timeline_model_id,
        }

        sync_model = align_to_spec(
            "SYNC_MODEL",
            pd.DataFrame(
                [
                    {
                        "subject_id": source.subject_id,
                        "sync_model_id": sync_model_id,
                        "source_run_id": source.run_id,
                        "source_device_type": source.device_type,
                        "target_run_id": target.run_id,
                        "target_device_type": target.device_type,
                        "source_timeline_model_id": source.timeline_model_id,
                        "target_timeline_model_id": target.timeline_model_id,
                        "model_name": sync_model_id,
                        "model_type": "piecewise_affine",
                        "extrapolation_policy": extrapolation_policy,
                        "parameters_json": json.dumps(parameters, sort_keys=True),
                        "notes": "Piecewise affine sync model fitted from canonical anchors.",
                    }
                ]
            ),
        )
        model_anchor = align_to_spec(
            "MODEL_ANCHOR",
            pd.DataFrame(
                [
                    {"subject_id": source.subject_id, "sync_model_id": sync_model_id, "anchor_id": pair.anchor_id}
                    for pair in model.anchor_pairs
                ]
            ),
        )
        mapping_parameters = {
            "top_k": int(top_k),
            "weak_support_threshold_ms": float(weak_support_threshold_ms),
            "max_allowed_delta_ms": float(max_allowed_delta_ms),
            "extrapolation_policy": extrapolation_policy,
            "primary_policy": primary_policy,
            "source_timeline_model_id": source.timeline_model_id,
            "target_timeline_model_id": target.timeline_model_id,
            "parent_mapping_version_id": parent_mapping_version_id,
        }
        mapping_version = align_to_spec(
            "MAPPING_VERSION",
            pd.DataFrame(
                [
                    {
                        **mapping_version_row(
                            source,
                            target,
                            mapping_version_id=mapping_version_id,
                            sync_model_id=sync_model_id,
                            mapping_name=mapping_version_id,
                            mapping_method="nearest_predicted_time",
                            parameters_json=json.dumps(mapping_parameters, sort_keys=True),
                        ),
                        "parent_mapping_version_id": parent_mapping_version_id,
                        "notes": "Mapping generated from a piecewise_affine sync model.",
                    }
                ]
            ),
        )

        if overwrite:
            self.store.delete_where("SAMPLE_MAPPING", version_key)
            self.store.delete_where("MODEL_ANCHOR", sync_key)
        self.store.replace_or_append_without_key_conflicts("SYNC_MODEL", sync_model)
        self.store.replace_or_append_without_key_conflicts("MODEL_ANCHOR", model_anchor)
        self.store.replace_or_append_without_key_conflicts("MAPPING_VERSION", mapping_version)
        self.store.replace_or_append_without_key_conflicts("SAMPLE_MAPPING", sample_mapping)

        diagnostics = self._diagnostics(sample_mapping, model)
        return PiecewiseSyncResult(
            sync_model=sync_model,
            model_anchor=model_anchor,
            mapping_version=mapping_version,
            sample_mapping=sample_mapping,
            anchor_pairs=pd.DataFrame([pair.to_dict() for pair in model.anchor_pairs]),
            diagnostics=diagnostics,
        )

    def _anchor_pairs(
        self,
        source: TimelineSelection,
        target: TimelineSelection,
        source_times: pd.DataFrame,
        target_times: pd.DataFrame,
    ) -> list[AnchorPair]:
        anchors = self.store.read_table("ANCHOR")
        members = self.store.read_table("ANCHOR_MEMBER")
        if anchors.empty or members.empty:
            raise ValueError("No anchors found. Place at least two pair anchors before fitting piecewise sync.")

        source_values, source_kind = timeline_numeric_values(source_times)
        target_values, target_kind = timeline_numeric_values(target_times)
        if source_kind == "missing" or target_kind == "missing":
            raise ValueError("Missing source or target timeline values.")
        source_lookup = self._time_lookup(source_times, source_values)
        target_lookup = self._time_lookup(target_times, target_values)

        members = members[members["subject_id"].astype(str) == source.subject_id].copy()
        src_members = members[
            (members["run_id"].astype(str) == source.run_id)
            & (members["device_type"].astype(str) == source.device_type)
        ]
        tgt_members = members[
            (members["run_id"].astype(str) == target.run_id)
            & (members["device_type"].astype(str) == target.device_type)
        ]
        paired = src_members.merge(tgt_members, on=["subject_id", "anchor_id"], suffixes=("_source", "_target"))
        if paired.empty:
            raise ValueError("No anchors contain both source and target members for this run pair.")

        pairs: list[AnchorPair] = []
        for row in paired.to_dict("records"):
            src_idx = int(row["sample_index_source"])
            tgt_idx = int(row["sample_index_target"])
            if src_idx not in source_lookup or tgt_idx not in target_lookup:
                continue
            st = float(source_lookup[src_idx])
            tt = float(target_lookup[tgt_idx])
            if not (np.isfinite(st) and np.isfinite(tt)):
                continue
            pairs.append(
                AnchorPair(
                    anchor_id=str(row["anchor_id"]),
                    source_sample_index=src_idx,
                    target_sample_index=tgt_idx,
                    source_time=st,
                    target_time=tt,
                )
            )
        if len(pairs) < 2:
            raise ValueError("Fewer than two anchors have finite source and target timeline values.")
        return pairs

    @staticmethod
    def _time_lookup(times: pd.DataFrame, values: pd.Series) -> dict[int, float]:
        tmp = times.copy()
        tmp["__value"] = pd.to_numeric(values, errors="coerce")
        return {int(row["sample_index"]): float(row["__value"]) for _, row in tmp.iterrows() if np.isfinite(row["__value"])}

    def _mapping_rows_from_model(
        self,
        source_times: pd.DataFrame,
        target_times: pd.DataFrame,
        source: TimelineSelection,
        target: TimelineSelection,
        model: PiecewiseAffineModel,
        mapping_version_id: str,
        config: NearestMappingConfig,
    ) -> pd.DataFrame:
        src_values, _ = timeline_numeric_values(source_times)
        tgt_values, _ = timeline_numeric_values(target_times)
        src = source_times.copy()
        tgt = target_times.copy()
        src["__t"] = pd.to_numeric(src_values, errors="coerce")
        tgt["__t"] = pd.to_numeric(tgt_values, errors="coerce")
        src = src[np.isfinite(src["__t"])].sort_values("sample_index")
        tgt = tgt[np.isfinite(tgt["__t"])].sort_values("__t")
        if src.empty or tgt.empty:
            return align_to_spec("SAMPLE_MAPPING", pd.DataFrame())
        target_values = tgt["__t"].to_numpy(dtype=float)
        target_sample_indices = tgt["sample_index"].astype(int).to_numpy()
        rows: list[dict[str, object]] = []
        for _, src_row in src.iterrows():
            pred = model.predict_one(float(src_row["__t"]))
            if not np.isfinite(pred.target_time):
                continue
            candidate_indices = _nearest_candidate_indices(target_values, pred.target_time, config.top_k)
            for rank, cand_idx in enumerate(candidate_indices, start=1):
                target_sample_index = int(target_sample_indices[cand_idx])
                delta_ms = float((target_values[cand_idx] - pred.target_time) * 1000.0)
                support = _support_status(pred.region, delta_ms, config.weak_support_threshold_ms)
                rows.append(
                    {
                        "subject_id": source.subject_id,
                        "mapping_version_id": mapping_version_id,
                        "source_run_id": source.run_id,
                        "source_device_type": source.device_type,
                        "source_sample_index": int(src_row["sample_index"]),
                        "target_run_id": target.run_id,
                        "target_device_type": target.device_type,
                        "target_sample_index": target_sample_index,
                        "predicted_minus_estimated_ms": delta_ms,
                        "rank": int(rank),
                        "is_primary": _is_primary(rank, support, delta_ms, config),
                        "mapping_region_type": pred.region,
                        "support_status": support,
                        "confidence_score": _confidence(delta_ms, support, config.max_allowed_delta_ms),
                    }
                )
        return align_to_spec("SAMPLE_MAPPING", pd.DataFrame(rows))

    def _select_times(self, selection: TimelineSelection) -> pd.DataFrame:
        times = self.store.read_table("SAMPLE_TIME_ESTIMATE")
        if times.empty:
            return times
        return times[
            (times["subject_id"].astype(str) == selection.subject_id)
            & (times["run_id"].astype(str) == selection.run_id)
            & (times["device_type"].astype(str) == selection.device_type)
            & (times["timeline_model_id"].astype(str) == selection.timeline_model_id)
        ].copy()

    @staticmethod
    def _diagnostics(sample_mapping: pd.DataFrame, model: PiecewiseAffineModel) -> pd.DataFrame:
        rows = [
            {"metric": "anchors", "value": int(len(model.anchor_pairs))},
            {"metric": "segments", "value": int(len(model.segments))},
            {"metric": "sample_mapping_rows", "value": int(len(sample_mapping))},
        ]
        if not sample_mapping.empty:
            primary = sample_mapping[sample_mapping["is_primary"].astype(bool)]
            rows.append({"metric": "primary_rows", "value": int(len(primary))})
            for status, count in sample_mapping["support_status"].value_counts(dropna=False).items():
                rows.append({"metric": f"support_status:{status}", "value": int(count)})
        loo = model.leave_one_out_errors()
        if not loo.empty and pd.to_numeric(loo["error_ms"], errors="coerce").notna().any():
            abs_err = pd.to_numeric(loo["error_ms"], errors="coerce").abs().dropna()
            rows.extend(
                [
                    {"metric": "loo_abs_error_ms_median", "value": float(abs_err.median())},
                    {"metric": "loo_abs_error_ms_max", "value": float(abs_err.max())},
                ]
            )
        return pd.DataFrame(rows)


def _nearest_candidate_indices(values: np.ndarray, predicted: float, top_k: int) -> list[int]:
    pos = int(np.searchsorted(values, predicted, side="left"))
    window = range(max(0, pos - top_k - 2), min(len(values), pos + top_k + 3))
    ranked = sorted(set(window), key=lambda i: (abs(values[i] - predicted), i))
    return ranked[: max(1, int(top_k))]


def _support_status(region: str, delta_ms: float, weak_support_threshold_ms: float) -> str:
    if region == "unsupported":
        return "outside_run"
    if region in {"left_extrapolation", "right_extrapolation"}:
        return "extrapolated"
    if abs(float(delta_ms)) > float(weak_support_threshold_ms):
        return "weak_support"
    return "supported"


def _is_primary(rank: int, support_status: str, delta_ms: float, config: NearestMappingConfig) -> bool:
    if int(rank) != 1:
        return False
    policy = str(config.primary_policy or "supported_only").replace("-", "_")
    if policy == "supported_only":
        return support_status == "supported"
    if policy == "within_max_delta":
        return support_status not in {"outside_run", "missing_source_time", "missing_target"} and abs(float(delta_ms)) <= float(config.max_allowed_delta_ms)
    if policy == "nearest_any":
        return support_status not in {"outside_run", "missing_source_time", "missing_target"}
    raise ValueError(f"Unknown primary_policy={config.primary_policy!r}")


def _confidence(delta_ms: float, support_status: str, max_allowed_delta_ms: float) -> float:
    if support_status in {"outside_run", "missing_source_time", "missing_target"}:
        return 0.0
    scale = max(float(max_allowed_delta_ms), 1e-9)
    base = max(0.0, 1.0 - abs(float(delta_ms)) / scale)
    if support_status in {"weak_support", "extrapolated"}:
        base *= 0.5
    return float(base)
