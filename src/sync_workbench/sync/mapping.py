"""Nearest-time mapping generation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from sync_workbench.core.ids import identity_sync_model_id
from sync_workbench.core.tables import align_to_spec
from sync_workbench.core.time_utils import datetime_to_epoch_seconds, utc_now_str


@dataclass(frozen=True)
class TimelineSelection:
    subject_id: str
    run_id: str
    device_type: str
    timeline_model_id: str


@dataclass(frozen=True)
class NearestMappingConfig:
    top_k: int = 3
    weak_support_threshold_ms: float = 75.0
    max_allowed_delta_ms: float = 200.0
    extrapolation_policy: str = "disallow"
    allow_numeric_identity: bool = False


def generate_identity_sync_model_row(
    source: TimelineSelection,
    target: TimelineSelection,
    *,
    sync_model_id: str | None = None,
    extrapolation_policy: str = "disallow",
    coordinate_kind: str = "datetime",
) -> dict[str, object]:
    sync_id = sync_model_id or identity_sync_model_id(
        source.run_id,
        source.device_type,
        source.timeline_model_id,
        target.run_id,
        target.device_type,
        target.timeline_model_id,
    )
    return {
        "subject_id": source.subject_id,
        "sync_model_id": sync_id,
        "source_run_id": source.run_id,
        "source_device_type": source.device_type,
        "target_run_id": target.run_id,
        "target_device_type": target.device_type,
        "source_timeline_model_id": source.timeline_model_id,
        "target_timeline_model_id": target.timeline_model_id,
        "model_name": "Identity-time mapping between selected timelines",
        "model_type": "identity_time",
        "extrapolation_policy": extrapolation_policy,
        "parameters_json": json.dumps({"target_time": "source_time", "coordinate_kind": coordinate_kind}, sort_keys=True),
        "notes": "v0.1 trivial sync model used for nearest-time initial mapping.",
    }


def nearest_mapping_rows(
    source_times: pd.DataFrame,
    target_times: pd.DataFrame,
    source: TimelineSelection,
    target: TimelineSelection,
    *,
    mapping_version_id: str,
    config: NearestMappingConfig,
) -> pd.DataFrame:
    """Generate SAMPLE_MAPPING rows using identity-time nearest-neighbour search."""
    src_numeric, src_kind = _timeline_numeric_values(source_times)
    tgt_numeric, tgt_kind = _timeline_numeric_values(target_times)

    if src_kind != tgt_kind:
        raise ValueError(f"Timeline coordinate mismatch: source={src_kind}, target={tgt_kind}")
    if src_kind == "numeric" and not config.allow_numeric_identity:
        raise ValueError(
            "Numeric identity mapping is disabled by default. Use wallclock-like datetime timelines, "
            "or explicitly enable allow_numeric_identity when the numeric coordinates are compatible."
        )

    src = source_times.copy()
    tgt = target_times.copy()
    src["_t"] = src_numeric
    tgt["_t"] = tgt_numeric
    src = src[np.isfinite(src["_t"])].sort_values("sample_index")
    tgt = tgt[np.isfinite(tgt["_t"])].sort_values("_t")
    if tgt.empty:
        return align_to_spec("SAMPLE_MAPPING", pd.DataFrame())

    target_values = tgt["_t"].to_numpy(dtype=float)
    target_sample_indices = tgt["sample_index"].astype(int).to_numpy()
    t_min = float(np.nanmin(target_values))
    t_max = float(np.nanmax(target_values))

    rows: list[dict[str, object]] = []
    for _, src_row in src.iterrows():
        predicted = float(src_row["_t"])
        source_sample_index = int(src_row["sample_index"])
        candidate_indices = _nearest_candidate_indices(target_values, predicted, config.top_k)
        for rank, cand_idx in enumerate(candidate_indices, start=1):
            target_sample_index = int(target_sample_indices[cand_idx])
            delta_ms = float((target_values[cand_idx] - predicted) * 1000.0)
            region, support = _support_labels(
                predicted,
                delta_ms,
                t_min,
                t_max,
                config.weak_support_threshold_ms,
                config.extrapolation_policy,
            )
            rows.append(
                {
                    "subject_id": source.subject_id,
                    "mapping_version_id": mapping_version_id,
                    "source_run_id": source.run_id,
                    "source_device_type": source.device_type,
                    "source_sample_index": source_sample_index,
                    "target_run_id": target.run_id,
                    "target_device_type": target.device_type,
                    "target_sample_index": target_sample_index,
                    "predicted_minus_estimated_ms": delta_ms,
                    "rank": int(rank),
                    "is_primary": bool(rank == 1 and support in {"supported", "weak_support"}),
                    "mapping_region_type": region,
                    "support_status": support,
                    "confidence_score": _confidence(delta_ms, support, config.max_allowed_delta_ms),
                }
            )
    return align_to_spec("SAMPLE_MAPPING", pd.DataFrame(rows))


def mapping_version_row(
    source: TimelineSelection,
    target: TimelineSelection,
    *,
    mapping_version_id: str,
    sync_model_id: str,
    mapping_name: str | None = None,
    mapping_method: str = "initial_nearest_for_anchoring"
) -> dict[str, object]:
    return {
        "subject_id": source.subject_id,
        "mapping_version_id": mapping_version_id,
        "source_run_id": source.run_id,
        "source_device_type": source.device_type,
        "target_run_id": target.run_id,
        "target_device_type": target.device_type,
        "mapping_name": mapping_name or mapping_version_id,
        "mapping_method": mapping_method, #"nearest_predicted_time"
        "created_at": utc_now_str(),
        "parent_mapping_version_id": "",
        "source_sync_model_id": sync_model_id,
        "notes": "Crude nearest-time mapping generated to help anchor placement/navigation.",
    }


def _timeline_numeric_values(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "time_value_datetime" in df.columns:
        dt_text = df["time_value_datetime"].fillna("").astype(str).str.strip()
        has_datetime = dt_text.ne("") & ~dt_text.str.lower().isin({"nan", "none", "nat"})
        if has_datetime.any():
            return df["time_value_datetime"].map(datetime_to_epoch_seconds), "datetime"
    if "time_value_sec" in df.columns and pd.to_numeric(df["time_value_sec"], errors="coerce").notna().any():
        return pd.to_numeric(df["time_value_sec"], errors="coerce"), "numeric"
    return pd.Series(np.nan, index=df.index), "missing"


def _nearest_candidate_indices(values: np.ndarray, predicted: float, top_k: int) -> list[int]:
    pos = int(np.searchsorted(values, predicted, side="left"))
    window = range(max(0, pos - top_k - 2), min(len(values), pos + top_k + 3))
    ranked = sorted(set(window), key=lambda i: (abs(values[i] - predicted), i))
    return ranked[: max(1, int(top_k))]


def _support_labels(
    predicted: float,
    delta_ms: float,
    target_min: float,
    target_max: float,
    weak_threshold_ms: float,
    extrapolation_policy: str,
) -> tuple[str, str]:
    if predicted < target_min:
        return "left_extrapolation", "outside_run" if extrapolation_policy == "disallow" else "extrapolated"
    if predicted > target_max:
        return "right_extrapolation", "outside_run" if extrapolation_policy == "disallow" else "extrapolated"
    if abs(delta_ms) > weak_threshold_ms:
        return "interpolation", "weak_support"
    return "interpolation", "supported"


def _confidence(delta_ms: float, support_status: str, max_allowed_delta_ms: float) -> float:
    if support_status in {"outside_run", "missing_source_time", "missing_target"}:
        return 0.0
    if max_allowed_delta_ms <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - abs(delta_ms) / max_allowed_delta_ms)))
