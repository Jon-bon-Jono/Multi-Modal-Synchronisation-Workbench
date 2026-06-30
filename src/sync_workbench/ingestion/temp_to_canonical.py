"""Transform temporary ingestion dataframes into canonical v0.1 tables."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sync_workbench.assets.asset_refs import is_probably_absolute_path, normalise_asset_ref
from sync_workbench.core.ids import asset_id
from sync_workbench.core.tables import TABLE_SPECS, align_to_spec, empty_canonical_tables
from sync_workbench.core.time_utils import (
    add_seconds_to_datetime,
    datetime_to_epoch_seconds,
    epoch_seconds_to_datetime_str,
    format_datetime,
    parse_datetime,
    utc_now_str,
)
from sync_workbench.ingestion.temp_package import TempPackage


@dataclass
class TransformResult:
    tables: dict[str, pd.DataFrame]
    reports: dict[str, pd.DataFrame]


class TempToCanonicalTransformer:
    """Converts the current temporary `.zst` package to canonical tables.

    Large array-like payloads from the temporary files are intentionally not
    embedded in the canonical core. They remain in the source package or should
    be exported as artifacts in a later implementation step.
    """

    def __init__(self, package: TempPackage):
        self.package = package
        self.created_at = utc_now_str()

    def transform(self) -> TransformResult:
        tables = empty_canonical_tables()
        reports: dict[str, pd.DataFrame] = {}

        tables["SUBJECT"] = self._subjects()
        tables["DEVICE_RUN"] = self._device_runs()

        sample_frames: list[pd.DataFrame] = []
        sample_time_frames: list[pd.DataFrame] = []
        timeline_model_frames: list[pd.DataFrame] = []
        run_asset_frames: list[pd.DataFrame] = []

        if self.package.rgb_samples is not None:
            rgb = self._prepare_rgb_samples(self.package.rgb_samples)
            sample_frames.append(self._run_samples_from_prepared(rgb, "kinect_rgb"))
            run_asset_frames.append(self._rgb_assets(rgb))
            rgb_models, rgb_times, rgb_reports = self._rgb_timeline_tables(rgb)
            timeline_model_frames.append(rgb_models)
            sample_time_frames.append(rgb_times)
            reports.update(rgb_reports)

        if self.package.radar_pc_samples is not None:
            pc = self._prepare_radar_pc_samples(self.package.radar_pc_samples)
            sample_frames.append(self._run_samples_from_prepared(pc, "radar_pc"))
            pc_models, pc_times, pc_reports = self._radar_pc_timeline_tables(pc)
            timeline_model_frames.append(pc_models)
            sample_time_frames.append(pc_times)
            reports.update(pc_reports)

        if sample_frames:
            tables["RUN_SAMPLE"] = align_to_spec("RUN_SAMPLE", pd.concat(sample_frames, ignore_index=True))
        if run_asset_frames:
            tables["RUN_ASSET"] = align_to_spec("RUN_ASSET", pd.concat(run_asset_frames, ignore_index=True).drop_duplicates("asset_id"))
        else:
            tables["RUN_ASSET"] = TABLE_SPECS["RUN_ASSET"].empty()
        if timeline_model_frames:
            tables["RUN_TIMELINE_MODEL"] = align_to_spec(
                "RUN_TIMELINE_MODEL", pd.concat(timeline_model_frames, ignore_index=True)
            )
        if sample_time_frames:
            tables["SAMPLE_TIME_ESTIMATE"] = align_to_spec(
                "SAMPLE_TIME_ESTIMATE", pd.concat(sample_time_frames, ignore_index=True)
            )

        reports["table_counts"] = self._table_counts(tables)
        reports["device_run_summary"] = self._device_run_summary(tables)
        reports["asset_path_warnings"] = self._asset_path_warnings(tables.get("RUN_ASSET", TABLE_SPECS["RUN_ASSET"].empty()))
        return TransformResult(tables=tables, reports=reports)

    def _subjects(self) -> pd.DataFrame:
        subject_ids = sorted(self.package.device_runs["subject_id"].astype(str).unique())
        return pd.DataFrame({"subject_id": subject_ids, "experiment_date": "", "notes": ""})

    def _device_runs(self) -> pd.DataFrame:
        cols = [
            "subject_id",
            "run_id",
            "device_type",
            "start_wallclock_est",
            "end_wallclock_est",
            "nominal_fps",
            "notes",
        ]
        out = self.package.device_runs.loc[:, cols].copy()
        out["start_wallclock_est"] = out["start_wallclock_est"].map(format_datetime)
        out["end_wallclock_est"] = out["end_wallclock_est"].map(format_datetime)
        return align_to_spec("DEVICE_RUN", out.drop_duplicates(["subject_id", "run_id", "device_type"]))

    def _prepare_rgb_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["device_type"] = "kinect_rgb"
        out = out.sort_values(["subject_id", "run_id", "frame_number"]).reset_index(drop=False)
        out["sample_index"] = out.groupby(["subject_id", "run_id"]).cumcount().astype(int)
        out["source_frame_number"] = out["frame_number"]
        return out

    def _prepare_radar_pc_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["device_type"] = "radar_pc"
        out = out.sort_values(["subject_id", "run_id", "frame_number"]).reset_index(drop=False)
        out["sample_index"] = out.groupby(["subject_id", "run_id"]).cumcount().astype(int)
        out["source_frame_number"] = out["frame_number"]
        return out

    def _run_samples_from_prepared(self, df: pd.DataFrame, device_type: str) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "subject_id": df["subject_id"].astype(str),
                "run_id": df["run_id"].astype(str),
                "device_type": device_type,
                "sample_index": df["sample_index"].astype(int),
                "sample_kind": df.get("sample_kind", "frame"),
                "notes": "source_frame_number=" + df["source_frame_number"].astype(str),
            }
        )
        return align_to_spec("RUN_SAMPLE", out)

    def _rgb_assets(self, rgb: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if "video_ref" not in rgb.columns:
            return TABLE_SPECS["RUN_ASSET"].empty()
        grouped = rgb.groupby(["subject_id", "run_id", "device_type", "video_ref"], dropna=False).size().reset_index(name="n")
        for row in grouped.itertuples(index=False):
            ref = normalise_asset_ref(row.video_ref)
            rows.append(
                {
                    "asset_id": asset_id(row.subject_id, row.run_id, row.device_type, "rgb_video"),
                    "subject_id": row.subject_id,
                    "run_id": row.run_id,
                    "device_type": row.device_type,
                    "asset_role": "rgb_video",
                    "storage_key": "rgb",
                    "asset_ref": ref,
                    "notes": f"registered from rgb_samples.zst; rows={row.n}",
                }
            )
        return align_to_spec("RUN_ASSET", pd.DataFrame(rows))

    def _timeline_model_row(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        model_id: str,
        name: str,
        model_type: str,
        source_time_basis: str,
        params: dict[str, Any] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        return {
            "subject_id": subject_id,
            "run_id": run_id,
            "device_type": device_type,
            "timeline_model_id": model_id,
            "timeline_model_name": name,
            "timeline_model_type": model_type,
            "source_time_basis": source_time_basis,
            "parameters_json": json.dumps(params or {}, sort_keys=True),
            "created_at": self.created_at,
            "notes": notes,
        }

    def _time_row(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        model_id: str,
        sample_index: int,
        time_kind: str,
        *,
        time_value_datetime: str = "",
        time_value_sec: float | None = None,
        reference_time_datetime: str = "",
        residual_ms: float | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        return {
            "subject_id": subject_id,
            "run_id": run_id,
            "device_type": device_type,
            "timeline_model_id": model_id,
            "sample_index": int(sample_index),
            "time_value_datetime": time_value_datetime or "",
            "time_value_sec": np.nan if time_value_sec is None else float(time_value_sec),
            "time_kind": time_kind,
            "reference_time_datetime": reference_time_datetime or "",
            "residual_ms": np.nan if residual_ms is None else float(residual_ms),
            "notes": notes,
        }

    def _rgb_timeline_tables(self, rgb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
        model_rows: list[dict[str, Any]] = []
        time_rows: list[dict[str, Any]] = []
        reports: dict[str, pd.DataFrame] = {}
        device_runs = self._device_runs().set_index(["subject_id", "run_id", "device_type"])

        for (subject_id, run_id), group in rgb.groupby(["subject_id", "run_id"], sort=False):
            device_type = "kinect_rgb"
            run_key = (subject_id, run_id, device_type)
            reference_time = ""
            if run_key in device_runs.index:
                reference_time = device_runs.loc[run_key, "start_wallclock_est"]

            model_rows.append(
                self._timeline_model_row(
                    subject_id,
                    run_id,
                    device_type,
                    "rgb_wallclock_from_pts",
                    "RGB wallclock estimate from run start + PTS",
                    "identity_observed",
                    "wallclock_est",
                    {"dense": True},
                )
            )
            model_rows.append(
                self._timeline_model_row(
                    subject_id,
                    run_id,
                    device_type,
                    "rgb_pts_elapsed",
                    "RGB video PTS elapsed seconds",
                    "identity_observed",
                    "pts_sec",
                    {"dense": True, "coordinate": "relative_seconds"},
                )
            )
            for row in group.itertuples(index=False):
                time_rows.append(
                    self._time_row(
                        subject_id,
                        run_id,
                        device_type,
                        "rgb_wallclock_from_pts",
                        row.sample_index,
                        "estimated_wallclock",
                        time_value_datetime=format_datetime(getattr(row, "wallclock_est")),
                        reference_time_datetime=reference_time,
                    )
                )
                time_rows.append(
                    self._time_row(
                        subject_id,
                        run_id,
                        device_type,
                        "rgb_pts_elapsed",
                        row.sample_index,
                        "pts_based",
                        time_value_sec=float(getattr(row, "pts_sec")),
                        reference_time_datetime=reference_time,
                    )
                )

            # Sparse Kinect elapsed observations.
            if "kinect_internal_elapsed_sec" in group.columns:
                obs = group.dropna(subset=["kinect_internal_elapsed_sec"])
                if not obs.empty:
                    model_rows.append(
                        self._timeline_model_row(
                            subject_id,
                            run_id,
                            device_type,
                            "rgb_kinect_elapsed_observed",
                            "Observed Kinect internal elapsed seconds",
                            "identity_observed",
                            "kinect_internal_elapsed_sec",
                            {"dense": False, "num_observed": int(len(obs))},
                        )
                    )
                    for row in obs.itertuples(index=False):
                        time_rows.append(
                            self._time_row(
                                subject_id,
                                run_id,
                                device_type,
                                "rgb_kinect_elapsed_observed",
                                row.sample_index,
                                "device_elapsed",
                                time_value_sec=float(row.kinect_internal_elapsed_sec),
                                reference_time_datetime=reference_time,
                            )
                        )
                    fit = _fit_linear(group["pts_sec"], group["kinect_internal_elapsed_sec"])
                    if fit is not None:
                        a, b, residuals_ms, used = fit
                        model_rows.append(
                            self._timeline_model_row(
                                subject_id,
                                run_id,
                                device_type,
                                "rgb_kinect_elapsed_from_pts",
                                "Derived Kinect internal elapsed seconds from PTS",
                                "regressed_from_pts",
                                "pts_sec + observed kinect_internal_elapsed_sec",
                                {
                                    "dense": True,
                                    "slope": a,
                                    "intercept": b,
                                    "num_fit_points": int(used),
                                    "residual_unit": "ms",
                                },
                                "Derived estimate; does not recover original missing Kinect timestamps.",
                            )
                        )
                        pred = a * group["pts_sec"].astype(float) + b
                        obs_resid = _residual_lookup(group, "kinect_internal_elapsed_sec", pred, scale=1000.0)
                        for row, yhat in zip(group.itertuples(index=False), pred):
                            time_rows.append(
                                self._time_row(
                                    subject_id,
                                    run_id,
                                    device_type,
                                    "rgb_kinect_elapsed_from_pts",
                                    row.sample_index,
                                    "device_elapsed",
                                    time_value_sec=float(yhat),
                                    reference_time_datetime=reference_time,
                                    residual_ms=obs_resid.get(int(row.sample_index)),
                                    notes="derived_from_pts",
                                )
                            )
                        reports[f"rgb_kinect_elapsed_from_pts_{subject_id}_{run_id}"] = pd.DataFrame(
                            {
                                "metric": ["num_fit_points", "slope", "intercept", "rmse_ms"],
                                "value": [used, a, b, float(np.sqrt(np.nanmean(residuals_ms**2)))],
                            }
                        )

            # Sparse SmartCup OS observations and derived dense wallclock model.
            if "smartcup_os_time" in group.columns:
                group_os = group.copy()
                group_os["_smartcup_epoch"] = group_os["smartcup_os_time"].map(datetime_to_epoch_seconds)
                obs = group_os.dropna(subset=["_smartcup_epoch"])
                if not obs.empty:
                    model_rows.append(
                        self._timeline_model_row(
                            subject_id,
                            run_id,
                            device_type,
                            "rgb_smartcup_os_observed",
                            "Observed Host 1 SmartCup OS timestamps",
                            "identity_observed",
                            "smartcup_os_time",
                            {"dense": False, "num_observed": int(len(obs))},
                        )
                    )
                    for row in obs.itertuples(index=False):
                        time_rows.append(
                            self._time_row(
                                subject_id,
                                run_id,
                                device_type,
                                "rgb_smartcup_os_observed",
                                row.sample_index,
                                "observed_wallclock",
                                time_value_datetime=format_datetime(row.smartcup_os_time),
                                reference_time_datetime=reference_time,
                            )
                        )
                    fit = _fit_linear(group_os["pts_sec"], group_os["_smartcup_epoch"])
                    if fit is not None:
                        a, b, residuals_ms, used = fit
                        model_rows.append(
                            self._timeline_model_row(
                                subject_id,
                                run_id,
                                device_type,
                                "rgb_smartcup_os_from_pts",
                                "Derived Host 1 SmartCup OS wallclock from PTS",
                                "regressed_from_pts",
                                "pts_sec + observed smartcup_os_time",
                                {
                                    "dense": True,
                                    "slope": a,
                                    "intercept_epoch_seconds": b,
                                    "num_fit_points": int(used),
                                    "residual_unit": "ms",
                                },
                                "Derived estimate; does not recover original missing SmartCup OS timestamps.",
                            )
                        )
                        pred_epoch = a * group_os["pts_sec"].astype(float) + b
                        obs_resid = _residual_lookup(group_os, "_smartcup_epoch", pred_epoch, scale=1000.0)
                        for row, yhat in zip(group_os.itertuples(index=False), pred_epoch):
                            time_rows.append(
                                self._time_row(
                                    subject_id,
                                    run_id,
                                    device_type,
                                    "rgb_smartcup_os_from_pts",
                                    row.sample_index,
                                    "estimated_wallclock",
                                    time_value_datetime=epoch_seconds_to_datetime_str(float(yhat)),
                                    reference_time_datetime=reference_time,
                                    residual_ms=obs_resid.get(int(row.sample_index)),
                                    notes="derived_from_pts",
                                )
                            )
                        reports[f"rgb_smartcup_os_from_pts_{subject_id}_{run_id}"] = pd.DataFrame(
                            {
                                "metric": ["num_fit_points", "slope", "intercept_epoch_seconds", "rmse_ms"],
                                "value": [used, a, b, float(np.sqrt(np.nanmean(residuals_ms**2)))],
                            }
                        )

        return (
            align_to_spec("RUN_TIMELINE_MODEL", pd.DataFrame(model_rows)),
            align_to_spec("SAMPLE_TIME_ESTIMATE", pd.DataFrame(time_rows)),
            reports,
        )

    def _radar_pc_timeline_tables(self, pc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
        model_rows: list[dict[str, Any]] = []
        time_rows: list[dict[str, Any]] = []
        reports: dict[str, pd.DataFrame] = {}
        device_runs = self._device_runs().set_index(["subject_id", "run_id", "device_type"])

        for (subject_id, run_id), group in pc.groupby(["subject_id", "run_id"], sort=False):
            device_type = "radar_pc"
            run_key = (subject_id, run_id, device_type)
            reference_time = ""
            if run_key in device_runs.index:
                reference_time = device_runs.loc[run_key, "start_wallclock_est"]

            model_rows.append(
                self._timeline_model_row(
                    subject_id,
                    run_id,
                    device_type,
                    "radar_pc_observed_wallclock",
                    "Observed radar PC parser-completion wallclock",
                    "identity_observed",
                    "observed_wallclock",
                    {"dense": True},
                )
            )
            group = group.copy()
            group["_observed_epoch"] = group["observed_wallclock"].map(datetime_to_epoch_seconds)
            for row in group.itertuples(index=False):
                time_rows.append(
                    self._time_row(
                        subject_id,
                        run_id,
                        device_type,
                        "radar_pc_observed_wallclock",
                        row.sample_index,
                        "observed_wallclock",
                        time_value_datetime=format_datetime(row.observed_wallclock),
                        reference_time_datetime=reference_time,
                    )
                )
            fit = _fit_linear(group["sample_index"].astype(float), group["_observed_epoch"])
            if fit is not None:
                a, b, residuals_ms, used = fit
                model_rows.append(
                    self._timeline_model_row(
                        subject_id,
                        run_id,
                        device_type,
                        "radar_pc_linear_from_index",
                        "Radar PC latent acquisition wallclock from frame index",
                        "regressed_observed_vs_index",
                        "observed_wallclock + sample_index",
                        {
                            "dense": True,
                            "slope_seconds_per_sample": a,
                            "intercept_epoch_seconds": b,
                            "num_fit_points": int(used),
                            "residual_unit": "ms",
                        },
                        "Smoothed timeline; observed residuals reflect parser/buffer jitter plus fit error.",
                    )
                )
                pred_epoch = a * group["sample_index"].astype(float) + b
                obs_resid = _residual_lookup(group, "_observed_epoch", pred_epoch, scale=1000.0)
                for row, yhat in zip(group.itertuples(index=False), pred_epoch):
                    time_rows.append(
                        self._time_row(
                            subject_id,
                            run_id,
                            device_type,
                            "radar_pc_linear_from_index",
                            row.sample_index,
                            "latent_acquisition_time",
                            time_value_datetime=epoch_seconds_to_datetime_str(float(yhat)),
                            reference_time_datetime=reference_time,
                            residual_ms=obs_resid.get(int(row.sample_index)),
                            notes="linear_fit_observed_wallclock_vs_sample_index",
                        )
                    )
                reports[f"radar_pc_linear_from_index_{subject_id}_{run_id}"] = pd.DataFrame(
                    {
                        "metric": ["num_fit_points", "seconds_per_sample", "fps_est", "intercept_epoch_seconds", "rmse_ms"],
                        "value": [used, a, 1.0 / a if a else np.nan, b, float(np.sqrt(np.nanmean(residuals_ms**2)))],
                    }
                )

        return (
            align_to_spec("RUN_TIMELINE_MODEL", pd.DataFrame(model_rows)),
            align_to_spec("SAMPLE_TIME_ESTIMATE", pd.DataFrame(time_rows)),
            reports,
        )

    def _table_counts(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"table": name, "rows": int(len(df))} for name, df in tables.items()]
        ).sort_values("table")

    def _device_run_summary(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        runs = tables.get("DEVICE_RUN", TABLE_SPECS["DEVICE_RUN"].empty())
        samples = tables.get("RUN_SAMPLE", TABLE_SPECS["RUN_SAMPLE"].empty())
        if runs.empty:
            return pd.DataFrame()
        counts = (
            samples.groupby(["subject_id", "run_id", "device_type"]).size().reset_index(name="sample_count")
            if not samples.empty
            else pd.DataFrame(columns=["subject_id", "run_id", "device_type", "sample_count"])
        )
        out = runs.merge(counts, on=["subject_id", "run_id", "device_type"], how="left")
        out["sample_count"] = out["sample_count"].fillna(0).astype(int)
        return out

    def _asset_path_warnings(self, run_assets: pd.DataFrame) -> pd.DataFrame:
        if run_assets.empty:
            return pd.DataFrame(columns=["asset_id", "asset_ref", "warning"])
        rows = []
        for row in run_assets.itertuples(index=False):
            if is_probably_absolute_path(row.asset_ref):
                rows.append({"asset_id": row.asset_id, "asset_ref": row.asset_ref, "warning": "asset_ref looks absolute"})
        return pd.DataFrame(rows, columns=["asset_id", "asset_ref", "warning"])


def _fit_linear(x: pd.Series, y: pd.Series) -> tuple[float, float, np.ndarray, int] | None:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2 or len(np.unique(x_arr[mask])) < 2:
        return None
    a, b = np.polyfit(x_arr[mask], y_arr[mask], deg=1)
    pred = a * x_arr[mask] + b
    residuals_ms = (y_arr[mask] - pred) * 1000.0
    return float(a), float(b), residuals_ms, int(mask.sum())


def _residual_lookup(group: pd.DataFrame, observed_col: str, predicted: pd.Series | np.ndarray, *, scale: float) -> dict[int, float]:
    observed = pd.to_numeric(group[observed_col], errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(predicted, dtype=float)
    sample_indices = group["sample_index"].astype(int).to_numpy()
    mask = np.isfinite(observed) & np.isfinite(pred)
    return {int(idx): float((obs - yhat) * scale) for idx, obs, yhat in zip(sample_indices[mask], observed[mask], pred[mask])}
