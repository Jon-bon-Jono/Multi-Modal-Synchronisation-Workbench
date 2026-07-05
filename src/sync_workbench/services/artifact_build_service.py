"""Build artifact bundles from temporary ingestion payload columns."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sync_workbench.core.ids import asset_id
from sync_workbench.core.tables import TABLE_SPECS, align_to_spec
from sync_workbench.core.time_utils import utc_now_str
from sync_workbench.ingestion.temp_package import TempPackage
from sync_workbench.storage.artifact_store import ArtifactStore
from sync_workbench.storage.jsonl_index import IndexedJsonlWriter
from sync_workbench.storage.ragged_npz import RaggedNpzWriter
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


@dataclass
class ArtifactBuildResult:
    run_assets: pd.DataFrame
    sample_artifacts: pd.DataFrame
    sample_summary: pd.DataFrame
    reports: dict[str, pd.DataFrame] = field(default_factory=dict)


class ArtifactBuildService:
    """Create run-level payload bundles and canonical metadata references."""

    storage_key = "artifact_store"

    def build_from_temp_package(
        self,
        input_temp: str | Path,
        sqlite_path: str | Path,
        artifact_root: str | Path,
        *,
        overwrite: bool = False,
        subject_id: str | None = None,
        devices: list[str] | None = None,
    ) -> ArtifactBuildResult:
        package = TempPackage.read(input_temp)
        store = SQLiteCoreStore(sqlite_path)
        artifact_store = ArtifactStore(artifact_root)
        created_at = utc_now_str()
        allowed_devices = set(devices or ["kinect_rgb", "radar_pc"])

        run_assets: list[dict[str, Any]] = []
        sample_artifacts: list[dict[str, Any]] = []
        sample_summary: list[dict[str, Any]] = []
        report_rows: list[dict[str, Any]] = []

        if "kinect_rgb" in allowed_devices and package.rgb_samples is not None:
            rgb = _prepare_samples(package.rgb_samples, "kinect_rgb")
            if subject_id is not None:
                rgb = rgb[rgb["subject_id"].astype(str) == str(subject_id)]
            for (subj, run_id), group in rgb.groupby(["subject_id", "run_id"], sort=False):
                assets, artifacts, summary, rows = self._build_rgb_run(
                    artifact_store, group, str(subj), str(run_id), created_at, overwrite=overwrite
                )
                run_assets.extend(assets)
                sample_artifacts.extend(artifacts)
                sample_summary.extend(summary)
                report_rows.extend(rows)

        if "radar_pc" in allowed_devices and package.radar_pc_samples is not None:
            pc = _prepare_samples(package.radar_pc_samples, "radar_pc")
            if subject_id is not None:
                pc = pc[pc["subject_id"].astype(str) == str(subject_id)]
            for (subj, run_id), group in pc.groupby(["subject_id", "run_id"], sort=False):
                assets, artifacts, summary, rows = self._build_radar_pc_run(
                    artifact_store, group, str(subj), str(run_id), created_at, overwrite=overwrite
                )
                run_assets.extend(assets)
                sample_artifacts.extend(artifacts)
                sample_summary.extend(summary)
                report_rows.extend(rows)

        run_assets_df = align_to_spec("RUN_ASSET", pd.DataFrame(run_assets)) if run_assets else TABLE_SPECS["RUN_ASSET"].empty()
        sample_artifacts_df = (
            align_to_spec("SAMPLE_ARTIFACT", pd.DataFrame(sample_artifacts))
            if sample_artifacts
            else TABLE_SPECS["SAMPLE_ARTIFACT"].empty()
        )
        sample_summary_df = (
            align_to_spec("SAMPLE_SUMMARY", pd.DataFrame(sample_summary))
            if sample_summary
            else TABLE_SPECS["SAMPLE_SUMMARY"].empty()
        )

        if not run_assets_df.empty:
            store.replace_or_append_without_key_conflicts("RUN_ASSET", run_assets_df)
        if not sample_artifacts_df.empty:
            store.replace_or_append_without_key_conflicts("SAMPLE_ARTIFACT", sample_artifacts_df)
        if not sample_summary_df.empty:
            store.replace_or_append_without_key_conflicts("SAMPLE_SUMMARY", sample_summary_df)

        reports = {
            "artifact_build_summary": pd.DataFrame(report_rows),
            "table_counts": pd.DataFrame(
                [
                    {"table": "RUN_ASSET", "rows_written": int(len(run_assets_df))},
                    {"table": "SAMPLE_ARTIFACT", "rows_written": int(len(sample_artifacts_df))},
                    {"table": "SAMPLE_SUMMARY", "rows_written": int(len(sample_summary_df))},
                ]
            ),
        }
        return ArtifactBuildResult(run_assets_df, sample_artifacts_df, sample_summary_df, reports)

    def _build_rgb_run(
        self,
        artifact_store: ArtifactStore,
        group: pd.DataFrame,
        subject_id: str,
        run_id: str,
        created_at: str,
        *,
        overwrite: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        device_type = "kinect_rgb"
        run_assets: list[dict[str, Any]] = []
        sample_artifacts: list[dict[str, Any]] = []
        report_rows: list[dict[str, Any]] = []

        payload_specs = [
            ("pose2d", "pose2d_bundle", "pose2d.npz", (26, 3), "float64"),
            ("conf2d", "conf2d_bundle", "conf2d.npz", tuple(), "float64"),
            ("pose3d", "pose3d_bundle", "pose3d.npz", (32, 4), "float64"),
        ]
        manifest_rows: list[dict[str, Any]] = []

        for column, asset_role, filename, tail_shape, dtype in payload_specs:
            if column not in group.columns:
                continue
            ref = artifact_store.ref_for(subject_id, run_id, device_type, filename)
            path = artifact_store.path_for_ref(ref)
            _guard_overwrite(path, overwrite)
            payloads = [(int(row.sample_index), _as_payload_array(getattr(row, column), tail_shape, dtype)) for row in group.itertuples(index=False)]
            infos = RaggedNpzWriter.write(path, payloads, tail_shape=tail_shape, dtype=dtype)
            aid = asset_id(subject_id, run_id, device_type, asset_role)
            run_assets.append(self._run_asset_row(aid, subject_id, run_id, device_type, asset_role, ref, f"{column} run-level ragged NPZ bundle"))
            for info in infos:
                row = self._sample_artifact_row(
                    subject_id,
                    run_id,
                    device_type,
                    info.sample_index,
                    column,
                    aid,
                    ref,
                    {"sample_index": info.sample_index},
                    "ragged_npz",
                    info.shape,
                    info.dtype,
                    info.nbytes,
                    created_at,
                    "payload stored in run-level ragged NPZ bundle",
                )
                sample_artifacts.append(row)
                manifest_rows.append(row)
            report_rows.append({"subject_id": subject_id, "run_id": run_id, "device_type": device_type, "artifact_role": asset_role, "samples": len(infos), "artifact_ref": ref})

        if "activity" in group.columns:
            ref = artifact_store.ref_for(subject_id, run_id, device_type, "activity.jsonl")
            path = artifact_store.path_for_ref(ref)
            _guard_overwrite(path, overwrite)
            infos = IndexedJsonlWriter.write(path, [(int(r.sample_index), r.activity) for r in group.itertuples(index=False)])
            aid = asset_id(subject_id, run_id, device_type, "activity_jsonl")
            run_assets.append(self._run_asset_row(aid, subject_id, run_id, device_type, "activity_jsonl", ref, "activity per-sample JSONL bundle"))
            for info in infos:
                payload = group.loc[group["sample_index"].astype(int) == info.sample_index, "activity"].iloc[0]
                row = self._sample_artifact_row(
                    subject_id,
                    run_id,
                    device_type,
                    info.sample_index,
                    "activity",
                    aid,
                    ref,
                    {"byte_offset": info.byte_offset, "nbytes": info.nbytes},
                    "jsonl",
                    _payload_shape(payload),
                    "json",
                    info.nbytes,
                    created_at,
                    "payload stored in indexed JSONL bundle",
                )
                sample_artifacts.append(row)
                manifest_rows.append(row)
            report_rows.append({"subject_id": subject_id, "run_id": run_id, "device_type": device_type, "artifact_role": "activity_jsonl", "samples": len(infos), "artifact_ref": ref})

        manifest_asset = self._write_manifest(artifact_store, subject_id, run_id, device_type, manifest_rows, overwrite)
        if manifest_asset is not None:
            run_assets.append(manifest_asset)

        summary_rows = [_rgb_summary_row(row, created_at) for row in group.itertuples(index=False)]
        return run_assets, sample_artifacts, summary_rows, report_rows

    def _build_radar_pc_run(
        self,
        artifact_store: ArtifactStore,
        group: pd.DataFrame,
        subject_id: str,
        run_id: str,
        created_at: str,
        *,
        overwrite: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        device_type = "radar_pc"
        run_assets: list[dict[str, Any]] = []
        sample_artifacts: list[dict[str, Any]] = []
        report_rows: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []

        if "points" in group.columns:
            ref = artifact_store.ref_for(subject_id, run_id, device_type, "points.npz")
            path = artifact_store.path_for_ref(ref)
            _guard_overwrite(path, overwrite)
            payloads = [(int(row.sample_index), _as_payload_array(row.points, (6,), "float64")) for row in group.itertuples(index=False)]
            infos = RaggedNpzWriter.write(path, payloads, tail_shape=(6,), dtype="float64")
            aid = asset_id(subject_id, run_id, device_type, "radar_points_bundle")
            run_assets.append(self._run_asset_row(aid, subject_id, run_id, device_type, "radar_points_bundle", ref, "radar point-cloud ragged NPZ bundle"))
            for info in infos:
                row = self._sample_artifact_row(
                    subject_id,
                    run_id,
                    device_type,
                    info.sample_index,
                    "radar_points",
                    aid,
                    ref,
                    {"sample_index": info.sample_index},
                    "ragged_npz",
                    info.shape,
                    info.dtype,
                    info.nbytes,
                    created_at,
                    "payload stored in run-level ragged NPZ bundle",
                )
                sample_artifacts.append(row)
                manifest_rows.append(row)
            report_rows.append({"subject_id": subject_id, "run_id": run_id, "device_type": device_type, "artifact_role": "radar_points_bundle", "samples": len(infos), "artifact_ref": ref})

        manifest_asset = self._write_manifest(artifact_store, subject_id, run_id, device_type, manifest_rows, overwrite)
        if manifest_asset is not None:
            run_assets.append(manifest_asset)

        summary_rows = [_radar_pc_summary_row(row, created_at) for row in group.itertuples(index=False)]
        return run_assets, sample_artifacts, summary_rows, report_rows

    def _write_manifest(
        self,
        artifact_store: ArtifactStore,
        subject_id: str,
        run_id: str,
        device_type: str,
        manifest_rows: list[dict[str, Any]],
        overwrite: bool,
    ) -> dict[str, Any] | None:
        if not manifest_rows:
            return None
        manifest = pd.DataFrame(manifest_rows)
        parquet_ref = artifact_store.ref_for(subject_id, run_id, device_type, "sample_payload_manifest.parquet")
        parquet_path = artifact_store.path_for_ref(parquet_ref)
        _guard_overwrite(parquet_path, overwrite)
        aid = asset_id(subject_id, run_id, device_type, "sample_payload_manifest")
        try:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_parquet(parquet_path, index=False)
            return self._run_asset_row(aid, subject_id, run_id, device_type, "sample_payload_manifest", parquet_ref, "per-sample payload manifest parquet")
        except ImportError:
            csv_ref = artifact_store.ref_for(subject_id, run_id, device_type, "sample_payload_manifest.csv")
            csv_path = artifact_store.path_for_ref(csv_ref)
            _guard_overwrite(csv_path, overwrite)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(csv_path, index=False)
            return self._run_asset_row(aid, subject_id, run_id, device_type, "sample_payload_manifest", csv_ref, "per-sample payload manifest CSV fallback; install pyarrow for parquet")

    def _run_asset_row(self, aid: str, subject_id: str, run_id: str, device_type: str, asset_role: str, ref: str, notes: str) -> dict[str, Any]:
        return {
            "asset_id": aid,
            "subject_id": subject_id,
            "run_id": run_id,
            "device_type": device_type,
            "asset_role": asset_role,
            "storage_key": self.storage_key,
            "asset_ref": ref,
            "notes": notes,
        }

    def _sample_artifact_row(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        sample_index: int,
        role: str,
        aid: str,
        ref: str,
        member_key: dict[str, Any],
        artifact_format: str,
        shape: tuple[int, ...] | list[int],
        dtype: str,
        nbytes: int,
        created_at: str,
        notes: str,
    ) -> dict[str, Any]:
        return {
            "subject_id": subject_id,
            "run_id": run_id,
            "device_type": device_type,
            "sample_index": int(sample_index),
            "artifact_role": role,
            "artifact_id": aid,
            "storage_key": self.storage_key,
            "artifact_ref": ref,
            "artifact_member_key": json.dumps(member_key, sort_keys=True),
            "artifact_format": artifact_format,
            "payload_shape": json.dumps(list(shape)),
            "payload_dtype": str(dtype),
            "payload_bytes": int(nbytes),
            "created_at": created_at,
            "notes": notes,
        }


def _prepare_samples(df: pd.DataFrame, device_type: str) -> pd.DataFrame:
    out = df.copy()
    out["device_type"] = device_type
    out = out.sort_values(["subject_id", "run_id", "frame_number"]).reset_index(drop=False)
    out["sample_index"] = out.groupby(["subject_id", "run_id"]).cumcount().astype(int)
    return out


def _guard_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {path}. Pass overwrite=True or --overwrite to replace it.")


def _as_payload_array(value: Any, tail_shape: tuple[int, ...], dtype: str) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    expected_ndim = 1 + len(tail_shape)
    if arr.size == 0:
        return np.empty((0, *tail_shape), dtype=dtype)
    if arr.ndim != expected_ndim or tuple(arr.shape[1:]) != tuple(tail_shape):
        raise ValueError(f"Payload has shape {arr.shape}; expected (*, {tail_shape}).")
    return arr


def _payload_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict):
        return (len(value),)
    if isinstance(value, (list, tuple)):
        return (len(value),)
    if value is None:
        return tuple()
    return tuple(np.asarray(value).shape)


def _first_dim(value: Any, tail_shape: tuple[int, ...], dtype: str = "float64") -> int:
    try:
        return int(_as_payload_array(value, tail_shape, dtype).shape[0])
    except Exception:
        return 0


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _rgb_summary_row(row: Any, created_at: str) -> dict[str, Any]:
    pose2d_n = _first_dim(getattr(row, "pose2d", None), (26, 3))
    conf2d_n = _first_dim(getattr(row, "conf2d", None), tuple())
    pose3d_n = _first_dim(getattr(row, "pose3d", None), (32, 4))
    return {
        "subject_id": str(row.subject_id),
        "run_id": str(row.run_id),
        "device_type": "kinect_rgb",
        "sample_index": int(row.sample_index),
        "num_people": _as_int(getattr(row, "num_people", 0)),
        "num_2d": _as_int(getattr(row, "num_2d", pose2d_n)),
        "num_3d": _as_int(getattr(row, "num_3d", pose3d_n)),
        "point_count": pd.NA,
        "point_count_filtered": pd.NA,
        "has_pose2d": bool(pose2d_n > 0),
        "has_conf2d": bool(conf2d_n > 0),
        "has_pose3d": bool(pose3d_n > 0),
        "has_activity": hasattr(row, "activity"),
        "has_points": False,
        "created_at": created_at,
        "notes": "derived from rgb_samples payload columns",
    }


def _radar_pc_summary_row(row: Any, created_at: str) -> dict[str, Any]:
    point_n = _first_dim(getattr(row, "points", None), (6,))
    return {
        "subject_id": str(row.subject_id),
        "run_id": str(row.run_id),
        "device_type": "radar_pc",
        "sample_index": int(row.sample_index),
        "num_people": pd.NA,
        "num_2d": pd.NA,
        "num_3d": pd.NA,
        "point_count": _as_int(getattr(row, "point_count", point_n)),
        "point_count_filtered": _as_int(getattr(row, "point_count_filtered", 0)),
        "has_pose2d": False,
        "has_conf2d": False,
        "has_pose3d": False,
        "has_activity": False,
        "has_points": bool(point_n > 0),
        "created_at": created_at,
        "notes": "derived from radar_pc_samples payload columns",
    }
