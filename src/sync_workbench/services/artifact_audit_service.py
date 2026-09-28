"""Audit artifact metadata and bundle files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from sync_workbench.ingestion.raw_point_cloud_package import file_hash

from sync_workbench.assets.asset_refs import is_probably_absolute_path
from sync_workbench.storage.artifact_store import ArtifactStore
from sync_workbench.storage.jsonl_index import IndexedJsonlReader
from sync_workbench.storage.ragged_npz import RaggedNpzReader
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


class ArtifactAuditService:
    def __init__(self, sqlite_path: str | Path, artifact_root: str | Path):
        self.store = SQLiteCoreStore(sqlite_path)
        self.artifact_store = ArtifactStore(artifact_root)

    def audit(self) -> pd.DataFrame:
        issues: list[dict[str, Any]] = []
        self._audit_run_assets(issues)
        self._audit_sample_artifacts(issues)
        self._audit_summary_consistency(issues)
        self._audit_cloud_versions(issues)
        return pd.DataFrame(issues, columns=["severity", "table", "subject_id", "run_id", "device_type", "sample_index", "point_cloud_version_id", "artifact_role", "artifact_ref", "issue"])

    def audit_cloud_versions(self) -> pd.DataFrame:
        """Audit version registration, coverage, payload counts and bundle hashes."""
        issues = []
        self._audit_cloud_versions(issues)
        return pd.DataFrame(issues)

    def _audit_cloud_versions(self, issues):
        versions = self.store.read_table("POINT_CLOUD_VERSION")
        if "POINT_CLOUD_VERSION" not in self.store.list_tables():
            return  # Explicit migration has not been run on this legacy store.
        artifacts = self.store.read_table("SAMPLE_ARTIFACT")
        summaries = self.store.read_table("SAMPLE_SUMMARY")
        keys = ["subject_id", "run_id", "device_type", "point_cloud_version_id"]
        for name, frame in (("SAMPLE_ARTIFACT", artifacts), ("SAMPLE_SUMMARY", summaries)):
            radar = frame[frame.device_type.isin(["radar_pc", "radar_raw"])]
            missing = radar[keys].drop_duplicates().merge(versions[keys], on=keys, how="left", indicator=True)
            for row in missing[missing._merge == "left_only"].itertuples(index=False):
                issues.append(_issue("error", name, row, issue="cloud version is not registered"))
        for version in versions.itertuples(index=False):
            if not str(version.point_cloud_version_id).startswith("raw_") or version.point_cloud_version_id == "raw_legacy":
                continue
            try:
                manifest = json.loads(version.provenance_json)["version"]
                ref = self.artifact_store.path_for_ref(version.artifact_ref)
                if file_hash(ref) != version.artifact_sha256:
                    raise ValueError("bundle checksum mismatch")
                reader = RaggedNpzReader(ref)
                errors = reader.validate()
                if errors:
                    raise ValueError("; ".join(errors))
                if reader.values.dtype != np.float32 or reader.values.shape[1:] != (6,):
                    raise ValueError("raw bundle must have six float32 columns")
                amask = np.ones(len(artifacts), dtype=bool)
                smask = np.ones(len(summaries), dtype=bool)
                for key in keys:
                    amask &= artifacts[key].eq(getattr(version, key)).to_numpy()
                    smask &= summaries[key].eq(getattr(version, key)).to_numpy()
                a, s = artifacts[amask], summaries[smask].copy()
                s["sample_index"] = pd.to_numeric(s.sample_index).astype(int)
                s = s.sort_values("sample_index")
                if s.sample_index.tolist() != list(range(manifest["acquisition_frame_count"])):
                    raise ValueError("summary coverage differs from full acquisition")
                expected = (np.asarray(manifest["source_frame_ids"], dtype=np.int64) - manifest["source_frame_number_base"]).tolist()
                if s.loc[s.point_status == "available", "sample_index"].tolist() != expected or reader.sample_index.tolist() != expected:
                    raise ValueError("available coverage differs from provenance/bundle")
                if sorted(pd.to_numeric(a.sample_index).astype(int)) != expected or not a.artifact_ref.eq(version.artifact_ref).all() or not a.artifact_role.eq("radar_points").all():
                    raise ValueError("artifact references differ from version coverage/bundle")
                for row in a.itertuples(index=False):
                    cloud = reader.get(int(row.sample_index))
                    if (json.loads(row.artifact_member_key).get("sample_index") != int(row.sample_index)
                            or row.artifact_format != "ragged_npz" or row.payload_dtype != "float32"
                            or json.loads(row.payload_shape) != list(cloud.shape)
                            or int(row.payload_bytes) != cloud.nbytes):
                        raise ValueError(f"artifact member metadata differs at sample {row.sample_index}")
                for row in s.itertuples(index=False):
                    if row.point_status == "unprocessed":
                        if pd.notna(row.point_count) or pd.notna(row.point_count_filtered):
                            raise ValueError("unprocessed frame has point counts")
                        continue
                    points = reader.get(row.sample_index)
                    if row.point_status != "available" or int(row.point_count) != len(points) or int(row.point_count_filtered) != np.count_nonzero(~np.isin(points[:, 5], [253, 254, 255])):
                        raise ValueError(f"point counts/status differ at sample {row.sample_index}")
                if len(reader.values) != manifest["point_count"]:
                    raise ValueError("bundle point total differs from provenance")
            except Exception as exc:
                issues.append(_issue("error", "POINT_CLOUD_VERSION", version, issue=str(exc)))

    def _audit_run_assets(self, issues: list[dict[str, Any]]) -> None:
        assets = self.store.read_table("RUN_ASSET")
        if assets.empty:
            issues.append(_issue("warning", "RUN_ASSET", issue="RUN_ASSET is empty"))
            return
        artifact_assets = assets[assets.get("storage_key", "").astype(str) == "artifact_store"]
        for row in artifact_assets.itertuples(index=False):
            ref = str(row.asset_ref)
            if is_probably_absolute_path(ref):
                issues.append(_issue("warning", "RUN_ASSET", row, issue="asset_ref appears to be an absolute path"))
            path = self.artifact_store.path_for_ref(ref)
            if not path.exists():
                issues.append(_issue("error", "RUN_ASSET", row, issue=f"artifact file missing: {path}"))

    def _audit_sample_artifacts(self, issues: list[dict[str, Any]]) -> None:
        artifacts = self.store.read_table("SAMPLE_ARTIFACT")
        if artifacts.empty:
            issues.append(_issue("warning", "SAMPLE_ARTIFACT", issue="SAMPLE_ARTIFACT is empty"))
            return

        npz_checked: set[Path] = set()
        for row in artifacts.itertuples(index=False):
            path = self.artifact_store.path_for_ref(str(row.artifact_ref))
            if not path.exists():
                issues.append(_issue("error", "SAMPLE_ARTIFACT", row, issue=f"artifact file missing: {path}"))
                continue
            if str(row.artifact_format) == "ragged_npz" and path not in npz_checked:
                try:
                    reader = RaggedNpzReader(path)
                    for msg in reader.validate():
                        issues.append(_issue("error", "SAMPLE_ARTIFACT", row, issue=f"invalid ragged NPZ: {msg}"))
                except Exception as exc:
                    issues.append(_issue("error", "SAMPLE_ARTIFACT", row, issue=f"cannot read ragged NPZ: {exc}"))
                npz_checked.add(path)
            elif str(row.artifact_format) == "jsonl":
                try:
                    member_key = json.loads(row.artifact_member_key or "{}")
                    IndexedJsonlReader(path).read_at(int(member_key["byte_offset"]), int(member_key.get("nbytes", 0)) or None)
                except Exception as exc:
                    issues.append(_issue("error", "SAMPLE_ARTIFACT", row, issue=f"cannot read JSONL member: {exc}"))

    def _audit_summary_consistency(self, issues: list[dict[str, Any]]) -> None:
        artifacts = self.store.read_table("SAMPLE_ARTIFACT")
        summaries = self.store.read_table("SAMPLE_SUMMARY")
        if artifacts.empty or summaries.empty:
            return
        keys = ["subject_id", "run_id", "device_type", "sample_index", "point_cloud_version_id"]
        artifact_keys = artifacts[keys].drop_duplicates()
        merged = artifact_keys.merge(summaries[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        missing = merged[merged["_merge"] == "left_only"]
        for row in missing.itertuples(index=False):
            issues.append(_issue("warning", "SAMPLE_SUMMARY", row, issue="sample has artifacts but no SAMPLE_SUMMARY row"))


def _issue(severity: str, table: str, row: Any | None = None, *, issue: str) -> dict[str, Any]:
    return {
        "severity": severity,
        "table": table,
        "subject_id": str(getattr(row, "subject_id", "")) if row is not None else "",
        "run_id": str(getattr(row, "run_id", "")) if row is not None else "",
        "device_type": str(getattr(row, "device_type", "")) if row is not None else "",
        "sample_index": getattr(row, "sample_index", "") if row is not None else "",
        "point_cloud_version_id": str(getattr(row, "point_cloud_version_id", "")) if row is not None else "",
        "artifact_role": str(getattr(row, "artifact_role", getattr(row, "asset_role", ""))) if row is not None else "",
        "artifact_ref": str(getattr(row, "artifact_ref", getattr(row, "asset_ref", ""))) if row is not None else "",
        "issue": issue,
    }
