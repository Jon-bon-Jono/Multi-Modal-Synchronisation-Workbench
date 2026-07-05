"""Audit artifact metadata and bundle files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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
        return pd.DataFrame(issues, columns=["severity", "table", "subject_id", "run_id", "device_type", "sample_index", "artifact_role", "artifact_ref", "issue"])

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
        keys = ["subject_id", "run_id", "device_type", "sample_index"]
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
        "artifact_role": str(getattr(row, "artifact_role", getattr(row, "asset_role", ""))) if row is not None else "",
        "artifact_ref": str(getattr(row, "artifact_ref", getattr(row, "asset_ref", ""))) if row is not None else "",
        "issue": issue,
    }
