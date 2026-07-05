"""Mapped-pair inspection service for CLI and future GUI workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sync_workbench.services.payload_service import PayloadService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


class PairInspectionService:
    def __init__(self, sqlite_path: str | Path, artifact_root: str | Path):
        self.store = SQLiteCoreStore(sqlite_path)
        self.payloads = PayloadService(sqlite_path, artifact_root)

    def inspect_pair(
        self,
        mapping_version_id: str,
        source_sample_index: int,
        *,
        subject_id: str | None = None,
        primary_only: bool = True,
        include_payloads: bool = False,
    ) -> dict[str, Any]:
        subject = subject_id or self._infer_unique_subject(mapping_version_id)
        pair = self.payloads.get_mapped_pair_payloads(
            subject,
            mapping_version_id,
            source_sample_index,
            primary_only=primary_only,
        )
        source_summary = self._summary_row(**pair["source"])
        target_summary = self._summary_row(**pair["target"])
        out = {
            "mapping": pair["mapping"],
            "source": {**pair["source"], "summary": source_summary, "payload_roles": sorted(pair["source_payloads"].keys())},
            "target": {**pair["target"], "summary": target_summary, "payload_roles": sorted(pair["target_payloads"].keys())},
            "payload_shapes": {
                "source": {k: _describe_payload(v) for k, v in pair["source_payloads"].items()},
                "target": {k: _describe_payload(v) for k, v in pair["target_payloads"].items()},
            },
        }
        if include_payloads:
            out["payloads"] = {"source": pair["source_payloads"], "target": pair["target_payloads"]}
        return out

    def _infer_unique_subject(self, mapping_version_id: str) -> str:
        versions = self.store.read_table("MAPPING_VERSION")
        if versions.empty:
            raise KeyError("MAPPING_VERSION is empty; pass --subject if the database is incomplete.")
        matches = versions[versions["mapping_version_id"].astype(str) == str(mapping_version_id)]
        subjects = sorted(matches["subject_id"].astype(str).unique())
        if len(subjects) != 1:
            raise ValueError(
                f"Could not infer a unique subject for mapping_version_id={mapping_version_id!r}; found {subjects}. Pass --subject."
            )
        return subjects[0]

    def _summary_row(self, subject_id: str, run_id: str, device_type: str, sample_index: int) -> dict[str, Any]:
        summary = self.store.read_table("SAMPLE_SUMMARY")
        if summary.empty:
            return {}
        mask = (
            (summary["subject_id"].astype(str) == str(subject_id))
            & (summary["run_id"].astype(str) == str(run_id))
            & (summary["device_type"].astype(str) == str(device_type))
            & (pd.to_numeric(summary["sample_index"], errors="coerce").astype("Int64") == int(sample_index))
        )
        rows = summary.loc[mask]
        return {} if rows.empty else rows.iloc[0].to_dict()


def _describe_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, np.ndarray):
        return {"type": "ndarray", "shape": list(payload.shape), "dtype": str(payload.dtype), "nbytes": int(payload.nbytes)}
    if isinstance(payload, dict):
        return {"type": "dict", "keys": sorted(map(str, payload.keys()))}
    if isinstance(payload, list):
        return {"type": "list", "len": len(payload)}
    return {"type": type(payload).__name__}
