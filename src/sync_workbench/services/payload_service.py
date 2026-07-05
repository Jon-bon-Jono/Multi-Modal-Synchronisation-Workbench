"""Service-layer access to sample payload artifacts.

Future frontends should call this service rather than reading temporary zst files
or artifact bundles directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sync_workbench.storage.artifact_store import ArtifactStore
from sync_workbench.storage.jsonl_index import IndexedJsonlReader
from sync_workbench.storage.ragged_npz import RaggedNpzReader
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


class PayloadService:
    def __init__(self, sqlite_path: str | Path, artifact_root: str | Path):
        self.store = SQLiteCoreStore(sqlite_path)
        self.artifact_store = ArtifactStore(artifact_root)
        self._npz_cache: dict[Path, RaggedNpzReader] = {}
        self._jsonl_cache: dict[Path, IndexedJsonlReader] = {}

    def get_sample_artifact_rows(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        sample_index: int,
    ) -> pd.DataFrame:
        artifacts = self.store.read_table("SAMPLE_ARTIFACT")
        if artifacts.empty:
            return artifacts
        mask = (
            (artifacts["subject_id"].astype(str) == str(subject_id))
            & (artifacts["run_id"].astype(str) == str(run_id))
            & (artifacts["device_type"].astype(str) == str(device_type))
            & (pd.to_numeric(artifacts["sample_index"], errors="coerce").astype("Int64") == int(sample_index))
        )
        return artifacts.loc[mask].copy()

    def available_payload_roles(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        sample_index: int,
    ) -> list[str]:
        rows = self.get_sample_artifact_rows(subject_id, run_id, device_type, sample_index)
        if rows.empty:
            return []
        return sorted(rows["artifact_role"].astype(str).unique())

    def get_payload(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        sample_index: int,
        artifact_role: str,
    ) -> Any:
        rows = self.get_sample_artifact_rows(subject_id, run_id, device_type, sample_index)
        rows = rows[rows["artifact_role"].astype(str) == str(artifact_role)]
        if rows.empty:
            raise KeyError(
                f"No artifact role {artifact_role!r} for {subject_id}/{run_id}/{device_type}/sample {sample_index}."
            )
        row = rows.iloc[0]
        return self._read_payload(row)

    def get_sample_payloads(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        sample_index: int,
        *,
        roles: list[str] | None = None,
    ) -> dict[str, Any]:
        rows = self.get_sample_artifact_rows(subject_id, run_id, device_type, sample_index)
        if roles is not None:
            wanted = set(map(str, roles))
            rows = rows[rows["artifact_role"].astype(str).isin(wanted)]
        out: dict[str, Any] = {}
        for row in rows.itertuples(index=False):
            out[str(row.artifact_role)] = self._read_payload(row)
        return out

    def get_mapped_pair_payloads(
        self,
        subject_id: str,
        mapping_version_id: str,
        source_sample_index: int,
        *,
        primary_only: bool = True,
        source_roles: list[str] | None = None,
        target_roles: list[str] | None = None,
    ) -> dict[str, Any]:
        mappings = self.store.read_table("SAMPLE_MAPPING")
        if mappings.empty:
            raise KeyError("SAMPLE_MAPPING is empty.")
        mask = (
            (mappings["subject_id"].astype(str) == str(subject_id))
            & (mappings["mapping_version_id"].astype(str) == str(mapping_version_id))
            & (pd.to_numeric(mappings["source_sample_index"], errors="coerce").astype("Int64") == int(source_sample_index))
        )
        rows = mappings.loc[mask].copy()
        if rows.empty:
            raise KeyError(
                f"No mapping rows for subject={subject_id}, mapping_version={mapping_version_id}, source_sample={source_sample_index}."
            )
        if primary_only:
            primary = rows[rows["is_primary"].astype(str).str.lower().isin({"true", "1"})]
            if not primary.empty:
                rows = primary
        if "rank" in rows.columns:
            rows["__rank"] = pd.to_numeric(rows["rank"], errors="coerce").fillna(1_000_000)
            rows = rows.sort_values("__rank")
        selected = rows.iloc[0]
        source = {
            "subject_id": str(selected.subject_id),
            "run_id": str(selected.source_run_id),
            "device_type": str(selected.source_device_type),
            "sample_index": int(selected.source_sample_index),
        }
        target = {
            "subject_id": str(selected.subject_id),
            "run_id": str(selected.target_run_id),
            "device_type": str(selected.target_device_type),
            "sample_index": int(selected.target_sample_index),
        }
        return {
            "mapping": selected.drop(labels=["__rank"], errors="ignore").to_dict(),
            "source": source,
            "target": target,
            "source_payloads": self.get_sample_payloads(**source, roles=source_roles),
            "target_payloads": self.get_sample_payloads(**target, roles=target_roles),
        }

    def _read_payload(self, row: Any) -> Any:
        artifact_ref = str(getattr(row, "artifact_ref"))
        path = self.artifact_store.path_for_ref(artifact_ref)
        artifact_format = str(getattr(row, "artifact_format", ""))
        member_key_raw = getattr(row, "artifact_member_key", "{}") or "{}"
        member_key = json.loads(member_key_raw) if isinstance(member_key_raw, str) else dict(member_key_raw)

        if artifact_format == "ragged_npz":
            reader = self._npz_cache.get(path)
            if reader is None:
                reader = RaggedNpzReader(path)
                self._npz_cache[path] = reader
            return reader.get(int(member_key.get("sample_index", getattr(row, "sample_index"))))

        if artifact_format == "jsonl":
            reader = self._jsonl_cache.get(path)
            if reader is None:
                reader = IndexedJsonlReader(path)
                self._jsonl_cache[path] = reader
            return reader.read_at(int(member_key["byte_offset"]), int(member_key.get("nbytes", 0)) or None)

        raise ValueError(f"Unsupported artifact_format={artifact_format!r} for {artifact_ref}")
