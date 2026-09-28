"""Service-layer access to sample payload artifacts.

Future frontends should call this service rather than reading temporary zst files
or artifact bundles directly.
"""
from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sync_workbench.core.tables import align_to_spec
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
        self._bounds_cache: dict[tuple[str, str, str], tuple[int, int]] = {}

    def get_sample_artifact_rows(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        sample_index: int,
        *,
        point_cloud_version_id: str | None = None,
    ) -> pd.DataFrame:
        return self._sample_rows("SAMPLE_ARTIFACT", subject_id, run_id, device_type, sample_index, point_cloud_version_id)

    def _sample_rows(self, table, subject, run, device, sample, version):
        with closing(sqlite3.connect(self.store.path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            where = "subject_id=? AND run_id=? AND device_type=? AND sample_index=?"
            params = [subject, run, device, int(sample)]
            if device == "radar_pc":
                version = version or "online_original"
            elif device == "radar_raw" and version is None:
                if "point_cloud_version_id" in cols:
                    registered = {r[0] for r in conn.execute("SELECT point_cloud_version_id FROM POINT_CLOUD_VERSION WHERE subject_id=? AND run_id=? AND device_type=?", params[:3])}
                    if registered - {"raw_legacy"}:
                        raise ValueError("Select point_cloud_version_id explicitly for offline raw payloads")
                version = "raw_legacy"
            elif device not in {"radar_pc", "radar_raw"}:
                if version:
                    raise ValueError("Point-cloud versions only apply to radar payloads")
                version = ""
            if "point_cloud_version_id" in cols:
                if device == "radar_raw" and version != "raw_legacy":
                    found = conn.execute("SELECT 1 FROM POINT_CLOUD_VERSION WHERE subject_id=? AND run_id=? AND device_type=? AND point_cloud_version_id=?", (*params[:3], version)).fetchone()
                    if found is None:
                        raise KeyError("Point-cloud version is not registered for this acquisition")
                where += " AND point_cloud_version_id=?"
                params.append(version)
            rows = align_to_spec(table, pd.read_sql_query(f'SELECT * FROM "{table}" WHERE ' + where, conn, params=params))
            return rows[rows.point_cloud_version_id == version].copy()

    def list_point_cloud_versions(self, subject_id, run_id, device_type):
        with closing(sqlite3.connect(self.store.path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            return pd.read_sql_query("SELECT * FROM POINT_CLOUD_VERSION WHERE subject_id=? AND run_id=? AND device_type=? ORDER BY point_cloud_version_id", conn, params=(subject_id, run_id, device_type))

    def get_sample_summary(self, subject_id, run_id, device_type, sample_index, *, point_cloud_version_id=None):
        rows = self._sample_rows("SAMPLE_SUMMARY", subject_id, run_id, device_type, sample_index, point_cloud_version_id)
        if len(rows) > 1:
            raise ValueError("Duplicate versioned sample summary")
        return {} if rows.empty else rows.iloc[0].to_dict()

    def clear_cache(self):
        self._npz_cache.clear()
        self._jsonl_cache.clear()
        self._bounds_cache.clear()

    def available_payload_roles(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        sample_index: int,
        *,
        point_cloud_version_id: str | None = None,
    ) -> list[str]:
        rows = self.get_sample_artifact_rows(subject_id, run_id, device_type, sample_index, point_cloud_version_id=point_cloud_version_id)
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
        *,
        point_cloud_version_id: str | None = None,
    ) -> Any:
        rows = self.get_sample_artifact_rows(subject_id, run_id, device_type, sample_index, point_cloud_version_id=point_cloud_version_id)
        rows = rows[rows["artifact_role"].astype(str) == str(artifact_role)]
        if rows.empty:
            raise KeyError(
                f"No artifact role {artifact_role!r} for {subject_id}/{run_id}/{device_type}/sample {sample_index}."
            )
        if len(rows) != 1:
            raise ValueError("Duplicate versioned sample artifact role")
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
        point_cloud_version_id: str | None = None,
    ) -> dict[str, Any]:
        rows = self.get_sample_artifact_rows(subject_id, run_id, device_type, sample_index, point_cloud_version_id=point_cloud_version_id)
        if roles is not None:
            wanted = set(map(str, roles))
            rows = rows[rows["artifact_role"].astype(str).isin(wanted)]
        if rows.artifact_role.duplicated().any():
            raise ValueError("Duplicate versioned sample artifact role")
        out: dict[str, Any] = {}
        for row in rows.itertuples(index=False):
            out[str(row.artifact_role)] = self._read_payload(row)
        return out

    def get_ragged_payload_window(
        self,
        subject_id: str,
        run_id: str,
        device_type: str,
        center_sample_index: int,
        artifact_role: str,
        *,
        radius: int = 0,
        point_cloud_version_id: str | None = None,
        min_sample_index: int | None = None,
        max_sample_index: int | None = None,
    ) -> Any:
        """Read a contiguous sample-index window from a ragged NPZ bundle."""
        radius = max(0, int(radius or 0))
        center = int(center_sample_index)
        start_sample = center - radius
        end_sample = center + radius

        if min_sample_index is not None:
            start_sample = max(int(min_sample_index), start_sample)

        if max_sample_index is not None:
            end_sample = min(int(max_sample_index), end_sample)

        rows = self.get_sample_artifact_rows(subject_id, run_id, device_type, center, point_cloud_version_id=point_cloud_version_id)
        rows = rows[rows["artifact_role"].astype(str) == str(artifact_role)]

        if rows.empty:
            raise KeyError(
                f"No artifact role {artifact_role!r} for "
                f"{subject_id}/{run_id}/{device_type}/sample {center}."
            )

        if len(rows) != 1:
            raise ValueError("Duplicate versioned sample artifact role")
        row = rows.iloc[0]
        artifact_format = str(getattr(row, "artifact_format", ""))

        if artifact_format != "ragged_npz":
            if radius == 0:
                return self._read_payload(row)
            raise ValueError(
                f"Window reads only support ragged_npz artifacts, got {artifact_format!r}."
            )

        path = self.artifact_store.path_for_ref(str(getattr(row, "artifact_ref")))

        reader = self._npz_cache.get(path)
        if reader is None:
            reader = RaggedNpzReader(path)
            self._npz_cache[path] = reader

        if device_type == "radar_raw" and point_cloud_version_id not in (None, "raw_legacy"):
            run_key = (subject_id, run_id, device_type)
            if run_key not in self._bounds_cache:
                with closing(sqlite3.connect(self.store.path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
                    bounds = conn.execute("SELECT MIN(CAST(sample_index AS INTEGER)), MAX(CAST(sample_index AS INTEGER)) FROM RUN_SAMPLE WHERE subject_id=? AND run_id=? AND device_type=?", run_key).fetchone()
                self._bounds_cache[run_key] = tuple(map(int, bounds))
            bounds = self._bounds_cache[run_key]
            start_sample, end_sample = max(start_sample, bounds[0]), min(end_sample, bounds[1])
            left = np.searchsorted(reader.sample_index, start_sample, side="left")
            right = np.searchsorted(reader.sample_index, end_sample, side="right")
            if right - left != max(0, end_sample - start_sample + 1):
                raise KeyError("Raw cloud window includes unprocessed frames")
        return reader.get_index_range(start_sample, end_sample)
    
    def get_mapped_pair_payloads(
        self,
        subject_id: str,
        mapping_version_id: str,
        source_sample_index: int,
        *,
        primary_only: bool = True,
        source_roles: list[str] | None = None,
        target_roles: list[str] | None = None,
        source_point_cloud_version_id: str | None = None,
        target_point_cloud_version_id: str | None = None,
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
            "source_payloads": self.get_sample_payloads(**source, roles=source_roles, point_cloud_version_id=source_point_cloud_version_id),
            "target_payloads": self.get_sample_payloads(**target, roles=target_roles, point_cloud_version_id=target_point_cloud_version_id),
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
