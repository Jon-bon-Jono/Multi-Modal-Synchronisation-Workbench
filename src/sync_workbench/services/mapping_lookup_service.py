"""Lookup helpers for GUI navigation over SAMPLE_MAPPING rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import TimelineSelection, timeline_numeric_values


class MappingLookupService:
    def __init__(self, sqlite_path: str | Path):
        self.store = SQLiteCoreStore(sqlite_path)
        self._mapping_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._time_cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}

    def get_mapping_context(self, subject_id: str, mapping_version_id: str) -> dict[str, object]:
        versions = self.store.read_table("MAPPING_VERSION")
        rows = versions[
            (versions["subject_id"].astype(str) == str(subject_id))
            & (versions["mapping_version_id"].astype(str) == str(mapping_version_id))
        ]
        if rows.empty:
            raise KeyError(f"Mapping version not found: {subject_id}/{mapping_version_id}")
        return rows.iloc[0].to_dict()

    def map_source_to_target(self, subject_id: str, mapping_version_id: str, source_sample_index: int) -> dict[str, object]:
        mappings = self._mapping_rows(subject_id, mapping_version_id)
        rows = mappings[pd.to_numeric(mappings["source_sample_index"], errors="coerce").astype("Int64") == int(source_sample_index)].copy()
        if rows.empty:
            raise KeyError(f"No mapping row for source sample {source_sample_index} in {subject_id}/{mapping_version_id}")
        return self._best_row(rows).to_dict()

    def map_target_to_source(self, subject_id: str, mapping_version_id: str, target_sample_index: int) -> dict[str, object]:
        mappings = self._mapping_rows(subject_id, mapping_version_id)
        rows = mappings[pd.to_numeric(mappings["target_sample_index"], errors="coerce").astype("Int64") == int(target_sample_index)].copy()
        if rows.empty:
            # Fallback: choose nearest by target sample index if the exact target was not selected.
            mappings["__target_int"] = pd.to_numeric(mappings["target_sample_index"], errors="coerce")
            mappings = mappings[np.isfinite(mappings["__target_int"])].copy()
            if mappings.empty:
                raise KeyError(f"No inverse mapping rows exist for {subject_id}/{mapping_version_id}")
            mappings["__target_dist"] = (mappings["__target_int"] - int(target_sample_index)).abs()
            rows = mappings.sort_values("__target_dist").head(20).copy()
        return self._best_row(rows).drop(labels=["__target_int", "__target_dist"], errors="ignore").to_dict()

    def nearest_sample_by_time(
        self,
        *,
        subject_id: str,
        run_id: str,
        device_type: str,
        timeline_model_id: str,
        time_value: float,
    ) -> int:
        selection = TimelineSelection(subject_id, run_id, device_type, timeline_model_id)
        times = self._select_times(selection)
        values, _ = timeline_numeric_values(times)
        values = pd.to_numeric(values, errors="coerce")
        times = times.copy()
        times["__t"] = values
        times = times[np.isfinite(times["__t"])].copy()
        if times.empty:
            raise KeyError(f"No finite timeline values for {selection}")
        idx = (times["__t"] - float(time_value)).abs().idxmin()
        return int(times.loc[idx, "sample_index"])

    def _mapping_rows(self, subject_id: str, mapping_version_id: str) -> pd.DataFrame:
        cache_key = (str(subject_id), str(mapping_version_id))
        cached = self._mapping_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        mappings = self.store.read_table("SAMPLE_MAPPING")
        if mappings.empty:
            raise KeyError("SAMPLE_MAPPING is empty.")
        rows = mappings[
            (mappings["subject_id"].astype(str) == str(subject_id))
            & (mappings["mapping_version_id"].astype(str) == str(mapping_version_id))
        ].copy()
        if rows.empty:
            raise KeyError(f"No SAMPLE_MAPPING rows for {subject_id}/{mapping_version_id}")
        self._mapping_cache[cache_key] = rows.copy()
        return rows.copy()

    @staticmethod
    def _best_row(rows: pd.DataFrame) -> pd.Series:
        rows = rows.copy()
        rows["__is_primary"] = rows["is_primary"].astype(str).str.lower().isin({"true", "1"})
        rows["__rank"] = pd.to_numeric(rows["rank"], errors="coerce").fillna(1_000_000)
        rows["__abs_delta"] = pd.to_numeric(rows["predicted_minus_estimated_ms"], errors="coerce").abs().fillna(1_000_000)
        rows = rows.sort_values(["__is_primary", "__rank", "__abs_delta"], ascending=[False, True, True])
        return rows.iloc[0].drop(labels=["__is_primary", "__rank", "__abs_delta"], errors="ignore")

    def _select_times(self, selection: TimelineSelection) -> pd.DataFrame:
        cache_key = (str(selection.subject_id), str(selection.run_id), str(selection.device_type), str(selection.timeline_model_id))
        cached = self._time_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        times = self.store.read_table("SAMPLE_TIME_ESTIMATE")
        if times.empty:
            return times
        selected = times[
            (times["subject_id"].astype(str) == str(selection.subject_id))
            & (times["run_id"].astype(str) == str(selection.run_id))
            & (times["device_type"].astype(str) == str(selection.device_type))
            & (times["timeline_model_id"].astype(str) == str(selection.timeline_model_id))
        ].copy()
        self._time_cache[cache_key] = selected.copy()
        return selected.copy()
