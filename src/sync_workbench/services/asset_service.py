"""Small service for resolving canonical RUN_ASSET references."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sync_workbench.assets.asset_refs import normalise_asset_ref
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


class AssetService:
    """Resolve run-level assets through simple storage roots.

    This is intentionally small. v0.2.2 only needs roots such as
    {"rgb": Path("rgb_videos"), "artifact_store": Path("artifact_store")}.
    """

    def __init__(self, sqlite_path: str | Path, roots: dict[str, str | Path] | None = None):
        self.store = SQLiteCoreStore(sqlite_path)
        self.roots = {str(k): Path(v) for k, v in (roots or {}).items()}

    def get_run_asset_rows(
        self,
        *,
        subject_id: str,
        run_id: str,
        device_type: str,
        asset_role: str | None = None,
    ) -> pd.DataFrame:
        assets = self.store.read_table("RUN_ASSET")
        if assets.empty:
            return assets
        mask = (
            (assets["subject_id"].astype(str) == str(subject_id))
            & (assets["run_id"].astype(str) == str(run_id))
            & (assets["device_type"].astype(str) == str(device_type))
        )
        if asset_role is not None:
            mask &= assets["asset_role"].astype(str) == str(asset_role)
        return assets.loc[mask].copy()

    def get_run_asset_ref(
        self,
        *,
        subject_id: str,
        run_id: str,
        device_type: str,
        asset_role: str,
    ) -> str:
        rows = self.get_run_asset_rows(
            subject_id=subject_id,
            run_id=run_id,
            device_type=device_type,
            asset_role=asset_role,
        )
        if rows.empty:
            raise KeyError(
                f"No RUN_ASSET for subject={subject_id}, run={run_id}, device={device_type}, role={asset_role}."
            )
        return str(rows.iloc[0]["asset_ref"])

    def resolve_run_asset(
        self,
        *,
        subject_id: str,
        run_id: str,
        device_type: str,
        asset_role: str,
        must_exist: bool = True,
    ) -> Path:
        rows = self.get_run_asset_rows(
            subject_id=subject_id,
            run_id=run_id,
            device_type=device_type,
            asset_role=asset_role,
        )
        if rows.empty:
            raise KeyError(
                f"No RUN_ASSET for subject={subject_id}, run={run_id}, device={device_type}, role={asset_role}."
            )
        row = rows.iloc[0]
        storage_key = str(row.get("storage_key", "") or "")
        asset_ref = normalise_asset_ref(str(row["asset_ref"]))
        root = self.roots.get(storage_key)
        if root is None:
            raise KeyError(
                f"No local root configured for storage_key={storage_key!r}. "
                f"Known roots: {sorted(self.roots)}"
            )
        path = root / asset_ref
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Resolved asset path does not exist: {path}")
        return path
