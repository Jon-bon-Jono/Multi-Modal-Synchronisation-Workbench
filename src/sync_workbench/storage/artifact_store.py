"""Filesystem helper for the Sync Workbench artifact store."""
from __future__ import annotations

from pathlib import Path

from sync_workbench.assets.asset_refs import is_probably_absolute_path, normalise_asset_ref


class ArtifactStore:
    """Resolve portable artifact refs relative to a local artifact root."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for_ref(self, artifact_ref: str) -> Path:
        ref = normalise_asset_ref(artifact_ref)
        if is_probably_absolute_path(ref):
            return Path(ref)
        return self.root / ref

    def ref_for(self, subject_id: str, run_id: str, device_type: str, filename: str) -> str:
        return normalise_asset_ref(f"subjects/{subject_id}/{device_type}/{run_id}/{filename}")

    def path_for(self, subject_id: str, run_id: str, device_type: str, filename: str) -> Path:
        return self.path_for_ref(self.ref_for(subject_id, run_id, device_type, filename))
