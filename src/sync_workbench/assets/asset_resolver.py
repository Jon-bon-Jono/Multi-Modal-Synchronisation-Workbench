"""Resolve portable asset references to local file-system paths."""
from __future__ import annotations

from pathlib import Path

from .asset_refs import normalise_asset_ref
from .roots_config import RootsConfig


class AssetResolver:
    def __init__(self, config: RootsConfig):
        self.config = config

    def candidates(self, storage_key: str | None, asset_ref: str) -> list[Path]:
        ref = normalise_asset_ref(asset_ref)
        keys = []
        if storage_key:
            keys.append(storage_key)
        keys.append("default")
        seen: set[Path] = set()
        out: list[Path] = []
        for key in keys:
            for root in self.config.roots.get(key, []):
                candidate = root / ref
                if candidate not in seen:
                    out.append(candidate)
                    seen.add(candidate)
        return out

    def resolve_first_existing(self, storage_key: str | None, asset_ref: str) -> Path | None:
        for candidate in self.candidates(storage_key, asset_ref):
            if candidate.exists():
                return candidate
        return None
