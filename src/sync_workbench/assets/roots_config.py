"""Load local asset-root configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RootsConfig:
    roots: dict[str, list[Path]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RootsConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        roots_raw: dict[str, Any] = raw.get("roots", {})
        roots = {key: [Path(p).expanduser() for p in vals] for key, vals in roots_raw.items()}
        return cls(roots=roots)

    @classmethod
    def empty(cls) -> "RootsConfig":
        return cls(roots={})
