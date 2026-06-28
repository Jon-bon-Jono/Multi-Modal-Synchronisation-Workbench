"""Reader for the temporary ingestion package."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TempPackage:
    root: Path
    device_runs: pd.DataFrame
    rgb_samples: pd.DataFrame | None = None
    radar_pc_samples: pd.DataFrame | None = None
    radar_raw_samples: pd.DataFrame | None = None
    run_assets: pd.DataFrame | None = None

    @classmethod
    def read(cls, root: str | Path) -> "TempPackage":
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(root_path)
        device_runs = _read_required(root_path / "device_runs.zst")
        return cls(
            root=root_path,
            device_runs=device_runs,
            rgb_samples=_read_optional(root_path / "rgb_samples.zst"),
            radar_pc_samples=_read_optional(root_path / "radar_pc_samples.zst"),
            radar_raw_samples=_read_optional(root_path / "radar_raw_samples.zst"),
            run_assets=_read_optional(root_path / "run_assets.zst"),
        )


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required temporary ingestion file not found: {path}")
    return _read_pickle_lenient(path)


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return _read_pickle_lenient(path)


def _read_pickle_lenient(path: Path) -> pd.DataFrame:
    """Read a pandas pickle.

    Production files are expected to be zstd-compressed when using `.zst`.
    Tests sometimes use an uncompressed pickle with the same name to avoid an
    optional zstandard dependency, so this falls back to compression=None.
    """
    try:
        return pd.read_pickle(path, compression="infer")
    except Exception:
        return pd.read_pickle(path, compression=None)
