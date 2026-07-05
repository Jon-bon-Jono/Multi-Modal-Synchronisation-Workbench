"""Report entry points for synthetic feasibility workflows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sync_workbench.experimental.feasibility.piecewise_probe import run_synthetic_report


def build_piecewise_synthetic_report(output_dir: str | Path) -> pd.DataFrame:
    return run_synthetic_report(output_dir)
