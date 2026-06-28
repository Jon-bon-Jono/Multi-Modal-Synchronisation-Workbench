"""Parquet/CSV export helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path, *, fallback_csv: bool = True) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = output / f"{name.lower()}.parquet"
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:
            if not fallback_csv:
                raise
            csv_path = output / f"{name.lower()}.csv"
            df.to_csv(csv_path, index=False)
            (output / f"{name.lower()}.parquet.ERROR.txt").write_text(
                "Parquet export failed. Install sync-workbench[parquet] or pyarrow.\n"
                f"Original error: {exc}\n",
                encoding="utf-8",
            )
