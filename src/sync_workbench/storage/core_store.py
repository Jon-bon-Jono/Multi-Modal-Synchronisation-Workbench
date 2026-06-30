"""Storage interface for canonical tables."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class CoreStore(Protocol):
    def write_table(self, name: str, df: pd.DataFrame, *, if_exists: str = "replace") -> None:
        ...

    def read_table(self, name: str) -> pd.DataFrame:
        ...

    def list_tables(self) -> list[str]:
        ...
