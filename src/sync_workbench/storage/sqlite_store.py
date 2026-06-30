"""SQLite implementation of the canonical store."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from sync_workbench.core.tables import TABLE_SPECS, align_to_spec


class SQLiteCoreStore:
    """Simple SQLite-backed store for canonical v0.1 tables.

    The v0.1 implementation deliberately keeps SQL constraints light and relies
    on explicit validation reports. That keeps the code portable while the schema
    is still settling.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialise_empty(self) -> None:
        with self.connect() as conn:
            for name, spec in TABLE_SPECS.items():
                empty = spec.empty()
                empty.to_sql(name, conn, if_exists="replace", index=False)

    def write_table(self, name: str, df: pd.DataFrame, *, if_exists: str = "replace") -> None:
        if name not in TABLE_SPECS:
            raise KeyError(f"Unknown canonical table: {name}")
        out = align_to_spec(name, df)
        with self.connect() as conn:
            out.to_sql(name, conn, if_exists=if_exists, index=False)

    def read_table(self, name: str) -> pd.DataFrame:
        if name not in TABLE_SPECS:
            raise KeyError(f"Unknown canonical table: {name}")
        with self.connect() as conn:
            try:
                return pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
            except Exception:
                return TABLE_SPECS[name].empty()

    def list_tables(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def row_exists(self, name: str, filters: dict[str, object]) -> bool:
        """Return True if at least one row exists matching all filter values."""
        if name not in TABLE_SPECS:
            raise KeyError(f"Unknown canonical table: {name}")
        if not filters:
            raise ValueError("row_exists requires at least one filter.")

        unknown = set(filters) - set(TABLE_SPECS[name].columns)
        if unknown:
            raise KeyError(f"Unknown columns for {name}: {sorted(unknown)}")

        where = " AND ".join(f'"{col}" = ?' for col in filters)
        values = tuple(filters.values())

        with self.connect() as conn:
            try:
                row = conn.execute(
                    f'SELECT 1 FROM "{name}" WHERE {where} LIMIT 1',
                    values,
                ).fetchone()
            except sqlite3.OperationalError:
                return False

        return row is not None

    def delete_where(self, name: str, filters: dict[str, object]) -> int:
        """Delete rows matching all filter values and return number deleted."""
        if name not in TABLE_SPECS:
            raise KeyError(f"Unknown canonical table: {name}")
        if not filters:
            raise ValueError("delete_where requires at least one filter.")

        unknown = set(filters) - set(TABLE_SPECS[name].columns)
        if unknown:
            raise KeyError(f"Unknown columns for {name}: {sorted(unknown)}")

        where = " AND ".join(f'"{col}" = ?' for col in filters)
        values = tuple(filters.values())

        with self.connect() as conn:
            try:
                cursor = conn.execute(
                    f'DELETE FROM "{name}" WHERE {where}',
                    values,
                )
            except sqlite3.OperationalError:
                return 0

        return int(cursor.rowcount or 0)
    
    def replace_or_append_without_key_conflicts(self, name: str, new_rows: pd.DataFrame) -> None:
        """Append rows after deleting existing rows with matching logical keys."""
        spec = TABLE_SPECS[name]
        current = self.read_table(name)
        if current.empty:
            self.write_table(name, new_rows, if_exists="append")
            return
        if new_rows.empty:
            return
        key_cols = list(spec.key)
        current_keyed = current.merge(new_rows[key_cols].drop_duplicates(), on=key_cols, how="left", indicator=True)
        kept = current_keyed.loc[current_keyed["_merge"] == "left_only", current.columns]
        combined = pd.concat([kept, align_to_spec(name, new_rows)], ignore_index=True)
        self.write_table(name, combined, if_exists="replace")
