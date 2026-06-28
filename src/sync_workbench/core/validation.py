"""Validation helpers for canonical tables and ingestion inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .tables import TABLE_SPECS


@dataclass
class ValidationIssue:
    severity: str
    table: str
    message: str


def check_required_columns(table_name: str, df: pd.DataFrame) -> list[ValidationIssue]:
    spec = TABLE_SPECS[table_name]
    missing = [c for c in spec.required if c not in df.columns]
    if not missing:
        return []
    return [ValidationIssue("error", table_name, f"Missing required columns: {missing}")]


def check_duplicate_keys(table_name: str, df: pd.DataFrame) -> list[ValidationIssue]:
    spec = TABLE_SPECS[table_name]
    if not all(c in df.columns for c in spec.key):
        return []
    dup_mask = df.duplicated(list(spec.key), keep=False)
    if not dup_mask.any():
        return []
    return [
        ValidationIssue(
            "error",
            table_name,
            f"Duplicate logical keys found: {int(dup_mask.sum())} rows affected; key={spec.key}",
        )
    ]


def check_null_required(table_name: str, df: pd.DataFrame) -> list[ValidationIssue]:
    spec = TABLE_SPECS[table_name]
    issues: list[ValidationIssue] = []
    for col in spec.required:
        if col not in df.columns:
            continue
        nullish = df[col].isna() | (df[col].astype(str) == "")
        if nullish.any():
            issues.append(
                ValidationIssue(
                    "error",
                    table_name,
                    f"Required column {col!r} has {int(nullish.sum())} null/empty values.",
                )
            )
    return issues


def validate_canonical_tables(tables: dict[str, pd.DataFrame]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, df in tables.items():
        if name not in TABLE_SPECS:
            issues.append(ValidationIssue("warning", name, "Unknown table name."))
            continue
        issues.extend(check_required_columns(name, df))
        issues.extend(check_null_required(name, df))
        issues.extend(check_duplicate_keys(name, df))
    return issues


def issues_to_frame(issues: Iterable[ValidationIssue]) -> pd.DataFrame:
    return pd.DataFrame([issue.__dict__ for issue in issues], columns=["severity", "table", "message"])
