"""Datetime helpers for the workbench."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime(DATETIME_FORMAT)


def parse_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse a workbench datetime string into a pandas Timestamp.

    Empty strings and null-like values become NaT. The project convention is a
    timezone-naive ISO-like string with microseconds, but this parser is mildly
    forgiving for practical ingestion.
    """
    if value is None:
        return pd.NaT
    if isinstance(value, float) and pd.isna(value):
        return pd.NaT
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, str) and not value.strip():
        return pd.NaT
    try:
        return pd.to_datetime(value, format=DATETIME_FORMAT, errors="raise")
    except Exception:
        return pd.to_datetime(value, errors="coerce")


def format_datetime(value: Any) -> str:
    ts = parse_datetime(value)
    if pd.isna(ts):
        return ""
    # pandas can keep timezone information. Canonical v0.1 stores naive strings.
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return ts.strftime(DATETIME_FORMAT)


def datetime_to_epoch_seconds(value: Any) -> float:
    ts = parse_datetime(value)
    if pd.isna(ts):
        return float("nan")
    return float(ts.timestamp())


def epoch_seconds_to_datetime_str(value: float) -> str:
    if pd.isna(value):
        return ""
    return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None).strftime(DATETIME_FORMAT)


def add_seconds_to_datetime(value: Any, seconds: float) -> str:
    ts = parse_datetime(value)
    if pd.isna(ts) or pd.isna(seconds):
        return ""
    return (ts + pd.to_timedelta(float(seconds), unit="s")).strftime(DATETIME_FORMAT)
