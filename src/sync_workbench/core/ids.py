"""Small deterministic ID helpers."""
from __future__ import annotations

import re


def slugify(value: object, *, max_len: int = 80) -> str:
    '''
        Converts to URL-friendly string ('slug') by replacing special chars (not a number/letter, 
        not a hyphen/underscore/period) with underscores and capping length
    '''
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] or "unknown"


def asset_id(subject_id: str, run_id: str, device_type: str, role: str) -> str:
    return f"asset_{slugify(subject_id)}_{slugify(run_id)}_{slugify(device_type)}_{slugify(role)}"


def identity_sync_model_id(
    source_run_id: str,
    source_device_type: str,
    source_timeline_model_id: str,
    target_run_id: str,
    target_device_type: str,
    target_timeline_model_id: str,
) -> str:
    return (
        "sync_identity_"
        f"{slugify(source_run_id)}_{slugify(source_device_type)}_{slugify(source_timeline_model_id)}__"
        f"{slugify(target_run_id)}_{slugify(target_device_type)}_{slugify(target_timeline_model_id)}"
    )
