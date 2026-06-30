"""Portable asset-reference helpers."""
from __future__ import annotations

import re
from pathlib import PurePosixPath


def normalise_asset_ref(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    # Collapse duplicate separators without destroying URI-like prefixes. Asset
    # refs should not be full URIs in v0.1, so a simple POSIX normalisation is ok.
    text = re.sub(r"/+", "/", text)
    return str(PurePosixPath(text)) if text not in {"", "."} else ""


def is_probably_absolute_path(value: object) -> bool:
    text = str(value).strip()
    if not text:
        return False
    # Windows drive path or POSIX root path.
    return bool(re.match(r"^[A-Za-z]:[\\/]", text)) or text.startswith("/")
