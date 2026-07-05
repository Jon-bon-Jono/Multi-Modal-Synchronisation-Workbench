"""Indexed JSONL helpers for per-sample dictionary-like payloads."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class JsonlPayloadInfo:
    sample_index: int
    byte_offset: int
    nbytes: int


class IndexedJsonlWriter:
    @staticmethod
    def write(path: str | Path, records: Iterable[tuple[int, Any]]) -> list[JsonlPayloadInfo]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        infos: list[JsonlPayloadInfo] = []
        with path.open("wb") as f:
            for sample_index, payload in sorted(((int(i), p) for i, p in records), key=lambda x: x[0]):
                offset = f.tell()
                line_obj = {"sample_index": int(sample_index), "payload": _json_safe(payload)}
                line = json.dumps(line_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                raw = line.encode("utf-8")
                f.write(raw)
                infos.append(JsonlPayloadInfo(sample_index=int(sample_index), byte_offset=int(offset), nbytes=len(raw)))
        return infos


class IndexedJsonlReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def read_at(self, byte_offset: int, nbytes: int | None = None) -> Any:
        with self.path.open("rb") as f:
            f.seek(int(byte_offset))
            raw = f.read(int(nbytes)) if nbytes is not None and int(nbytes) > 0 else f.readline()
        obj = json.loads(raw.decode("utf-8"))
        return obj.get("payload")


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/Pandas scalars and containers into JSON-safe values."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:  # pragma: no cover - imports are expected in this project
        np = None
        pd = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
