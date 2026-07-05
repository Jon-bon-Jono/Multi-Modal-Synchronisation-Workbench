"""Ragged per-sample NumPy payload bundles.

The bundle layout is intentionally simple and stable:

- ``sample_index``: one canonical sample index per payload row
- ``offsets``: start/end offsets into the first dimension of ``values``
- ``values``: concatenated payload rows
- ``tail_shape``: payload shape after the ragged first dimension

For a frame-level point cloud with shape ``(P, 6)``, ``tail_shape`` is ``(6,)``.
For 2D poses with shape ``(N, 26, 3)``, ``tail_shape`` is ``(26, 3)``.
For confidence vectors with shape ``(N,)``, ``tail_shape`` is empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class RaggedPayloadInfo:
    sample_index: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


class RaggedNpzWriter:
    """Write variable-length per-sample arrays to one NPZ bundle."""

    @staticmethod
    def write(
        path: str | Path,
        payloads: Mapping[int, object] | Iterable[tuple[int, object]],
        *,
        tail_shape: tuple[int, ...] | None = None,
        dtype: str | np.dtype | None = None,
    ) -> list[RaggedPayloadInfo]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        items = list(payloads.items() if isinstance(payloads, Mapping) else payloads)
        items = [(int(idx), value) for idx, value in items]
        items.sort(key=lambda x: x[0])

        arrays: list[np.ndarray] = []
        inferred_tail = tail_shape
        inferred_dtype = np.dtype(dtype) if dtype is not None else None

        for _, value in items:
            arr = np.asarray(value, dtype=inferred_dtype)
            if inferred_tail is None:
                if arr.ndim == 0:
                    raise ValueError("Ragged payload entries must be arrays with at least one dimension.")
                inferred_tail = tuple(arr.shape[1:])
            expected_ndim = 1 + len(inferred_tail)
            if arr.ndim != expected_ndim or tuple(arr.shape[1:]) != tuple(inferred_tail):
                raise ValueError(
                    f"Inconsistent payload shape {arr.shape}; expected (*, {tuple(inferred_tail)})."
                )
            if inferred_dtype is None:
                inferred_dtype = arr.dtype
            elif arr.dtype != inferred_dtype:
                arr = arr.astype(inferred_dtype, copy=False)
            arrays.append(arr)

        if inferred_tail is None:
            inferred_tail = tuple(tail_shape or ())
        if inferred_dtype is None:
            inferred_dtype = np.dtype(dtype or "float64")

        sample_index = np.asarray([idx for idx, _ in items], dtype=np.int64)
        lengths = np.asarray([int(arr.shape[0]) for arr in arrays], dtype=np.int64)
        offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])

        if arrays:
            values = np.concatenate(arrays, axis=0) if int(offsets[-1]) else np.empty((0, *inferred_tail), dtype=inferred_dtype)
        else:
            values = np.empty((0, *inferred_tail), dtype=inferred_dtype)

        np.savez_compressed(
            path,
            sample_index=sample_index,
            offsets=offsets,
            values=values,
            tail_shape=np.asarray(inferred_tail, dtype=np.int64),
        )

        return [
            RaggedPayloadInfo(
                sample_index=idx,
                shape=tuple(arr.shape),
                dtype=str(arr.dtype),
                nbytes=int(arr.nbytes),
            )
            for (idx, _), arr in zip(items, arrays)
        ]


class RaggedNpzReader:
    """Read arrays from a ragged NPZ bundle."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._npz = np.load(self.path, allow_pickle=False)
        self.sample_index = self._npz["sample_index"].astype(np.int64)
        self.offsets = self._npz["offsets"].astype(np.int64)
        self.values = self._npz["values"]
        self._lookup = {int(idx): i for i, idx in enumerate(self.sample_index)}

    def get(self, sample_index: int) -> np.ndarray:
        try:
            pos = self._lookup[int(sample_index)]
        except KeyError as exc:
            raise KeyError(f"Sample {sample_index} not present in {self.path}") from exc
        start = int(self.offsets[pos])
        end = int(self.offsets[pos + 1])
        return self.values[start:end]

    def validate(self) -> list[str]:
        issues: list[str] = []
        if len(self.offsets) != len(self.sample_index) + 1:
            issues.append("offset length is not sample_count + 1")
        if len(self.offsets) and int(self.offsets[0]) != 0:
            issues.append("first offset is not zero")
        if np.any(np.diff(self.offsets) < 0):
            issues.append("offsets are not monotonic")
        if len(self.offsets) and int(self.offsets[-1]) != int(self.values.shape[0]):
            issues.append("last offset does not match values length")
        if len(set(map(int, self.sample_index))) != len(self.sample_index):
            issues.append("duplicate sample_index values")
        return issues
