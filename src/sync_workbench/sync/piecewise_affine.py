"""Official piecewise-affine synchronisation model primitives.

This module is deliberately independent of SQLite, CLI, and GUI code. It takes
anchor pairs expressed in numeric source/target timeline coordinates and returns
an inspectable piecewise affine mapping.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

Region = Literal["interpolation", "left_extrapolation", "right_extrapolation", "unsupported"]


@dataclass(frozen=True)
class AnchorPair:
    anchor_id: str
    source_sample_index: int
    target_sample_index: int
    source_time: float
    target_time: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PiecewiseAffineSegment:
    left_anchor_id: str
    right_anchor_id: str
    source_left: float
    source_right: float
    target_left: float
    target_right: float
    slope: float
    intercept: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Prediction:
    source_time: float
    target_time: float
    region: Region
    segment_index: int | None


@dataclass(frozen=True)
class PiecewiseAffineModel:
    anchor_pairs: tuple[AnchorPair, ...]
    segments: tuple[PiecewiseAffineSegment, ...]
    extrapolation_policy: str = "disallow"

    def predict_one(self, source_time: float) -> Prediction:
        t = float(source_time)
        if not np.isfinite(t):
            return Prediction(t, float("nan"), "unsupported", None)

        segments = self.segments
        if not segments:
            return Prediction(t, float("nan"), "unsupported", None)

        first = segments[0]
        last = segments[-1]
        allow_extrapolation = str(self.extrapolation_policy).replace("-", "_") in {
            "allow_linear",
            "allow_with_penalty",
            "allow_nearest_only",
        }

        if t < first.source_left:
            if not allow_extrapolation:
                return Prediction(t, float("nan"), "unsupported", None)
            return Prediction(t, first.slope * t + first.intercept, "left_extrapolation", 0)

        if t > last.source_right:
            if not allow_extrapolation:
                return Prediction(t, float("nan"), "unsupported", None)
            return Prediction(t, last.slope * t + last.intercept, "right_extrapolation", len(segments) - 1)

        # Include the right-most anchor in the final segment.
        for idx, segment in enumerate(segments):
            is_final = idx == len(segments) - 1
            if segment.source_left <= t < segment.source_right or (is_final and t <= segment.source_right):
                return Prediction(t, segment.slope * t + segment.intercept, "interpolation", idx)

        return Prediction(t, float("nan"), "unsupported", None)

    def predict_many(self, source_times: Iterable[float]) -> pd.DataFrame:
        rows = [asdict(self.predict_one(float(t))) for t in source_times]
        return pd.DataFrame(rows)

    def anchor_residuals(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for pair in self.anchor_pairs:
            pred = self.predict_one(pair.source_time)
            residual = pred.target_time - pair.target_time if np.isfinite(pred.target_time) else float("nan")
            rows.append(
                {
                    "anchor_id": pair.anchor_id,
                    "source_sample_index": pair.source_sample_index,
                    "target_sample_index": pair.target_sample_index,
                    "source_time": pair.source_time,
                    "target_time": pair.target_time,
                    "predicted_target_time": pred.target_time,
                    "residual_sec": residual,
                    "residual_ms": residual * 1000.0 if np.isfinite(residual) else float("nan"),
                    "region": pred.region,
                }
            )
        return pd.DataFrame(rows)

    def leave_one_out_errors(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        pairs = list(self.anchor_pairs)
        for idx, held_out in enumerate(pairs):
            train = [pair for j, pair in enumerate(pairs) if j != idx]
            row = {
                "anchor_id": held_out.anchor_id,
                "source_sample_index": held_out.source_sample_index,
                "target_sample_index": held_out.target_sample_index,
                "source_time": held_out.source_time,
                "target_time": held_out.target_time,
            }
            if len(train) < 2:
                row.update(
                    {
                        "predicted_target_time": float("nan"),
                        "error_sec": float("nan"),
                        "error_ms": float("nan"),
                        "region": "unsupported",
                        "status": "insufficient_training_anchors",
                    }
                )
            else:
                try:
                    model = fit_piecewise_affine(
                        train,
                        extrapolation_policy="allow_linear",
                        require_monotonic_target=False,
                    )
                    pred = model.predict_one(held_out.source_time)
                    error = pred.target_time - held_out.target_time if np.isfinite(pred.target_time) else float("nan")
                    row.update(
                        {
                            "predicted_target_time": pred.target_time,
                            "error_sec": error,
                            "error_ms": error * 1000.0 if np.isfinite(error) else float("nan"),
                            "region": pred.region,
                            "status": "ok" if np.isfinite(error) else "unsupported",
                        }
                    )
                except ValueError as exc:
                    row.update(
                        {
                            "predicted_target_time": float("nan"),
                            "error_sec": float("nan"),
                            "error_ms": float("nan"),
                            "region": "unsupported",
                            "status": f"failed:{exc}",
                        }
                    )
            rows.append(row)
        return pd.DataFrame(rows)

    def to_parameters_dict(self) -> dict[str, object]:
        return {
            "model_type": "piecewise_affine",
            "extrapolation_policy": self.extrapolation_policy,
            "anchor_pairs": [pair.to_dict() for pair in self.anchor_pairs],
            "segments": [segment.to_dict() for segment in self.segments],
        }


def fit_piecewise_affine(
    anchor_pairs: Iterable[AnchorPair],
    *,
    extrapolation_policy: str = "disallow",
    require_monotonic_target: bool = True,
) -> PiecewiseAffineModel:
    pairs = sorted(list(anchor_pairs), key=lambda p: (float(p.source_time), str(p.anchor_id)))
    if len(pairs) < 2:
        raise ValueError("Piecewise affine synchronisation requires at least two anchor pairs.")

    source_times = np.asarray([float(p.source_time) for p in pairs], dtype=float)
    target_times = np.asarray([float(p.target_time) for p in pairs], dtype=float)

    if not np.all(np.isfinite(source_times)) or not np.all(np.isfinite(target_times)):
        raise ValueError("Anchor pair times must be finite numeric values.")

    source_deltas = np.diff(source_times)
    if np.any(source_deltas <= 0):
        raise ValueError("Anchor source times must be strictly increasing.")

    target_deltas = np.diff(target_times)
    if require_monotonic_target and np.any(target_deltas <= 0):
        raise ValueError("Anchor target times must be strictly increasing for the default piecewise model.")

    segments: list[PiecewiseAffineSegment] = []
    for left, right in zip(pairs[:-1], pairs[1:]):
        dx = float(right.source_time - left.source_time)
        if dx <= 0:
            raise ValueError("Anchor source times must be strictly increasing.")
        slope = float((right.target_time - left.target_time) / dx)
        intercept = float(left.target_time - slope * left.source_time)
        segments.append(
            PiecewiseAffineSegment(
                left_anchor_id=left.anchor_id,
                right_anchor_id=right.anchor_id,
                source_left=float(left.source_time),
                source_right=float(right.source_time),
                target_left=float(left.target_time),
                target_right=float(right.target_time),
                slope=slope,
                intercept=intercept,
            )
        )

    return PiecewiseAffineModel(tuple(pairs), tuple(segments), extrapolation_policy=extrapolation_policy)
