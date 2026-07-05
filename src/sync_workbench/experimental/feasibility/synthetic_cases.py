"""Synthetic source-target timeline cases for the piecewise-affine sandbox."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    source_times: np.ndarray
    target_times: np.ndarray
    truth_target_times: np.ndarray
    anchor_source_times: np.ndarray
    anchor_target_times: np.ndarray
    description: str


def make_synthetic_cases() -> list[SyntheticCase]:
    return [
        _identity(),
        _constant_offset(),
        _global_affine_drift(),
        _piecewise_drift(),
        _sparse_anchors(),
        _bad_anchor(),
        _partial_overlap(),
        _before_after_anchor_extrapolation(),
        _target_gap_or_dropped_segment(),
        _different_frame_rates(),
    ]


def case_to_frames(case: SyntheticCase) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.DataFrame({"sample_index": np.arange(len(case.source_times)), "time_sec": case.source_times})
    target = pd.DataFrame({"sample_index": np.arange(len(case.target_times)), "time_sec": case.target_times})
    anchors = pd.DataFrame(
        {
            "anchor_id": [f"{case.name}_a{i:02d}" for i in range(len(case.anchor_source_times))],
            "source_time": case.anchor_source_times,
            "target_time": case.anchor_target_times,
        }
    )
    return source, target, anchors


def _base_source(duration: float = 20.0, fps: float = 15.0) -> np.ndarray:
    return np.arange(0.0, duration + 1e-9, 1.0 / fps)


def _target_grid(start: float, end: float, fps: float = 20.0) -> np.ndarray:
    return np.arange(start, end + 1e-9, 1.0 / fps)


def _identity() -> SyntheticCase:
    s = _base_source()
    anchors = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    return SyntheticCase("identity", s, _target_grid(0, 20), s, anchors, anchors.copy(), "Perfect identity mapping.")


def _constant_offset() -> SyntheticCase:
    s = _base_source()
    truth = s + 1.25
    anchors = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    return SyntheticCase("constant_offset", s, _target_grid(1.25, 21.25), truth, anchors, anchors + 1.25, "Target is source plus constant offset.")


def _global_affine_drift() -> SyntheticCase:
    s = _base_source()
    truth = 0.98 * s + 0.5
    anchors = np.array([0.0, 6.0, 12.0, 20.0])
    return SyntheticCase("global_affine_drift", s, _target_grid(0.5, 20.1), truth, anchors, 0.98 * anchors + 0.5, "Global scale drift plus offset.")


def _piecewise_drift() -> SyntheticCase:
    s = _base_source()
    truth = np.where(s <= 10.0, 1.02 * s + 0.2, 10.4 + 0.94 * (s - 10.0))
    anchors = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    a_truth = np.where(anchors <= 10.0, 1.02 * anchors + 0.2, 10.4 + 0.94 * (anchors - 10.0))
    return SyntheticCase("piecewise_drift", s, _target_grid(0.2, 19.8), truth, anchors, a_truth, "Different drift before and after the midpoint.")


def _sparse_anchors() -> SyntheticCase:
    s = _base_source()
    truth = s + 0.4 * np.sin(s / 3.0)
    anchors = np.array([2.0, 18.0])
    return SyntheticCase("sparse_anchors", s, _target_grid(0, 20), truth, anchors, anchors + 0.4 * np.sin(anchors / 3.0), "Only two poorly distributed anchors.")


def _bad_anchor() -> SyntheticCase:
    s = _base_source()
    truth = s + 0.75
    anchors = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    target = anchors + 0.75
    target[2] += 1.2
    return SyntheticCase("bad_anchor", s, _target_grid(0.75, 20.75), truth, anchors, target, "One deliberately bad middle anchor.")


def _partial_overlap() -> SyntheticCase:
    s = _base_source(duration=24)
    truth = s - 2.0
    anchors = np.array([4.0, 10.0, 16.0, 22.0])
    return SyntheticCase("partial_overlap", s, _target_grid(2.0, 18.0), truth, anchors, anchors - 2.0, "Target only covers part of source timeline.")


def _before_after_anchor_extrapolation() -> SyntheticCase:
    s = _base_source(duration=20)
    truth = 1.05 * s - 0.5
    anchors = np.array([5.0, 10.0, 15.0])
    return SyntheticCase("before_after_anchor_extrapolation", s, _target_grid(-0.5, 20.5), truth, anchors, 1.05 * anchors - 0.5, "Anchors do not cover beginning or end.")


def _target_gap_or_dropped_segment() -> SyntheticCase:
    s = _base_source(duration=20)
    truth = s + 0.3
    target = np.concatenate([_target_grid(0.3, 8.0), _target_grid(10.0, 20.3)])
    anchors = np.array([0.0, 5.0, 12.0, 20.0])
    return SyntheticCase("target_gap_or_dropped_segment", s, target, truth, anchors, anchors + 0.3, "Target has a dropped/gap segment.")


def _different_frame_rates() -> SyntheticCase:
    s = _base_source(duration=20, fps=15)
    truth = 1.01 * s + 0.1
    anchors = np.array([0.0, 7.0, 14.0, 20.0])
    return SyntheticCase("different_frame_rates_15fps_vs_20fps", s, _target_grid(0.1, 20.3, fps=20), truth, anchors, 1.01 * anchors + 0.1, "15 FPS source and 20 FPS target.")
