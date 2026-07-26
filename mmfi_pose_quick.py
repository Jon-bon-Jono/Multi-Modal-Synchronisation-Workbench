#!/usr/bin/env python3
"""Quick MM-Fi mmWave point-cloud -> 3D pose experiment.

This version uses:
- the native 17-joint MM-Fi topology defined in MMFI17_NAMES;
- corrected packed MM-Fi point order [x, y, z, Doppler, intensity];
- centred odd-sized radar windows (default: five 10 Hz samples spanning 0.4 s);
- frame-balanced, permutation-invariant geometric point sampling;
- masked padding instead of duplicated sparse-frame points;
- configurable return-strength handling: none, window-robust, or window-rank;
- one relative-time feature per point;
- a point-token Transformer with learned joint queries and 3-axis SimCC;
- a robust radar-cloud anchor with anchor-relative pelvis and
  pelvis-relative remaining-joint targets by default;
- joint geometric augmentation applied consistently to radar points and pose;
- fast SyncWB inference through cached run-level ragged-NPZ readers;
- optional Kinect-to-radar rigid extrinsics for SyncWB evaluation;
- tqdm progress bars when tqdm is installed;
- SyncWB metrics only for frames whose SAMPLE_SUMMARY.num_people == 1.

Recommended geometry + Doppler baseline:
    python mmfi_pose_quick.py train \
        --packed-root "D:/backup_data/MM-Fi/packed_data" \
        --out runs/mmfi_pose_anchor_v4 \
        --split cross_environment \
        --signal-mode none \
        --spatial-mode cloud_anchor \
        --target-mode cloud_anchor_relative \
        --epochs 25

Return-strength ablation using domain-compatible within-window normalisation:
    python mmfi_pose_quick.py train \
        --packed-root "D:/backup_data/MM-Fi/packed_data" \
        --out runs/mmfi_pose_anchor_v4_robust_signal \
        --signal-mode robust \
        --spatial-mode cloud_anchor \
        --target-mode cloud_anchor_relative \
        --epochs 25

Apply the checkpoint to one SyncWB mapping version:
    python mmfi_pose_quick.py infer-syncwb \
        --checkpoint runs/mmfi_pose_anchor_v4/best.pt \
        --sqlite workbench.sqlite \
        --artifact-root artifact_store \
        --subject 19_MM \
        --mapping-version piecewise_rgb_to_pc_v001_map \
        --out runs/19_MM_mmfi_pose_anchor_v4

Supervised fine-tuning on all selected subject runs/mapping versions:
    python mmfi_pose_quick.py finetune-syncwb \
        --checkpoint runs/mmfi_pose_anchor_v4/best.pt \
        --sqlite workbench.sqlite \
        --artifact-root artifact_store \
        --subjects 07_SW \
        --mapping-methods initial_nearest_for_anchoring \
        --out runs/mmfi_pose_anchor_v4_finetuned_07_SW

Inference/testing on all selected subject runs/mapping versions:
    python mmfi_pose_quick.py infer-syncwb-all \
        --checkpoint runs/mmfi_pose_anchor_v4_finetuned_07_SW/best.pt \
        --sqlite workbench.sqlite \
        --artifact-root artifact_store \
        --subjects 19_MM \
        --mapping-methods nearest_predicted_time \
        --out runs/19_MM_mmfi_pose_anchor_v4_finetuned_07_SW

Packed MM-Fi conversions:
    point cloud raw -> internal: [1, 0, 2, 4, 3]
    internal point columns:       [x, y, z, Doppler, intensity]
    keypoints raw -> internal:    [0, 2, 1]

Assumptions:
- packed MM-Fi point_clouds.npy has shape [N, P, >=5];
- packed MM-Fi kpts.npy has shape [N, 17, 3];
- SyncWB radar payloads use [x, y, z, Doppler, SNR, target_id];
- SyncWB Kinect pose3d uses Azure Kinect 32-joint order and millimetres;
- SyncWB SAMPLE_SUMMARY contains num_people for Kinect/RGB samples;
- MM-Fi intensity and SyncWB SNR are treated as semantically equivalent only
  when a within-window normalisation mode is explicitly enabled.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable=None, **_: object):
        return iterable


# -----------------------------------------------------------------------------
# Reproducibility and topology
# -----------------------------------------------------------------------------

MMFI17_NAMES = [
    "pelvis_center",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
    "thorax",
    "neck",
    "face_center",  # provisional anatomical interpretation
    "head",         # provisional anatomical interpretation
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]

# The two uncertain face/head joints are omitted from the more defensible body15
# metrics, but remain part of the trained 17-joint output.
BODY15_INDICES = np.asarray([i for i in range(17) if i not in {9, 10}], dtype=np.int64)

# Azure Kinect 32-joint order -> native MM-Fi 17-joint order.
KINECT32_TO_MMFI17 = np.asarray([
    0,   # pelvis_center <- pelvis
    18,  # left_hip
    19,  # left_knee
    20,  # left_ankle
    22,  # right_hip
    23,  # right_knee
    24,  # right_ankle
    2,   # thorax <- spine_chest
    3,   # neck
    27,  # face_center <- nose (provisional)
    26,  # head <- head (provisional)
    12,  # right_shoulder
    13,  # right_elbow
    14,  # right_wrist
    5,   # left_shoulder
    6,   # left_elbow
    7,   # left_wrist
], dtype=np.int64)

MMFI17_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (7, 11), (11, 12), (12, 13),
    (7, 14), (14, 15), (15, 16),
]

MMFI_POINT_ORDER = np.asarray([1, 0, 2, 4, 3], dtype=np.int64)
MMFI_POSE_ORDER = np.asarray([0, 2, 1], dtype=np.int64)
NOISE_TARGET_IDS = {253.0, 254.0, 255.0}
SIGNAL_MODES = ("none", "robust", "rank")
SPATIAL_MODES = ("legacy", "cloud_anchor")
TARGET_MODES = ("absolute", "pelvis_relative", "cloud_anchor_relative")
SCRIPT_VERSION = "domain-gap-v4"
SYNCWB_FINETUNE_VERSION = "syncwb-supervised-finetune-v1"
MMFI_SAMPLE_PERIOD_SEC = 0.10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Packed MM-Fi discovery and preprocessing
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PackedRecord:
    environment: str
    subject: str
    point_path: str
    pose_path: str
    activity_path: str | None
    n_frames: int


@dataclass(frozen=True)
class FeatureConfig:
    signal_mode: str = "none"
    doppler_limit_mps: float = 3.0
    spatial_mode: str = "legacy"

    def __post_init__(self) -> None:
        if self.signal_mode not in SIGNAL_MODES:
            raise ValueError(
                f"signal_mode must be one of {SIGNAL_MODES}, got {self.signal_mode!r}"
            )
        if not np.isfinite(self.doppler_limit_mps) or self.doppler_limit_mps <= 0:
            raise ValueError("doppler_limit_mps must be finite and positive")
        if self.spatial_mode not in SPATIAL_MODES:
            raise ValueError(
                f"spatial_mode must be one of {SPATIAL_MODES}, "
                f"got {self.spatial_mode!r}"
            )

    @property
    def input_dim(self) -> int:
        if self.spatial_mode == "legacy":
            # absolute xyz + centred xyz + Doppler + optional signal + relative time
            return 8 + int(self.signal_mode != "none")
        # cloud-anchor-relative xyz + sensor range + Doppler + optional signal
        # + relative time
        return 6 + int(self.signal_mode != "none")


@dataclass
class PoseBounds:
    """Per-joint, per-axis SimCC bounds in the encoded target representation."""

    low: list[list[float]]
    high: list[list[float]]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.asarray(self.low, dtype=np.float32)
        high = np.asarray(self.high, dtype=np.float32)
        if low.shape != (17, 3) or high.shape != (17, 3):
            raise ValueError(
                f"PoseBounds must have shape [17,3], got {low.shape} and {high.shape}"
            )
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("PoseBounds must be finite")
        if np.any(high <= low):
            raise ValueError("Every PoseBounds high value must exceed low")
        return low, high


def discover_packed_records(root: Path) -> list[PackedRecord]:
    records: list[PackedRecord] = []
    for point_path in sorted(root.glob("*/S*/point_clouds.npy")):
        subject_dir = point_path.parent
        pose_path = subject_dir / "kpts.npy"
        activity_path = subject_dir / "act_classes.npy"
        if not pose_path.exists():
            continue

        pc = np.load(point_path, mmap_mode="r")
        kp = np.load(pose_path, mmap_mode="r")
        if pc.ndim != 3 or pc.shape[-1] < 5:
            raise ValueError(f"Unexpected point-cloud shape {pc.shape}: {point_path}")
        if kp.ndim != 3 or kp.shape[1:] != (17, 3):
            raise ValueError(f"Unexpected keypoint shape {kp.shape}: {pose_path}")
        if len(pc) != len(kp):
            raise ValueError(f"Frame count mismatch in {subject_dir}: {len(pc)} != {len(kp)}")

        records.append(PackedRecord(
            environment=subject_dir.parent.name,
            subject=subject_dir.name,
            point_path=str(point_path),
            pose_path=str(pose_path),
            activity_path=str(activity_path) if activity_path.exists() else None,
            n_frames=len(pc),
        ))

    if not records:
        raise FileNotFoundError(
            f"No */S*/point_clouds.npy + kpts.npy pairs found under {root}"
        )
    return records


def split_records(
    records: Sequence[PackedRecord],
    split: str,
    seed: int,
    test_env: str,
    val_env: str | None,
) -> tuple[list[PackedRecord], list[PackedRecord], list[PackedRecord]]:
    if split == "cross_subject":
        subjects = sorted({r.subject for r in records})
        rng = random.Random(seed)
        rng.shuffle(subjects)
        n_test = max(1, int(round(0.20 * len(subjects))))
        n_val = max(1, int(round(0.10 * len(subjects))))
        test_subjects = set(subjects[:n_test])
        val_subjects = set(subjects[n_test:n_test + n_val])
        train_subjects = set(subjects[n_test + n_val:])
        train = [r for r in records if r.subject in train_subjects]
        val = [r for r in records if r.subject in val_subjects]
        test = [r for r in records if r.subject in test_subjects]
    elif split == "cross_environment":
        environments = sorted({r.environment for r in records})
        if test_env not in environments:
            raise ValueError(f"test_env={test_env!r} not found; available={environments}")
        if val_env is None:
            candidates = [e for e in environments if e != test_env]
            if len(candidates) < 2:
                raise ValueError("cross_environment needs at least three environments")
            val_env = candidates[-1]
        if val_env == test_env or val_env not in environments:
            raise ValueError(f"Invalid val_env={val_env!r}; available={environments}")
        train = [r for r in records if r.environment not in {test_env, val_env}]
        val = [r for r in records if r.environment == val_env]
        test = [r for r in records if r.environment == test_env]
    else:
        raise ValueError(f"Unknown split: {split}")

    if not train or not val or not test:
        raise ValueError(
            f"Empty split: train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test


def transform_mmfi_points(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)[..., MMFI_POINT_ORDER]


def transform_mmfi_pose(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float32)[..., MMFI_POSE_ORDER]


def point_valid_mask(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] < 5:
        raise ValueError(f"Expected [points, >=5], got {p.shape}")
    finite = np.all(np.isfinite(p[:, :5]), axis=1)
    # XYZ determines whether a row is spatial padding. A legitimate stationary
    # point may have zero Doppler and a return-strength value of zero.
    non_padding = np.any(np.abs(p[:, :3]) > 1e-8, axis=1)
    return finite & non_padding


def stack_radar_window(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate an odd-sized radar window and append relative frame time.

    Input frame columns: [x, y, z, Doppler, intensity/SNR, ...]
    Output columns:      [x, y, z, Doppler, intensity/SNR, relative_time]
    """
    if not frames or len(frames) % 2 == 0:
        raise ValueError("Radar window must contain a positive odd number of frames")

    half = len(frames) // 2
    stacked: list[np.ndarray] = []
    for position, frame in enumerate(frames):
        points = np.asarray(frame, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 5:
            raise ValueError(f"Expected radar frame [points, >=5], got {points.shape}")
        relative_time = 0.0 if half == 0 else (position - half) / half
        time_column = np.full((len(points), 1), relative_time, dtype=np.float32)
        stacked.append(np.concatenate([points[:, :5], time_column], axis=1))

    return np.concatenate(stacked, axis=0)


def geometric_median(
    xyz: np.ndarray,
    *,
    tolerance: float = 1e-5,
    max_iterations: int = 64,
) -> np.ndarray:
    """Return a deterministic, permutation-invariant geometric median.

    The Weiszfeld update is translation/rotation equivariant and has a 50%
    breakdown point. It is used only as a cloud reference, not as an anatomical
    pelvis estimate.
    """
    points = np.asarray(xyz, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        raise ValueError("geometric_median requires at least one finite point")
    if len(points) == 1:
        return points[0].astype(np.float32)

    estimate = np.median(points, axis=0)
    for _ in range(max_iterations):
        distances = np.linalg.norm(points - estimate, axis=1)
        coincident = distances <= tolerance
        if np.any(coincident):
            candidate = points[np.flatnonzero(coincident)[0]]
        else:
            weights = 1.0 / np.maximum(distances, tolerance)
            candidate = np.sum(points * weights[:, None], axis=0) / weights.sum()
        if np.linalg.norm(candidate - estimate) <= tolerance:
            estimate = candidate
            break
        estimate = candidate
    return estimate.astype(np.float32)


def robust_cloud_anchor(stacked_points: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate the cloud position at the centre timestamp.

    A geometric median is calculated independently for every non-empty frame.
    A coordinate-wise Theil-Sen line is then fitted through those per-frame
    anchors and evaluated at relative time zero. Equal per-frame treatment
    prevents a dense frame from dominating the result.
    """
    p = np.asarray(stacked_points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 6:
        raise ValueError(f"Expected stacked points [points, >=6], got {p.shape}")

    valid = point_valid_mask(p)
    relative_times = np.unique(p[:, 5])
    frame_times: list[float] = []
    frame_anchors: list[np.ndarray] = []
    frame_counts: list[int] = []
    for relative_time in np.sort(relative_times):
        frame = p[valid & np.isclose(p[:, 5], relative_time, atol=1e-6)]
        if len(frame) == 0:
            continue
        frame_times.append(float(relative_time))
        frame_anchors.append(geometric_median(frame[:, :3]))
        frame_counts.append(int(len(frame)))

    if not frame_anchors:
        raise ValueError("Radar window contains no finite non-padding points")

    times = np.asarray(frame_times, dtype=np.float64)
    anchors = np.asarray(frame_anchors, dtype=np.float64)
    centre = robust_anchor_from_frame_anchors(times, anchors)

    centre_index = int(np.argmin(np.abs(times)))
    diagnostics: dict[str, object] = {
        "anchor": centre,
        "frame_times": times.astype(np.float32),
        "frame_counts": np.asarray(frame_counts, dtype=np.int64),
        "valid_window_frames": int(len(frame_counts)),
        "window_point_count": int(sum(frame_counts)),
        "center_point_count": int(frame_counts[centre_index]),
    }
    return centre, diagnostics


def robust_anchor_from_frame_anchors(
    frame_times: np.ndarray,
    frame_anchors: np.ndarray,
) -> np.ndarray:
    """Fit the robust t=0 trajectory once frame medians are available."""
    times = np.asarray(frame_times, dtype=np.float64)
    anchors = np.asarray(frame_anchors, dtype=np.float64)
    valid = np.isfinite(times) & np.all(np.isfinite(anchors), axis=1)
    times, anchors = times[valid], anchors[valid]
    if len(anchors) == 0:
        raise ValueError("At least one finite per-frame anchor is required")
    if len(anchors) == 1:
        centre = anchors[0]
    else:
        centre = np.empty(3, dtype=np.float64)
        for axis in range(3):
            slopes: list[float] = []
            for left in range(len(times) - 1):
                for right in range(left + 1, len(times)):
                    delta = times[right] - times[left]
                    if abs(delta) > 1e-12:
                        slopes.append(
                            float((anchors[right, axis] - anchors[left, axis]) / delta)
                        )
            slope = float(np.median(slopes)) if slopes else 0.0
            centre[axis] = float(np.median(anchors[:, axis] - slope * times))
    return centre.astype(np.float32)


def radar_window_support(
    stacked_points: np.ndarray,
    window_size: int,
) -> dict[str, object]:
    """Count real points at each expected relative timestamp."""
    p = np.asarray(stacked_points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 6:
        raise ValueError(f"Expected stacked points [points, >=6], got {p.shape}")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    expected_times = np.linspace(-1.0, 1.0, window_size, dtype=np.float32)
    valid = point_valid_mask(p)
    frame_counts = np.asarray([
        np.count_nonzero(valid & np.isclose(p[:, 5], relative_time, atol=1e-6))
        for relative_time in expected_times
    ], dtype=np.int64)
    return {
        "frame_counts": frame_counts,
        "valid_window_frames": int(np.count_nonzero(frame_counts)),
        "window_point_count": int(frame_counts.sum()),
        "center_point_count": int(frame_counts[window_size // 2]),
    }


def _canonical_point_order(frame: np.ndarray) -> np.ndarray:
    """Canonicalise identical point sets before deterministic geometric sampling."""
    p = np.asarray(frame, dtype=np.float32)
    # x is the primary key, then y/z/Doppler/signal/time. Equal rows are
    # interchangeable, so their internal tie order cannot affect the output.
    order = np.lexsort((
        p[:, 5],
        p[:, 4],
        p[:, 3],
        p[:, 2],
        p[:, 1],
        p[:, 0],
    ))
    return p[order]


def _farthest_point_indices(frame: np.ndarray, count: int) -> np.ndarray:
    """Deterministic XYZ farthest-point sampling after canonical ordering."""
    p = _canonical_point_order(frame)
    if count >= len(p):
        return np.arange(len(p), dtype=np.int64)

    xyz = p[:, :3].astype(np.float64)
    centre = geometric_median(xyz).astype(np.float64)
    start_distances = np.linalg.norm(xyz - centre, axis=1)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmin(start_distances))
    min_distance_sq = np.sum((xyz - xyz[selected[0]]) ** 2, axis=1)
    min_distance_sq[selected[0]] = -1.0
    for position in range(1, count):
        selected[position] = int(np.argmax(min_distance_sq))
        distance_sq = np.sum((xyz - xyz[selected[position]]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected[: position + 1]] = -1.0
    return selected


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile ranks mapped to [-1, 1], without SciPy."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return (2.0 * ranks / len(x) - 1.0).astype(np.float32)


def robust_feature_normalize(
    points: np.ndarray,
    feature_config: FeatureConfig,
    cloud_anchor: np.ndarray | None = None,
) -> np.ndarray:
    """Construct model features from a sampled radar window.

    Input columns:
        [x, y, z, Doppler, intensity/SNR, relative_time]

    ``legacy`` spatial features (8):
        absolute xyz / 4 m, window-median-centred xyz / 2 m, clipped Doppler,
        and relative frame time.

    ``cloud_anchor`` spatial features (6):
        robust-cloud-anchor-relative xyz / 2 m, physical sensor range / 4 m,
        clipped Doppler, and relative frame time.

    Optional return-strength feature (1):
        ``robust``: within-window median/IQR, clipped to +/-5 IQR and /5;
        ``rank``: within-window average-tie percentile rank mapped to [-1,1].

    Both optional modes are invariant to positive affine rescaling of the raw
    return-strength channel, which is the compatibility assumption used for
    MM-Fi intensity versus SyncWB SNR. ``none`` is the recommended baseline.
    """
    p = np.asarray(points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 6:
        raise ValueError(f"Expected sampled points [points, >=6], got {p.shape}")

    mask = point_valid_mask(p)
    xyz = p[:, :3]
    doppler = p[:, 3]
    signal = p[:, 4]
    relative_time = p[:, 5]

    if feature_config.spatial_mode == "cloud_anchor":
        if cloud_anchor is None:
            raise ValueError("cloud_anchor spatial features require a cloud anchor")
        centre = np.asarray(cloud_anchor, dtype=np.float32)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError(f"Expected a finite cloud anchor [3], got {centre}")
    else:
        centre = (
            np.median(xyz[mask], axis=0)
            if np.any(mask)
            else np.zeros(3, dtype=np.float32)
        )

    absolute_xyz = np.clip(xyz / 4.0, -2.0, 2.0)
    centred_xyz = np.clip((xyz - centre) / 2.0, -2.0, 2.0)
    sensor_range = np.clip(np.linalg.norm(xyz, axis=1) / 4.0, 0.0, 2.0)[:, None]
    doppler_n = (
        np.clip(
            doppler,
            -feature_config.doppler_limit_mps,
            feature_config.doppler_limit_mps,
        )
        / feature_config.doppler_limit_mps
    )[:, None]
    time_n = np.clip(relative_time, -1.0, 1.0)[:, None]

    if feature_config.spatial_mode == "legacy":
        parts = [absolute_xyz, centred_xyz, doppler_n]
    else:
        parts = [centred_xyz, sensor_range, doppler_n]
    if feature_config.signal_mode == "robust":
        signal_n = np.zeros(len(p), dtype=np.float32)
        if np.any(mask):
            valid_signal = signal[mask]
            median = float(np.median(valid_signal))
            q25, q75 = np.percentile(valid_signal, [25, 75])
            scale = max(float(q75 - q25), 1e-3)
            signal_n[mask] = (
                np.clip((valid_signal - median) / scale, -5.0, 5.0) / 5.0
            )
        parts.append(signal_n[:, None])
    elif feature_config.signal_mode == "rank":
        signal_n = np.zeros(len(p), dtype=np.float32)
        if np.any(mask):
            signal_n[mask] = percentile_rank(signal[mask])
        parts.append(signal_n[:, None])
    elif feature_config.signal_mode != "none":  # defensive; dataclass validates
        raise ValueError(f"Unsupported signal_mode={feature_config.signal_mode!r}")
    parts.append(time_n)

    out = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    out[~mask] = 0.0
    if out.shape[1] != feature_config.input_dim:
        raise RuntimeError(
            f"Feature construction produced {out.shape[1]} columns; "
            f"expected {feature_config.input_dim}"
        )
    return out


def _frame_quotas(num_points: int, window_size: int) -> list[int]:
    if num_points < 1:
        raise ValueError("num_points must be positive")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    base, remainder = divmod(num_points, window_size)
    quotas = [base] * window_size
    # Give remainder tokens to frames nearest the centre, symmetrically where possible.
    centre = window_size // 2
    order = sorted(range(window_size), key=lambda i: (abs(i - centre), i))
    for i in order[:remainder]:
        quotas[i] += 1
    return quotas


def frame_balanced_sample(
    stacked_points: np.ndarray,
    num_points: int,
    window_size: int,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an equal token budget from every temporal frame.

    Sparse frames are padded with masked zero tokens rather than duplicated
    points. Dense frames use random sampling during training and deterministic
    geometric sampling during evaluation. Canonical ordering makes both paths
    independent of the input array's arbitrary row order.
    """
    p = np.asarray(stacked_points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 6:
        raise ValueError(f"Expected stacked points [points, >=6], got {p.shape}")

    half = window_size // 2
    expected_times = [0.0 if half == 0 else (i - half) / half for i in range(window_size)]
    quotas = _frame_quotas(num_points, window_size)
    valid = point_valid_mask(p)
    sampled: list[np.ndarray] = []
    sampled_masks: list[np.ndarray] = []

    for relative_time, quota in zip(expected_times, quotas):
        if quota == 0:
            continue
        frame = p[valid & np.isclose(p[:, 5], relative_time, atol=1e-6)]
        if len(frame) == 0:
            sampled.append(np.zeros((quota, p.shape[1]), dtype=np.float32))
            sampled_masks.append(np.zeros(quota, dtype=bool))
            continue

        canonical = _canonical_point_order(frame)
        if len(canonical) <= quota:
            selected = canonical
        else:
            if rng is None:
                ids = _farthest_point_indices(canonical, quota)
            else:
                ids = rng.choice(len(canonical), size=quota, replace=False)
            selected = canonical[ids]

        pad = quota - len(selected)
        if pad:
            selected = np.concatenate([
                selected,
                np.zeros((pad, p.shape[1]), dtype=np.float32),
            ], axis=0)
        sampled.append(selected)
        sampled_masks.append(np.concatenate([
            np.ones(quota - pad, dtype=bool),
            np.zeros(pad, dtype=bool),
        ]))

    out = np.concatenate(sampled, axis=0)
    token_mask = np.concatenate(sampled_masks, axis=0)
    if len(out) != num_points:
        raise RuntimeError(f"Frame-balanced sampler produced {len(out)} != {num_points}")
    if not np.any(token_mask):
        token_mask[0] = True  # avoid an entirely masked Transformer input
    return out.astype(np.float32, copy=False), token_mask


def rotation_matrix_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Column-vector XYZ Euler rotation returned for row-vector use via ``@ R.T``."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def augment_geometry(
    points: np.ndarray,
    pose: np.ndarray,
    cloud_anchor: np.ndarray,
    rng: np.random.Generator,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    translation_m: Sequence[float],
    body_scale_min: float,
    body_scale_max: float,
    x_reflection_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one physically consistent geometric augmentation to points and pose.

    Rotation, reflection and body-scale are applied about the pelvis so the
    absolute pelvis remains stable. Translation is then applied to both sensor
    points and pose. This increases orientation/range robustness without
    corrupting radar-to-pose correspondence.
    """
    out_points = np.asarray(points, dtype=np.float32).copy()
    out_pose = np.asarray(pose, dtype=np.float32).copy()
    out_anchor = np.asarray(cloud_anchor, dtype=np.float32).copy()
    if out_anchor.shape != (3,):
        raise ValueError(f"Expected cloud_anchor [3], got {out_anchor.shape}")
    valid = point_valid_mask(out_points)
    pelvis = out_pose[0, :3].copy()

    roll = math.radians(float(rng.uniform(-roll_deg, roll_deg)))
    pitch = math.radians(float(rng.uniform(-pitch_deg, pitch_deg)))
    yaw = math.radians(float(rng.uniform(-yaw_deg, yaw_deg)))
    rotation = rotation_matrix_xyz(roll, pitch, yaw)
    scale = float(rng.uniform(body_scale_min, body_scale_max))

    out_pose[:, :3] = (out_pose[:, :3] - pelvis) * scale @ rotation.T + pelvis
    out_anchor = (out_anchor - pelvis) * scale @ rotation.T + pelvis
    if np.any(valid):
        out_points[valid, :3] = (
            (out_points[valid, :3] - pelvis) * scale @ rotation.T + pelvis
        )

    if rng.random() < x_reflection_probability:
        out_pose[:, 0] = 2.0 * pelvis[0] - out_pose[:, 0]
        out_anchor[0] = 2.0 * pelvis[0] - out_anchor[0]
        if np.any(valid):
            out_points[valid, 0] = 2.0 * pelvis[0] - out_points[valid, 0]

    translation_limit = np.asarray(translation_m, dtype=np.float32)
    translation = rng.uniform(-translation_limit, translation_limit).astype(np.float32)
    out_pose[:, :3] += translation
    out_anchor += translation
    if np.any(valid):
        out_points[valid, :3] += translation
    return out_points, out_pose, out_anchor


def encode_pose_target(
    pose: np.ndarray,
    target_mode: str,
    cloud_anchor: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(pose, dtype=np.float32)
    if y.shape[-2:] != (17, 3):
        raise ValueError(f"Expected pose [...,17,3], got {y.shape}")
    if target_mode == "absolute":
        return y.copy()
    if target_mode == "pelvis_relative":
        encoded = y.copy()
        encoded[..., 1:, :] -= y[..., :1, :]
        # Joint 0 deliberately remains the absolute pelvis.
        return encoded
    if target_mode == "cloud_anchor_relative":
        if cloud_anchor is None:
            raise ValueError("cloud_anchor_relative targets require a cloud anchor")
        anchor = np.asarray(cloud_anchor, dtype=np.float32)
        if anchor.shape != y.shape[:-2] + (3,):
            raise ValueError(
                f"Cloud anchor shape {anchor.shape} is incompatible with pose {y.shape}"
            )
        encoded = y.copy()
        encoded[..., 0, :] -= anchor
        encoded[..., 1:, :] -= y[..., :1, :]
        return encoded
    raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")


def decode_pose_target(
    encoded: np.ndarray | torch.Tensor,
    target_mode: str,
    cloud_anchor: np.ndarray | torch.Tensor | None = None,
):
    if target_mode == "absolute":
        return encoded
    if target_mode not in {"pelvis_relative", "cloud_anchor_relative"}:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
    if isinstance(encoded, torch.Tensor):
        pose = encoded.clone()
        if target_mode == "cloud_anchor_relative":
            if cloud_anchor is None:
                raise ValueError("cloud_anchor_relative decoding requires a cloud anchor")
            anchor = torch.as_tensor(
                cloud_anchor,
                dtype=pose.dtype,
                device=pose.device,
            )
            pose[..., 0, :] = pose[..., 0, :] + anchor
        pose[..., 1:, :] = pose[..., 1:, :] + pose[..., :1, :]
        return pose
    pose = np.asarray(encoded).copy()
    if target_mode == "cloud_anchor_relative":
        if cloud_anchor is None:
            raise ValueError("cloud_anchor_relative decoding requires a cloud anchor")
        pose[..., 0, :] += np.asarray(cloud_anchor, dtype=pose.dtype)
    pose[..., 1:, :] += pose[..., :1, :]
    return pose


def record_segments(record: PackedRecord) -> list[tuple[int, int]]:
    """Return half-open contiguous activity segments for one packed subject."""
    if record.activity_path is None:
        return [(0, record.n_frames)]
    activities = np.asarray(np.load(record.activity_path, mmap_mode="r"))
    if activities.ndim != 1 or len(activities) != record.n_frames:
        raise ValueError(
            f"Unexpected activity array {activities.shape}: {record.activity_path}"
        )
    boundaries = np.flatnonzero(activities[1:] != activities[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(activities)]])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def valid_record_centres(record: PackedRecord, window_size: int) -> np.ndarray:
    half = window_size // 2
    chunks = [
        np.arange(start + half, stop - half, dtype=np.int64)
        for start, stop in record_segments(record)
        if stop - start >= window_size
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def estimate_pose_bounds(
    records: Sequence[PackedRecord],
    max_frames: int,
    seed: int,
    target_mode: str,
    translation_m: Sequence[float],
    body_scale_max: float,
    window_size: int,
) -> PoseBounds:
    """Estimate per-joint SimCC bounds in encoded target space.

    Pelvis-relative joints receive symmetric radius-based bounds so pitch/roll,
    left-right reflection and body-scale augmentation cannot clip a coordinate.
    The absolute pelvis bounds are expanded by the configured translation range.
    """
    rng = np.random.default_rng(seed)
    total = sum(r.n_frames for r in records)
    samples: list[np.ndarray] = []
    anchor_samples: list[np.ndarray] = []

    for record in records:
        valid_centres = valid_record_centres(record, window_size)
        if len(valid_centres) == 0:
            continue
        n = max(1, int(round(max_frames * record.n_frames / total)))
        n = min(n, len(valid_centres))
        ids = rng.choice(valid_centres, size=n, replace=False)
        poses_all = np.load(record.pose_path, mmap_mode="r")
        points_all = (
            np.load(record.point_path, mmap_mode="r")
            if target_mode == "cloud_anchor_relative"
            else None
        )
        half = window_size // 2
        for centre_id in ids:
            pose = transform_mmfi_pose(poses_all[int(centre_id)])
            if not np.all(np.isfinite(pose)):
                continue
            if points_all is not None:
                frames = [
                    transform_mmfi_points(points_all[frame_id])
                    for frame_id in range(int(centre_id) - half, int(centre_id) + half + 1)
                ]
                try:
                    anchor, _ = robust_cloud_anchor(stack_radar_window(frames))
                except ValueError:
                    continue
                anchor_samples.append(anchor)
            samples.append(pose)

    if not samples:
        raise ValueError("No finite MM-Fi pose values found")

    poses = np.stack(samples).astype(np.float32)
    anchors = (
        np.stack(anchor_samples).astype(np.float32)
        if target_mode == "cloud_anchor_relative"
        else None
    )
    if target_mode == "cloud_anchor_relative" and (
        anchors is None or len(anchors) != len(poses)
    ):
        raise RuntimeError("Pose/anchor sample count mismatch while estimating bounds")
    encoded = encode_pose_target(poses, target_mode, anchors)
    low = np.percentile(encoded, 0.1, axis=0)
    high = np.percentile(encoded, 99.9, axis=0)
    margin = np.maximum(0.10 * (high - low), 0.05)
    low -= margin
    high += margin

    translation_limit = np.asarray(translation_m, dtype=np.float32)
    if target_mode in {"pelvis_relative", "cloud_anchor_relative"}:
        if target_mode == "pelvis_relative":
            low[0] -= translation_limit
            high[0] += translation_limit
        else:
            root_offsets = poses[:, 0, :] - anchors
            root_radius = max(
                float(np.percentile(np.linalg.norm(root_offsets, axis=-1), 99.9))
                * body_scale_max
                * 1.10,
                0.25,
            )
            low[0] = -root_radius
            high[0] = root_radius
        relative = poses[:, 1:, :] - poses[:, :1, :]
        radii = np.percentile(np.linalg.norm(relative, axis=-1), 99.9, axis=0)
        radii = np.maximum(radii * body_scale_max * 1.10, 0.15)
        low[1:] = -radii[:, None]
        high[1:] = radii[:, None]
    else:
        # Make absolute-target bounds safe under rotations/reflection about the
        # pelvis by enclosing each joint in a source-derived radius around the
        # observed pelvis support.
        root_low = np.percentile(poses[:, 0, :], 0.1, axis=0)
        root_high = np.percentile(poses[:, 0, :], 99.9, axis=0)
        relative = poses - poses[:, :1, :]
        radii = np.percentile(np.linalg.norm(relative, axis=-1), 99.9, axis=0)
        radii = np.maximum(radii * body_scale_max * 1.10, 0.05)
        low = root_low[None, :] - radii[:, None] - translation_limit[None, :]
        high = root_high[None, :] + radii[:, None] + translation_limit[None, :]

    return PoseBounds(low.tolist(), high.tolist())


class PackedMMFiDataset(Dataset):
    """Centred sliding windows that never cross an activity boundary."""

    def __init__(
        self,
        records: Sequence[PackedRecord],
        bounds: PoseBounds,
        bins: int,
        num_points: int,
        window_size: int,
        training: bool,
        seed: int,
        feature_config: FeatureConfig,
        target_mode: str,
        yaw_aug_deg: float,
        pitch_aug_deg: float,
        roll_aug_deg: float,
        translation_aug_m: Sequence[float],
        body_scale_min: float,
        body_scale_max: float,
        x_reflection_probability: float,
        signal_dropout_probability: float,
    ):
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}")
        if not 0.0 <= signal_dropout_probability <= 1.0:
            raise ValueError("signal_dropout_probability must be in [0,1]")

        self.records = list(records)
        self.bounds = bounds
        self.bins = int(bins)
        self.num_points = int(num_points)
        self.window_size = int(window_size)
        self.half_window = self.window_size // 2
        self.training = bool(training)
        self.seed = int(seed)
        self.feature_config = feature_config
        self.target_mode = target_mode
        self.yaw_aug_deg = float(yaw_aug_deg)
        self.pitch_aug_deg = float(pitch_aug_deg)
        self.roll_aug_deg = float(roll_aug_deg)
        self.translation_aug_m = tuple(float(v) for v in translation_aug_m)
        self.body_scale_min = float(body_scale_min)
        self.body_scale_max = float(body_scale_max)
        self.x_reflection_probability = float(x_reflection_probability)
        self.signal_dropout_probability = float(signal_dropout_probability)
        self.low, self.high = bounds.arrays()
        # Shared storage lets persistent Windows DataLoader workers observe the
        # parent process's epoch while retaining their per-frame anchor caches.
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()

        self.segments: list[tuple[int, int, np.ndarray]] = []
        self.excluded_empty_or_invalid_windows = 0
        for record_index, record in enumerate(self.records):
            point_clouds = np.load(record.point_path, mmap_mode="r")
            poses = np.load(record.pose_path, mmap_mode="r")
            raw_xyz = np.asarray(point_clouds[..., :3])
            frame_has_points = np.any(
                np.all(np.isfinite(raw_xyz), axis=-1)
                & (np.linalg.norm(raw_xyz, axis=-1) > 1e-8),
                axis=1,
            )
            finite_pose = np.all(np.isfinite(poses), axis=(1, 2))
            for segment_start, segment_stop in record_segments(record):
                window_count = segment_stop - segment_start - self.window_size + 1
                if window_count > 0:
                    support_count = np.convolve(
                        frame_has_points[segment_start:segment_stop].astype(np.int16),
                        np.ones(self.window_size, dtype=np.int16),
                        mode="valid",
                    )
                    centre_ids = (
                        np.arange(window_count, dtype=np.int64)
                        + segment_start
                        + self.half_window
                    )
                    valid_offsets = np.flatnonzero(
                        (support_count > 0) & finite_pose[centre_ids]
                    ).astype(np.int64)
                    self.excluded_empty_or_invalid_windows += int(
                        window_count - len(valid_offsets)
                    )
                    if len(valid_offsets) == 0:
                        continue
                    self.segments.append((
                        int(record_index),
                        int(segment_start),
                        valid_offsets,
                    ))
        if not self.segments:
            raise ValueError("No activity segment is long enough for the requested window")
        self.ends = np.cumsum([len(segment[2]) for segment in self.segments]).tolist()
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._frame_anchor_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return int(self.ends[-1])

    def set_epoch(self, epoch: int) -> None:
        self._epoch_state.fill_(int(epoch))

    def _arrays(self, record_index: int) -> tuple[np.ndarray, np.ndarray]:
        if record_index not in self._cache:
            record = self.records[record_index]
            self._cache[record_index] = (
                np.load(record.point_path, mmap_mode="r"),
                np.load(record.pose_path, mmap_mode="r"),
            )
        return self._cache[record_index]

    def _cloud_anchor(
        self,
        record_index: int,
        window_start: int,
        frames: Sequence[np.ndarray],
    ) -> np.ndarray:
        if record_index not in self._frame_anchor_cache:
            n_frames = self.records[record_index].n_frames
            self._frame_anchor_cache[record_index] = (
                np.full((n_frames, 3), np.nan, dtype=np.float32),
                np.zeros(n_frames, dtype=bool),
            )
        anchors, computed = self._frame_anchor_cache[record_index]
        window_anchors = np.empty((self.window_size, 3), dtype=np.float32)
        for offset, frame in enumerate(frames):
            frame_index = window_start + offset
            if not computed[frame_index]:
                valid = point_valid_mask(frame)
                if np.any(valid):
                    anchors[frame_index] = geometric_median(frame[valid, :3])
                computed[frame_index] = True
            window_anchors[offset] = anchors[frame_index]
        frame_times = np.linspace(-1.0, 1.0, self.window_size, dtype=np.float32)
        return robust_anchor_from_frame_anchors(frame_times, window_anchors)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        segment_index = bisect.bisect_right(self.ends, int(index))
        segment_dataset_start = 0 if segment_index == 0 else self.ends[segment_index - 1]
        record_index, segment_start, valid_offsets = self.segments[segment_index]
        local_index = int(index) - int(segment_dataset_start)
        window_start = segment_start + int(valid_offsets[local_index])
        centre_frame = window_start + self.half_window
        window_end = window_start + self.window_size

        point_clouds, poses = self._arrays(record_index)
        frames = [
            transform_mmfi_points(point_clouds[frame_index])
            for frame_index in range(window_start, window_end)
        ]
        points = stack_radar_window(frames)
        pose = transform_mmfi_pose(poses[centre_frame])
        cloud_anchor = self._cloud_anchor(record_index, window_start, frames)

        rng = np.random.default_rng(
            self.seed
            + int(index) * 104729
            + int(self._epoch_state.item()) * 1000003
        )
        if self.training:
            points, pose, cloud_anchor = augment_geometry(
                points,
                pose,
                cloud_anchor,
                rng,
                yaw_deg=self.yaw_aug_deg,
                pitch_deg=self.pitch_aug_deg,
                roll_deg=self.roll_aug_deg,
                translation_m=self.translation_aug_m,
                body_scale_min=self.body_scale_min,
                body_scale_max=self.body_scale_max,
                x_reflection_probability=self.x_reflection_probability,
            )
            valid = point_valid_mask(points)
            if np.any(valid):
                points[valid, :3] += rng.normal(
                    0.0, 0.008, size=(valid.sum(), 3)
                ).astype(np.float32)
                points[valid, 3] += rng.normal(
                    0.0, 0.03, size=valid.sum()
                ).astype(np.float32)
                drop = rng.random(valid.sum()) < 0.08
                if np.all(drop):
                    # Sparse source windows must retain a real geometric
                    # reference; masked padding is not a valid cloud anchor.
                    drop[int(rng.integers(0, len(drop)))] = False
                points[np.flatnonzero(valid)[drop]] = 0.0

        anchor_diagnostics = radar_window_support(points, self.window_size)
        sampled, token_mask = frame_balanced_sample(
            points,
            self.num_points,
            self.window_size,
            rng if self.training else None,
        )
        features = robust_feature_normalize(
            sampled,
            self.feature_config,
            cloud_anchor=(
                cloud_anchor
                if self.feature_config.spatial_mode == "cloud_anchor"
                else None
            ),
        )
        if (
            self.training
            and self.feature_config.signal_mode != "none"
            and rng.random() < self.signal_dropout_probability
        ):
            # Signal is immediately before relative time in both signal modes.
            features[:, -2] = 0.0

        encoded_pose = encode_pose_target(
            pose,
            self.target_mode,
            cloud_anchor=(
                cloud_anchor
                if self.target_mode == "cloud_anchor_relative"
                else None
            ),
        )
        normalized = np.clip(
            (encoded_pose - self.low) / (self.high - self.low),
            0.0,
            1.0,
        )
        labels = np.rint(normalized * (self.bins - 1)).astype(np.int64)

        return {
            "points": torch.from_numpy(features),
            "token_mask": torch.from_numpy(token_mask),
            "labels": torch.from_numpy(labels),
            "pose": torch.from_numpy(pose.astype(np.float32)),
            "cloud_anchor": torch.from_numpy(cloud_anchor),
            "window_point_count": int(anchor_diagnostics["window_point_count"]),
            "center_point_count": int(anchor_diagnostics["center_point_count"]),
            "index": int(index),
            "centre_frame_index": int(centre_frame),
        }


def cap_dataset(dataset: Dataset, maximum: int, seed: int) -> Dataset:
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(dataset), size=maximum, replace=False))
    return Subset(dataset, ids.tolist())


def set_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    """Propagate augmentation epoch through a possible Subset wrapper."""
    if isinstance(dataset, Subset):
        set_dataset_epoch(dataset.dataset, epoch)
    elif hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


# -----------------------------------------------------------------------------
# Point-token Transformer + joint queries + 3-axis SimCC
# -----------------------------------------------------------------------------

class PointQuerySimCC(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        joints: int = 17,
        bins: int = 192,
        dim: int = 128,
        heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.joints = joints
        self.bins = bins
        self.point_embed = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.joint_queries = nn.Parameter(torch.randn(1, joints, dim) * 0.02)
        self.out_norm = nn.LayerNorm(dim)
        self.axis_heads = nn.ModuleList([nn.Linear(dim, bins) for _ in range(3)])

    def forward(
        self,
        points: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # token_mask=True means a real point; Transformer masks use True for padding.
        memory = self.encoder(
            self.point_embed(points),
            src_key_padding_mask=~token_mask.bool(),
        )
        queries = self.joint_queries.expand(points.shape[0], -1, -1)
        joints = self.out_norm(self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=~token_mask.bool(),
        ))
        return tuple(head(joints) for head in self.axis_heads)


def simcc_loss(logits: Sequence[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    losses = [
        F.cross_entropy(
            axis_logits.reshape(-1, axis_logits.shape[-1]),
            labels[..., axis].reshape(-1),
            label_smoothing=0.03,
        )
        for axis, axis_logits in enumerate(logits)
    ]
    return sum(losses) / len(losses)


def decode_simcc(
    logits: Sequence[torch.Tensor],
    low: torch.Tensor,
    high: torch.Tensor,
    target_mode: str,
    cloud_anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode per-joint SimCC coordinates and reconstruct absolute pose."""
    if low.shape != (17, 3) or high.shape != (17, 3):
        raise ValueError(f"Expected [17,3] bounds, got {low.shape} and {high.shape}")
    bins = logits[0].shape[-1]
    grid = torch.linspace(
        0.0,
        1.0,
        bins,
        device=logits[0].device,
        dtype=logits[0].dtype,
    )
    coordinates = []
    for axis, axis_logits in enumerate(logits):
        unit = (axis_logits.softmax(dim=-1) * grid).sum(dim=-1)
        axis_low = low[:, axis][None, :]
        axis_high = high[:, axis][None, :]
        coordinates.append(axis_low + unit * (axis_high - axis_low))
    encoded = torch.stack(coordinates, dim=-1)
    return decode_pose_target(encoded, target_mode, cloud_anchor)


# -----------------------------------------------------------------------------
# Pose metrics
# -----------------------------------------------------------------------------

def pose_root(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose)[..., 0, :]


def similarity_align_one(
    pred: np.ndarray,
    target: np.ndarray,
    joint_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Similarity-align pred to target using selected joints, then transform all joints."""
    x_all = np.asarray(pred, dtype=np.float64)
    y_all = np.asarray(target, dtype=np.float64)
    ids = np.arange(len(x_all)) if joint_indices is None else np.asarray(joint_indices)
    x = x_all[ids]
    y = y_all[ids]

    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    x_centered, y_centered = x - mean_x, y - mean_y
    norm_x = np.linalg.norm(x_centered)
    norm_y = np.linalg.norm(y_centered)
    if norm_x < 1e-9 or norm_y < 1e-9:
        return x_all.astype(np.float32)

    x_normalized = x_centered / norm_x
    y_normalized = y_centered / norm_y
    u, singular_values, vt = np.linalg.svd(x_normalized.T @ y_normalized)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    scale = norm_y / norm_x * float(singular_values.sum())
    translation = mean_y - scale * mean_x @ rotation
    return (scale * x_all @ rotation + translation).astype(np.float32)


def fit_global_similarity(
    pred: np.ndarray,
    target: np.ndarray,
    joint_indices: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    ids = np.arange(pred.shape[1]) if joint_indices is None else np.asarray(joint_indices)
    x = np.asarray(pred, dtype=np.float64)[:, ids].reshape(-1, 3)
    y = np.asarray(target, dtype=np.float64)[:, ids].reshape(-1, 3)
    valid = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        raise ValueError("Not enough valid joints to fit a global similarity transform")

    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    x_centered, y_centered = x - mean_x, y - mean_y
    u, singular_values, vt = np.linalg.svd(x_centered.T @ y_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    denominator = float(np.sum(x_centered * x_centered))
    scale = float(singular_values.sum() / max(denominator, 1e-12))
    translation = mean_y - scale * mean_x @ rotation
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def apply_global_similarity(
    pred: np.ndarray,
    transform: tuple[float, np.ndarray, np.ndarray],
) -> np.ndarray:
    scale, rotation, translation = transform
    return (scale * np.asarray(pred) @ rotation + translation).astype(np.float32)


def pose_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    joint_indices: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid_frames = (
        np.all(np.isfinite(pred), axis=(1, 2))
        & np.all(np.isfinite(target), axis=(1, 2))
    )
    pred = pred[valid_frames]
    target = target[valid_frames]
    if len(pred) == 0:
        return {f"{prefix}frames": 0}

    ids = np.arange(pred.shape[1]) if joint_indices is None else np.asarray(joint_indices)
    error = np.linalg.norm(pred[:, ids] - target[:, ids], axis=-1)

    pred_root_relative = pred - pose_root(pred)[:, None, :]
    target_root_relative = target - pose_root(target)[:, None, :]
    root_relative_error = np.linalg.norm(
        pred_root_relative[:, ids] - target_root_relative[:, ids],
        axis=-1,
    )

    aligned = np.stack([
        similarity_align_one(p, t, joint_indices=ids)
        for p, t in zip(pred, target)
    ])
    aligned_error = np.linalg.norm(aligned[:, ids] - target[:, ids], axis=-1)

    return {
        f"{prefix}frames": int(len(pred)),
        f"{prefix}mpjpe_mm": float(error.mean() * 1000.0),
        f"{prefix}median_joint_error_mm": float(np.median(error) * 1000.0),
        f"{prefix}root_relative_mpjpe_mm": float(root_relative_error.mean() * 1000.0),
        f"{prefix}pa_mpjpe_mm": float(aligned_error.mean() * 1000.0),
        f"{prefix}pck_50mm": float((error <= 0.05).mean()),
        f"{prefix}pck_100mm": float((error <= 0.10).mean()),
        f"{prefix}pck_150mm": float((error <= 0.15).mean()),
    }


def pose_metric_bundle(pred: np.ndarray, target: np.ndarray) -> dict[str, object]:
    return {
        "full17_provisional": pose_metrics(pred, target),
        "body15_excluding_uncertain_face_head": pose_metrics(
            pred,
            target,
            joint_indices=BODY15_INDICES,
        ),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    bounds: PoseBounds,
    target_mode: str,
    device: torch.device,
    amp: bool,
    progress: bool,
    description: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    low_np, high_np = bounds.arrays()
    low = torch.tensor(low_np, device=device)
    high = torch.tensor(high_np, device=device)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    losses: list[float] = []

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=description,
        leave=False,
        disable=not progress,
    )
    for batch in iterator:
        points = batch["points"].to(device, non_blocking=True)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        cloud_anchor = batch["cloud_anchor"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(points, token_mask)
            loss = simcc_loss(logits, labels)
            pred = decode_simcc(logits, low, high, target_mode, cloud_anchor)
        losses.append(float(loss.item()))
        predictions.append(pred.float().cpu().numpy())
        targets.append(batch["pose"].numpy())
        indices.append(np.asarray(batch["index"]))

    pred_np = np.concatenate(predictions)
    target_np = np.concatenate(targets)
    index_np = np.concatenate(indices)
    metrics: dict[str, object] = {
        "loss": float(np.mean(losses)),
        **pose_metric_bundle(pred_np, target_np),
    }
    return metrics, pred_np, target_np, index_np


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        # Epoch state lives in shared memory, so persistent workers can retain
        # expensive per-frame anchor caches without repeating augmentations.
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def train_command(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    root = Path(args.packed_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    feature_config = FeatureConfig(
        signal_mode=args.signal_mode,
        doppler_limit_mps=args.doppler_limit_mps,
        spatial_mode=args.spatial_mode,
    )
    if (
        args.target_mode == "cloud_anchor_relative"
        and feature_config.spatial_mode != "cloud_anchor"
    ):
        raise ValueError(
            "cloud_anchor_relative targets require --spatial-mode cloud_anchor"
        )
    if not (0.0 < args.body_scale_min <= args.body_scale_max):
        raise ValueError("Require 0 < body_scale_min <= body_scale_max")
    if not 0.0 <= args.x_reflection_probability <= 1.0:
        raise ValueError("x_reflection_probability must be in [0,1]")
    if not 0.0 <= args.signal_dropout_probability <= 1.0:
        raise ValueError("signal_dropout_probability must be in [0,1]")
    translation_aug_m = (
        args.translation_aug_x_m,
        args.translation_aug_y_m,
        args.translation_aug_z_m,
    )
    if any(v < 0 for v in translation_aug_m):
        raise ValueError("Translation augmentation limits must be non-negative")
    if any(v < 0 for v in (args.roll_aug_deg, args.pitch_aug_deg, args.yaw_aug_deg)):
        raise ValueError("Rotation augmentation limits must be non-negative")
    if args.sample_period_sec <= 0:
        raise ValueError("sample_period_sec must be positive")
    if args.inference_min_center_points < 1 or args.inference_min_window_points < 1:
        raise ValueError("Inference point-count defaults must be positive")
    if not 1 <= args.inference_min_valid_window_frames <= args.window_size:
        raise ValueError(
            "inference_min_valid_window_frames must be between 1 and window_size"
        )

    records = discover_packed_records(root)
    train_records, val_records, test_records = split_records(
        records,
        args.split,
        args.seed,
        args.test_env,
        args.val_env,
    )
    bounds = estimate_pose_bounds(
        train_records,
        args.stats_frames,
        args.seed,
        args.target_mode,
        translation_aug_m,
        args.body_scale_max,
        args.window_size,
    )

    preprocessing_contract = {
        "script_version": SCRIPT_VERSION,
        "mmfi_point_order": MMFI_POINT_ORDER.tolist(),
        "mmfi_internal_columns": ["x", "y", "z", "doppler", "intensity"],
        "mmfi_pose_order": MMFI_POSE_ORDER.tolist(),
        "feature_config": asdict(feature_config),
        "frame_balanced_sampling": True,
        "sampling_strategy": "per_frame_masked_padding_random_train_fps_eval",
        "activity_boundary_windows": True,
        "sample_period_sec": args.sample_period_sec,
        "cloud_anchor_algorithm": (
            "per_frame_geometric_median_coordinate_theil_sen_at_t0"
            if feature_config.spatial_mode == "cloud_anchor"
            else None
        ),
        "target_mode": args.target_mode,
        "translation_aug_m": list(translation_aug_m),
        "body_scale_range": [args.body_scale_min, args.body_scale_max],
        "rotation_aug_deg": {
            "roll": args.roll_aug_deg,
            "pitch": args.pitch_aug_deg,
            "yaw": args.yaw_aug_deg,
        },
        "x_reflection_probability": args.x_reflection_probability,
        "signal_dropout_probability": args.signal_dropout_probability,
    }
    split_manifest = {
        "train": [asdict(r) for r in train_records],
        "val": [asdict(r) for r in val_records],
        "test": [asdict(r) for r in test_records],
        "pose_bounds": asdict(bounds),
        "mmfi17_names": MMFI17_NAMES,
        "preprocessing_contract": preprocessing_contract,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
    }
    (out / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )

    dataset_kwargs = dict(
        bounds=bounds,
        bins=args.bins,
        num_points=args.num_points,
        window_size=args.window_size,
        seed=args.seed,
        feature_config=feature_config,
        target_mode=args.target_mode,
        yaw_aug_deg=args.yaw_aug_deg,
        pitch_aug_deg=args.pitch_aug_deg,
        roll_aug_deg=args.roll_aug_deg,
        translation_aug_m=translation_aug_m,
        body_scale_min=args.body_scale_min,
        body_scale_max=args.body_scale_max,
        x_reflection_probability=args.x_reflection_probability,
        signal_dropout_probability=args.signal_dropout_probability,
    )
    train_base = PackedMMFiDataset(
        records=train_records,
        training=True,
        **dataset_kwargs,
    )
    val_base = PackedMMFiDataset(
        records=val_records,
        training=False,
        **dataset_kwargs,
    )
    test_base = PackedMMFiDataset(
        records=test_records,
        training=False,
        **dataset_kwargs,
    )
    train_ds: Dataset = train_base
    val_ds: Dataset = val_base
    test_ds: Dataset = test_base

    train_ds = cap_dataset(train_ds, args.max_train_frames, args.seed)
    val_ds = cap_dataset(val_ds, args.max_val_frames, args.seed + 1)
    test_ds = cap_dataset(test_ds, args.max_test_frames, args.seed + 2)

    train_loader = make_loader(train_ds, args.batch_size, args.workers, True, device)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, False, device)
    test_loader = make_loader(test_ds, args.batch_size, args.workers, False, device)

    model_config = {
        "input_dim": feature_config.input_dim,
        "joints": 17,
        "bins": args.bins,
        "dim": args.dim,
        "heads": args.heads,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
    }
    model = PointQuerySimCC(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )

    low_np, high_np = bounds.arrays()
    low = torch.tensor(low_np, device=device)
    high = torch.tensor(high_np, device=device)
    best_val = float("inf")
    history: list[dict[str, float | int]] = []

    print(json.dumps({
        "script_version": SCRIPT_VERSION,
        "device": str(device),
        "records": len(records),
        "windows": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "source_windows_excluded_empty_or_invalid": {
            "train": train_base.excluded_empty_or_invalid_windows,
            "val": val_base.excluded_empty_or_invalid_windows,
            "test": test_base.excluded_empty_or_invalid_windows,
        },
        "window_size": args.window_size,
        "num_points": args.num_points,
        "feature_config": asdict(feature_config),
        "target_mode": args.target_mode,
        "model_input_dim": feature_config.input_dim,
        "parameters": sum(p.numel() for p in model.parameters()),
        "preprocessing_contract": preprocessing_contract,
        "bounds_shape": list(low_np.shape),
    }, indent=2))

    for epoch in range(1, args.epochs + 1):
        set_dataset_epoch(train_ds, epoch)
        model.train()
        train_losses: list[float] = []
        iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train {epoch}/{args.epochs}",
            leave=False,
            disable=not args.progress,
        )
        for batch in iterator:
            points = batch["points"].to(device, non_blocking=True)
            token_mask = batch["token_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            target_pose = batch["pose"].to(device, non_blocking=True)
            cloud_anchor = batch["cloud_anchor"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(points, token_mask)
                loss = simcc_loss(logits, labels)
                pred_pose = decode_simcc(
                    logits,
                    low,
                    high,
                    args.target_mode,
                    cloud_anchor,
                )
                pred_bones = torch.stack([
                    pred_pose[:, b] - pred_pose[:, a]
                    for a, b in MMFI17_EDGES
                ], dim=1)
                true_bones = torch.stack([
                    target_pose[:, b] - target_pose[:, a]
                    for a, b in MMFI17_EDGES
                ], dim=1)
                loss = loss + args.bone_weight * F.smooth_l1_loss(
                    pred_bones,
                    true_bones,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            if args.progress:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val_metrics, _, _, _ = evaluate_loader(
            model,
            val_loader,
            bounds,
            args.target_mode,
            device,
            args.amp,
            args.progress,
            "Validate",
        )
        val_mpjpe = float(
            val_metrics["body15_excluding_uncertain_face_head"]["mpjpe_mm"]
        )
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(train_losses)),
            "val_body15_mpjpe_mm": val_mpjpe,
            "val_full17_mpjpe_mm": float(
                val_metrics["full17_provisional"]["mpjpe_mm"]
            ),
            "val_loss": float(val_metrics["loss"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(json.dumps(row))

        if val_mpjpe < best_val:
            best_val = val_mpjpe
            torch.save({
                "script_version": SCRIPT_VERSION,
                "model": model.state_dict(),
                "model_config": model_config,
                "pose_bounds": asdict(bounds),
                "target_mode": args.target_mode,
                "feature_config": asdict(feature_config),
                "frame_balanced_sampling": True,
                "sampling_strategy": (
                    "per_frame_masked_padding_random_train_fps_eval"
                ),
                "sample_period_sec": args.sample_period_sec,
                "inference_quality_defaults": {
                    "min_center_points": args.inference_min_center_points,
                    "min_window_points": args.inference_min_window_points,
                    "min_valid_window_frames": args.inference_min_valid_window_frames,
                },
                "inference_quality_default_provenance": (
                    "Conservative MM-Fi lower-tail support defaults (the centre "
                    "threshold is its per-frame p10) checked against 19_MM radar "
                    "point-count diagnostics; no 19_MM pose labels were used."
                ),
                "window_size": args.window_size,
                "num_points": args.num_points,
                "mmfi_point_order": MMFI_POINT_ORDER.tolist(),
                "mmfi_pose_order": MMFI_POSE_ORDER.tolist(),
                "mmfi17_names": MMFI17_NAMES,
                "mmfi17_edges": MMFI17_EDGES,
                "uncertain_joint_indices": [9, 10],
                "preprocessing_contract": preprocessing_contract,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, out / "best.pt")

    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, pred, target, indices = evaluate_loader(
        model,
        test_loader,
        bounds,
        args.target_mode,
        device,
        args.amp,
        args.progress,
        "Test",
    )
    (out / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        out / "test_predictions.npz",
        pred=pred,
        target=target,
        index=indices,
        joint_names=np.asarray(MMFI17_NAMES),
    )
    print(json.dumps({"test": test_metrics}, indent=2))


# -----------------------------------------------------------------------------
# Checkpoint loading and generic inference
# -----------------------------------------------------------------------------

def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[nn.Module, PoseBounds, dict, FeatureConfig, str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    names = checkpoint.get("mmfi17_names")
    if names != MMFI17_NAMES:
        raise ValueError(
            "Checkpoint topology is absent or does not match MMFI17_NAMES. "
            "Retrain with this script; checkpoints with a different joint topology are not accepted."
        )
    if checkpoint.get("mmfi_point_order") != MMFI_POINT_ORDER.tolist():
        raise ValueError(
            "Checkpoint MM-Fi point order does not match corrected "
            "[1,0,2,4,3] -> [x,y,z,Doppler,intensity]. Retrain with this script."
        )
    if int(checkpoint.get("window_size", 0)) < 1:
        raise ValueError("Checkpoint does not contain a valid window_size")
    if checkpoint.get("frame_balanced_sampling") is not True:
        raise ValueError(
            "Checkpoint predates frame-balanced temporal sampling. Retrain with this script."
        )

    try:
        feature_config = FeatureConfig(**checkpoint["feature_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Checkpoint has no valid feature_config; retraining is required") from exc
    target_mode = str(checkpoint.get("target_mode", ""))
    if target_mode not in TARGET_MODES:
        raise ValueError("Checkpoint has no valid target_mode; retraining is required")
    model_config = checkpoint.get("model_config", {})
    if int(model_config.get("input_dim", -1)) != feature_config.input_dim:
        raise ValueError(
            "Checkpoint model input_dim does not match its feature_config: "
            f"{model_config.get('input_dim')} != {feature_config.input_dim}"
        )

    bounds = PoseBounds(**checkpoint["pose_bounds"])
    bounds.arrays()  # validate exact [17,3] contract before constructing model
    model = PointQuerySimCC(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, bounds, checkpoint, feature_config, target_mode


@torch.no_grad()
def predict_point_clouds(
    model: nn.Module,
    bounds: PoseBounds,
    point_clouds: Iterable[np.ndarray],
    num_points: int,
    window_size: int,
    feature_config: FeatureConfig,
    target_mode: str,
    batch_size: int,
    device: torch.device,
    amp: bool,
    *,
    total: int | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    low_np, high_np = bounds.arrays()
    low = torch.tensor(low_np, device=device)
    high = torch.tensor(high_np, device=device)
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    output_anchors: list[np.ndarray] = []

    def flush() -> None:
        if not features:
            return
        x = torch.from_numpy(np.stack(features)).to(device)
        mask = torch.from_numpy(np.stack(masks)).to(device)
        anchor = torch.from_numpy(np.stack(anchors)).to(device)
        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            pred = decode_simcc(model(x, mask), low, high, target_mode, anchor)
        outputs.append(pred.float().cpu().numpy())
        output_anchors.append(anchor.float().cpu().numpy())
        features.clear()
        masks.clear()
        anchors.clear()

    iterator = tqdm(
        point_clouds,
        total=total,
        desc="Model inference",
        leave=False,
        disable=not progress,
    )
    for points in iterator:
        cloud_anchor, _ = robust_cloud_anchor(points)
        sampled, token_mask = frame_balanced_sample(
            np.asarray(points, dtype=np.float32),
            num_points,
            window_size,
            None,
        )
        features.append(robust_feature_normalize(
            sampled,
            feature_config,
            cloud_anchor=(
                cloud_anchor
                if feature_config.spatial_mode == "cloud_anchor"
                else None
            ),
        ))
        masks.append(token_mask)
        anchors.append(cloud_anchor)
        if len(features) >= batch_size:
            flush()
    flush()

    if not outputs:
        return (
            np.empty((0, 17, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )
    return np.concatenate(outputs, axis=0), np.concatenate(output_anchors, axis=0)


# -----------------------------------------------------------------------------
# SyncWB inference and single-person evaluation
# -----------------------------------------------------------------------------

def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def kinect_pose32_to_radar_mmfi17(
    pose3d: np.ndarray,
    person_index: int = 0,
    *,
    extrinsic_euler_deg: Sequence[float] = (0.0, 0.0, 0.0),
    extrinsic_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray | None:
    """Convert Azure Kinect pose into the radar frame used for evaluation.

    The axis-basis conversion [x,y,z] mm -> [x,z,-y] m is always applied.
    A measured residual Kinect-to-radar rigid transform can then be supplied as
    XYZ Euler angles and translation. Defaults are zero for backward behaviour,
    but zero extrinsics should not be interpreted as calibrated co-location.
    """
    arr = np.asarray(pose3d, dtype=np.float32)
    if (
        arr.ndim != 3
        or arr.shape[0] <= person_index
        or arr.shape[1] < 32
        or arr.shape[2] < 3
    ):
        return None

    xyz = arr[person_index, :, :3][:, [0, 2, 1]].copy()
    xyz[:, 2] *= -1.0
    xyz *= 1e-3

    euler = np.asarray(extrinsic_euler_deg, dtype=np.float32)
    translation = np.asarray(extrinsic_translation_m, dtype=np.float32)
    if euler.shape != (3,) or translation.shape != (3,):
        raise ValueError("Extrinsic Euler angles and translation must each have three values")
    rotation = rotation_matrix_xyz(*np.radians(euler).tolist())
    xyz = xyz @ rotation.T + translation
    return xyz[KINECT32_TO_MMFI17]


def filter_syncwb_radar(points: np.ndarray, filter_noise: bool) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 5:
        return np.empty((0, 5), dtype=np.float32)

    mask = np.all(np.isfinite(p[:, :5]), axis=1)
    if filter_noise and p.shape[1] >= 6:
        target_id = p[:, 5]
        mask &= np.isfinite(target_id)
        for bad in NOISE_TARGET_IDS:
            mask &= target_id != bad
    return p[mask, :5]


def attach_source_num_people(
    rows: pd.DataFrame,
    sample_summary: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "subject_id",
        "run_id",
        "device_type",
        "sample_index",
        "num_people",
    }
    if sample_summary.empty or not required.issubset(sample_summary.columns):
        raise ValueError(
            "SAMPLE_SUMMARY is missing or has no num_people column. "
            "Single-person-only evaluation cannot be enforced."
        )

    summary = sample_summary[list(required)].copy()
    summary["sample_index"] = pd.to_numeric(
        summary["sample_index"], errors="coerce"
    ).astype("Int64")
    summary["num_people"] = pd.to_numeric(summary["num_people"], errors="coerce")
    summary = summary.rename(columns={
        "run_id": "source_run_id",
        "device_type": "source_device_type",
        "sample_index": "source_sample_index",
    })

    merged = rows.merge(
        summary,
        on=[
            "subject_id",
            "source_run_id",
            "source_device_type",
            "source_sample_index",
        ],
        how="left",
        validate="many_to_one",
    )
    return merged


def lag_sweep_root_mpjpe(
    pred: np.ndarray,
    target: np.ndarray,
    source_sample_indices: np.ndarray,
    eligible_mask: np.ndarray,
    maximum: int,
) -> pd.DataFrame:
    """Compare prediction at source index i with GT at source index i+shift.

    Using explicit sample indices avoids treating gaps from skipped/multi-person
    frames as if they were consecutive frames.
    """
    pred_root_relative = pred - pose_root(pred)[:, None, :]
    target_root_relative = target - pose_root(target)[:, None, :]
    indices = np.asarray(source_sample_indices, dtype=np.int64)
    eligible = np.asarray(eligible_mask, dtype=bool)

    target_by_sample = {
        int(indices[i]): target_root_relative[i]
        for i in np.flatnonzero(eligible)
    }

    rows: list[dict[str, float | int]] = []
    for shift in range(-maximum, maximum + 1):
        errors: list[float] = []
        for i in np.flatnonzero(eligible):
            shifted_target = target_by_sample.get(int(indices[i]) + shift)
            if shifted_target is None:
                continue
            error = np.linalg.norm(
                pred_root_relative[i, BODY15_INDICES]
                - shifted_target[BODY15_INDICES],
                axis=-1,
            ).mean()
            errors.append(float(error))

        if errors:
            rows.append({
                "rgb_frame_shift": int(shift),
                "body15_root_relative_mpjpe_mm": float(np.mean(errors) * 1000.0),
                "frames": int(len(errors)),
            })

    return pd.DataFrame(rows)



def open_syncwb_ragged_readers(
    run_assets: pd.DataFrame,
    artifact_root: str | Path,
    required_payloads: set[tuple[str, str, str, str]],
) -> dict[tuple[str, str, str, str], object]:
    """Open each required run-level payload bundle exactly once.

    Per-sample payload access is convenient for interactive use, but its
    current artifact lookup scans SAMPLE_ARTIFACT for every sample request. A
    five-frame sliding-window pass can therefore trigger hundreds of thousands
    of full-table scans. For offline inference, resolve the small RUN_ASSET table
    once and read directly through RaggedNpzReader.
    """
    from sync_workbench.storage.artifact_store import ArtifactStore
    from sync_workbench.storage.ragged_npz import RaggedNpzReader

    payload_asset_roles = {
        "radar_points": "radar_points_bundle",
        "pose3d": "pose3d_bundle",
    }

    required_columns = {
        "subject_id",
        "run_id",
        "device_type",
        "asset_role",
        "asset_ref",
    }
    if run_assets.empty or not required_columns.issubset(run_assets.columns):
        raise ValueError(
            "RUN_ASSET is empty or missing columns required for fast payload loading"
        )

    assets = run_assets[list(required_columns)].copy()
    for column in ("subject_id", "run_id", "device_type", "asset_role", "asset_ref"):
        assets[column] = assets[column].astype(str)
    assets = assets.drop_duplicates()

    artifact_store = ArtifactStore(artifact_root)
    readers: dict[tuple[str, str, str, str], object] = {}

    for key in sorted(required_payloads):
        subject_id, run_id, device_type, payload_role = key
        asset_role = payload_asset_roles.get(payload_role)
        if asset_role is None:
            raise ValueError(f"Unsupported fast payload role: {payload_role!r}")

        matches = assets[
            (assets["subject_id"] == subject_id)
            & (assets["run_id"] == run_id)
            & (assets["device_type"] == device_type)
            & (assets["asset_role"] == asset_role)
        ]
        if matches.empty:
            raise KeyError(
                f"No RUN_ASSET role {asset_role!r} for "
                f"{subject_id}/{run_id}/{device_type}"
            )

        refs = matches["asset_ref"].dropna().astype(str).unique().tolist()
        if len(refs) != 1:
            raise ValueError(
                f"Expected one {asset_role!r} bundle for "
                f"{subject_id}/{run_id}/{device_type}, found {refs}"
            )

        path = artifact_store.path_for_ref(refs[0])
        readers[key] = RaggedNpzReader(path)

    return readers


def select_syncwb_mapping_versions(
    store: object,
    subjects: Sequence[str],
    mapping_methods: Sequence[str],
) -> pd.DataFrame:
    """Select every mapping version matching a subject/method allow-list."""
    subject_values = list(dict.fromkeys(str(value) for value in subjects))
    method_values = list(dict.fromkeys(str(value) for value in mapping_methods))
    if not subject_values:
        raise ValueError("At least one SyncWB subject is required")
    if not method_values:
        raise ValueError("At least one SyncWB mapping method is required")

    versions = store.read_table("MAPPING_VERSION")
    required = {
        "subject_id",
        "mapping_version_id",
        "source_run_id",
        "source_device_type",
        "target_run_id",
        "target_device_type",
        "mapping_method",
    }
    if versions.empty or not required.issubset(versions.columns):
        raise ValueError(
            "MAPPING_VERSION is empty or missing columns required for selection"
        )
    selected = versions[
        versions["subject_id"].astype(str).isin(subject_values)
        & versions["mapping_method"].astype(str).isin(method_values)
    ].copy()
    if selected.empty:
        available = (
            versions[versions["subject_id"].astype(str).isin(subject_values)]
            [["subject_id", "mapping_method"]]
            .drop_duplicates()
            .sort_values(["subject_id", "mapping_method"])
            .to_dict("records")
        )
        raise ValueError(
            "No MAPPING_VERSION rows matched the requested subjects and methods. "
            f"Available for those subjects: {available}"
        )
    for column in required:
        selected[column] = selected[column].astype(str)
    return selected.sort_values(
        ["subject_id", "mapping_version_id"]
    ).reset_index(drop=True)


def primary_rows_for_mapping_versions(
    store: object,
    versions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load primary mappings and enforce known-single-person source frames."""
    mappings = store.read_table("SAMPLE_MAPPING")
    keys = versions[[
        "subject_id",
        "mapping_version_id",
        "mapping_method",
    ]].drop_duplicates()
    rows = mappings.merge(
        keys,
        on=["subject_id", "mapping_version_id"],
        how="inner",
        validate="many_to_one",
    )
    if "is_primary" in rows.columns:
        rows = rows[bool_series(rows["is_primary"])].copy()
    if rows.empty:
        raise ValueError("No primary SAMPLE_MAPPING rows matched the selected versions")

    for column in (
        "subject_id",
        "mapping_version_id",
        "source_run_id",
        "source_device_type",
        "target_run_id",
        "target_device_type",
    ):
        rows[column] = rows[column].astype(str)
    for column in ("source_sample_index", "target_sample_index"):
        rows[column] = pd.to_numeric(rows[column], errors="raise").astype(int)

    rows = attach_source_num_people(rows, store.read_table("SAMPLE_SUMMARY"))
    people = pd.to_numeric(rows["num_people"], errors="coerce")
    known_single = people.eq(1)
    counts = {
        "primary_mapping_rows": int(len(rows)),
        "known_single_person_rows": int(known_single.sum()),
        "excluded_multi_person_rows": int(people.gt(1).sum()),
        "excluded_unknown_or_zero_person_rows": int(
            (~known_single & ~people.gt(1)).sum()
        ),
    }
    rows = rows.loc[known_single].copy()
    if rows.empty:
        raise ValueError(
            "No selected primary mapping rows have SAMPLE_SUMMARY.num_people == 1"
        )
    return (
        rows.sort_values(
            ["subject_id", "mapping_version_id", "source_sample_index"]
        ).reset_index(drop=True),
        counts,
    )


def target_run_fps(
    rows: pd.DataFrame,
    device_runs: pd.DataFrame,
    override_fps: float | None,
) -> tuple[dict[tuple[str, str, str], float], str]:
    """Resolve radar FPS independently for every selected target run."""
    target_keys = rows[[
        "subject_id",
        "target_run_id",
        "target_device_type",
    ]].drop_duplicates()
    resolved: dict[tuple[str, str, str], float] = {}
    if override_fps is not None:
        if float(override_fps) <= 0:
            raise ValueError("target_fps must be positive")
        for row in target_keys.itertuples(index=False):
            resolved[(str(row.subject_id), str(row.target_run_id), str(row.target_device_type))] = (
                float(override_fps)
            )
        return resolved, "command_line"

    for row in target_keys.itertuples(index=False):
        key = (
            str(row.subject_id),
            str(row.target_run_id),
            str(row.target_device_type),
        )
        matches = device_runs[
            (device_runs["subject_id"].astype(str) == key[0])
            & (device_runs["run_id"].astype(str) == key[1])
            & (device_runs["device_type"].astype(str) == key[2])
        ]
        values = pd.to_numeric(
            matches.get("nominal_fps", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().unique()
        values = [float(value) for value in values if float(value) > 0]
        if len(values) != 1:
            raise ValueError(
                "Could not derive exactly one positive nominal_fps for target run "
                f"{key}; supply --target-fps explicitly"
            )
        resolved[key] = values[0]
    return resolved, "DEVICE_RUN.nominal_fps"


@dataclass
class SyncWBWindowCollection:
    windows: list[np.ndarray]
    poses: np.ndarray
    cloud_anchors: np.ndarray
    rows: pd.DataFrame
    selected_versions: pd.DataFrame
    counts: dict[str, int]
    fps_source: str


def load_syncwb_supervised_windows(
    *,
    store: object,
    artifact_root: str | Path,
    versions: pd.DataFrame,
    checkpoint: dict,
    person_index: int,
    target_fps_override: float | None,
    min_center_points: int,
    min_window_points: int,
    min_valid_window_frames: int,
    filter_noise: bool,
    extrinsic_euler_deg: Sequence[float],
    extrinsic_translation_m: Sequence[float],
    max_frames: int,
    seed: int,
    progress: bool,
) -> SyncWBWindowCollection:
    """Materialise quality-filtered radar windows with finite Kinect labels."""
    rows, counts = primary_rows_for_mapping_versions(store, versions)
    if max_frames > 0 and len(rows) > max_frames:
        # A proportional sample prevents a diagnostic cap from silently selecting
        # only the first mapping version.
        rng = np.random.default_rng(seed)
        selected_ids: list[int] = []
        for _, group in rows.groupby(
            ["subject_id", "mapping_version_id"], sort=False
        ):
            quota = max(1, int(round(max_frames * len(group) / len(rows))))
            quota = min(quota, len(group))
            selected_ids.extend(
                rng.choice(group.index.to_numpy(), size=quota, replace=False).tolist()
            )
        if len(selected_ids) > max_frames:
            selected_ids = rng.choice(
                np.asarray(selected_ids), size=max_frames, replace=False
            ).tolist()
        rows = rows.loc[sorted(selected_ids)].reset_index(drop=True)
        counts["rows_after_max_frames_cap"] = int(len(rows))

    window_size = int(checkpoint["window_size"])
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("Checkpoint window_size must be a positive odd integer")
    if min_center_points < 1 or min_window_points < 1:
        raise ValueError("Point-count thresholds must be positive")
    if not 1 <= min_valid_window_frames <= window_size:
        raise ValueError("min_valid_window_frames must be between 1 and window_size")
    half_window = window_size // 2
    sample_period_sec = float(
        checkpoint.get("sample_period_sec", MMFI_SAMPLE_PERIOD_SEC)
    )
    fps_by_run, fps_source = target_run_fps(
        rows,
        store.read_table("DEVICE_RUN"),
        target_fps_override,
    )
    stride_by_run = {
        key: max(1, int(round(sample_period_sec * fps)))
        for key, fps in fps_by_run.items()
    }

    run_assets = store.read_table("RUN_ASSET")
    required_payloads: set[tuple[str, str, str, str]] = set()
    radar_requests: set[tuple[str, str, str, int]] = set()
    pose_requests: set[tuple[str, str, str, int]] = set()
    for row in rows.itertuples(index=False):
        subject_id = str(row.subject_id)
        target_key = (
            subject_id,
            str(row.target_run_id),
            str(row.target_device_type),
        )
        source_key = (
            subject_id,
            str(row.source_run_id),
            str(row.source_device_type),
        )
        required_payloads.add((*target_key, "radar_points"))
        required_payloads.add((*source_key, "pose3d"))
        stride = stride_by_run[target_key]
        target_centre = int(row.target_sample_index)
        for offset in range(-half_window, half_window + 1):
            target_index = target_centre + offset * stride
            if target_index >= 0:
                radar_requests.add((*target_key, target_index))
        pose_requests.add((*source_key, int(row.source_sample_index)))

    readers = open_syncwb_ragged_readers(
        run_assets,
        artifact_root,
        required_payloads,
    )
    radar_cache: dict[tuple[str, str, str, int], np.ndarray | None] = {}
    for subject_id, run_id, device_type, sample_index in tqdm(
        sorted(radar_requests),
        total=len(radar_requests),
        desc="Cache fine-tuning radar frames",
        leave=False,
        disable=not progress,
    ):
        reader = readers[(subject_id, run_id, device_type, "radar_points")]
        try:
            radar_cache[(subject_id, run_id, device_type, sample_index)] = (
                filter_syncwb_radar(reader.get(sample_index), filter_noise)
            )
        except KeyError:
            radar_cache[(subject_id, run_id, device_type, sample_index)] = None

    pose_cache: dict[tuple[str, str, str, int], np.ndarray | None] = {}
    for subject_id, run_id, device_type, sample_index in tqdm(
        sorted(pose_requests),
        total=len(pose_requests),
        desc="Cache fine-tuning Kinect poses",
        leave=False,
        disable=not progress,
    ):
        reader = readers[(subject_id, run_id, device_type, "pose3d")]
        try:
            pose_cache[(subject_id, run_id, device_type, sample_index)] = (
                reader.get(sample_index)
            )
        except KeyError:
            pose_cache[(subject_id, run_id, device_type, sample_index)] = None

    windows: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    kept_rows: list[dict[str, object]] = []
    skipped = {
        "incomplete_radar_window": 0,
        "too_few_center_points": 0,
        "too_few_window_points": 0,
        "too_few_valid_window_frames": 0,
        "missing_pose_payload": 0,
        "invalid_pose": 0,
    }

    for row in tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc="Load supervised SyncWB windows",
        disable=not progress,
    ):
        subject_id = str(row.subject_id)
        target_key = (
            subject_id,
            str(row.target_run_id),
            str(row.target_device_type),
        )
        source_key = (
            subject_id,
            str(row.source_run_id),
            str(row.source_device_type),
        )
        stride = stride_by_run[target_key]
        target_centre = int(row.target_sample_index)
        frame_indices = [
            target_centre + offset * stride
            for offset in range(-half_window, half_window + 1)
        ]
        frames = [
            radar_cache.get((*target_key, frame_index))
            for frame_index in frame_indices
        ]
        if any(frame is None for frame in frames):
            skipped["incomplete_radar_window"] += 1
            continue
        stacked = stack_radar_window(frames)
        support = radar_window_support(stacked, window_size)
        if int(support["center_point_count"]) < min_center_points:
            skipped["too_few_center_points"] += 1
            continue
        if int(support["window_point_count"]) < min_window_points:
            skipped["too_few_window_points"] += 1
            continue
        if int(support["valid_window_frames"]) < min_valid_window_frames:
            skipped["too_few_valid_window_frames"] += 1
            continue

        pose3d = pose_cache.get((*source_key, int(row.source_sample_index)))
        if pose3d is None:
            skipped["missing_pose_payload"] += 1
            continue
        pose = kinect_pose32_to_radar_mmfi17(
            pose3d,
            person_index,
            extrinsic_euler_deg=extrinsic_euler_deg,
            extrinsic_translation_m=extrinsic_translation_m,
        )
        if pose is None or not np.all(np.isfinite(pose)):
            skipped["invalid_pose"] += 1
            continue
        try:
            cloud_anchor, _ = robust_cloud_anchor(stacked)
        except ValueError:
            skipped["incomplete_radar_window"] += 1
            continue

        windows.append(stacked)
        poses.append(np.asarray(pose, dtype=np.float32))
        anchors.append(cloud_anchor)
        kept_rows.append({
            "subject_id": subject_id,
            "mapping_version_id": str(row.mapping_version_id),
            "mapping_method": str(row.mapping_method),
            "source_run_id": source_key[1],
            "source_device_type": source_key[2],
            "source_sample_index": int(row.source_sample_index),
            "target_run_id": target_key[1],
            "target_device_type": target_key[2],
            "target_sample_index": target_centre,
            "target_window_start": int(frame_indices[0]),
            "target_window_end": int(frame_indices[-1]),
            "target_frame_stride": int(stride),
            "target_fps": float(fps_by_run[target_key]),
            "center_point_count": int(support["center_point_count"]),
            "window_point_count": int(support["window_point_count"]),
            "valid_window_frames": int(support["valid_window_frames"]),
        })

    if not windows:
        raise ValueError(
            "No finite single-person Kinect labels passed the radar-window quality policy"
        )
    counts.update({f"skipped_{key}": int(value) for key, value in skipped.items()})
    counts["supervised_windows"] = int(len(windows))
    return SyncWBWindowCollection(
        windows=windows,
        poses=np.stack(poses).astype(np.float32),
        cloud_anchors=np.stack(anchors).astype(np.float32),
        rows=pd.DataFrame(kept_rows),
        selected_versions=versions.copy(),
        counts=counts,
        fps_source=fps_source,
    )


class SyncWBWindowDataset(Dataset):
    """Preloaded SyncWB radar windows paired with converted MM-Fi-17 Kinect pose."""

    def __init__(
        self,
        collection: SyncWBWindowCollection,
        indices: Sequence[int],
        *,
        bounds: PoseBounds,
        bins: int,
        num_points: int,
        window_size: int,
        training: bool,
        seed: int,
        feature_config: FeatureConfig,
        target_mode: str,
        yaw_aug_deg: float,
        pitch_aug_deg: float,
        roll_aug_deg: float,
        translation_aug_m: Sequence[float],
        body_scale_min: float,
        body_scale_max: float,
        x_reflection_probability: float,
        signal_dropout_probability: float,
    ):
        self.collection = collection
        self.indices = np.asarray(indices, dtype=np.int64)
        self.bounds = bounds
        self.bins = int(bins)
        self.num_points = int(num_points)
        self.window_size = int(window_size)
        self.training = bool(training)
        self.seed = int(seed)
        self.feature_config = feature_config
        self.target_mode = str(target_mode)
        self.yaw_aug_deg = float(yaw_aug_deg)
        self.pitch_aug_deg = float(pitch_aug_deg)
        self.roll_aug_deg = float(roll_aug_deg)
        self.translation_aug_m = tuple(float(v) for v in translation_aug_m)
        self.body_scale_min = float(body_scale_min)
        self.body_scale_max = float(body_scale_max)
        self.x_reflection_probability = float(x_reflection_probability)
        self.signal_dropout_probability = float(signal_dropout_probability)
        self.low, self.high = bounds.arrays()
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()
        if len(self.indices) == 0:
            raise ValueError("SyncWBWindowDataset received no indices")

    def __len__(self) -> int:
        return int(len(self.indices))

    def set_epoch(self, epoch: int) -> None:
        self._epoch_state.fill_(int(epoch))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        collection_index = int(self.indices[int(index)])
        points = self.collection.windows[collection_index].copy()
        pose = self.collection.poses[collection_index].copy()
        cloud_anchor = self.collection.cloud_anchors[collection_index].copy()
        rng = np.random.default_rng(
            self.seed
            + collection_index * 104729
            + int(self._epoch_state.item()) * 1000003
        )
        if self.training:
            points, pose, cloud_anchor = augment_geometry(
                points,
                pose,
                cloud_anchor,
                rng,
                yaw_deg=self.yaw_aug_deg,
                pitch_deg=self.pitch_aug_deg,
                roll_deg=self.roll_aug_deg,
                translation_m=self.translation_aug_m,
                body_scale_min=self.body_scale_min,
                body_scale_max=self.body_scale_max,
                x_reflection_probability=self.x_reflection_probability,
            )
            valid = point_valid_mask(points)
            if np.any(valid):
                points[valid, :3] += rng.normal(
                    0.0, 0.008, size=(valid.sum(), 3)
                ).astype(np.float32)
                points[valid, 3] += rng.normal(
                    0.0, 0.03, size=valid.sum()
                ).astype(np.float32)
                drop = rng.random(valid.sum()) < 0.08
                if np.all(drop):
                    drop[int(rng.integers(0, len(drop)))] = False
                points[np.flatnonzero(valid)[drop]] = 0.0

        sampled, token_mask = frame_balanced_sample(
            points,
            self.num_points,
            self.window_size,
            rng if self.training else None,
        )
        features = robust_feature_normalize(
            sampled,
            self.feature_config,
            cloud_anchor=(
                cloud_anchor
                if self.feature_config.spatial_mode == "cloud_anchor"
                else None
            ),
        )
        if (
            self.training
            and self.feature_config.signal_mode != "none"
            and rng.random() < self.signal_dropout_probability
        ):
            features[:, -2] = 0.0

        encoded_pose = encode_pose_target(
            pose,
            self.target_mode,
            cloud_anchor=(
                cloud_anchor
                if self.target_mode == "cloud_anchor_relative"
                else None
            ),
        )
        normalized = np.clip(
            (encoded_pose - self.low) / (self.high - self.low),
            0.0,
            1.0,
        )
        labels = np.rint(normalized * (self.bins - 1)).astype(np.int64)
        return {
            "points": torch.from_numpy(features),
            "token_mask": torch.from_numpy(token_mask),
            "labels": torch.from_numpy(labels),
            "pose": torch.from_numpy(pose),
            "cloud_anchor": torch.from_numpy(cloud_anchor),
            "index": collection_index,
        }


def split_syncwb_train_validation(
    rows: pd.DataFrame,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a held-out temporal tail from every mapping, with a window gap."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    train_ids: list[int] = []
    val_ids: list[int] = []
    for _, group in rows.groupby(
        ["subject_id", "mapping_version_id"], sort=False
    ):
        ordered = group.sort_values("target_sample_index")
        if len(ordered) < 10:
            raise ValueError(
                "Every selected mapping needs at least 10 supervised windows "
                "for a temporal train/validation split"
            )
        n_val = max(1, int(round(len(ordered) * validation_fraction)))
        n_val = min(n_val, len(ordered) - 2)
        validation = ordered.iloc[-n_val:]
        validation_start = int(validation["target_window_start"].min())
        training = ordered[
            ordered["target_window_end"].astype(int) < validation_start
        ]
        if training.empty:
            raise ValueError(
                "Temporal validation gap removed every training window for "
                f"{ordered.iloc[0]['mapping_version_id']}"
            )
        train_ids.extend(training.index.astype(int).tolist())
        val_ids.extend(validation.index.astype(int).tolist())
    return (
        np.asarray(sorted(train_ids), dtype=np.int64),
        np.asarray(sorted(val_ids), dtype=np.int64),
    )


def set_point_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for module in (model.point_embed, model.encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def finetune_syncwb_command(args: argparse.Namespace) -> None:
    """Supervised target-domain fine-tuning without changing model topology."""
    from sync_workbench.storage.sqlite_store import SQLiteCoreStore

    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, bounds, source_checkpoint, feature_config, target_mode = load_checkpoint(
        Path(args.checkpoint),
        device,
    )
    window_size = int(source_checkpoint["window_size"])
    num_points = int(source_checkpoint.get("num_points", 320))
    model_config = dict(source_checkpoint["model_config"])
    bins = int(model_config["bins"])
    quality_defaults = source_checkpoint.get("inference_quality_defaults", {})
    min_center_points = (
        int(args.min_center_points)
        if args.min_center_points is not None
        else int(quality_defaults.get("min_center_points", 10))
    )
    min_window_points = (
        int(args.min_window_points)
        if args.min_window_points is not None
        else int(quality_defaults.get("min_window_points", 64))
    )
    min_valid_window_frames = (
        int(args.min_valid_window_frames)
        if args.min_valid_window_frames is not None
        else int(quality_defaults.get("min_valid_window_frames", 3))
    )
    if not 0.0 < args.body_scale_min <= args.body_scale_max:
        raise ValueError("Require 0 < body_scale_min <= body_scale_max")
    if not 0.0 <= args.x_reflection_probability <= 1.0:
        raise ValueError("x_reflection_probability must be in [0,1]")
    if not 0.0 <= args.signal_dropout_probability <= 1.0:
        raise ValueError("signal_dropout_probability must be in [0,1]")

    store = SQLiteCoreStore(args.sqlite)
    versions = select_syncwb_mapping_versions(
        store,
        args.subjects,
        args.mapping_methods,
    )
    collection = load_syncwb_supervised_windows(
        store=store,
        artifact_root=args.artifact_root,
        versions=versions,
        checkpoint=source_checkpoint,
        person_index=args.person_index,
        target_fps_override=args.target_fps,
        min_center_points=min_center_points,
        min_window_points=min_window_points,
        min_valid_window_frames=min_valid_window_frames,
        filter_noise=args.filter_noise,
        extrinsic_euler_deg=(
            args.kinect_to_radar_roll_deg,
            args.kinect_to_radar_pitch_deg,
            args.kinect_to_radar_yaw_deg,
        ),
        extrinsic_translation_m=(
            args.kinect_to_radar_tx_m,
            args.kinect_to_radar_ty_m,
            args.kinect_to_radar_tz_m,
        ),
        max_frames=args.max_frames,
        seed=args.seed,
        progress=args.progress,
    )
    bound_low, bound_high = bounds.arrays()
    encoded_target = encode_pose_target(
        collection.poses,
        target_mode,
        cloud_anchor=(
            collection.cloud_anchors
            if target_mode == "cloud_anchor_relative"
            else None
        ),
    )
    outside_bounds = (encoded_target < bound_low) | (encoded_target > bound_high)
    bound_diagnostics = {
        "coordinate_fraction_outside_pretrained_bounds": float(
            outside_bounds.mean()
        ),
        "frame_fraction_with_any_coordinate_outside_pretrained_bounds": float(
            outside_bounds.any(axis=(1, 2)).mean()
        ),
        "joint_axis_counts_below": (
            encoded_target < bound_low
        ).sum(axis=0).astype(int).tolist(),
        "joint_axis_counts_above": (
            encoded_target > bound_high
        ).sum(axis=0).astype(int).tolist(),
        "policy": (
            "Retain pretrained SimCC bins/bounds; out-of-range labels are clipped "
            "rather than reinterpreting pretrained head logits."
        ),
    }
    if bound_diagnostics[
        "frame_fraction_with_any_coordinate_outside_pretrained_bounds"
    ] > 0.01:
        warnings.warn(
            "More than 1% of SyncWB frames contain a target coordinate outside "
            "the pretrained SimCC bounds; inspect finetune_manifest.json.",
            RuntimeWarning,
        )
    train_indices, val_indices = split_syncwb_train_validation(
        collection.rows,
        args.validation_fraction,
    )
    if args.max_train_frames > 0 and len(train_indices) > args.max_train_frames:
        rng = np.random.default_rng(args.seed)
        train_indices = np.sort(
            rng.choice(
                train_indices,
                size=args.max_train_frames,
                replace=False,
            )
        )
    if args.max_val_frames > 0 and len(val_indices) > args.max_val_frames:
        rng = np.random.default_rng(args.seed + 1)
        val_indices = np.sort(
            rng.choice(val_indices, size=args.max_val_frames, replace=False)
        )

    dataset_kwargs = {
        "collection": collection,
        "bounds": bounds,
        "bins": bins,
        "num_points": num_points,
        "window_size": window_size,
        "seed": args.seed,
        "feature_config": feature_config,
        "target_mode": target_mode,
        "yaw_aug_deg": args.yaw_aug_deg,
        "pitch_aug_deg": args.pitch_aug_deg,
        "roll_aug_deg": args.roll_aug_deg,
        "translation_aug_m": (
            args.translation_aug_x_m,
            args.translation_aug_y_m,
            args.translation_aug_z_m,
        ),
        "body_scale_min": args.body_scale_min,
        "body_scale_max": args.body_scale_max,
        "x_reflection_probability": args.x_reflection_probability,
        "signal_dropout_probability": args.signal_dropout_probability,
    }
    train_ds = SyncWBWindowDataset(
        indices=train_indices,
        training=True,
        **dataset_kwargs,
    )
    val_ds = SyncWBWindowDataset(
        indices=val_indices,
        training=False,
        **dataset_kwargs,
    )
    train_loader = make_loader(
        train_ds,
        args.batch_size,
        args.workers,
        True,
        device,
    )
    val_loader = make_loader(
        val_ds,
        args.batch_size,
        args.workers,
        False,
        device,
    )
    before_metrics, before_pred, val_target, val_collection_indices = evaluate_loader(
        model,
        val_loader,
        bounds,
        target_mode,
        device,
        args.amp,
        args.progress,
        "Validate pretrained",
    )

    split_manifest = {
        "adaptation_type": "supervised_target_domain_finetuning",
        "adaptation_version": SYNCWB_FINETUNE_VERSION,
        "pretrained_checkpoint": str(Path(args.checkpoint).resolve()),
        "subjects": list(args.subjects),
        "mapping_methods": list(args.mapping_methods),
        "selected_mapping_versions": versions.to_dict("records"),
        "counts": collection.counts,
        "pretrained_pose_bound_diagnostics": bound_diagnostics,
        "train_windows": int(len(train_indices)),
        "validation_windows": int(len(val_indices)),
        "split_policy": (
            "temporal tail per mapping version with overlapping-window boundary gap"
        ),
        "kinect_joint_conversion": {
            "source": "Azure Kinect 32 joints",
            "target": "native MM-Fi 17-joint ordering",
            "indices": KINECT32_TO_MMFI17.tolist(),
            "axis_basis": "[x,y,z] mm -> [x,z,-y] m, then optional rigid extrinsic",
        },
        "kinect_to_radar_extrinsic": {
            "euler_xyz_deg": [
                args.kinect_to_radar_roll_deg,
                args.kinect_to_radar_pitch_deg,
                args.kinect_to_radar_yaw_deg,
            ],
            "translation_xyz_m": [
                args.kinect_to_radar_tx_m,
                args.kinect_to_radar_ty_m,
                args.kinect_to_radar_tz_m,
            ],
            "warning": (
                "Default zero values are an uncalibrated co-location assumption, "
                "not measured sensor extrinsics."
            ),
        },
        "args": {key: value for key, value in vars(args).items() if key != "func"},
    }
    (out / "finetune_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )
    collection.rows.assign(
        split=np.where(
            collection.rows.index.isin(set(val_indices.tolist())),
            "validation",
            np.where(
                collection.rows.index.isin(set(train_indices.tolist())),
                "train",
                "gap_or_cap_excluded",
            ),
        )
    ).to_csv(out / "windows.csv", index=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )
    low_np, high_np = bounds.arrays()
    low = torch.tensor(low_np, device=device)
    high = torch.tensor(high_np, device=device)
    history: list[dict[str, float | int | bool]] = []
    best_val = float("inf")

    print(json.dumps({
        "adaptation_version": SYNCWB_FINETUNE_VERSION,
        "device": str(device),
        "selected_mapping_versions": int(len(versions)),
        "selected_target_runs": int(
            versions[["subject_id", "target_run_id"]].drop_duplicates().shape[0]
        ),
        "windows": {"train": len(train_ds), "validation": len(val_ds)},
        "pretrained_validation": before_metrics,
        "pretrained_pose_bound_diagnostics": bound_diagnostics,
        "architecture_changed": False,
        "joint_head": "existing native MM-Fi 17-joint SimCC head",
    }, indent=2))

    for epoch in range(1, args.epochs + 1):
        backbone_trainable = epoch > args.freeze_backbone_epochs
        set_point_backbone_trainable(model, backbone_trainable)
        set_dataset_epoch(train_ds, epoch)
        model.train()
        if not backbone_trainable:
            model.point_embed.eval()
            model.encoder.eval()
        train_losses: list[float] = []
        iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Fine-tune {epoch}/{args.epochs}",
            leave=False,
            disable=not args.progress,
        )
        for batch in iterator:
            points = batch["points"].to(device, non_blocking=True)
            token_mask = batch["token_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            target_pose = batch["pose"].to(device, non_blocking=True)
            cloud_anchor = batch["cloud_anchor"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(points, token_mask)
                loss = simcc_loss(logits, labels)
                pred_pose = decode_simcc(
                    logits,
                    low,
                    high,
                    target_mode,
                    cloud_anchor,
                )
                pred_bones = torch.stack([
                    pred_pose[:, b] - pred_pose[:, a]
                    for a, b in MMFI17_EDGES
                ], dim=1)
                true_bones = torch.stack([
                    target_pose[:, b] - target_pose[:, a]
                    for a, b in MMFI17_EDGES
                ], dim=1)
                loss = loss + args.bone_weight * F.smooth_l1_loss(
                    pred_bones,
                    true_bones,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            if args.progress:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val_metrics, val_pred, val_target, val_collection_indices = evaluate_loader(
            model,
            val_loader,
            bounds,
            target_mode,
            device,
            args.amp,
            args.progress,
            "Validate fine-tuned",
        )
        criterion = float(
            val_metrics["body15_excluding_uncertain_face_head"][
                "root_relative_mpjpe_mm"
            ]
        )
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "backbone_trainable": backbone_trainable,
            "train_loss": float(np.mean(train_losses)),
            "val_body15_root_relative_mpjpe_mm": criterion,
            "val_body15_mpjpe_mm": float(
                val_metrics["body15_excluding_uncertain_face_head"]["mpjpe_mm"]
            ),
            "val_loss": float(val_metrics["loss"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(json.dumps(row))
        if criterion < best_val:
            best_val = criterion
            adapted = dict(source_checkpoint)
            adapted.update({
                "script_version": SCRIPT_VERSION,
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "mmfi_val_metrics_before_finetune": source_checkpoint.get(
                    "val_metrics", {}
                ),
                "adaptation": {
                    "type": "supervised_target_domain_finetuning",
                    "version": SYNCWB_FINETUNE_VERSION,
                    "pretrained_checkpoint": str(Path(args.checkpoint).resolve()),
                    "subjects": list(args.subjects),
                    "mapping_methods": list(args.mapping_methods),
                    "mapping_version_ids": versions[
                        "mapping_version_id"
                    ].astype(str).tolist(),
                    "kinect32_to_mmfi17": KINECT32_TO_MMFI17.tolist(),
                    "pretrained_pose_bound_diagnostics": bound_diagnostics,
                    "validation_selection_metric": (
                        "body15_root_relative_mpjpe_mm"
                    ),
                    "architecture_changed": False,
                },
            })
            torch.save(adapted, out / "best.pt")
            np.savez_compressed(
                out / "validation_predictions.npz",
                pred=val_pred,
                target=val_target,
                collection_index=val_collection_indices,
                joint_names=np.asarray(MMFI17_NAMES),
            )

    best_checkpoint = torch.load(
        out / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"])
    final_metrics, _, _, _ = evaluate_loader(
        model,
        val_loader,
        bounds,
        target_mode,
        device,
        args.amp,
        args.progress,
        "Validate best",
    )
    result = {
        "pretrained_validation": before_metrics,
        "fine_tuned_validation": final_metrics,
        "selection_metric": "body15_root_relative_mpjpe_mm",
        "best_epoch": int(best_checkpoint["epoch"]),
    }
    (out / "validation_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"


def infer_syncwb_all_command(args: argparse.Namespace) -> None:
    """Run the existing per-mapping inference for every selected version."""
    from sync_workbench.storage.sqlite_store import SQLiteCoreStore

    store = SQLiteCoreStore(args.sqlite)
    versions = select_syncwb_mapping_versions(
        store,
        args.subjects,
        args.mapping_methods,
    )
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for version in versions.itertuples(index=False):
        output = (
            root
            / safe_path_component(str(version.subject_id))
            / safe_path_component(str(version.mapping_version_id))
        )
        child = argparse.Namespace(**vars(args))
        child.subject = str(version.subject_id)
        child.mapping_version = str(version.mapping_version_id)
        child.out = str(output)
        infer_syncwb_command(child)
        metrics_path = output / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results.append({
            "subject_id": child.subject,
            "mapping_version_id": child.mapping_version,
            "mapping_method": str(version.mapping_method),
            "source_run_id": str(version.source_run_id),
            "target_run_id": str(version.target_run_id),
            "output": str(output),
            "single_person_raw": metrics.get("single_person_raw", {}),
            "single_person_globally_aligned_heldout": metrics.get(
                "single_person_globally_aligned_heldout", {}
            ),
        })
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "subjects": list(args.subjects),
        "mapping_methods": list(args.mapping_methods),
        "mapping_versions_run": len(results),
        "runs": results,
    }
    (root / "all_runs_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def infer_syncwb_command(args: argparse.Namespace) -> None:
    # Deferred imports keep MM-Fi-only training independent of SyncWB.
    from sync_workbench.storage.sqlite_store import SQLiteCoreStore

    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, bounds, checkpoint, feature_config, target_mode = load_checkpoint(
        Path(args.checkpoint), device
    )

    checkpoint_window_size = int(checkpoint["window_size"])
    if args.window_size is not None and args.window_size != checkpoint_window_size:
        raise ValueError(
            f"Checkpoint window_size={checkpoint_window_size}, "
            f"but --window-size={args.window_size} was requested"
        )
    window_size = checkpoint_window_size
    half_window = window_size // 2
    num_points = (
        int(args.num_points)
        if args.num_points is not None
        else int(checkpoint.get("num_points", 320))
    )
    quality_defaults = checkpoint.get("inference_quality_defaults", {})
    min_center_points = (
        int(args.min_center_points)
        if args.min_center_points is not None
        else int(quality_defaults.get("min_center_points", 10))
    )
    min_window_points = (
        int(args.min_window_points)
        if args.min_window_points is not None
        else int(quality_defaults.get("min_window_points", 64))
    )
    min_valid_window_frames = (
        int(args.min_valid_window_frames)
        if args.min_valid_window_frames is not None
        else int(quality_defaults.get("min_valid_window_frames", 3))
    )
    if min_center_points < 1 or min_window_points < 1:
        raise ValueError("Point-count inference thresholds must be positive")
    if not 1 <= min_valid_window_frames <= window_size:
        raise ValueError("min_valid_window_frames must be between 1 and window_size")

    store = SQLiteCoreStore(args.sqlite)
    mappings = store.read_table("SAMPLE_MAPPING")
    mapping_mask = (
        (mappings["subject_id"].astype(str) == str(args.subject))
        & (mappings["mapping_version_id"].astype(str) == str(args.mapping_version))
    )
    rows = mappings.loc[mapping_mask].copy()
    if "is_primary" in rows.columns:
        rows = rows[bool_series(rows["is_primary"])]
    if rows.empty:
        raise ValueError("No primary mapping rows matched the requested subject/mapping version")

    rows["subject_id"] = rows["subject_id"].astype(str)
    rows["source_run_id"] = rows["source_run_id"].astype(str)
    rows["source_device_type"] = rows["source_device_type"].astype(str)
    rows["target_run_id"] = rows["target_run_id"].astype(str)
    rows["target_device_type"] = rows["target_device_type"].astype(str)
    rows["source_sample_index"] = pd.to_numeric(
        rows["source_sample_index"], errors="raise"
    ).astype(int)
    rows["target_sample_index"] = pd.to_numeric(
        rows["target_sample_index"], errors="raise"
    ).astype(int)
    rows = rows.sort_values("source_sample_index")

    sample_summary = store.read_table("SAMPLE_SUMMARY")
    rows = attach_source_num_people(rows, sample_summary)
    primary_mapping_rows = int(len(rows))
    people_numeric = pd.to_numeric(rows["num_people"], errors="coerce")
    known_single_mask = people_numeric.eq(1)
    excluded_multi_person_rows = int(people_numeric.gt(1).sum())
    excluded_unknown_or_zero_person_rows = int(
        (~known_single_mask & ~people_numeric.gt(1)).sum()
    )
    rows = rows.loc[known_single_mask].copy()
    if args.max_frames > 0:
        rows = rows.iloc[:args.max_frames]
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise ValueError(
            "No primary mapping rows have SAMPLE_SUMMARY.num_people == 1"
        )

    device_runs = store.read_table("DEVICE_RUN")
    if args.target_fps is not None:
        target_fps = float(args.target_fps)
        fps_source = "command_line"
    else:
        target_keys = rows[[
            "subject_id",
            "target_run_id",
            "target_device_type",
        ]].drop_duplicates()
        run_fps: list[float] = []
        for target_key in target_keys.itertuples(index=False):
            matches = device_runs[
                (device_runs["subject_id"].astype(str) == str(target_key.subject_id))
                & (device_runs["run_id"].astype(str) == str(target_key.target_run_id))
                & (
                    device_runs["device_type"].astype(str)
                    == str(target_key.target_device_type)
                )
            ]
            values = pd.to_numeric(
                matches.get("nominal_fps", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna().unique()
            run_fps.extend(float(value) for value in values if float(value) > 0)
        unique_fps = sorted(set(run_fps))
        if len(unique_fps) != 1:
            raise ValueError(
                "Could not derive one target nominal_fps from DEVICE_RUN; "
                "supply --target-fps explicitly"
            )
        target_fps = unique_fps[0]
        fps_source = "DEVICE_RUN.nominal_fps"
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    sample_period_sec = float(
        checkpoint.get("sample_period_sec", MMFI_SAMPLE_PERIOD_SEC)
    )
    target_frame_stride = max(1, int(round(sample_period_sec * target_fps)))
    target_offsets = np.arange(-half_window, half_window + 1) * target_frame_stride

    # Resolve the two run-level NPZ bundles once. The previous implementation
    # called per-sample payload lookup up to six times per mapped row. Each
    # call performed a DataFrame scan over SAMPLE_ARTIFACT, which caused the
    # multi-hour estimate for a full session.
    run_assets = store.read_table("RUN_ASSET")
    required_payloads: set[tuple[str, str, str, str]] = set()
    for row in rows.itertuples(index=False):
        required_payloads.add((
            str(row.subject_id),
            str(row.target_run_id),
            str(row.target_device_type),
            "radar_points",
        ))
        required_payloads.add((
            str(row.subject_id),
            str(row.source_run_id),
            str(row.source_device_type),
            "pose3d",
        ))
    readers = open_syncwb_ragged_readers(
        run_assets,
        args.artifact_root,
        required_payloads,
    )

    # Cache only unique source/target frames. Target offsets preserve the
    # physical MM-Fi sample period rather than merely copying frame count.
    radar_requests: set[tuple[str, str, str, int]] = set()
    pose_requests: set[tuple[str, str, str, int]] = set()
    for row in rows.itertuples(index=False):
        subject_id = str(row.subject_id)
        target_run_id = str(row.target_run_id)
        target_device_type = str(row.target_device_type)
        target_centre = int(row.target_sample_index)
        for target_offset in target_offsets:
            target_index = target_centre + int(target_offset)
            if target_index >= 0:
                radar_requests.add((
                    subject_id,
                    target_run_id,
                    target_device_type,
                    target_index,
                ))
        pose_requests.add((
            subject_id,
            str(row.source_run_id),
            str(row.source_device_type),
            int(row.source_sample_index),
        ))

    radar_cache: dict[tuple[str, str, str, int], np.ndarray | None] = {}
    radar_iterator = tqdm(
        sorted(radar_requests),
        total=len(radar_requests),
        desc="Cache unique radar frames",
        leave=False,
        disable=not args.progress,
    )
    for subject_id, run_id, device_type, sample_index in radar_iterator:
        reader = readers[(subject_id, run_id, device_type, "radar_points")]
        try:
            raw = reader.get(sample_index)
        except KeyError:
            radar_cache[(subject_id, run_id, device_type, sample_index)] = None
        else:
            radar_cache[(subject_id, run_id, device_type, sample_index)] = (
                filter_syncwb_radar(raw, args.filter_noise)
            )

    pose_cache: dict[tuple[str, str, str, int], np.ndarray | None] = {}
    pose_iterator = tqdm(
        sorted(pose_requests),
        total=len(pose_requests),
        desc="Cache unique Kinect poses",
        leave=False,
        disable=not args.progress,
    )
    for subject_id, run_id, device_type, sample_index in pose_iterator:
        reader = readers[(subject_id, run_id, device_type, "pose3d")]
        try:
            pose_cache[(subject_id, run_id, device_type, sample_index)] = (
                reader.get(sample_index)
            )
        except KeyError:
            pose_cache[(subject_id, run_id, device_type, sample_index)] = None

    radar_windows: list[np.ndarray] = []
    gt_frames: list[np.ndarray] = []
    kept_rows: list[dict[str, object]] = []
    skipped = {
        "incomplete_radar_window": 0,
        "too_few_center_points": 0,
        "too_few_window_points": 0,
        "too_few_valid_window_frames": 0,
    }
    unavailable_ground_truth = {
        "missing_pose_payload": 0,
        "invalid_pose": 0,
    }

    iterator = tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc="Load SyncWB windows",
        disable=not args.progress,
    )
    for row in iterator:
        subject_id = str(row.subject_id)
        target_run_id = str(row.target_run_id)
        target_device_type = str(row.target_device_type)
        target_centre = int(row.target_sample_index)

        window_frames: list[np.ndarray] = []
        for target_offset in target_offsets:
            target_index = target_centre + int(target_offset)
            frame = radar_cache.get((
                subject_id,
                target_run_id,
                target_device_type,
                target_index,
            ))
            if frame is None:
                window_frames = []
                break
            window_frames.append(frame)

        if len(window_frames) != window_size:
            skipped["incomplete_radar_window"] += 1
            continue

        stacked_window = stack_radar_window(window_frames)
        support = radar_window_support(stacked_window, window_size)
        if int(support["center_point_count"]) < min_center_points:
            skipped["too_few_center_points"] += 1
            continue
        if int(support["window_point_count"]) < min_window_points:
            skipped["too_few_window_points"] += 1
            continue
        if int(support["valid_window_frames"]) < min_valid_window_frames:
            skipped["too_few_valid_window_frames"] += 1
            continue

        pose3d = pose_cache.get((
            subject_id,
            str(row.source_run_id),
            str(row.source_device_type),
            int(row.source_sample_index),
        ))
        if pose3d is None:
            unavailable_ground_truth["missing_pose_payload"] += 1
            gt = np.full((17, 3), np.nan, dtype=np.float32)
            ground_truth_available = False
        else:
            converted_gt = kinect_pose32_to_radar_mmfi17(
                pose3d,
                args.person_index,
                extrinsic_euler_deg=(
                    args.kinect_to_radar_roll_deg,
                    args.kinect_to_radar_pitch_deg,
                    args.kinect_to_radar_yaw_deg,
                ),
                extrinsic_translation_m=(
                    args.kinect_to_radar_tx_m,
                    args.kinect_to_radar_ty_m,
                    args.kinect_to_radar_tz_m,
                ),
            )
            ground_truth_available = bool(
                converted_gt is not None
                and np.all(np.isfinite(converted_gt))
            )
            if ground_truth_available:
                gt = np.asarray(converted_gt, dtype=np.float32)
            else:
                unavailable_ground_truth["invalid_pose"] += 1
                gt = np.full((17, 3), np.nan, dtype=np.float32)

        num_people = 1.0
        radar_windows.append(stacked_window)
        gt_frames.append(gt)
        kept_rows.append({
            "source_sample_index": int(row.source_sample_index),
            "target_sample_index": target_centre,
            "target_window_start": target_centre + int(target_offsets[0]),
            "target_window_end": target_centre + int(target_offsets[-1]),
            "target_frame_stride": target_frame_stride,
            "center_point_count": int(support["center_point_count"]),
            "window_point_count": int(support["window_point_count"]),
            "valid_window_frames": int(support["valid_window_frames"]),
            "num_people": num_people,
            "ground_truth_available": ground_truth_available,
            "metric_eligible_single_person": ground_truth_available,
            "predicted_minus_estimated_ms": getattr(
                row,
                "predicted_minus_estimated_ms",
                np.nan,
            ),
        })

    if not radar_windows:
        raise ValueError(
            "No known-single-person rows passed the radar-window quality policy"
        )

    pred, cloud_anchors = predict_point_clouds(
        model=model,
        bounds=bounds,
        point_clouds=radar_windows,
        num_points=num_points,
        window_size=window_size,
        feature_config=feature_config,
        target_mode=target_mode,
        batch_size=args.batch_size,
        device=device,
        amp=args.amp,
        total=len(radar_windows),
        progress=args.progress,
    )
    gt = np.stack(gt_frames).astype(np.float32)
    pair_df = pd.DataFrame(kept_rows)
    metric_mask = pair_df["metric_eligible_single_person"].to_numpy(dtype=bool)

    single_indices = np.flatnonzero(metric_mask)
    if len(single_indices) > 0:
        requested_calibration = max(
            1,
            int(round(len(single_indices) * args.calibration_fraction)),
        )
        if len(single_indices) >= 11:
            n_calibration = min(
                max(10, requested_calibration),
                len(single_indices) - 1,
            )
        else:
            n_calibration = max(1, len(single_indices) - 1)

        calibration_indices = single_indices[:n_calibration]
        evaluation_indices = single_indices[n_calibration:]
        if len(evaluation_indices) == 0:
            warnings.warn(
                "Too few ground-truth frames for held-out aligned evaluation; "
                "using all eligible frames for aligned metrics.",
                RuntimeWarning,
            )
            evaluation_indices = single_indices
        transform = fit_global_similarity(
            pred[calibration_indices],
            gt[calibration_indices],
            joint_indices=BODY15_INDICES,
        )
        pred_global = apply_global_similarity(pred, transform)
    else:
        calibration_indices = np.empty(0, dtype=np.int64)
        evaluation_indices = np.empty(0, dtype=np.int64)
        transform = (
            1.0,
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        pred_global = pred.copy()

    heldout_mask = np.zeros(len(pair_df), dtype=bool)
    heldout_mask[evaluation_indices] = True

    summary: dict[str, object] = {
        "checkpoint_mmfi_validation": checkpoint.get(
            "mmfi_val_metrics_before_finetune",
            checkpoint.get("val_metrics", {}),
        ),
        "checkpoint_adaptation": checkpoint.get("adaptation"),
        "checkpoint_adaptation_validation": (
            checkpoint.get("val_metrics", {})
            if checkpoint.get("adaptation") is not None
            else None
        ),
        "checkpoint_preprocessing": {
            "script_version": checkpoint.get("script_version"),
            "feature_config": asdict(feature_config),
            "target_mode": target_mode,
            "frame_balanced_sampling": checkpoint.get("frame_balanced_sampling"),
            "sampling_strategy": checkpoint.get("sampling_strategy"),
            "sample_period_sec": sample_period_sec,
            "mmfi_point_order": checkpoint.get("mmfi_point_order"),
        },
        "syncwb_radar_window": {
            "target_fps": target_fps,
            "target_fps_source": fps_source,
            "target_frame_stride": target_frame_stride,
            "effective_sample_period_sec": target_frame_stride / target_fps,
            "window_span_sec": (
                (window_size - 1) * target_frame_stride / target_fps
            ),
            "minimum_center_points": min_center_points,
            "minimum_window_points": min_window_points,
            "minimum_valid_window_frames": min_valid_window_frames,
            "quality_default_provenance": checkpoint.get(
                "inference_quality_default_provenance",
                (
                    "Conservative MM-Fi lower-tail support defaults (the centre "
                    "threshold is its per-frame p10) checked against 19_MM radar "
                    "point-count diagnostics; no 19_MM pose labels were used "
                    "(explicit checkpoint field absent)."
                ),
            ),
        },
        "kinect_to_radar_extrinsic_used_for_ground_truth": {
            "euler_xyz_deg": [
                args.kinect_to_radar_roll_deg,
                args.kinect_to_radar_pitch_deg,
                args.kinect_to_radar_yaw_deg,
            ],
            "translation_xyz_m": [
                args.kinect_to_radar_tx_m,
                args.kinect_to_radar_ty_m,
                args.kinect_to_radar_tz_m,
            ],
            "warning": (
                "Zero values preserve previous behaviour but do not imply that "
                "Kinect and radar were extrinsically calibrated."
            ),
        },
        "counts": {
            "primary_mapping_rows_considered": primary_mapping_rows,
            "known_single_person_rows_considered": int(len(rows)),
            "valid_predictions_saved": int(len(pred)),
            "single_person_metric_frames": int(metric_mask.sum()),
            "predictions_without_ground_truth": int((~metric_mask).sum()),
            "excluded_non_single_person_before_inference": (
                excluded_multi_person_rows + excluded_unknown_or_zero_person_rows
            ),
            "excluded_multi_person_before_inference": excluded_multi_person_rows,
            "excluded_unknown_or_zero_person_before_inference": (
                excluded_unknown_or_zero_person_rows
            ),
            **{f"skipped_{key}": int(value) for key, value in skipped.items()},
            **{
                f"ground_truth_unavailable_{key}": int(value)
                for key, value in unavailable_ground_truth.items()
            },
        },
        "evaluation_policy": {
            "syncwb_metrics_require_num_people_equal_to": 1,
            "predictions_do_not_require_kinect_ground_truth": True,
            "full17_face_head_joints_are_provisional": [9, 10],
            "global_similarity_fit_joint_set": "body15",
            "global_similarity_calibration_frames": int(len(calibration_indices)),
            "globally_aligned_evaluation_frames": int(len(evaluation_indices)),
        },
        "single_person_raw": pose_metric_bundle(pred[metric_mask], gt[metric_mask]),
        "single_person_globally_aligned_heldout": pose_metric_bundle(
            pred_global[evaluation_indices],
            gt[evaluation_indices],
        ),
    }

    sweep = lag_sweep_root_mpjpe(
        pred_global,
        gt,
        pair_df["source_sample_index"].to_numpy(dtype=np.int64),
        heldout_mask,
        args.lag_sweep,
    )
    if not sweep.empty:
        best = sweep.loc[
            sweep["body15_root_relative_mpjpe_mm"].idxmin()
        ].to_dict()
        summary["best_single_person_body15_root_relative_lag"] = best
        sweep.to_csv(out / "lag_sweep.csv", index=False)

    pair_df.to_csv(out / "pairs.csv", index=False)
    np.savez_compressed(
        out / "predictions.npz",
        pred=pred,
        cloud_anchor=cloud_anchors,
        pred_globally_aligned=pred_global,
        target=gt,
        source_sample_index=pair_df["source_sample_index"].to_numpy(),
        target_sample_index=pair_df["target_sample_index"].to_numpy(),
        num_people=pair_df["num_people"].to_numpy(),
        metric_eligible_single_person=metric_mask,
        ground_truth_available=pair_df["ground_truth_available"].to_numpy(dtype=bool),
        globally_aligned_heldout_mask=heldout_mask,
        joint_names=np.asarray(MMFI17_NAMES),
        global_scale=np.asarray(transform[0]),
        global_rotation=transform[1],
        global_translation=transform[2],
        signal_mode=np.asarray(feature_config.signal_mode),
        target_mode=np.asarray(target_mode),
        spatial_mode=np.asarray(feature_config.spatial_mode),
        target_fps=np.asarray(target_fps),
        target_frame_stride=np.asarray(target_frame_stride),
        sample_period_sec=np.asarray(sample_period_sec),
        center_point_count=pair_df["center_point_count"].to_numpy(),
        window_point_count=pair_df["window_point_count"].to_numpy(),
        valid_window_frames=pair_df["valid_window_frames"].to_numpy(),
        kinect_to_radar_euler_xyz_deg=np.asarray([
            args.kinect_to_radar_roll_deg,
            args.kinect_to_radar_pitch_deg,
            args.kinect_to_radar_yaw_deg,
        ], dtype=np.float32),
        kinect_to_radar_translation_xyz_m=np.asarray([
            args.kinect_to_radar_tx_m,
            args.kinect_to_radar_ty_m,
            args.kinect_to_radar_tz_m,
        ], dtype=np.float32),
    )
    (out / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def self_test_command(_: argparse.Namespace) -> None:
    """Run deterministic preprocessing/model/checkpoint smoke tests on CPU."""
    import tempfile

    results: dict[str, object] = {}

    raw = np.asarray([[1.0, 2.0, 3.0, 17.5, -0.604349]], dtype=np.float32)
    transformed = transform_mmfi_points(raw)
    expected = np.asarray([[2.0, 1.0, 3.0, -0.604349, 17.5]], dtype=np.float32)
    np.testing.assert_allclose(transformed, expected, atol=1e-6)
    results["corrected_mmfi_channel_order"] = True

    frames = []
    for frame_index, count in enumerate((3, 7, 2, 9, 4)):
        frame = np.zeros((count, 5), dtype=np.float32)
        frame[:, 0] = np.linspace(0.1, 0.2, count) + frame_index
        frame[:, 1] = 2.0 + frame_index * 0.1
        frame[:, 2] = np.linspace(-0.2, 0.2, count)
        frame[:, 3] = np.linspace(-1.0, 1.0, count)
        frame[:, 4] = np.linspace(10.0, 30.0, count)
        frames.append(frame)
    stacked = stack_radar_window(frames)
    sampled, token_mask = frame_balanced_sample(stacked, 25, 5, np.random.default_rng(3))
    unique_times, time_counts = np.unique(sampled[token_mask, 5], return_counts=True)
    np.testing.assert_allclose(unique_times, [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(time_counts, [3, 5, 2, 5, 4])
    permuted = stacked[np.random.default_rng(41).permutation(len(stacked))]
    sampled_permuted, mask_permuted = frame_balanced_sample(
        permuted, 25, 5, np.random.default_rng(3)
    )
    np.testing.assert_array_equal(token_mask, mask_permuted)
    np.testing.assert_allclose(sampled, sampled_permuted, atol=0.0)
    eval_sampled, eval_mask = frame_balanced_sample(stacked, 25, 5, None)
    eval_permuted, eval_permuted_mask = frame_balanced_sample(permuted, 25, 5, None)
    np.testing.assert_array_equal(eval_mask, eval_permuted_mask)
    np.testing.assert_allclose(eval_sampled, eval_permuted, atol=0.0)
    results["frame_balanced_temporal_sampling"] = True
    results["permutation_invariant_sampling"] = True

    cloud_anchor, anchor_diagnostics = robust_cloud_anchor(stacked)
    support = radar_window_support(stacked, 5)
    assert anchor_diagnostics["valid_window_frames"] == 5
    assert support["center_point_count"] == 2
    for spatial_mode, base_dim in (("legacy", 8), ("cloud_anchor", 6)):
        for signal_mode in SIGNAL_MODES:
            expected_dim = base_dim + int(signal_mode != "none")
            config = FeatureConfig(
                signal_mode=signal_mode,
                doppler_limit_mps=3.0,
                spatial_mode=spatial_mode,
            )
            features = robust_feature_normalize(
                sampled,
                config,
                cloud_anchor=cloud_anchor if spatial_mode == "cloud_anchor" else None,
            )
            assert features.shape == (25, expected_dim)
            doppler_index = 4 if spatial_mode == "cloud_anchor" else 6
            assert np.max(np.abs(features[:, doppler_index])) <= 1.0 + 1e-6
            if signal_mode != "none":
                affine = sampled.copy()
                affine[:, 4] = affine[:, 4] * 7.0 + 13.0
                affine_features = robust_feature_normalize(
                    affine,
                    config,
                    cloud_anchor=(
                        cloud_anchor if spatial_mode == "cloud_anchor" else None
                    ),
                )
                np.testing.assert_allclose(
                    features[:, -2], affine_features[:, -2], atol=1e-5
                )
    results["signal_modes_and_affine_compatibility"] = True
    results["robust_cloud_anchor_and_support"] = True

    pose = np.zeros((17, 3), dtype=np.float32)
    pose[0] = [0.2, 3.0, 0.1]
    pose[1:] = pose[0] + np.linspace(-0.8, 0.8, 16)[:, None] * np.asarray(
        [0.2, 0.1, 0.3], dtype=np.float32
    )
    for target_mode in TARGET_MODES:
        target_anchor = (
            np.asarray([0.1, 2.8, -0.1], dtype=np.float32)
            if target_mode == "cloud_anchor_relative"
            else None
        )
        encoded = encode_pose_target(pose, target_mode, target_anchor)
        decoded = decode_pose_target(encoded, target_mode, target_anchor)
        np.testing.assert_allclose(decoded, pose, atol=1e-6)
    results["pose_target_round_trip"] = True

    low = np.full((17, 3), -5.0, dtype=np.float32)
    high = np.full((17, 3), 5.0, dtype=np.float32)
    bounds = PoseBounds(low.tolist(), high.tolist())
    device = torch.device("cpu")
    for spatial_mode in SPATIAL_MODES:
        for signal_mode in SIGNAL_MODES:
            feature_config = FeatureConfig(
                signal_mode=signal_mode,
                spatial_mode=spatial_mode,
            )
            model_config = {
                "input_dim": feature_config.input_dim,
                "joints": 17,
                "bins": 16,
                "dim": 32,
                "heads": 4,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "dropout": 0.0,
            }
            model = PointQuerySimCC(**model_config).eval()
            x = torch.zeros((2, 25, feature_config.input_dim), dtype=torch.float32)
            mask = torch.ones((2, 25), dtype=torch.bool)
            with torch.no_grad():
                logits = model(x, mask)
                assert all(axis.shape == (2, 17, 16) for axis in logits)
                decode_mode = (
                    "cloud_anchor_relative"
                    if spatial_mode == "cloud_anchor"
                    else "pelvis_relative"
                )
                anchors = (
                    torch.zeros((2, 3))
                    if decode_mode == "cloud_anchor_relative"
                    else None
                )
                decoded = decode_simcc(
                    logits,
                    torch.tensor(low),
                    torch.tensor(high),
                    decode_mode,
                    anchors,
                )
                assert decoded.shape == (2, 17, 3)
    results["model_forward_all_signal_modes"] = True

    feature_config = FeatureConfig(signal_mode="none", spatial_mode="cloud_anchor")
    model_config = {
        "input_dim": feature_config.input_dim,
        "joints": 17,
        "bins": 16,
        "dim": 32,
        "heads": 4,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "dropout": 0.0,
    }
    model = PointQuerySimCC(**model_config)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "smoke.pt"
        torch.save({
            "script_version": SCRIPT_VERSION,
            "model": model.state_dict(),
            "model_config": model_config,
            "pose_bounds": asdict(bounds),
            "target_mode": "cloud_anchor_relative",
            "feature_config": asdict(feature_config),
            "frame_balanced_sampling": True,
            "window_size": 5,
            "num_points": 25,
            "mmfi_point_order": MMFI_POINT_ORDER.tolist(),
            "mmfi_pose_order": MMFI_POSE_ORDER.tolist(),
            "mmfi17_names": MMFI17_NAMES,
        }, checkpoint_path)
        loaded_model, loaded_bounds, _, loaded_features, loaded_target_mode = load_checkpoint(
            checkpoint_path, device
        )
        assert isinstance(loaded_model, PointQuerySimCC)
        assert loaded_bounds.arrays()[0].shape == (17, 3)
        assert loaded_features == feature_config
        assert loaded_target_mode == "cloud_anchor_relative"
    results["checkpoint_contract_round_trip"] = True

    pose32 = np.zeros((1, 32, 3), dtype=np.float32)
    transformed_pose = kinect_pose32_to_radar_mmfi17(
        pose32,
        extrinsic_translation_m=(0.1, -0.2, 0.3),
    )
    assert transformed_pose is not None
    np.testing.assert_allclose(
        transformed_pose,
        np.tile(np.asarray([0.1, -0.2, 0.3], dtype=np.float32), (17, 1)),
        atol=1e-6,
    )
    results["kinect_to_radar_extrinsic_application"] = True

    print(json.dumps({
        "script_version": SCRIPT_VERSION,
        "status": "passed",
        "tests": results,
    }, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument("--packed-root", required=True)
    train.add_argument("--out", default="runs/mmfi_pose_anchor_v4")
    train.add_argument(
        "--split",
        choices=["cross_subject", "cross_environment"],
        default="cross_subject",
    )
    train.add_argument("--test-env", default="E04")
    train.add_argument("--val-env", default=None)
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--bone-weight", type=float, default=0.10)
    train.add_argument("--bins", type=int, default=192)
    train.add_argument("--num-points", type=int, default=250)
    train.add_argument("--window-size", type=int, default=5)
    train.add_argument(
        "--sample-period-sec",
        type=float,
        default=MMFI_SAMPLE_PERIOD_SEC,
        help="Physical time between adjacent source frames represented by the window",
    )
    train.add_argument(
        "--signal-mode",
        choices=SIGNAL_MODES,
        default="none",
        help=(
            "Return-strength feature: none is the recommended cross-domain baseline; "
            "robust uses within-window median/IQR; rank uses within-window percentile rank"
        ),
    )
    train.add_argument(
        "--doppler-limit-mps",
        type=float,
        default=3.0,
        help="Clip physical Doppler to +/- this value, then scale to [-1,1]",
    )
    train.add_argument(
        "--spatial-mode",
        choices=SPATIAL_MODES,
        default="cloud_anchor",
        help=(
            "cloud_anchor uses robust-anchor-relative xyz plus physical range; "
            "legacy preserves the earlier absolute+median-centred features"
        ),
    )
    train.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default="cloud_anchor_relative",
        help=(
            "Default predicts pelvis relative to the robust cloud anchor and "
            "the remaining joints relative to pelvis"
        ),
    )
    train.add_argument("--yaw-aug-deg", type=float, default=15.0)
    train.add_argument(
        "--pitch-aug-deg",
        type=float,
        default=35.0,
        help="Covers the approximately 30-degree source/target frame-orientation gap",
    )
    train.add_argument("--roll-aug-deg", type=float, default=5.0)
    train.add_argument("--translation-aug-x-m", type=float, default=0.35)
    train.add_argument("--translation-aug-y-m", type=float, default=0.75)
    train.add_argument("--translation-aug-z-m", type=float, default=0.25)
    train.add_argument("--body-scale-min", type=float, default=0.85)
    train.add_argument("--body-scale-max", type=float, default=1.15)
    train.add_argument(
        "--x-reflection-probability",
        type=float,
        default=0.50,
        help="Coordinate-frame reflection augmentation applied consistently to points and pose",
    )
    train.add_argument(
        "--signal-dropout-probability",
        type=float,
        default=0.25,
        help="When signal is enabled, randomly remove it for whole training windows",
    )
    train.add_argument("--dim", type=int, default=128)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--encoder-layers", type=int, default=4)
    train.add_argument("--decoder-layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--stats-frames", type=int, default=20000)
    train.add_argument("--max-train-frames", type=int, default=0)
    train.add_argument("--max-val-frames", type=int, default=30000)
    train.add_argument("--max-test-frames", type=int, default=30000)
    train.add_argument("--inference-min-center-points", type=int, default=10)
    train.add_argument("--inference-min-window-points", type=int, default=64)
    train.add_argument("--inference-min-valid-window-frames", type=int, default=3)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--cpu", action="store_true")
    train.set_defaults(func=train_command)

    finetune = subparsers.add_parser(
        "finetune-syncwb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=(
            "Supervised fine-tuning on every selected SyncWB mapping version; "
            "the pretrained architecture and MM-Fi-17 output head are retained"
        ),
    )
    finetune.add_argument("--checkpoint", required=True)
    finetune.add_argument("--sqlite", required=True)
    finetune.add_argument("--artifact-root", required=True)
    finetune.add_argument("--subjects", nargs="+", required=True)
    finetune.add_argument(
        "--mapping-methods",
        nargs="+",
        default=["initial_nearest_for_anchoring"],
    )
    finetune.add_argument("--out", default="runs/mmfi_pose_syncwb_finetuned")
    finetune.add_argument("--epochs", type=int, default=10)
    finetune.add_argument("--batch-size", type=int, default=64)
    finetune.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 avoids duplicating preloaded SyncWB windows on Windows",
    )
    finetune.add_argument("--lr", type=float, default=3e-5)
    finetune.add_argument("--weight-decay", type=float, default=1e-4)
    finetune.add_argument("--bone-weight", type=float, default=0.10)
    finetune.add_argument("--freeze-backbone-epochs", type=int, default=1)
    finetune.add_argument("--validation-fraction", type=float, default=0.10)
    finetune.add_argument("--max-frames", type=int, default=0)
    finetune.add_argument("--max-train-frames", type=int, default=0)
    finetune.add_argument("--max-val-frames", type=int, default=10000)
    finetune.add_argument("--person-index", type=int, default=0)
    finetune.add_argument("--target-fps", type=float, default=None)
    finetune.add_argument("--min-center-points", type=int, default=None)
    finetune.add_argument("--min-window-points", type=int, default=None)
    finetune.add_argument("--min-valid-window-frames", type=int, default=None)
    finetune.add_argument("--kinect-to-radar-roll-deg", type=float, default=0.0)
    finetune.add_argument("--kinect-to-radar-pitch-deg", type=float, default=0.0)
    finetune.add_argument("--kinect-to-radar-yaw-deg", type=float, default=0.0)
    finetune.add_argument("--kinect-to-radar-tx-m", type=float, default=0.0)
    finetune.add_argument("--kinect-to-radar-ty-m", type=float, default=0.0)
    finetune.add_argument("--kinect-to-radar-tz-m", type=float, default=0.0)
    finetune.add_argument("--yaw-aug-deg", type=float, default=10.0)
    finetune.add_argument("--pitch-aug-deg", type=float, default=5.0)
    finetune.add_argument("--roll-aug-deg", type=float, default=5.0)
    finetune.add_argument("--translation-aug-x-m", type=float, default=0.10)
    finetune.add_argument("--translation-aug-y-m", type=float, default=0.10)
    finetune.add_argument("--translation-aug-z-m", type=float, default=0.10)
    finetune.add_argument("--body-scale-min", type=float, default=0.95)
    finetune.add_argument("--body-scale-max", type=float, default=1.05)
    finetune.add_argument("--x-reflection-probability", type=float, default=0.0)
    finetune.add_argument("--signal-dropout-probability", type=float, default=0.0)
    finetune.add_argument(
        "--filter-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    finetune.add_argument("--seed", type=int, default=7)
    finetune.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    finetune.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    finetune.add_argument("--cpu", action="store_true")
    finetune.set_defaults(func=finetune_syncwb_command)

    infer_all = subparsers.add_parser(
        "infer-syncwb-all",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run inference/testing for every selected subject mapping version",
    )
    infer_all.add_argument("--checkpoint", required=True)
    infer_all.add_argument("--sqlite", required=True)
    infer_all.add_argument("--artifact-root", required=True)
    infer_all.add_argument("--subjects", nargs="+", required=True)
    infer_all.add_argument(
        "--mapping-methods",
        nargs="+",
        default=["initial_nearest_for_anchoring"],
    )
    infer_all.add_argument("--out", default="runs/syncwb_pose_all")
    infer_all.add_argument("--person-index", type=int, default=0)
    infer_all.add_argument("--kinect-to-radar-roll-deg", type=float, default=0.0)
    infer_all.add_argument("--kinect-to-radar-pitch-deg", type=float, default=0.0)
    infer_all.add_argument("--kinect-to-radar-yaw-deg", type=float, default=0.0)
    infer_all.add_argument("--kinect-to-radar-tx-m", type=float, default=0.0)
    infer_all.add_argument("--kinect-to-radar-ty-m", type=float, default=0.0)
    infer_all.add_argument("--kinect-to-radar-tz-m", type=float, default=0.0)
    infer_all.add_argument("--window-size", type=int, default=None)
    infer_all.add_argument("--num-points", type=int, default=None)
    infer_all.add_argument("--target-fps", type=float, default=None)
    infer_all.add_argument("--min-center-points", type=int, default=None)
    infer_all.add_argument("--min-window-points", type=int, default=None)
    infer_all.add_argument("--min-valid-window-frames", type=int, default=None)
    infer_all.add_argument("--batch-size", type=int, default=256)
    infer_all.add_argument("--max-frames", type=int, default=0)
    infer_all.add_argument("--calibration-fraction", type=float, default=0.10)
    infer_all.add_argument("--lag-sweep", type=int, default=20)
    infer_all.add_argument(
        "--filter-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    infer_all.add_argument("--seed", type=int, default=7)
    infer_all.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    infer_all.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    infer_all.add_argument("--cpu", action="store_true")
    infer_all.set_defaults(func=infer_syncwb_all_command)

    infer = subparsers.add_parser(
        "infer-syncwb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--sqlite", required=True)
    infer.add_argument("--artifact-root", required=True)
    infer.add_argument("--subject", required=True)
    infer.add_argument("--mapping-version", required=True)
    infer.add_argument("--out", default="runs/syncwb_pose")
    infer.add_argument("--person-index", type=int, default=0)
    infer.add_argument(
        "--kinect-to-radar-roll-deg",
        type=float,
        default=0.0,
        help="Measured residual Kinect->radar extrinsic roll after x,z,-y axis conversion",
    )
    infer.add_argument("--kinect-to-radar-pitch-deg", type=float, default=0.0)
    infer.add_argument("--kinect-to-radar-yaw-deg", type=float, default=0.0)
    infer.add_argument("--kinect-to-radar-tx-m", type=float, default=0.0)
    infer.add_argument("--kinect-to-radar-ty-m", type=float, default=0.0)
    infer.add_argument("--kinect-to-radar-tz-m", type=float, default=0.0)
    infer.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Must match the checkpoint; omitted uses checkpoint value",
    )
    infer.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Omitted uses the checkpoint training value",
    )
    infer.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Override target radar FPS; omitted reads DEVICE_RUN.nominal_fps",
    )
    infer.add_argument(
        "--min-center-points",
        type=int,
        default=None,
        help="Omitted uses the source-training checkpoint default",
    )
    infer.add_argument(
        "--min-window-points",
        type=int,
        default=None,
        help="Omitted uses the source-training checkpoint default",
    )
    infer.add_argument(
        "--min-valid-window-frames",
        type=int,
        default=None,
        help="Omitted uses the source-training checkpoint default",
    )
    infer.add_argument("--batch-size", type=int, default=256)
    infer.add_argument("--max-frames", type=int, default=0)
    infer.add_argument("--calibration-fraction", type=float, default=0.10)
    infer.add_argument("--lag-sweep", type=int, default=20)
    infer.add_argument(
        "--filter-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    infer.add_argument("--seed", type=int, default=7)
    infer.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    infer.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    infer.add_argument("--cpu", action="store_true")
    infer.set_defaults(func=infer_syncwb_command)

    self_test = subparsers.add_parser(
        "self-test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run deterministic CPU smoke tests for preprocessing/model/checkpoint contracts",
    )
    self_test.set_defaults(func=self_test_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
