#!/usr/bin/env python3
"""Quick MM-Fi mmWave point-cloud -> 3D pose experiment.

This version uses:
- the native 17-joint MM-Fi topology defined in MMFI17_NAMES;
- corrected packed MM-Fi point order [x, y, z, Doppler, intensity];
- centred odd-sized radar windows (default: five frames);
- frame-balanced point sampling so each temporal frame contributes equally;
- configurable return-strength handling: none, window-robust, or window-rank;
- one relative-time feature per point;
- a point-token Transformer with learned joint queries and 3-axis SimCC;
- pelvis-relative joint targets with an absolute pelvis token by default;
- joint geometric augmentation applied consistently to radar points and pose;
- fast SyncWB inference through cached run-level ragged-NPZ readers;
- optional Kinect-to-radar rigid extrinsics for SyncWB evaluation;
- tqdm progress bars when tqdm is installed;
- SyncWB metrics only for frames whose SAMPLE_SUMMARY.num_people == 1.

Recommended geometry + Doppler baseline:
    python mmfi_pose_quick.py train \
        --packed-root "D:/backup_data/MM-Fi/packed_data" \
        --out runs/mmfi_pose_window5_xyz_doppler \
        --split cross_subject \
        --signal-mode none \
        --target-mode pelvis_relative \
        --epochs 25

Return-strength ablation using domain-compatible within-window normalisation:
    python mmfi_pose_quick.py train \
        --packed-root "D:/backup_data/MM-Fi/packed_data" \
        --out runs/mmfi_pose_window5_robust_signal \
        --signal-mode robust \
        --target-mode pelvis_relative \
        --epochs 25

Apply the checkpoint to one SyncWB mapping version:
    python mmfi_pose_quick.py infer-syncwb \
        --checkpoint runs/mmfi_pose_window5_xyz_doppler/best.pt \
        --sqlite workbench.sqlite \
        --artifact-root artifact_store \
        --subject 19_MM \
        --mapping-version piecewise_rgb_to_pc_v001_map \
        --out runs/subject_19_pose

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
TARGET_MODES = ("absolute", "pelvis_relative")
SCRIPT_VERSION = "domain-gap-v3"


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
    n_frames: int


@dataclass(frozen=True)
class FeatureConfig:
    signal_mode: str = "none"
    doppler_limit_mps: float = 3.0

    def __post_init__(self) -> None:
        if self.signal_mode not in SIGNAL_MODES:
            raise ValueError(
                f"signal_mode must be one of {SIGNAL_MODES}, got {self.signal_mode!r}"
            )
        if not np.isfinite(self.doppler_limit_mps) or self.doppler_limit_mps <= 0:
            raise ValueError("doppler_limit_mps must be finite and positive")

    @property
    def input_dim(self) -> int:
        # absolute xyz + centred xyz + Doppler + optional signal + relative time
        return 8 + int(self.signal_mode != "none")


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
) -> np.ndarray:
    """Construct model features from a sampled radar window.

    Input columns:
        [x, y, z, Doppler, intensity/SNR, relative_time]

    Always-emitted features (8):
        absolute xyz / 4 m,
        window-centred xyz / 2 m,
        physically clipped Doppler / ``doppler_limit_mps``,
        relative frame time.

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

    centre = np.median(xyz[mask], axis=0) if np.any(mask) else np.zeros(3, dtype=np.float32)
    absolute_xyz = np.clip(xyz / 4.0, -2.0, 2.0)
    centred_xyz = np.clip((xyz - centre) / 2.0, -2.0, 2.0)
    doppler_n = (
        np.clip(
            doppler,
            -feature_config.doppler_limit_mps,
            feature_config.doppler_limit_mps,
        )
        / feature_config.doppler_limit_mps
    )[:, None]
    time_n = np.clip(relative_time, -1.0, 1.0)[:, None]

    parts = [absolute_xyz, centred_xyz, doppler_n]
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

    Sampling with replacement is used when a source frame contains fewer points
    than its quota. This removes the strong source/target cue caused by MM-Fi
    windows being padded while denser SyncWB windows were fully populated.
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

        replace = len(frame) < quota
        if rng is None:
            if replace:
                ids = np.arange(quota, dtype=np.int64) % len(frame)
            else:
                ids = np.linspace(0, len(frame) - 1, quota, dtype=np.int64)
        else:
            ids = rng.choice(len(frame), size=quota, replace=replace)
        sampled.append(frame[ids])
        sampled_masks.append(np.ones(quota, dtype=bool))

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
    rng: np.random.Generator,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    translation_m: Sequence[float],
    body_scale_min: float,
    body_scale_max: float,
    x_reflection_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one physically consistent geometric augmentation to points and pose.

    Rotation, reflection and body-scale are applied about the pelvis so the
    absolute pelvis remains stable. Translation is then applied to both sensor
    points and pose. This increases orientation/range robustness without
    corrupting radar-to-pose correspondence.
    """
    out_points = np.asarray(points, dtype=np.float32).copy()
    out_pose = np.asarray(pose, dtype=np.float32).copy()
    valid = point_valid_mask(out_points)
    pelvis = out_pose[0, :3].copy()

    roll = math.radians(float(rng.uniform(-roll_deg, roll_deg)))
    pitch = math.radians(float(rng.uniform(-pitch_deg, pitch_deg)))
    yaw = math.radians(float(rng.uniform(-yaw_deg, yaw_deg)))
    rotation = rotation_matrix_xyz(roll, pitch, yaw)
    scale = float(rng.uniform(body_scale_min, body_scale_max))

    out_pose[:, :3] = (out_pose[:, :3] - pelvis) * scale @ rotation.T + pelvis
    if np.any(valid):
        out_points[valid, :3] = (
            (out_points[valid, :3] - pelvis) * scale @ rotation.T + pelvis
        )

    if rng.random() < x_reflection_probability:
        out_pose[:, 0] = 2.0 * pelvis[0] - out_pose[:, 0]
        if np.any(valid):
            out_points[valid, 0] = 2.0 * pelvis[0] - out_points[valid, 0]

    translation_limit = np.asarray(translation_m, dtype=np.float32)
    translation = rng.uniform(-translation_limit, translation_limit).astype(np.float32)
    out_pose[:, :3] += translation
    if np.any(valid):
        out_points[valid, :3] += translation
    return out_points, out_pose


def encode_pose_target(pose: np.ndarray, target_mode: str) -> np.ndarray:
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
    raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")


def decode_pose_target(encoded: np.ndarray | torch.Tensor, target_mode: str):
    if target_mode == "absolute":
        return encoded
    if target_mode != "pelvis_relative":
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
    if isinstance(encoded, torch.Tensor):
        pose = encoded.clone()
        pose[..., 1:, :] = pose[..., 1:, :] + pose[..., :1, :]
        return pose
    pose = np.asarray(encoded).copy()
    pose[..., 1:, :] += pose[..., :1, :]
    return pose


def estimate_pose_bounds(
    records: Sequence[PackedRecord],
    max_frames: int,
    seed: int,
    target_mode: str,
    translation_m: Sequence[float],
    body_scale_max: float,
) -> PoseBounds:
    """Estimate per-joint SimCC bounds in encoded target space.

    Pelvis-relative joints receive symmetric radius-based bounds so pitch/roll,
    left-right reflection and body-scale augmentation cannot clip a coordinate.
    The absolute pelvis bounds are expanded by the configured translation range.
    """
    rng = np.random.default_rng(seed)
    total = sum(r.n_frames for r in records)
    samples: list[np.ndarray] = []

    for record in records:
        n = max(1, int(round(max_frames * record.n_frames / total)))
        n = min(n, record.n_frames)
        ids = rng.choice(record.n_frames, size=n, replace=False)
        pose = np.load(record.pose_path, mmap_mode="r")[ids]
        pose = transform_mmfi_pose(pose)
        pose = pose[np.all(np.isfinite(pose), axis=(1, 2))]
        if len(pose):
            samples.append(pose)

    if not samples:
        raise ValueError("No finite MM-Fi pose values found")

    poses = np.concatenate(samples, axis=0)
    encoded = encode_pose_target(poses, target_mode)
    low = np.percentile(encoded, 0.1, axis=0)
    high = np.percentile(encoded, 99.9, axis=0)
    margin = np.maximum(0.10 * (high - low), 0.05)
    low -= margin
    high += margin

    translation_limit = np.asarray(translation_m, dtype=np.float32)
    if target_mode == "pelvis_relative":
        low[0] -= translation_limit
        high[0] += translation_limit
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
    """Centred sliding windows that never cross a packed-record boundary."""

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

        for record in self.records:
            if record.n_frames < self.window_size:
                raise ValueError(
                    f"{record.point_path} contains {record.n_frames} frames, "
                    f"less than window_size={self.window_size}"
                )

        self.window_counts = [r.n_frames - self.window_size + 1 for r in self.records]
        self.ends = np.cumsum(self.window_counts).tolist()
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return int(self.ends[-1])

    def _arrays(self, record_index: int) -> tuple[np.ndarray, np.ndarray]:
        if record_index not in self._cache:
            record = self.records[record_index]
            self._cache[record_index] = (
                np.load(record.point_path, mmap_mode="r"),
                np.load(record.pose_path, mmap_mode="r"),
            )
        return self._cache[record_index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        record_index = bisect.bisect_right(self.ends, int(index))
        record_start = 0 if record_index == 0 else self.ends[record_index - 1]
        window_start = int(index) - int(record_start)
        centre_frame = window_start + self.half_window
        window_end = window_start + self.window_size

        point_clouds, poses = self._arrays(record_index)
        frames = [
            transform_mmfi_points(point_clouds[frame_index])
            for frame_index in range(window_start, window_end)
        ]
        points = stack_radar_window(frames)
        pose = transform_mmfi_pose(poses[centre_frame])

        rng = np.random.default_rng(self.seed + int(index) * 104729)
        if self.training:
            points, pose = augment_geometry(
                points,
                pose,
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
                points[np.flatnonzero(valid)[drop]] = 0.0

        sampled, token_mask = frame_balanced_sample(
            points,
            self.num_points,
            self.window_size,
            rng if self.training else None,
        )
        features = robust_feature_normalize(sampled, self.feature_config)
        if (
            self.training
            and self.feature_config.signal_mode != "none"
            and rng.random() < self.signal_dropout_probability
        ):
            # Signal is immediately before relative time in both signal modes.
            features[:, -2] = 0.0

        encoded_pose = encode_pose_target(pose, self.target_mode)
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
            "index": int(index),
            "centre_frame_index": int(centre_frame),
        }


def cap_dataset(dataset: Dataset, maximum: int, seed: int) -> Dataset:
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(dataset), size=maximum, replace=False))
    return Subset(dataset, ids.tolist())


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
    return decode_pose_target(encoded, target_mode)


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
        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(points, token_mask)
            loss = simcc_loss(logits, labels)
            pred = decode_simcc(logits, low, high, target_mode)
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
    )

    preprocessing_contract = {
        "script_version": SCRIPT_VERSION,
        "mmfi_point_order": MMFI_POINT_ORDER.tolist(),
        "mmfi_internal_columns": ["x", "y", "z", "doppler", "intensity"],
        "mmfi_pose_order": MMFI_POSE_ORDER.tolist(),
        "feature_config": asdict(feature_config),
        "frame_balanced_sampling": True,
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
    train_ds: Dataset = PackedMMFiDataset(
        records=train_records,
        training=True,
        **dataset_kwargs,
    )
    val_ds: Dataset = PackedMMFiDataset(
        records=val_records,
        training=False,
        **dataset_kwargs,
    )
    test_ds: Dataset = PackedMMFiDataset(
        records=test_records,
        training=False,
        **dataset_kwargs,
    )

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

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(points, token_mask)
                loss = simcc_loss(logits, labels)
                pred_pose = decode_simcc(logits, low, high, args.target_mode)
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
) -> np.ndarray:
    low_np, high_np = bounds.arrays()
    low = torch.tensor(low_np, device=device)
    high = torch.tensor(high_np, device=device)
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    outputs: list[np.ndarray] = []

    def flush() -> None:
        if not features:
            return
        x = torch.from_numpy(np.stack(features)).to(device)
        mask = torch.from_numpy(np.stack(masks)).to(device)
        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            pred = decode_simcc(model(x, mask), low, high, target_mode)
        outputs.append(pred.float().cpu().numpy())
        features.clear()
        masks.clear()

    iterator = tqdm(
        point_clouds,
        total=total,
        desc="Model inference",
        leave=False,
        disable=not progress,
    )
    for points in iterator:
        sampled, token_mask = frame_balanced_sample(
            np.asarray(points, dtype=np.float32),
            num_points,
            window_size,
            None,
        )
        features.append(robust_feature_normalize(sampled, feature_config))
        masks.append(token_mask)
        if len(features) >= batch_size:
            flush()
    flush()

    if not outputs:
        return np.empty((0, 17, 3), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


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
    if args.max_frames > 0:
        rows = rows.iloc[:args.max_frames]
    rows = rows.reset_index(drop=True)

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

    # Cache only unique source/target frames. With a five-frame sliding window,
    # adjacent examples share four radar frames, so this reduces roughly
    # 5 * mapping_rows reads to approximately the number of unique radar frames.
    radar_requests: set[tuple[str, str, str, int]] = set()
    pose_requests: set[tuple[str, str, str, int]] = set()
    for row in rows.itertuples(index=False):
        subject_id = str(row.subject_id)
        target_run_id = str(row.target_run_id)
        target_device_type = str(row.target_device_type)
        target_centre = int(row.target_sample_index)
        for target_index in range(
            target_centre - half_window,
            target_centre + half_window + 1,
        ):
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
        for target_index in range(
            target_centre - half_window,
            target_centre + half_window + 1,
        ):
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

        pose3d = pose_cache.get((
            subject_id,
            str(row.source_run_id),
            str(row.source_device_type),
            int(row.source_sample_index),
        ))
        if pose3d is None:
            skipped["missing_pose_payload"] += 1
            continue

        gt = kinect_pose32_to_radar_mmfi17(
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
        if gt is None or not np.all(np.isfinite(gt)):
            skipped["invalid_pose"] += 1
            continue

        num_people = (
            float(row.num_people)
            if pd.notna(row.num_people)
            else np.nan
        )
        metric_eligible = bool(np.isfinite(num_people) and int(num_people) == 1)

        radar_windows.append(stack_radar_window(window_frames))
        gt_frames.append(gt)
        kept_rows.append({
            "source_sample_index": int(row.source_sample_index),
            "target_sample_index": target_centre,
            "target_window_start": target_centre - half_window,
            "target_window_end": target_centre + half_window,
            "num_people": num_people,
            "metric_eligible_single_person": metric_eligible,
            "predicted_minus_estimated_ms": getattr(
                row,
                "predicted_minus_estimated_ms",
                np.nan,
            ),
        })

    if not radar_windows:
        raise ValueError("No mapped rows contained a complete radar window and valid pose3d")

    pred = predict_point_clouds(
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

    if metric_mask.sum() == 0:
        raise ValueError(
            "No valid predictions had SAMPLE_SUMMARY.num_people == 1; "
            "SyncWB evaluation metrics cannot be reported"
        )

    single_indices = np.flatnonzero(metric_mask)
    requested_calibration = max(
        1,
        int(round(len(single_indices) * args.calibration_fraction)),
    )
    if len(single_indices) >= 11:
        n_calibration = min(max(10, requested_calibration), len(single_indices) - 1)
    else:
        n_calibration = max(1, len(single_indices) - 1)

    calibration_indices = single_indices[:n_calibration]
    evaluation_indices = single_indices[n_calibration:]
    if len(evaluation_indices) == 0:
        warnings.warn(
            "Too few single-person frames for held-out aligned evaluation; "
            "using all single-person frames for aligned metrics.",
            RuntimeWarning,
        )
        evaluation_indices = single_indices

    transform = fit_global_similarity(
        pred[calibration_indices],
        gt[calibration_indices],
        joint_indices=BODY15_INDICES,
    )
    pred_global = apply_global_similarity(pred, transform)

    heldout_mask = np.zeros(len(pair_df), dtype=bool)
    heldout_mask[evaluation_indices] = True

    summary: dict[str, object] = {
        "checkpoint_mmfi_validation": checkpoint.get("val_metrics", {}),
        "checkpoint_preprocessing": {
            "script_version": checkpoint.get("script_version"),
            "feature_config": asdict(feature_config),
            "target_mode": target_mode,
            "frame_balanced_sampling": checkpoint.get("frame_balanced_sampling"),
            "mmfi_point_order": checkpoint.get("mmfi_point_order"),
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
            "primary_mapping_rows_considered": int(len(rows)),
            "valid_predictions_saved": int(len(pred)),
            "single_person_metric_frames": int(metric_mask.sum()),
            "excluded_non_single_person_predictions": int((~metric_mask).sum()),
            "excluded_multi_person_predictions": int(
                (pd.to_numeric(pair_df["num_people"], errors="coerce") > 1).sum()
            ),
            "excluded_unknown_or_zero_person_predictions": int(
                (
                    ~pd.to_numeric(pair_df["num_people"], errors="coerce").eq(1)
                    & ~pd.to_numeric(pair_df["num_people"], errors="coerce").gt(1)
                ).sum()
            ),
            **{f"skipped_{key}": int(value) for key, value in skipped.items()},
        },
        "evaluation_policy": {
            "syncwb_metrics_require_num_people_equal_to": 1,
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
        pred_globally_aligned=pred_global,
        target=gt,
        source_sample_index=pair_df["source_sample_index"].to_numpy(),
        target_sample_index=pair_df["target_sample_index"].to_numpy(),
        num_people=pair_df["num_people"].to_numpy(),
        metric_eligible_single_person=metric_mask,
        globally_aligned_heldout_mask=heldout_mask,
        joint_names=np.asarray(MMFI17_NAMES),
        global_scale=np.asarray(transform[0]),
        global_rotation=transform[1],
        global_translation=transform[2],
        signal_mode=np.asarray(feature_config.signal_mode),
        target_mode=np.asarray(target_mode),
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
    np.testing.assert_array_equal(time_counts, [5, 5, 5, 5, 5])
    results["frame_balanced_temporal_sampling"] = True

    for signal_mode, expected_dim in (("none", 8), ("robust", 9), ("rank", 9)):
        config = FeatureConfig(signal_mode=signal_mode, doppler_limit_mps=3.0)
        features = robust_feature_normalize(sampled, config)
        assert features.shape == (25, expected_dim)
        assert np.max(np.abs(features[:, 6])) <= 1.0 + 1e-6
        if signal_mode != "none":
            affine = sampled.copy()
            affine[:, 4] = affine[:, 4] * 7.0 + 13.0
            affine_features = robust_feature_normalize(affine, config)
            np.testing.assert_allclose(features[:, -2], affine_features[:, -2], atol=1e-5)
    results["signal_modes_and_affine_compatibility"] = True

    pose = np.zeros((17, 3), dtype=np.float32)
    pose[0] = [0.2, 3.0, 0.1]
    pose[1:] = pose[0] + np.linspace(-0.8, 0.8, 16)[:, None] * np.asarray(
        [0.2, 0.1, 0.3], dtype=np.float32
    )
    for target_mode in TARGET_MODES:
        encoded = encode_pose_target(pose, target_mode)
        decoded = decode_pose_target(encoded, target_mode)
        np.testing.assert_allclose(decoded, pose, atol=1e-6)
    results["pose_target_round_trip"] = True

    low = np.full((17, 3), -5.0, dtype=np.float32)
    high = np.full((17, 3), 5.0, dtype=np.float32)
    bounds = PoseBounds(low.tolist(), high.tolist())
    device = torch.device("cpu")
    for signal_mode in SIGNAL_MODES:
        feature_config = FeatureConfig(signal_mode=signal_mode)
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
            decoded = decode_simcc(
                logits,
                torch.tensor(low),
                torch.tensor(high),
                "pelvis_relative",
            )
            assert decoded.shape == (2, 17, 3)
    results["model_forward_all_signal_modes"] = True

    feature_config = FeatureConfig(signal_mode="none")
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
            "target_mode": "pelvis_relative",
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
        assert loaded_target_mode == "pelvis_relative"
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
    train.add_argument("--out", default="runs/mmfi_pose_window5")
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
        "--target-mode",
        choices=TARGET_MODES,
        default="pelvis_relative",
        help="Default predicts an absolute pelvis plus pelvis-relative remaining joints",
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
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--cpu", action="store_true")
    train.set_defaults(func=train_command)

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
