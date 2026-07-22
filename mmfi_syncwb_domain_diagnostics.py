#!/usr/bin/env python3
"""Investigate MM-Fi -> SyncWB radar/pose domain shift.

This script compares paired MM-Fi radar point clouds + 3D poses against paired
SyncWB radar point clouds + Azure Kinect poses. It is intended to diagnose the
large zero-shot domain gap seen when applying an MM-Fi-trained pose model to a
SyncWB subject.

It investigates:

1. Axis conventions inside each dataset
   - all 48 signed XYZ permutations are tested;
   - pose-root motion is compared with robust radar-cloud-centroid motion;
   - a point-to-skeleton proximity score provides a second heuristic;
   - for SyncWB, the expected Kinect -> radar mapping is x,z,-y.

2. Cross-domain coordinate conventions
   - root-relative MM-Fi and SyncWB skeleton distributions are compared;
   - pelvis-centred point-cloud distributions are compared;
   - all signed permutations are ranked, with identity expected after the
     current conversions.

3. Frame-of-reference / extrinsic-offset evidence
   - pelvis-to-radar-cloud-centroid offsets are reported per domain;
   - their variability indicates how reliable that offset is as an extrinsic
     proxy (it is not a formal sensor calibration).

4. Feature distributions
   - point-cloud x/y/z, Doppler and return signal (MM-Fi intensity vs SyncWB SNR);
   - absolute and root-relative pose x/y/z;
   - point counts, bone lengths and pose scale.

5. Normalisation
   - raw feature gaps;
   - the exact point-feature normalisation used by mmfi_pose_quick.py;
   - exact window-level robust signal normalisation used by the model;
   - window-level percentile-rank signal normalisation as an alternative;
   - clipping/saturation rates.

Outputs
-------
The output directory contains:

- summary.txt
- report.json
- feature_statistics.csv
- frame_statistics.csv
- model_feature_statistics.csv
- model_feature_gap.csv
- signal_relationships.csv
- channel_quantisation.csv
- signal_proximity_summary.csv
- signal_proximity_deciles.csv
- signal_proximity_strength_groups.csv
- signal_proximity_range_bins.csv
- signal_normalisation_comparison.csv
- signal_proximity_samples.npz
- axis_search_mmfi_pose_to_radar.csv
- axis_search_syncwb_kinect_to_radar.csv
- cross_domain_pose_axis_search.csv
- cross_domain_point_axis_search.csv
- plots/*.png

Example
-------
python mmfi_syncwb_domain_diagnostics.py \
    --packed-root "D:/backup_data/MM-Fi/packed_data" \
    --sqlite workbench.sqlite \
    --artifact-root artifact_store \
    --subject 19_MM \
    --mapping-version piecewise_rgb_to_pc_v001_map \
    --out runs/domain_diagnostics_19_MM \
    --max-mmfi-frames 5000 \
    --max-syncwb-frames 5000

Notes
-----
- Axis-search scores are diagnostics, not a replacement for formal extrinsic
  calibration. Radar-cloud centroids depend on CFAR, body reflectivity, motion,
  occlusion and multipath.
- MM-Fi "intensity" and SyncWB "SNR" are not assumed to be physically
  equivalent. The script compares distributions and relationships only.
- SyncWB analysis uses frames with SAMPLE_SUMMARY.num_people == 1.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_: object):
        return iterable

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


# -----------------------------------------------------------------------------
# Shared topology and current project conventions
# -----------------------------------------------------------------------------

SCRIPT_VERSION = "signal-v2.1"

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
    "face_center",
    "head",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]

MMFI17_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (7, 11), (11, 12), (12, 13),
    (7, 14), (14, 15), (15, 16),
]

BODY15_INDICES = np.asarray([i for i in range(17) if i not in {9, 10}], dtype=np.int64)

KINECT32_TO_MMFI17 = np.asarray([
    0,   # pelvis
    18,  # left hip
    19,  # left knee
    20,  # left ankle
    22,  # right hip
    23,  # right knee
    24,  # right ankle
    2,   # thorax / spine chest
    3,   # neck
    27,  # face centre <- nose, provisional
    26,  # head, provisional
    12,  # right shoulder
    13,  # right elbow
    14,  # right wrist
    5,   # left shoulder
    6,   # left elbow
    7,   # left wrist
], dtype=np.int64)

# Current training-script conversions.
MMFI_POINT_ORDER = np.asarray([1, 0, 2, 4, 3], dtype=np.int64)
MMFI_POSE_ORDER = np.asarray([0, 2, 1], dtype=np.int64)
NOISE_TARGET_IDS = {253.0, 254.0, 255.0}

POINT_FEATURES = ("x", "y", "z", "doppler", "signal")
POSE_FEATURES = ("x", "y", "z")
QUANTILES = np.asarray([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999])
QUANTILE_NAMES = ["p0_1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99_9"]


@dataclass(frozen=True)
class PackedRecord:
    environment: str
    subject: str
    point_path: str
    pose_path: str
    n_frames: int


@dataclass
class DomainSamples:
    name: str
    point_frames: list[np.ndarray]       # centre frames, each [P,5]
    point_windows: list[list[np.ndarray]]  # exact model windows, each list of [P,5]
    poses: np.ndarray                    # [F,17,3] in current radar convention
    pose_base_for_axis_search: np.ndarray  # [F,17,3], before candidate mapping
    frame_ids: np.ndarray
    return_signal_name: str
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if len(self.point_frames) != len(self.poses):
            raise ValueError(
                f"{self.name}: point/pose frame mismatch "
                f"{len(self.point_frames)} != {len(self.poses)}"
            )
        if len(self.point_windows) != len(self.poses):
            raise ValueError(f"{self.name}: point-window/pose frame mismatch")
        if len(self.pose_base_for_axis_search) != len(self.poses):
            raise ValueError(f"{self.name}: axis-search pose frame mismatch")


@dataclass(frozen=True)
class AxisTransform:
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]

    @property
    def name(self) -> str:
        axis = ("x", "y", "z")
        parts = []
        for source_axis, sign in zip(self.permutation, self.signs):
            prefix = "-" if sign < 0 else ""
            parts.append(prefix + axis[source_axis])
        return ",".join(parts)

    @property
    def determinant(self) -> int:
        inversions = sum(
            self.permutation[i] > self.permutation[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )
        parity = -1 if inversions % 2 else 1
        return int(parity * np.prod(self.signs))

    def apply(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        return arr[..., list(self.permutation)] * np.asarray(self.signs, dtype=arr.dtype)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def signed_axis_transforms() -> list[AxisTransform]:
    return [
        AxisTransform(tuple(perm), tuple(signs))
        for perm in itertools.permutations(range(3))
        for signs in itertools.product((-1, 1), repeat=3)
    ]


def point_valid_mask(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] < 5:
        return np.zeros(len(p) if p.ndim else 0, dtype=bool)
    return (
        np.all(np.isfinite(p[:, :5]), axis=1)
        & np.any(np.abs(p[:, :5]) > 1e-8, axis=1)
    )


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


def robust_cloud_centroid(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.full(3, np.nan)
    return np.nanmedian(p[:, :3], axis=0)


def safe_iqr(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    q25 = np.nanpercentile(values, 25, axis=axis)
    q75 = np.nanpercentile(values, 75, axis=axis)
    return np.maximum(q75 - q25, 1e-6)


def finite_flat(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def describe_values(
    values: np.ndarray,
    *,
    domain: str,
    group: str,
    feature: str,
    units: str,
) -> dict[str, object]:
    x = finite_flat(values)
    row: dict[str, object] = {
        "domain": domain,
        "group": group,
        "feature": feature,
        "units": units,
        "count": int(len(x)),
    }
    if len(x) == 0:
        row.update({
            "min": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            **{name: np.nan for name in QUANTILE_NAMES},
        })
        return row

    q = np.quantile(x, QUANTILES)
    row.update({
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
        "iqr": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        **{name: float(value) for name, value in zip(QUANTILE_NAMES, q)},
    })
    return row


def reservoir_points(
    frames: Sequence[np.ndarray],
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Return an approximately frame-balanced point sample."""
    if not frames:
        return np.empty((0, 5), dtype=np.float32)
    rng = np.random.default_rng(seed)
    per_frame = max(1, int(math.ceil(max_points / len(frames))))
    chunks: list[np.ndarray] = []
    for frame in frames:
        p = np.asarray(frame, dtype=np.float32)
        if len(p) == 0:
            continue
        if len(p) > per_frame:
            ids = rng.choice(len(p), size=per_frame, replace=False)
            p = p[ids]
        chunks.append(p[:, :5])
    if not chunks:
        return np.empty((0, 5), dtype=np.float32)
    out = np.concatenate(chunks, axis=0)
    if len(out) > max_points:
        ids = rng.choice(len(out), size=max_points, replace=False)
        out = out[ids]
    return out


def select_evenly(total: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or total <= maximum:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.linspace(0, total - 1, maximum, dtype=np.int64))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = finite_flat(x)
    b = finite_flat(y)
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def paired_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values).reshape(-1)).rank(method="average").to_numpy(dtype=float)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3:
        return float("nan")
    return paired_corr(rankdata(a), rankdata(b))


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


# -----------------------------------------------------------------------------
# MM-Fi loading
# -----------------------------------------------------------------------------


def discover_packed_records(root: Path) -> list[PackedRecord]:
    records: list[PackedRecord] = []
    for point_path in sorted(root.glob("*/S*/point_clouds.npy")):
        pose_path = point_path.parent / "kpts.npy"
        if not pose_path.exists():
            continue
        points = np.load(point_path, mmap_mode="r")
        poses = np.load(pose_path, mmap_mode="r")
        if points.ndim != 3 or points.shape[-1] < 5:
            raise ValueError(f"Unexpected MM-Fi points shape {points.shape}: {point_path}")
        if poses.ndim != 3 or poses.shape[1:] != (17, 3):
            raise ValueError(f"Unexpected MM-Fi pose shape {poses.shape}: {pose_path}")
        if len(points) != len(poses):
            raise ValueError(f"MM-Fi frame count mismatch in {point_path.parent}")
        records.append(PackedRecord(
            environment=point_path.parent.parent.name,
            subject=point_path.parent.name,
            point_path=str(point_path),
            pose_path=str(pose_path),
            n_frames=len(points),
        ))
    if not records:
        raise FileNotFoundError(f"No packed MM-Fi records found under {root}")
    return records


def load_mmfi_samples(
    root: Path,
    max_frames: int,
    window_size: int,
    seed: int,
    progress: bool,
) -> DomainSamples:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    half = window_size // 2
    records = discover_packed_records(root)
    valid_centres_per_record = [max(0, r.n_frames - 2 * half) for r in records]
    total_centres = sum(valid_centres_per_record)
    if total_centres == 0:
        raise ValueError("No MM-Fi record is long enough for the requested window")

    selected_global = select_evenly(total_centres, max_frames)
    selected_set = set(map(int, selected_global))

    point_frames: list[np.ndarray] = []
    point_windows: list[list[np.ndarray]] = []
    poses: list[np.ndarray] = []
    frame_ids: list[str] = []
    global_start = 0

    iterator = tqdm(records, desc="Load MM-Fi samples", disable=not progress)
    for record, valid_count in zip(iterator, valid_centres_per_record):
        local_offsets = [
            global_id - global_start
            for global_id in selected_set
            if global_start <= global_id < global_start + valid_count
        ]
        global_start += valid_count
        if not local_offsets:
            continue

        pc_all = np.load(record.point_path, mmap_mode="r")
        pose_all = np.load(record.pose_path, mmap_mode="r")
        for local_offset in sorted(local_offsets):
            centre_id = int(local_offset + half)
            window: list[np.ndarray] = []
            complete = True
            for frame_id in range(centre_id - half, centre_id + half + 1):
                raw_points = np.asarray(pc_all[frame_id], dtype=np.float32)
                points = raw_points[..., MMFI_POINT_ORDER]
                points = points[point_valid_mask(points), :5]
                if len(points) == 0:
                    complete = False
                    break
                window.append(points)
            if not complete:
                continue
            pose = np.asarray(pose_all[centre_id], dtype=np.float32)[..., MMFI_POSE_ORDER]
            if not np.all(np.isfinite(pose)):
                continue
            point_frames.append(window[half])
            point_windows.append(window)
            poses.append(pose)
            frame_ids.append(f"{record.environment}/{record.subject}/{centre_id}")

    if not poses:
        raise ValueError("No valid MM-Fi paired samples were loaded")

    pose_array = np.stack(poses).astype(np.float32)
    return DomainSamples(
        name="MM-Fi",
        point_frames=point_frames,
        point_windows=point_windows,
        poses=pose_array,
        # The training script has already converted MM-Fi pose into the candidate
        # radar convention. Identity should therefore rank best here.
        pose_base_for_axis_search=pose_array.copy(),
        frame_ids=np.asarray(frame_ids),
        return_signal_name="intensity",
        metadata={
            "records": len(records),
            "available_window_centres": total_centres,
            "loaded_valid_frames": len(pose_array),
            "window_size": window_size,
            "point_axis_conversion": "raw columns [1,0,2,4,3] -> [x,y,z,Doppler,intensity]",
            "pose_axis_conversion": "raw axes [0,2,1]",
        },
    )


# -----------------------------------------------------------------------------
# SyncWB loading
# -----------------------------------------------------------------------------


def attach_source_num_people(rows: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "run_id", "device_type", "sample_index", "num_people"}
    if summary.empty or not required.issubset(summary.columns):
        raise ValueError("SAMPLE_SUMMARY is missing required num_people information")

    right = summary[list(required)].copy()
    for col in ("subject_id", "run_id", "device_type"):
        right[col] = right[col].astype(str)
    right["sample_index"] = pd.to_numeric(right["sample_index"], errors="coerce").astype("Int64")
    right["num_people"] = pd.to_numeric(right["num_people"], errors="coerce")
    right = right.rename(columns={
        "run_id": "source_run_id",
        "device_type": "source_device_type",
        "sample_index": "source_sample_index",
    })
    return rows.merge(
        right,
        on=["subject_id", "source_run_id", "source_device_type", "source_sample_index"],
        how="left",
        validate="many_to_one",
    )


def open_syncwb_ragged_readers(
    run_assets: pd.DataFrame,
    artifact_root: str | Path,
    required_payloads: set[tuple[str, str, str, str]],
) -> dict[tuple[str, str, str, str], object]:
    from sync_workbench.storage.artifact_store import ArtifactStore
    from sync_workbench.storage.ragged_npz import RaggedNpzReader

    roles = {
        "radar_points": "radar_points_bundle",
        "pose3d": "pose3d_bundle",
    }
    required_columns = {"subject_id", "run_id", "device_type", "asset_role", "asset_ref"}
    if run_assets.empty or not required_columns.issubset(run_assets.columns):
        raise ValueError("RUN_ASSET is missing columns required for direct bundle access")

    assets = run_assets[list(required_columns)].copy()
    for col in required_columns:
        assets[col] = assets[col].astype(str)
    assets = assets.drop_duplicates()
    store = ArtifactStore(artifact_root)
    readers: dict[tuple[str, str, str, str], object] = {}

    for key in sorted(required_payloads):
        subject_id, run_id, device_type, payload_role = key
        asset_role = roles[payload_role]
        matches = assets[
            (assets["subject_id"] == subject_id)
            & (assets["run_id"] == run_id)
            & (assets["device_type"] == device_type)
            & (assets["asset_role"] == asset_role)
        ]
        refs = matches["asset_ref"].dropna().astype(str).unique().tolist()
        if len(refs) != 1:
            raise ValueError(
                f"Expected one {asset_role} for {subject_id}/{run_id}/{device_type}, found {refs}"
            )
        readers[key] = RaggedNpzReader(store.path_for_ref(refs[0]))
    return readers


def raw_kinect_pose32_to_mmfi17_m(
    pose3d: np.ndarray,
    person_index: int,
) -> np.ndarray | None:
    arr = np.asarray(pose3d, dtype=np.float32)
    if (
        arr.ndim != 3
        or arr.shape[0] <= person_index
        or arr.shape[1] < 32
        or arr.shape[2] < 3
    ):
        return None
    return arr[person_index, KINECT32_TO_MMFI17, :3].astype(np.float32) * 1e-3


def current_kinect_to_radar(raw_pose_mmfi17_m: np.ndarray) -> np.ndarray:
    out = np.asarray(raw_pose_mmfi17_m, dtype=np.float32)[..., [0, 2, 1]].copy()
    out[..., 2] *= -1.0
    return out


def load_syncwb_samples(
    sqlite_path: Path,
    artifact_root: Path,
    subject: str,
    mapping_version: str,
    max_frames: int,
    window_size: int,
    person_index: int,
    filter_noise: bool,
    progress: bool,
) -> DomainSamples:
    from sync_workbench.storage.sqlite_store import SQLiteCoreStore

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    half = window_size // 2

    store = SQLiteCoreStore(sqlite_path)
    mappings = store.read_table("SAMPLE_MAPPING")
    mask = (
        (mappings["subject_id"].astype(str) == str(subject))
        & (mappings["mapping_version_id"].astype(str) == str(mapping_version))
    )
    rows = mappings.loc[mask].copy()
    if "is_primary" in rows.columns:
        rows = rows[bool_series(rows["is_primary"])]
    if rows.empty:
        raise ValueError("No primary SAMPLE_MAPPING rows matched the requested subject/version")

    for col in (
        "subject_id", "source_run_id", "source_device_type",
        "target_run_id", "target_device_type",
    ):
        rows[col] = rows[col].astype(str)
    rows["source_sample_index"] = pd.to_numeric(
        rows["source_sample_index"], errors="raise"
    ).astype(int)
    rows["target_sample_index"] = pd.to_numeric(
        rows["target_sample_index"], errors="raise"
    ).astype(int)
    rows = attach_source_num_people(rows, store.read_table("SAMPLE_SUMMARY"))
    rows = rows[pd.to_numeric(rows["num_people"], errors="coerce").eq(1)]
    rows = rows.sort_values("source_sample_index").reset_index(drop=True)
    if rows.empty:
        raise ValueError("No single-person mapped SyncWB rows were found")

    selected = select_evenly(len(rows), max_frames)
    rows = rows.iloc[selected].reset_index(drop=True)

    required_payloads: set[tuple[str, str, str, str]] = set()
    for row in rows.itertuples(index=False):
        required_payloads.add((
            str(row.subject_id), str(row.target_run_id), str(row.target_device_type), "radar_points"
        ))
        required_payloads.add((
            str(row.subject_id), str(row.source_run_id), str(row.source_device_type), "pose3d"
        ))
    readers = open_syncwb_ragged_readers(
        store.read_table("RUN_ASSET"), artifact_root, required_payloads
    )

    radar_requests: set[tuple[str, str, str, int]] = set()
    pose_requests: set[tuple[str, str, str, int]] = set()
    for row in rows.itertuples(index=False):
        subject_id = str(row.subject_id)
        target_run = str(row.target_run_id)
        target_device = str(row.target_device_type)
        centre = int(row.target_sample_index)
        for target_index in range(centre - half, centre + half + 1):
            if target_index >= 0:
                radar_requests.add((subject_id, target_run, target_device, target_index))
        pose_requests.add((
            subject_id, str(row.source_run_id), str(row.source_device_type),
            int(row.source_sample_index),
        ))

    radar_cache: dict[tuple[str, str, str, int], np.ndarray | None] = {}
    for subject_id, run_id, device_type, sample_index in tqdm(
        sorted(radar_requests),
        desc="Cache SyncWB radar frames",
        disable=not progress,
    ):
        reader = readers[(subject_id, run_id, device_type, "radar_points")]
        try:
            raw = reader.get(sample_index)
        except KeyError:
            radar_cache[(subject_id, run_id, device_type, sample_index)] = None
        else:
            filtered = filter_syncwb_radar(raw, filter_noise)
            radar_cache[(subject_id, run_id, device_type, sample_index)] = (
                filtered if len(filtered) else None
            )

    pose_cache: dict[tuple[str, str, str, int], np.ndarray | None] = {}
    for subject_id, run_id, device_type, sample_index in tqdm(
        sorted(pose_requests),
        desc="Cache SyncWB Kinect poses",
        disable=not progress,
    ):
        reader = readers[(subject_id, run_id, device_type, "pose3d")]
        try:
            pose_cache[(subject_id, run_id, device_type, sample_index)] = reader.get(sample_index)
        except KeyError:
            pose_cache[(subject_id, run_id, device_type, sample_index)] = None

    point_frames: list[np.ndarray] = []
    point_windows: list[list[np.ndarray]] = []
    current_poses: list[np.ndarray] = []
    raw_poses: list[np.ndarray] = []
    frame_ids: list[str] = []
    skipped = {
        "incomplete_radar_window": 0,
        "missing_pose": 0,
        "invalid_pose": 0,
    }

    iterator = tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc="Assemble SyncWB diagnostic pairs",
        disable=not progress,
    )
    for row in iterator:
        subject_id = str(row.subject_id)
        target_run = str(row.target_run_id)
        target_device = str(row.target_device_type)
        centre = int(row.target_sample_index)
        window: list[np.ndarray] = []
        for target_index in range(centre - half, centre + half + 1):
            frame = radar_cache.get((subject_id, target_run, target_device, target_index))
            if frame is None:
                window = []
                break
            window.append(frame)
        if len(window) != window_size:
            skipped["incomplete_radar_window"] += 1
            continue

        pose_payload = pose_cache.get((
            subject_id, str(row.source_run_id), str(row.source_device_type),
            int(row.source_sample_index),
        ))
        if pose_payload is None:
            skipped["missing_pose"] += 1
            continue
        raw_pose = raw_kinect_pose32_to_mmfi17_m(pose_payload, person_index)
        if raw_pose is None or not np.all(np.isfinite(raw_pose)):
            skipped["invalid_pose"] += 1
            continue

        point_frames.append(window[half])
        point_windows.append(window)
        raw_poses.append(raw_pose)
        current_poses.append(current_kinect_to_radar(raw_pose))
        frame_ids.append(
            f"{row.source_run_id}:{int(row.source_sample_index)}->"
            f"{row.target_run_id}:{int(row.target_sample_index)}"
        )

    if not current_poses:
        raise ValueError("No valid paired single-person SyncWB samples were loaded")

    return DomainSamples(
        name="SyncWB",
        point_frames=point_frames,
        point_windows=point_windows,
        poses=np.stack(current_poses).astype(np.float32),
        # Raw Kinect XYZ in metres. The signed-permutation search should recover
        # x,z,-y if the current assumed convention is correct.
        pose_base_for_axis_search=np.stack(raw_poses).astype(np.float32),
        frame_ids=np.asarray(frame_ids),
        return_signal_name="SNR",
        metadata={
            "subject": subject,
            "mapping_version": mapping_version,
            "eligible_single_person_rows_sampled": int(len(rows)),
            "loaded_valid_frames": len(current_poses),
            "window_size": window_size,
            "skipped": skipped,
            "current_kinect_to_radar_axis_conversion": "x,z,-y",
        },
    )


# -----------------------------------------------------------------------------
# Geometry diagnostics
# -----------------------------------------------------------------------------


def point_to_segments_distance(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Minimum distance from each point to the MM-Fi skeleton segments."""
    p = np.asarray(points, dtype=np.float64)[:, :3]
    y = np.asarray(pose, dtype=np.float64)
    if len(p) == 0:
        return np.empty(0, dtype=np.float64)

    minimum = np.full(len(p), np.inf, dtype=np.float64)
    for a, b in MMFI17_EDGES:
        start = y[a]
        end = y[b]
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom < 1e-12:
            distance = np.linalg.norm(p - start, axis=1)
        else:
            t = np.clip(((p - start) @ segment) / denom, 0.0, 1.0)
            nearest = start[None, :] + t[:, None] * segment[None, :]
            distance = np.linalg.norm(p - nearest, axis=1)
        minimum = np.minimum(minimum, distance)
    return minimum


def evaluate_axis_candidate(
    point_frames: Sequence[np.ndarray],
    pose_base: np.ndarray,
    transform: AxisTransform,
    max_geometry_frames: int,
    max_points_per_frame: int,
    seed: int,
) -> dict[str, object]:
    transformed = transform.apply(pose_base)
    centroids = np.stack([robust_cloud_centroid(p) for p in point_frames])
    roots = transformed[:, 0, :]
    valid = np.all(np.isfinite(centroids), axis=1) & np.all(np.isfinite(roots), axis=1)
    centroids = centroids[valid]
    roots = roots[valid]
    transformed_valid = transformed[valid]
    valid_frames = [point_frames[i] for i in np.flatnonzero(valid)]

    if len(roots) < 3:
        return {
            "transform": transform.name,
            "determinant": transform.determinant,
            "frames": int(len(roots)),
            "root_centroid_rmse_m": np.nan,
            "mean_axis_correlation": np.nan,
            "point_to_skeleton_q25_m": np.nan,
            "translation_x_m": np.nan,
            "translation_y_m": np.nan,
            "translation_z_m": np.nan,
        }

    translation = np.median(centroids - roots, axis=0)
    residual = roots + translation - centroids
    root_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

    axis_corrs = [paired_corr(roots[:, i], centroids[:, i]) for i in range(3)]
    finite_corrs = [c for c in axis_corrs if np.isfinite(c)]
    mean_corr = float(np.mean(finite_corrs)) if finite_corrs else np.nan

    ids = select_evenly(len(transformed_valid), max_geometry_frames)
    rng = np.random.default_rng(seed)
    frame_scores: list[float] = []
    for i in ids:
        points = np.asarray(valid_frames[int(i)], dtype=np.float64)
        if len(points) > max_points_per_frame:
            point_ids = rng.choice(len(points), size=max_points_per_frame, replace=False)
            points = points[point_ids]
        pose = transformed_valid[int(i)] + translation
        distances = point_to_segments_distance(points, pose)
        if len(distances):
            # Lower quartile emphasises the likely body-return subset instead of
            # forcing all background/multipath points to lie on the skeleton.
            frame_scores.append(float(np.quantile(distances, 0.25)))
    skeleton_q25 = float(np.median(frame_scores)) if frame_scores else np.nan

    return {
        "transform": transform.name,
        "determinant": transform.determinant,
        "frames": int(len(roots)),
        "root_centroid_rmse_m": root_rmse,
        "mean_axis_correlation": mean_corr,
        "corr_x": axis_corrs[0],
        "corr_y": axis_corrs[1],
        "corr_z": axis_corrs[2],
        "point_to_skeleton_q25_m": skeleton_q25,
        "translation_x_m": float(translation[0]),
        "translation_y_m": float(translation[1]),
        "translation_z_m": float(translation[2]),
    }


def minmax_metric(values: pd.Series, *, reverse: bool = False) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x)
    out = np.ones(len(x), dtype=float)
    if finite.sum() == 0:
        return out
    low, high = np.min(x[finite]), np.max(x[finite])
    if high - low < 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (x[finite] - low) / (high - low)
    if reverse:
        out[finite] = 1.0 - out[finite]
    return out


def run_within_domain_axis_search(
    domain: DomainSamples,
    max_geometry_frames: int,
    max_points_per_frame: int,
    seed: int,
    progress: bool,
) -> pd.DataFrame:
    rows = []
    iterator = tqdm(
        signed_axis_transforms(),
        desc=f"Axis search: {domain.name}",
        disable=not progress,
    )
    for transform in iterator:
        rows.append(evaluate_axis_candidate(
            domain.point_frames,
            domain.pose_base_for_axis_search,
            transform,
            max_geometry_frames,
            max_points_per_frame,
            seed,
        ))
    df = pd.DataFrame(rows)
    df["composite_score"] = (
        0.45 * minmax_metric(df["root_centroid_rmse_m"])
        + 0.45 * minmax_metric(df["point_to_skeleton_q25_m"])
        + 0.10 * minmax_metric(df["mean_axis_correlation"], reverse=True)
    )
    df = df.sort_values("composite_score").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def normalised_pose_shape(poses: np.ndarray) -> np.ndarray:
    p = np.asarray(poses, dtype=np.float64)
    root_relative = p - p[:, :1, :]
    scale = np.linalg.norm(root_relative[:, 8, :], axis=1)  # pelvis -> neck
    fallback = np.nanmedian([
        np.linalg.norm(root_relative[:, b] - root_relative[:, a], axis=1)
        for a, b in MMFI17_EDGES
    ], axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.2), scale, fallback)
    scale = np.maximum(scale, 1e-3)
    return root_relative / scale[:, None, None]


def pose_distribution_distance(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    ref = normalised_pose_shape(reference)[:, BODY15_INDICES]
    can = normalised_pose_shape(candidate)[:, BODY15_INDICES]
    ref_med = np.nanmedian(ref, axis=0)
    can_med = np.nanmedian(can, axis=0)
    ref_iqr = safe_iqr(ref, axis=0)
    can_iqr = safe_iqr(can, axis=0)
    median_gap = float(np.nanmean(np.abs(can_med - ref_med) / np.maximum(ref_iqr, 0.05)))
    iqr_gap = float(np.nanmean(np.abs(np.log((can_iqr + 1e-4) / (ref_iqr + 1e-4)))))
    return median_gap + 0.35 * iqr_gap, median_gap, iqr_gap


def run_cross_domain_pose_axis_search(
    mmfi: DomainSamples,
    syncwb: DomainSamples,
) -> pd.DataFrame:
    rows = []
    for transform in signed_axis_transforms():
        total, median_gap, iqr_gap = pose_distribution_distance(
            mmfi.poses,
            transform.apply(syncwb.poses),
        )
        rows.append({
            "transform_applied_to_syncwb_current_pose": transform.name,
            "determinant": transform.determinant,
            "score": total,
            "median_template_gap": median_gap,
            "iqr_gap": iqr_gap,
        })
    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def root_centred_point_sample(
    domain: DomainSamples,
    max_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_frame = max(1, int(math.ceil(max_points / len(domain.point_frames))))
    chunks: list[np.ndarray] = []
    for points, pose in zip(domain.point_frames, domain.poses):
        p = np.asarray(points, dtype=np.float32)
        if len(p) > per_frame:
            ids = rng.choice(len(p), size=per_frame, replace=False)
            p = p[ids]
        chunks.append(p[:, :3] - pose[0, :3])
    out = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.float32)
    if len(out) > max_points:
        ids = rng.choice(len(out), size=max_points, replace=False)
        out = out[ids]
    return out


def point_distribution_distance(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    q = np.asarray([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    ref_q = np.quantile(reference, q, axis=0)
    can_q = np.quantile(candidate, q, axis=0)
    scale = np.maximum(np.quantile(reference, 0.75, axis=0) - np.quantile(reference, 0.25, axis=0), 0.05)
    quantile_gap = float(np.mean(np.abs(can_q - ref_q) / scale[None, :]))

    ref_cov = np.cov(reference.T)
    can_cov = np.cov(candidate.T)
    covariance_gap = float(
        np.linalg.norm(can_cov - ref_cov, ord="fro")
        / max(np.linalg.norm(ref_cov, ord="fro"), 1e-6)
    )
    return quantile_gap + 0.25 * covariance_gap, covariance_gap


def run_cross_domain_point_axis_search(
    mmfi: DomainSamples,
    syncwb: DomainSamples,
    max_points: int,
    seed: int,
) -> pd.DataFrame:
    reference = root_centred_point_sample(mmfi, max_points, seed)
    target = root_centred_point_sample(syncwb, max_points, seed + 1)
    rows = []
    for transform in signed_axis_transforms():
        transformed = transform.apply(target)
        score, covariance_gap = point_distribution_distance(reference, transformed)
        rows.append({
            "transform_applied_to_syncwb_current_root_centred_points": transform.name,
            "determinant": transform.determinant,
            "score": score,
            "covariance_gap": covariance_gap,
        })
    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


# -----------------------------------------------------------------------------
# Feature / normalisation diagnostics
# -----------------------------------------------------------------------------


def frame_geometry_rows(domain: DomainSamples, max_points_per_frame: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for frame_id, points, pose in zip(domain.frame_ids, domain.point_frames, domain.poses):
        p = np.asarray(points, dtype=np.float64)
        centroid = robust_cloud_centroid(p)
        root = pose[0]
        offset = centroid - root
        sample = p
        if len(sample) > max_points_per_frame:
            sample = sample[rng.choice(len(sample), size=max_points_per_frame, replace=False)]
        distances = point_to_segments_distance(sample, pose)
        bone_lengths = np.asarray([
            np.linalg.norm(pose[b] - pose[a]) for a, b in MMFI17_EDGES
        ])
        rows.append({
            "domain": domain.name,
            "frame_id": str(frame_id),
            "point_count": int(len(p)),
            "cloud_centroid_x_m": float(centroid[0]),
            "cloud_centroid_y_m": float(centroid[1]),
            "cloud_centroid_z_m": float(centroid[2]),
            "pelvis_x_m": float(root[0]),
            "pelvis_y_m": float(root[1]),
            "pelvis_z_m": float(root[2]),
            "centroid_minus_pelvis_x_m": float(offset[0]),
            "centroid_minus_pelvis_y_m": float(offset[1]),
            "centroid_minus_pelvis_z_m": float(offset[2]),
            "point_to_skeleton_p25_m": float(np.quantile(distances, 0.25)) if len(distances) else np.nan,
            "point_to_skeleton_p50_m": float(np.quantile(distances, 0.50)) if len(distances) else np.nan,
            "pelvis_to_neck_m": float(np.linalg.norm(pose[8] - pose[0])),
            "median_bone_length_m": float(np.median(bone_lengths)),
            "return_signal_median": float(np.median(p[:, 4])),
            "doppler_abs_median_mps": float(np.median(np.abs(p[:, 3]))),
            "range_median_m": float(np.median(np.linalg.norm(p[:, :3], axis=1))),
        })
    return rows


def collect_feature_statistics(
    domain: DomainSamples,
    max_points: int,
    seed: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    points = reservoir_points(domain.point_frames, max_points, seed)
    rows: list[dict[str, object]] = []
    units = ("m", "m", "m", "m/s", domain.return_signal_name)
    for i, (feature, unit) in enumerate(zip(POINT_FEATURES, units)):
        rows.append(describe_values(
            points[:, i], domain=domain.name, group="point_raw", feature=feature, units=unit
        ))

    poses = np.asarray(domain.poses)
    root_relative = poses - poses[:, :1, :]
    for i, feature in enumerate(POSE_FEATURES):
        rows.append(describe_values(
            poses[..., i], domain=domain.name, group="pose_absolute", feature=feature, units="m"
        ))
        rows.append(describe_values(
            root_relative[..., i], domain=domain.name, group="pose_root_relative", feature=feature, units="m"
        ))
    return rows, points


def percentile_rank_per_frame(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return x
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=float)
    return 2.0 * ranks - 1.0


def current_model_features_for_window(frames: Sequence[np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the exact eight non-time features used by the current model.

    The training script concatenates all points from the centred radar window,
    then computes one shared XYZ centre and one shared signal median/IQR. The
    relative-time feature is intentionally omitted from the domain-gap table
    because it is deterministic and identical in both domains.
    """
    if not frames:
        return {}, np.empty((0, 5), dtype=np.float64)
    p = np.concatenate([np.asarray(frame, dtype=np.float64)[:, :5] for frame in frames], axis=0)
    xyz = p[:, :3]
    doppler = p[:, 3]
    signal = p[:, 4]
    centre = np.median(xyz, axis=0) if len(p) else np.zeros(3)
    signal_median = float(np.median(signal)) if len(p) else 0.0
    signal_iqr = (
        max(float(np.quantile(signal, 0.75) - np.quantile(signal, 0.25)), 1e-3)
        if len(p) else 1.0
    )

    return {
        "absolute_x": np.clip(xyz[:, 0] / 4.0, -2.0, 2.0),
        "absolute_y": np.clip(xyz[:, 1] / 4.0, -2.0, 2.0),
        "absolute_z": np.clip(xyz[:, 2] / 4.0, -2.0, 2.0),
        "centred_x": np.clip((xyz[:, 0] - centre[0]) / 2.0, -2.0, 2.0),
        "centred_y": np.clip((xyz[:, 1] - centre[1]) / 2.0, -2.0, 2.0),
        "centred_z": np.clip((xyz[:, 2] - centre[2]) / 2.0, -2.0, 2.0),
        "doppler": np.clip(doppler / 3.0, -2.0, 2.0),
        "signal_robust": np.clip((signal - signal_median) / signal_iqr, -5.0, 5.0) / 5.0,
        "signal_rank": percentile_rank_per_frame(signal),
    }, p


def collect_model_feature_values(
    domain: DomainSamples,
    max_points: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    per_window = max(1, int(math.ceil(max_points / len(domain.point_windows))))
    chunks: dict[str, list[np.ndarray]] = {}
    saturation_counts = {
        "absolute_xyz": [0, 0],
        "centred_xyz": [0, 0],
        "doppler": [0, 0],
        "signal_robust": [0, 0],
    }

    for frames in domain.point_windows:
        features, p = current_model_features_for_window(frames)
        if len(p) == 0:
            continue
        if len(p) > per_window:
            ids = rng.choice(len(p), size=per_window, replace=False)
        else:
            ids = np.arange(len(p))
        for name, values in features.items():
            chunks.setdefault(name, []).append(np.asarray(values)[ids])

        saturation_counts["absolute_xyz"][0] += int((np.abs(p[:, :3] / 4.0) >= 2.0).sum())
        saturation_counts["absolute_xyz"][1] += int(p[:, :3].size)
        centre = np.median(p[:, :3], axis=0)
        saturation_counts["centred_xyz"][0] += int(
            (np.abs((p[:, :3] - centre) / 2.0) >= 2.0).sum()
        )
        saturation_counts["centred_xyz"][1] += int(p[:, :3].size)
        saturation_counts["doppler"][0] += int((np.abs(p[:, 3] / 3.0) >= 2.0).sum())
        saturation_counts["doppler"][1] += int(len(p))
        signal = p[:, 4]
        med = np.median(signal)
        iqr = max(float(np.quantile(signal, 0.75) - np.quantile(signal, 0.25)), 1e-3)
        saturation_counts["signal_robust"][0] += int(
            (np.abs((signal - med) / iqr) >= 5.0).sum()
        )
        saturation_counts["signal_robust"][1] += int(len(p))

    values = {name: np.concatenate(parts) for name, parts in chunks.items() if parts}
    for name in list(values):
        if len(values[name]) > max_points:
            ids = rng.choice(len(values[name]), size=max_points, replace=False)
            values[name] = values[name][ids]

    saturation = {
        name: (float(hit / total) if total else np.nan)
        for name, (hit, total) in saturation_counts.items()
    }
    return values, saturation


def quantile_gap(reference: np.ndarray, target: np.ndarray) -> float:
    ref = finite_flat(reference)
    tar = finite_flat(target)
    if len(ref) == 0 or len(tar) == 0:
        return float("nan")
    q = np.asarray([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    ref_q = np.quantile(ref, q)
    tar_q = np.quantile(tar, q)
    scale = max(float(np.quantile(ref, 0.75) - np.quantile(ref, 0.25)), 0.05)
    return float(np.mean(np.abs(tar_q - ref_q)) / scale)


def collect_model_feature_tables(
    mmfi: DomainSamples,
    syncwb: DomainSamples,
    max_points: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    mmfi_values, mmfi_sat = collect_model_feature_values(mmfi, max_points, seed)
    sync_values, sync_sat = collect_model_feature_values(syncwb, max_points, seed + 1)

    stats: list[dict[str, object]] = []
    for domain_name, values in ((mmfi.name, mmfi_values), (syncwb.name, sync_values)):
        for feature, array in values.items():
            stats.append(describe_values(
                array,
                domain=domain_name,
                group="model_input_after_current_normalisation",
                feature=feature,
                units="normalised",
            ))

    gaps = []
    for feature in sorted(set(mmfi_values) & set(sync_values)):
        gaps.append({
            "feature": feature,
            "current_normalisation_quantile_gap_iqr_units": quantile_gap(
                mmfi_values[feature], sync_values[feature]
            ),
        })
    return pd.DataFrame(stats), pd.DataFrame(gaps), {
        mmfi.name: mmfi_sat,
        syncwb.name: sync_sat,
    }, mmfi_values, sync_values


def signal_relationship_rows(
    domain: DomainSamples,
    point_sample: np.ndarray,
    frame_df: pd.DataFrame,
) -> list[dict[str, object]]:
    p = np.asarray(point_sample, dtype=np.float64)
    if len(p) == 0:
        return []
    signal = p[:, 4]
    variables = {
        "range": np.linalg.norm(p[:, :3], axis=1),
        "abs_doppler": np.abs(p[:, 3]),
        "x": p[:, 0],
        "y": p[:, 1],
        "z": p[:, 2],
    }
    rows: list[dict[str, object]] = []
    for name, values in variables.items():
        rows.append({
            "domain": domain.name,
            "level": "point",
            "signal_name": domain.return_signal_name,
            "related_variable": name,
            "pearson": paired_corr(signal, values),
            "spearman": spearman_corr(signal, values),
            "samples": int(len(signal)),
        })

    subset = frame_df[frame_df["domain"] == domain.name]
    for name in ("point_count", "range_median_m", "doppler_abs_median_mps"):
        rows.append({
            "domain": domain.name,
            "level": "frame",
            "signal_name": f"median {domain.return_signal_name}",
            "related_variable": name,
            "pearson": paired_corr(subset["return_signal_median"], subset[name]),
            "spearman": spearman_corr(subset["return_signal_median"], subset[name]),
            "samples": int(len(subset)),
        })
    return rows



# -----------------------------------------------------------------------------
# Return-strength semantics: quantisation, body proximity and normalisation
# -----------------------------------------------------------------------------


def binary_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC without scipy/sklearn.

    Higher ``scores`` are assumed to predict ``positive=True``. Ties receive
    average ranks. Returns NaN when either class is absent.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(positive, dtype=bool).reshape(-1)
    valid = np.isfinite(s)
    s, y = s[valid], y[valid]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_pos = float(ranks[y].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(finite_flat(a))
    y = np.sort(finite_flat(b))
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def wasserstein_1d(a: np.ndarray, b: np.ndarray, quantiles: int = 2001) -> float:
    x = finite_flat(a)
    y = finite_flat(b)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, quantiles)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def partial_spearman_controlling_range(
    signal: np.ndarray,
    proximity: np.ndarray,
    range_m: np.ndarray,
) -> float:
    """Approximate partial Spearman correlation controlling nonlinear range.

    Ranks of signal and proximity are each residualised against a quadratic in
    ranked range, then Pearson correlation is computed between residuals.
    """
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    p = np.asarray(proximity, dtype=np.float64).reshape(-1)
    r = np.asarray(range_m, dtype=np.float64).reshape(-1)
    valid = np.isfinite(s) & np.isfinite(p) & np.isfinite(r)
    s, p, r = s[valid], p[valid], r[valid]
    if len(s) < 20:
        return float("nan")
    sr = rankdata(s)
    pr = rankdata(p)
    rr = rankdata(r)
    rr = (rr - rr.mean()) / max(rr.std(), 1e-12)
    design = np.column_stack([np.ones(len(rr)), rr, rr * rr])
    s_res = sr - design @ np.linalg.lstsq(design, sr, rcond=None)[0]
    p_res = pr - design @ np.linalg.lstsq(design, pr, rcond=None)[0]
    return paired_corr(s_res, p_res)


def channel_quantisation_rows(domain: DomainSamples, point_sample: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, feature, units in (
        (3, "doppler", "m/s"),
        (4, "signal", domain.return_signal_name),
    ):
        values = finite_flat(np.asarray(point_sample)[:, index])
        if len(values) == 0:
            continue
        rounded = np.round(values, 6)
        unique, counts = np.unique(rounded, return_counts=True)
        steps = np.diff(unique)
        steps = steps[steps > 1e-9]
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        rows.append({
            "domain": domain.name,
            "channel": feature,
            "semantic_name": "Doppler" if feature == "doppler" else domain.return_signal_name,
            "units": units,
            "samples": int(len(values)),
            "unique_values_rounded_1e_6": int(len(unique)),
            "unique_fraction": float(len(unique) / len(values)),
            "zero_fraction": float(np.mean(np.isclose(values, 0.0, atol=1e-8))),
            "negative_fraction": float(np.mean(values < 0.0)),
            "positive_fraction": float(np.mean(values > 0.0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "minimum_positive_unique_step": float(np.min(steps)) if len(steps) else np.nan,
            "median_positive_unique_step": float(np.median(steps)) if len(steps) else np.nan,
            "entropy_bits": entropy,
            "effective_levels_2_to_entropy": float(2.0 ** entropy),
        })
    return rows


def collect_signal_proximity_samples(
    domain: DomainSamples,
    max_points_per_frame: int,
    seed: int,
    progress: bool,
) -> pd.DataFrame:
    """Sample radar points with signal, range and point-to-skeleton distance."""
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    iterator = tqdm(
        zip(domain.frame_ids, domain.point_frames, domain.poses),
        total=len(domain.point_frames),
        desc=f"Signal/body proximity: {domain.name}",
        disable=not progress,
        leave=False,
    )
    for frame_id, points, pose in iterator:
        p = np.asarray(points, dtype=np.float64)
        if len(p) == 0:
            continue
        if max_points_per_frame > 0 and len(p) > max_points_per_frame:
            p = p[rng.choice(len(p), size=max_points_per_frame, replace=False)]
        distance = point_to_segments_distance(p, pose)
        signal = p[:, 4]
        frame_rank = pd.Series(signal).rank(method="average", pct=True).to_numpy(dtype=float)
        rows.append(pd.DataFrame({
            "domain": domain.name,
            "frame_id": str(frame_id),
            "signal": signal,
            "signal_frame_percentile": frame_rank,
            "doppler": p[:, 3],
            "range_m": np.linalg.norm(p[:, :3], axis=1),
            "skeleton_distance_m": distance,
        }))
    if not rows:
        return pd.DataFrame(columns=[
            "domain", "frame_id", "signal", "signal_frame_percentile",
            "doppler", "range_m", "skeleton_distance_m",
        ])
    return pd.concat(rows, ignore_index=True)


def signal_proximity_summary_rows(
    samples: pd.DataFrame,
    signal_name: str,
    near_threshold_m: float,
    clutter_threshold_m: float,
) -> list[dict[str, object]]:
    if samples.empty:
        return []
    signal = samples["signal"].to_numpy(dtype=float)
    rank_signal = samples["signal_frame_percentile"].to_numpy(dtype=float)
    distance = samples["skeleton_distance_m"].to_numpy(dtype=float)
    range_m = samples["range_m"].to_numpy(dtype=float)
    class_mask = (distance <= near_threshold_m) | (distance > clutter_threshold_m)
    near = distance[class_mask] <= near_threshold_m
    rows = []
    for representation, values in (
        ("raw", signal),
        ("within_frame_percentile", rank_signal),
    ):
        rows.append({
            "domain": str(samples["domain"].iloc[0]),
            "signal_name": signal_name,
            "representation": representation,
            "points": int(len(samples)),
            "near_body_threshold_m": float(near_threshold_m),
            "clutter_threshold_m": float(clutter_threshold_m),
            "near_body_points": int(np.sum(distance <= near_threshold_m)),
            "clutter_points": int(np.sum(distance > clutter_threshold_m)),
            "spearman_signal_vs_negative_distance": spearman_corr(values, -distance),
            "partial_spearman_controlling_range": partial_spearman_controlling_range(
                values, -distance, range_m
            ),
            "near_vs_clutter_auc": binary_auc(values[class_mask], near),
            "median_distance_bottom_quartile_m": float(
                np.median(distance[values <= np.quantile(values, 0.25)])
            ),
            "median_distance_top_quartile_m": float(
                np.median(distance[values >= np.quantile(values, 0.75)])
            ),
            "near_fraction_bottom_quartile": float(
                np.mean(distance[values <= np.quantile(values, 0.25)] <= near_threshold_m)
            ),
            "near_fraction_top_quartile": float(
                np.mean(distance[values >= np.quantile(values, 0.75)] <= near_threshold_m)
            ),
        })
    return rows


def signal_proximity_decile_rows(
    samples: pd.DataFrame,
    near_threshold_m: float,
) -> list[dict[str, object]]:
    if samples.empty:
        return []
    rows: list[dict[str, object]] = []
    percentiles = samples["signal_frame_percentile"].to_numpy(dtype=float)
    distances = samples["skeleton_distance_m"].to_numpy(dtype=float)
    for decile in range(10):
        low = decile / 10.0
        high = (decile + 1) / 10.0
        mask = (percentiles > low) & (percentiles <= high)
        if decile == 0:
            mask |= percentiles == 0.0
        if not np.any(mask):
            continue
        d = distances[mask]
        rows.append({
            "domain": str(samples["domain"].iloc[0]),
            "strength_decile": decile + 1,
            "percentile_low_exclusive": low,
            "percentile_high_inclusive": high,
            "points": int(len(d)),
            "median_skeleton_distance_m": float(np.median(d)),
            "p25_skeleton_distance_m": float(np.quantile(d, 0.25)),
            "p75_skeleton_distance_m": float(np.quantile(d, 0.75)),
            "fraction_within_10cm": float(np.mean(d <= 0.10)),
            "fraction_within_20cm": float(np.mean(d <= near_threshold_m)),
            "fraction_within_30cm": float(np.mean(d <= 0.30)),
        })
    return rows


def signal_strength_group_rows(
    samples: pd.DataFrame,
    near_threshold_m: float,
) -> list[dict[str, object]]:
    if samples.empty:
        return []
    rank = samples["signal_frame_percentile"].to_numpy(dtype=float)
    distance = samples["skeleton_distance_m"].to_numpy(dtype=float)
    groups = (
        ("bottom_25_percent", rank <= 0.25),
        ("middle_50_percent", (rank > 0.25) & (rank < 0.75)),
        ("top_25_percent", rank >= 0.75),
    )
    rows = []
    for name, mask in groups:
        d = distance[mask]
        if len(d) == 0:
            continue
        rows.append({
            "domain": str(samples["domain"].iloc[0]),
            "strength_group": name,
            "points": int(len(d)),
            "median_skeleton_distance_m": float(np.median(d)),
            "fraction_within_10cm": float(np.mean(d <= 0.10)),
            "fraction_within_20cm": float(np.mean(d <= near_threshold_m)),
            "fraction_within_30cm": float(np.mean(d <= 0.30)),
        })
    return rows


def signal_range_bin_rows(
    samples: pd.DataFrame,
    range_edges: Sequence[float],
    near_threshold_m: float,
    clutter_threshold_m: float,
) -> list[dict[str, object]]:
    if samples.empty:
        return []
    rows = []
    edges = list(map(float, range_edges))
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("signal range edges must be strictly increasing")
    for low, high in zip(edges[:-1], edges[1:]):
        subset = samples[(samples["range_m"] >= low) & (samples["range_m"] < high)]
        if len(subset) < 50:
            continue
        distance = subset["skeleton_distance_m"].to_numpy(dtype=float)
        class_mask = (distance <= near_threshold_m) | (distance > clutter_threshold_m)
        near = distance[class_mask] <= near_threshold_m
        for representation, column in (
            ("raw", "signal"),
            ("within_frame_percentile", "signal_frame_percentile"),
        ):
            values = subset[column].to_numpy(dtype=float)
            rows.append({
                "domain": str(samples["domain"].iloc[0]),
                "representation": representation,
                "range_low_m": low,
                "range_high_m": high,
                "points": int(len(subset)),
                "classified_points": int(class_mask.sum()),
                "spearman_signal_vs_negative_distance": spearman_corr(values, -distance),
                "near_vs_clutter_auc": binary_auc(values[class_mask], near),
            })
    return rows


def signal_candidate_values(samples: pd.DataFrame) -> dict[str, np.ndarray]:
    signal = samples["signal"].to_numpy(dtype=np.float64)
    robust_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    # Preserve row order by writing into arrays indexed by group row positions.
    robust = np.empty(len(samples), dtype=np.float64)
    rank = np.empty(len(samples), dtype=np.float64)
    for _, indices in samples.groupby("frame_id", sort=False).groups.items():
        ids = np.asarray(list(indices), dtype=np.int64)
        values = signal[ids]
        median = float(np.median(values))
        iqr = max(float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 1e-6)
        robust[ids] = np.clip((values - median) / iqr, -5.0, 5.0) / 5.0
        rank[ids] = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)
    positive = np.maximum(signal, 0.0)
    epsilon = max(float(np.quantile(positive[positive > 0], 0.01)) * 0.1, 1e-6) if np.any(positive > 0) else 1e-6
    return {
        "raw": signal,
        "log1p": np.log1p(positive),
        "db10": 10.0 * np.log10(positive + epsilon),
        "window_robust": robust,
        "window_percentile": rank,
    }


def range_matched_indices(
    mmfi_samples: pd.DataFrame,
    sync_samples: pd.DataFrame,
    range_edges: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mmfi_ids: list[np.ndarray] = []
    sync_ids: list[np.ndarray] = []
    edges = list(map(float, range_edges))
    for low, high in zip(edges[:-1], edges[1:]):
        a = np.flatnonzero(
            (mmfi_samples["range_m"].to_numpy() >= low)
            & (mmfi_samples["range_m"].to_numpy() < high)
        )
        b = np.flatnonzero(
            (sync_samples["range_m"].to_numpy() >= low)
            & (sync_samples["range_m"].to_numpy() < high)
        )
        n = min(len(a), len(b))
        if n == 0:
            continue
        mmfi_ids.append(rng.choice(a, size=n, replace=False))
        sync_ids.append(rng.choice(b, size=n, replace=False))
    if not mmfi_ids:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(mmfi_ids), np.concatenate(sync_ids)


def signal_normalisation_comparison_rows(
    mmfi_samples: pd.DataFrame,
    sync_samples: pd.DataFrame,
    range_edges: Sequence[float],
    seed: int,
) -> list[dict[str, object]]:
    mmfi_values = signal_candidate_values(mmfi_samples)
    sync_values = signal_candidate_values(sync_samples)
    mmfi_match, sync_match = range_matched_indices(
        mmfi_samples, sync_samples, range_edges, seed
    )
    rows = []
    for name in sorted(set(mmfi_values) & set(sync_values)):
        a = mmfi_values[name]
        b = sync_values[name]
        domain_labels = np.concatenate([
            np.zeros(len(a), dtype=bool), np.ones(len(b), dtype=bool)
        ])
        auc = binary_auc(np.concatenate([a, b]), domain_labels)
        auc_separability = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
        if len(mmfi_match) and len(sync_match):
            am = a[mmfi_match]
            bm = b[sync_match]
            labels_m = np.concatenate([
                np.zeros(len(am), dtype=bool), np.ones(len(bm), dtype=bool)
            ])
            auc_m = binary_auc(np.concatenate([am, bm]), labels_m)
            auc_m_sep = max(auc_m, 1.0 - auc_m) if np.isfinite(auc_m) else np.nan
        else:
            am = bm = np.empty(0)
            auc_m_sep = np.nan
        rows.append({
            "normalisation": name,
            "mmfi_points": int(len(a)),
            "syncwb_points": int(len(b)),
            "quantile_gap_mmfi_iqr_units": quantile_gap(a, b),
            "ks_statistic": ks_statistic(a, b),
            "wasserstein_distance": wasserstein_1d(a, b),
            "marginal_domain_auc_separability": auc_separability,
            "range_matched_points_per_domain": int(min(len(am), len(bm))),
            "range_matched_domain_auc_separability": auc_m_sep,
            "range_matched_ks_statistic": ks_statistic(am, bm) if len(am) else np.nan,
        })
    return rows


# -----------------------------------------------------------------------------
# Continuous frame-reference, support-overlap and per-joint diagnostics
# -----------------------------------------------------------------------------


def gui_sensor_to_world_row_rotation(pitch_down_deg: float) -> np.ndarray:
    """Return the row-vector rotation used by the SyncWB GUI.

    The GUI computes ``out = xyz @ rot.T`` where rot is R_x(-pitch).
    This function therefore returns ``rot.T`` for direct row-vector use.
    """
    theta = math.radians(float(pitch_down_deg))
    c, ss = math.cos(theta), math.sin(theta)
    rot = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, c, ss],
        [0.0, -ss, c],
    ], dtype=np.float64)
    return rot.T


def fit_template_kabsch_row_rotation(
    source_poses: np.ndarray,
    target_poses: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit one proper row-vector rotation between median body-15 templates."""
    source = median_pose_template(source_poses)[BODY15_INDICES]
    target = median_pose_template(target_poses)[BODY15_INDICES]
    source = source - source.mean(axis=0)
    target = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    before = float(np.sqrt(np.mean(np.sum((source - target) ** 2, axis=1))))
    after = float(np.sqrt(np.mean(np.sum((source @ rotation - target) ** 2, axis=1))))
    return rotation, before, after


def row_rotation_to_euler_xyz_deg(row_rotation: np.ndarray) -> np.ndarray:
    """Return approximate XYZ Euler angles for the equivalent column rotation."""
    r = np.asarray(row_rotation, dtype=np.float64).T
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def transform_domain_geometry(
    domain: DomainSamples,
    row_rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    rotation = np.asarray(row_rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    points = []
    for frame in domain.point_frames:
        out = np.asarray(frame, dtype=np.float64).copy()
        out[:, :3] = out[:, :3] @ rotation + translation
        points.append(out)
    poses = np.asarray(domain.poses, dtype=np.float64) @ rotation + translation
    return points, poses


def xyz_quantile_gap(reference: np.ndarray, target: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    tar = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    ref = ref[np.all(np.isfinite(ref), axis=1)]
    tar = tar[np.all(np.isfinite(tar), axis=1)]
    if len(ref) == 0 or len(tar) == 0:
        return float("nan")
    q = np.asarray([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    ref_q = np.quantile(ref, q, axis=0)
    tar_q = np.quantile(tar, q, axis=0)
    scale = np.maximum(
        np.quantile(ref, 0.75, axis=0) - np.quantile(ref, 0.25, axis=0),
        0.05,
    )
    return float(np.mean(np.abs(tar_q - ref_q) / scale[None, :]))


def root_centred_points_from_geometry(
    point_frames: Sequence[np.ndarray],
    poses: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_frame = max(1, int(math.ceil(max_points / len(point_frames))))
    chunks: list[np.ndarray] = []
    for points, pose in zip(point_frames, poses):
        p = np.asarray(points, dtype=np.float64)
        if len(p) > per_frame:
            p = p[rng.choice(len(p), size=per_frame, replace=False)]
        chunks.append(p[:, :3] - pose[0, :3])
    out = np.concatenate(chunks, axis=0)
    if len(out) > max_points:
        out = out[rng.choice(len(out), size=max_points, replace=False)]
    return out


def reference_frame_hypotheses(
    mmfi: DomainSamples,
    syncwb: DomainSamples,
    max_points: int,
    seed: int,
    pitch_down_deg: float,
    sensor_height_m: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    identity = np.eye(3, dtype=np.float64)
    gui_rotation = gui_sensor_to_world_row_rotation(pitch_down_deg)
    kabsch_rotation, kabsch_before, kabsch_after = fit_template_kabsch_row_rotation(
        syncwb.poses, mmfi.poses
    )
    kabsch_translation = (
        np.median(mmfi.poses[:, 0, :], axis=0)
        - np.median(syncwb.poses[:, 0, :] @ kabsch_rotation, axis=0)
    )

    hypotheses = [
        ("syncwb_sensor_current", identity, np.zeros(3)),
        ("syncwb_gui_world_rotation_only", gui_rotation, np.zeros(3)),
        (
            "syncwb_gui_world_full_height",
            gui_rotation,
            np.asarray([0.0, 0.0, sensor_height_m]),
        ),
        ("syncwb_best_pose_template_rotation_only", kabsch_rotation, np.zeros(3)),
        (
            "syncwb_best_pose_rotation_plus_distribution_translation",
            kabsch_rotation,
            kabsch_translation,
        ),
    ]

    mmfi_points = reservoir_points(mmfi.point_frames, max_points, seed)
    mmfi_root_points = root_centred_point_sample(mmfi, max_points, seed)
    rows: list[dict[str, object]] = []
    for name, rotation, translation in hypotheses:
        point_frames, poses = transform_domain_geometry(syncwb, rotation, translation)
        sync_points = reservoir_points(point_frames, max_points, seed + 1)
        sync_root_points = root_centred_points_from_geometry(
            point_frames, poses, max_points, seed + 1
        )
        pose_score, pose_median_gap, pose_iqr_gap = pose_distribution_distance(
            mmfi.poses, poses
        )
        point_score, point_cov_gap = point_distribution_distance(
            mmfi_root_points, sync_root_points
        )
        euler = row_rotation_to_euler_xyz_deg(rotation)
        rows.append({
            "hypothesis": name,
            "pose_root_relative_score": pose_score,
            "pose_median_template_gap": pose_median_gap,
            "pose_iqr_gap": pose_iqr_gap,
            "point_root_centred_score": point_score,
            "point_root_centred_covariance_gap": point_cov_gap,
            "pose_absolute_xyz_gap": xyz_quantile_gap(mmfi.poses, poses),
            "point_absolute_xyz_gap": xyz_quantile_gap(
                mmfi_points[:, :3], sync_points[:, :3]
            ),
            "rotation_euler_x_deg": float(euler[0]),
            "rotation_euler_y_deg": float(euler[1]),
            "rotation_euler_z_deg": float(euler[2]),
            "translation_x_m": float(translation[0]),
            "translation_y_m": float(translation[1]),
            "translation_z_m": float(translation[2]),
        })

    df = pd.DataFrame(rows)
    df["orientation_composite_score"] = (
        df["pose_root_relative_score"] + df["point_root_centred_score"]
    )
    df = df.sort_values("orientation_composite_score").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    details = {
        "kabsch_row_rotation": kabsch_rotation,
        "kabsch_euler_xyz_deg": row_rotation_to_euler_xyz_deg(kabsch_rotation),
        "kabsch_template_rmse_before": kabsch_before,
        "kabsch_template_rmse_after": kabsch_after,
        "distribution_translation_m": kabsch_translation,
        "gui_pitch_down_deg": pitch_down_deg,
        "gui_sensor_height_m": sensor_height_m,
    }
    return df, details


def support_overlap_rows(
    mmfi: DomainSamples,
    syncwb: DomainSamples,
    mmfi_points: np.ndarray,
    sync_points: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(group: str, feature: str, source_values: np.ndarray, target_values: np.ndarray) -> None:
        src = finite_flat(source_values)
        tar = finite_flat(target_values)
        if len(src) == 0 or len(tar) == 0:
            return
        low, high = np.quantile(src, [0.001, 0.999])
        rows.append({
            "group": group,
            "feature": feature,
            "mmfi_p0_1": float(low),
            "mmfi_p99_9": float(high),
            "syncwb_below_mmfi_support_fraction": float(np.mean(tar < low)),
            "syncwb_above_mmfi_support_fraction": float(np.mean(tar > high)),
            "syncwb_within_mmfi_support_fraction": float(np.mean((tar >= low) & (tar <= high))),
        })

    for i, feature in enumerate(POINT_FEATURES):
        add("point_raw", feature, mmfi_points[:, i], sync_points[:, i])

    mmfi_rel = mmfi.poses - mmfi.poses[:, :1]
    sync_rel = syncwb.poses - syncwb.poses[:, :1]
    for axis, feature in enumerate(POSE_FEATURES):
        add("pose_absolute", feature, mmfi.poses[..., axis], syncwb.poses[..., axis])
        add("pose_root_relative", feature, mmfi_rel[..., axis], sync_rel[..., axis])
    return rows


def pose_joint_statistics(domain: DomainSamples) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    absolute = np.asarray(domain.poses)
    relative = absolute - absolute[:, :1]
    for joint_index, joint_name in enumerate(MMFI17_NAMES):
        for axis, axis_name in enumerate(POSE_FEATURES):
            for group, values in (
                ("absolute", absolute[:, joint_index, axis]),
                ("root_relative", relative[:, joint_index, axis]),
            ):
                row = describe_values(
                    values,
                    domain=domain.name,
                    group=f"pose_joint_{group}",
                    feature=f"{joint_name}.{axis_name}",
                    units="m",
                )
                row["joint_index"] = joint_index
                row["joint_name"] = joint_name
                row["coordinate"] = axis_name
                rows.append(row)
    return rows


def checkpoint_pose_bound_report(
    checkpoint_path: str | None,
    syncwb_poses: np.ndarray,
) -> dict[str, object] | None:
    if not checkpoint_path:
        return None

    checkpoint_path_obj = Path(checkpoint_path)
    bounds: dict[str, object] | None = None
    source = None
    warning = None

    manifest_path = checkpoint_path_obj.parent / "split_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate = manifest.get("pose_bounds")
            if isinstance(candidate, dict) and "low" in candidate and "high" in candidate:
                bounds = candidate
                source = str(manifest_path)
        except Exception as exc:  # diagnostic must remain nonfatal
            warning = f"Could not read {manifest_path}: {exc}"

    if bounds is None:
        try:
            import torch
            checkpoint = torch.load(checkpoint_path_obj, map_location="cpu", weights_only=False)
            candidate = checkpoint.get("pose_bounds")
            if isinstance(candidate, dict) and "low" in candidate and "high" in candidate:
                bounds = candidate
                source = str(checkpoint_path_obj)
        except Exception as exc:  # includes Windows native-DLL import failures
            warning = f"Checkpoint pose-bound check unavailable: {type(exc).__name__}: {exc}"

    if bounds is None:
        return {
            "available": False,
            "checkpoint": str(checkpoint_path_obj),
            "warning": warning or "No pose_bounds.low/high found",
        }

    low = np.asarray(bounds["low"], dtype=np.float64)
    high = np.asarray(bounds["high"], dtype=np.float64)
    poses = np.asarray(syncwb_poses, dtype=np.float64)
    below = poses < low[None, None, :]
    above = poses > high[None, None, :]
    per_axis = []
    for axis, name in enumerate(POSE_FEATURES):
        per_axis.append({
            "axis": name,
            "low": float(low[axis]),
            "high": float(high[axis]),
            "below_fraction": float(np.mean(below[..., axis])),
            "above_fraction": float(np.mean(above[..., axis])),
            "outside_fraction": float(np.mean(below[..., axis] | above[..., axis])),
        })
    return {
        "available": True,
        "checkpoint": str(checkpoint_path_obj),
        "bounds_source": source,
        "warning": warning,
        "bounds": {"low": low, "high": high},
        "per_axis": per_axis,
        "any_coordinate_outside_fraction": float(np.mean(np.any(below | above, axis=-1))),
        "frames_with_any_joint_outside_fraction": float(
            np.mean(np.any(below | above, axis=(1, 2)))
        ),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_feature_quantile_plot(feature_stats: pd.DataFrame, path: Path) -> None:
    if plt is None:
        return
    points = feature_stats[feature_stats["group"] == "point_raw"]
    features = list(POINT_FEATURES)
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 14), constrained_layout=True)
    for ax, feature in zip(axes, features):
        subset = points[points["feature"] == feature]
        for _, row in subset.iterrows():
            x = [row["p1"], row["p25"], row["p50"], row["p75"], row["p99"]]
            ax.plot(x, [row["domain"]] * 5, marker="o", label=row["domain"])
        ax.set_title(f"Point feature: {feature}")
        ax.set_xlabel(str(subset["units"].iloc[0]) if len(subset) else "")
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_axis_search_plot(mmfi_axis: pd.DataFrame, sync_axis: pd.DataFrame, path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for ax, df, title in (
        (axes[0], mmfi_axis.head(12), "MM-Fi pose -> radar axis candidates"),
        (axes[1], sync_axis.head(12), "SyncWB raw Kinect pose -> radar axis candidates"),
    ):
        ax.barh(df["transform"].iloc[::-1], df["composite_score"].iloc[::-1])
        ax.set_title(title)
        ax.set_xlabel("Composite diagnostic score (lower is better)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def median_pose_template(poses: np.ndarray) -> np.ndarray:
    normalised = normalised_pose_shape(poses)
    return np.nanmedian(normalised, axis=0)


def save_pose_template_plot(mmfi: DomainSamples, syncwb: DomainSamples, path: Path) -> None:
    if plt is None:
        return
    mmfi_template = median_pose_template(mmfi.poses)
    sync_template = median_pose_template(syncwb.poses)
    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    for position, (template, title) in enumerate((
        (mmfi_template, "MM-Fi median root-relative pose"),
        (sync_template, "SyncWB median root-relative pose"),
    ), start=1):
        ax = fig.add_subplot(1, 2, position, projection="3d")
        ax.scatter(template[:, 0], template[:, 1], template[:, 2], s=18)
        for a, b in MMFI17_EDGES:
            ax.plot(
                [template[a, 0], template[b, 0]],
                [template[a, 1], template[b, 1]],
                [template[a, 2], template[b, 2]],
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_signal_plot(
    mmfi_values: dict[str, np.ndarray],
    sync_values: dict[str, np.ndarray],
    path: Path,
) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key, title in (
        (axes[0], "signal_robust", "Current window-level robust signal"),
        (axes[1], "signal_rank", "Per-frame percentile-rank signal"),
    ):
        for name, values in (("MM-Fi", mmfi_values[key]), ("SyncWB", sync_values[key])):
            ax.hist(values, bins=80, density=True, histtype="step", label=name)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)




def save_signal_body_proximity_plot(deciles: pd.DataFrame, path: Path) -> None:
    if plt is None or deciles.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for domain, subset in deciles.groupby("domain"):
        subset = subset.sort_values("strength_decile")
        axes[0].plot(
            subset["strength_decile"], subset["median_skeleton_distance_m"],
            marker="o", label=domain,
        )
        axes[1].plot(
            subset["strength_decile"], subset["fraction_within_20cm"],
            marker="o", label=domain,
        )
    axes[0].set_title("Body distance by within-frame strength decile")
    axes[0].set_xlabel("Strength decile (1=lowest, 10=highest)")
    axes[0].set_ylabel("Median point-to-skeleton distance (m)")
    axes[1].set_title("Near-body fraction by strength decile")
    axes[1].set_xlabel("Strength decile (1=lowest, 10=highest)")
    axes[1].set_ylabel("Fraction within 20 cm")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_signal_range_auc_plot(range_df: pd.DataFrame, path: Path) -> None:
    if plt is None or range_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for (domain, representation), subset in range_df.groupby(["domain", "representation"]):
        subset = subset.sort_values("range_low_m")
        centre = 0.5 * (subset["range_low_m"] + subset["range_high_m"])
        ax.plot(
            centre, subset["near_vs_clutter_auc"], marker="o",
            label=f"{domain} — {representation}",
        )
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_title("Return-strength near-body discrimination within range bins")
    ax.set_xlabel("Range-bin centre (m)")
    ax.set_ylabel("Near-body vs clutter AUROC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_signal_normalisation_candidates_plot(
    comparison_df: pd.DataFrame,
    path: Path,
) -> None:
    if plt is None or comparison_df.empty:
        return
    ordered = comparison_df.sort_values("range_matched_domain_auc_separability")
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(ordered))
    width = 0.38
    ax.bar(
        x - width / 2,
        ordered["marginal_domain_auc_separability"],
        width,
        label="Marginal",
    )
    ax.bar(
        x + width / 2,
        ordered["range_matched_domain_auc_separability"],
        width,
        label="Range matched",
    )
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(x, ordered["normalisation"], rotation=20)
    ax.set_ylim(0.45, 1.01)
    ax.set_ylabel("Domain-classification AUC separability")
    ax.set_title("How easily signal alone identifies the dataset")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Summary / report
# -----------------------------------------------------------------------------


def find_transform_row(df: pd.DataFrame, column: str, transform_name: str) -> dict[str, object]:
    matches = df[df[column] == transform_name]
    return matches.iloc[0].to_dict() if len(matches) else {}


def summarise_offset(frame_df: pd.DataFrame, domain: str) -> dict[str, object]:
    subset = frame_df[frame_df["domain"] == domain]
    cols = [
        "centroid_minus_pelvis_x_m",
        "centroid_minus_pelvis_y_m",
        "centroid_minus_pelvis_z_m",
    ]
    values = subset[cols].to_numpy(dtype=float)
    return {
        "median_m": np.nanmedian(values, axis=0),
        "iqr_m": np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0),
        "frames": int(len(values)),
    }


def build_summary_text(report: dict[str, object]) -> str:
    lines = [
        "MM-Fi -> SyncWB domain-shift diagnostics",
        "=" * 45,
        "",
        "Axis conventions",
        "----------------",
    ]
    axis = report["axis_conventions"]
    lines.append(
        f"MM-Fi best pose->radar candidate: {axis['mmfi_best']['transform']} "
        f"(current identity rank {axis['mmfi_current_identity']['rank']})."
    )
    lines.append(
        f"SyncWB best raw Kinect->radar candidate: {axis['syncwb_best']['transform']} "
        f"(current x,z,-y rank {axis['syncwb_current_x_z_minus_y']['rank']})."
    )
    lines.append(
        f"Cross-domain pose convention best residual transform: "
        f"{axis['cross_domain_pose_best']['transform_applied_to_syncwb_current_pose']} "
        f"(identity rank {axis['cross_domain_pose_identity']['rank']})."
    )
    lines.append(
        f"Cross-domain root-centred point convention best residual transform: "
        f"{axis['cross_domain_points_best']['transform_applied_to_syncwb_current_root_centred_points']} "
        f"(identity rank {axis['cross_domain_points_identity']['rank']})."
    )
    lines.extend([
        "",
        "Continuous frame-reference hypotheses",
        "-------------------------------------",
    ])
    hypotheses = report.get("reference_frame_hypotheses", [])
    for row in hypotheses:
        lines.append(
            f"{row['hypothesis']}: orientation score={row['orientation_composite_score']:.3f}, "
            f"pose-relative={row['pose_root_relative_score']:.3f}, "
            f"point-relative={row['point_root_centred_score']:.3f}."
        )
    details = report.get("reference_frame_details", {})
    if details:
        euler = details.get("kabsch_euler_xyz_deg", [float('nan')] * 3)
        lines.append(
            f"Best continuous pose-template rotation (SyncWB -> MM-Fi) has approximate "
            f"XYZ Euler angles [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] degrees."
        )

    lines.extend([
        "",
        "Frame-of-reference proxy",
        "------------------------",
    ])
    offsets = report["centroid_minus_pelvis_offset"]
    for domain in ("MM-Fi", "SyncWB"):
        med = offsets[domain]["median_m"]
        iqr = offsets[domain]["iqr_m"]
        lines.append(
            f"{domain}: median radar-centroid minus pelvis = "
            f"[{med[0]:.3f}, {med[1]:.3f}, {med[2]:.3f}] m; "
            f"IQR = [{iqr[0]:.3f}, {iqr[1]:.3f}, {iqr[2]:.3f}] m."
        )
    lines.extend([
        "",
        "Current model-input normalisation",
        "---------------------------------",
    ])
    gaps = report["model_feature_gaps"]
    for row in gaps[:12]:
        lines.append(
            f"{row['feature']}: cross-domain quantile gap = "
            f"{row['current_normalisation_quantile_gap_iqr_units']:.3f} MM-Fi IQR units."
        )
    lines.append("")
    support = report.get("support_overlap", [])
    worst_support = sorted(
        support,
        key=lambda row: row.get("syncwb_within_mmfi_support_fraction", 1.0),
    )[:8]
    if worst_support:
        lines.append("Lowest SyncWB coverage inside empirical MM-Fi p0.1-p99.9 support:")
        for row in worst_support:
            lines.append(
                f"  {row['group']}.{row['feature']}: "
                f"{100*row['syncwb_within_mmfi_support_fraction']:.2f}% inside."
            )
        lines.append("")

    checkpoint = report.get("checkpoint_pose_bounds")
    if checkpoint and checkpoint.get("available", True):
        lines.append(
            f"Checkpoint bounds: {100*checkpoint['frames_with_any_joint_outside_fraction']:.2f}% "
            f"of SyncWB frames have at least one joint coordinate outside the trained SimCC bounds."
        )
        lines.append("")
    elif checkpoint and checkpoint.get("warning"):
        lines.append(f"Checkpoint bounds unavailable: {checkpoint['warning']}")
        lines.append("")

    lines.append("Clipping/saturation rates:")
    for domain, values in report["model_feature_saturation"].items():
        lines.append(
            f"  {domain}: " + ", ".join(f"{k}={100*v:.3f}%" for k, v in values.items())
        )
    proximity = report.get("signal_proximity_summary", [])
    norm_rows = report.get("signal_normalisation_comparison", [])
    quant_rows = report.get("channel_quantisation", [])
    lines.extend(["", "Return-strength semantics", "-------------------------"])
    for row in proximity:
        if row.get("representation") == "within_frame_percentile":
            lines.append(
                f"{row['domain']} {row['signal_name']}: "
                f"near-vs-clutter AUC={row['near_vs_clutter_auc']:.3f}, "
                f"range-adjusted proximity Spearman={row['partial_spearman_controlling_range']:.3f}, "
                f"top-quartile near-body fraction={100*row['near_fraction_top_quartile']:.2f}%, "
                f"bottom-quartile={100*row['near_fraction_bottom_quartile']:.2f}%."
            )
    if norm_rows:
        best_norm = min(
            norm_rows,
            key=lambda row: row.get("range_matched_domain_auc_separability", 1.0)
            if row.get("range_matched_domain_auc_separability") is not None else 1.0,
        )
        lines.append(
            f"Lowest range-matched signal-only domain separability: "
            f"{best_norm['normalisation']} "
            f"(AUC={best_norm['range_matched_domain_auc_separability']:.3f}; 0.5 is indistinguishable)."
        )
    for row in quant_rows:
        if row.get("domain") == "MM-Fi":
            lines.append(
                f"MM-Fi {row['semantic_name']}: range "
                f"[{row['minimum']:.4f}, {row['maximum']:.4f}], "
                f"negative={100*row['negative_fraction']:.2f}%, "
                f"median unique step={row['median_positive_unique_step']:.6f}."
            )
    lines.extend([
        "",
        "Interpretation cautions",
        "-----------------------",
        "- A signed-axis search can reveal a convention mismatch, but radar-cloud centroids are not anatomical landmarks.",
        "- A stable pelvis-to-cloud-centroid offset is only an extrinsic clue, not a calibrated transform.",
        "- The fitted continuous pose-template rotation can also absorb differences in activity/posture distribution; inspect the plots before treating it as sensor pitch.",
        "- MM-Fi intensity and SyncWB SNR may differ non-linearly and may reflect different detector/firmware processing.",
        "- Window-level robust or rank normalisation can align marginal signal distributions but cannot make the physical quantities equivalent.",
    ])
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main command
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    print(json.dumps({
        "script_version": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "mmfi_point_order": MMFI_POINT_ORDER.tolist(),
        "mmfi_internal_columns": ["x", "y", "z", "doppler", "intensity"],
    }, indent=2))
    if MMFI_POINT_ORDER.tolist() != [1, 0, 2, 4, 3]:
        raise RuntimeError(
            "This signal-v2 diagnostic requires MMFI_POINT_ORDER=[1,0,2,4,3]"
        )
    out = Path(args.out)
    plots_dir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    mmfi = load_mmfi_samples(
        Path(args.packed_root),
        args.max_mmfi_frames,
        args.window_size,
        args.seed,
        args.progress,
    )
    syncwb = load_syncwb_samples(
        Path(args.sqlite),
        Path(args.artifact_root),
        str(args.subject),
        str(args.mapping_version),
        args.max_syncwb_frames,
        args.window_size,
        args.person_index,
        args.filter_noise,
        args.progress,
    )

    # Core feature and frame statistics.
    feature_rows: list[dict[str, object]] = []
    mmfi_stats, mmfi_point_sample = collect_feature_statistics(
        mmfi, args.max_points, args.seed
    )
    sync_stats, sync_point_sample = collect_feature_statistics(
        syncwb, args.max_points, args.seed + 1
    )
    feature_rows.extend(mmfi_stats)
    feature_rows.extend(sync_stats)
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(out / "feature_statistics.csv", index=False)

    joint_stats_df = pd.DataFrame(
        pose_joint_statistics(mmfi) + pose_joint_statistics(syncwb)
    )
    joint_stats_df.to_csv(out / "pose_joint_statistics.csv", index=False)

    support_df = pd.DataFrame(support_overlap_rows(
        mmfi, syncwb, mmfi_point_sample, sync_point_sample
    ))
    support_df.to_csv(out / "support_overlap.csv", index=False)

    frame_rows = frame_geometry_rows(
        mmfi, args.geometry_points_per_frame, args.seed
    ) + frame_geometry_rows(
        syncwb, args.geometry_points_per_frame, args.seed + 1
    )
    frame_df = pd.DataFrame(frame_rows)
    frame_df.to_csv(out / "frame_statistics.csv", index=False)

    # Axis and reference-frame diagnostics.
    mmfi_axis = run_within_domain_axis_search(
        mmfi,
        args.axis_search_frames,
        args.geometry_points_per_frame,
        args.seed,
        args.progress,
    )
    sync_axis = run_within_domain_axis_search(
        syncwb,
        args.axis_search_frames,
        args.geometry_points_per_frame,
        args.seed + 1,
        args.progress,
    )
    cross_pose_axis = run_cross_domain_pose_axis_search(mmfi, syncwb)
    cross_point_axis = run_cross_domain_point_axis_search(
        mmfi, syncwb, args.max_points, args.seed
    )
    frame_hypotheses, frame_hypothesis_details = reference_frame_hypotheses(
        mmfi,
        syncwb,
        args.max_points,
        args.seed,
        args.syncwb_pitch_down_deg,
        args.syncwb_sensor_height_m,
    )

    mmfi_axis.to_csv(out / "axis_search_mmfi_pose_to_radar.csv", index=False)
    sync_axis.to_csv(out / "axis_search_syncwb_kinect_to_radar.csv", index=False)
    cross_pose_axis.to_csv(out / "cross_domain_pose_axis_search.csv", index=False)
    cross_point_axis.to_csv(out / "cross_domain_point_axis_search.csv", index=False)
    frame_hypotheses.to_csv(out / "reference_frame_hypotheses.csv", index=False)

    # Exact current model-input normalisation and signal alternatives.
    model_stats, model_gap, saturation, mmfi_model_values, sync_model_values = (
        collect_model_feature_tables(mmfi, syncwb, args.max_points, args.seed)
    )
    model_stats.to_csv(out / "model_feature_statistics.csv", index=False)
    model_gap.to_csv(out / "model_feature_gap.csv", index=False)

    signal_rows = signal_relationship_rows(mmfi, mmfi_point_sample, frame_df)
    signal_rows += signal_relationship_rows(syncwb, sync_point_sample, frame_df)
    signal_df = pd.DataFrame(signal_rows)
    signal_df.to_csv(out / "signal_relationships.csv", index=False)

    # Corrected Doppler/intensity semantics and return-strength body-proximity tests.
    quantisation_df = pd.DataFrame(
        channel_quantisation_rows(mmfi, mmfi_point_sample)
        + channel_quantisation_rows(syncwb, sync_point_sample)
    )
    quantisation_df.to_csv(out / "channel_quantisation.csv", index=False)

    mmfi_proximity = collect_signal_proximity_samples(
        mmfi, args.signal_points_per_frame, args.seed, args.progress
    )
    sync_proximity = collect_signal_proximity_samples(
        syncwb, args.signal_points_per_frame, args.seed + 1, args.progress
    )
    proximity_samples = pd.concat([mmfi_proximity, sync_proximity], ignore_index=True)
    np.savez_compressed(
        out / "signal_proximity_samples.npz",
        domain=proximity_samples["domain"].astype(str).to_numpy(),
        frame_id=proximity_samples["frame_id"].astype(str).to_numpy(),
        signal=proximity_samples["signal"].to_numpy(dtype=np.float32),
        signal_frame_percentile=proximity_samples["signal_frame_percentile"].to_numpy(dtype=np.float32),
        doppler=proximity_samples["doppler"].to_numpy(dtype=np.float32),
        range_m=proximity_samples["range_m"].to_numpy(dtype=np.float32),
        skeleton_distance_m=proximity_samples["skeleton_distance_m"].to_numpy(dtype=np.float32),
    )

    proximity_summary_df = pd.DataFrame(
        signal_proximity_summary_rows(
            mmfi_proximity, mmfi.return_signal_name,
            args.near_body_threshold_m, args.clutter_threshold_m,
        )
        + signal_proximity_summary_rows(
            sync_proximity, syncwb.return_signal_name,
            args.near_body_threshold_m, args.clutter_threshold_m,
        )
    )
    proximity_summary_df.to_csv(out / "signal_proximity_summary.csv", index=False)

    proximity_deciles_df = pd.DataFrame(
        signal_proximity_decile_rows(mmfi_proximity, args.near_body_threshold_m)
        + signal_proximity_decile_rows(sync_proximity, args.near_body_threshold_m)
    )
    proximity_deciles_df.to_csv(out / "signal_proximity_deciles.csv", index=False)

    strength_groups_df = pd.DataFrame(
        signal_strength_group_rows(mmfi_proximity, args.near_body_threshold_m)
        + signal_strength_group_rows(sync_proximity, args.near_body_threshold_m)
    )
    strength_groups_df.to_csv(out / "signal_proximity_strength_groups.csv", index=False)

    range_edges = list(map(float, args.signal_range_edges))
    range_bins_df = pd.DataFrame(
        signal_range_bin_rows(
            mmfi_proximity, range_edges,
            args.near_body_threshold_m, args.clutter_threshold_m,
        )
        + signal_range_bin_rows(
            sync_proximity, range_edges,
            args.near_body_threshold_m, args.clutter_threshold_m,
        )
    )
    range_bins_df.to_csv(out / "signal_proximity_range_bins.csv", index=False)

    signal_norm_df = pd.DataFrame(signal_normalisation_comparison_rows(
        mmfi_proximity, sync_proximity, range_edges, args.seed
    ))
    signal_norm_df.to_csv(out / "signal_normalisation_comparison.csv", index=False)

    mmfi_current = find_transform_row(mmfi_axis, "transform", "x,y,z")
    sync_current = find_transform_row(sync_axis, "transform", "x,z,-y")
    cross_pose_identity = find_transform_row(
        cross_pose_axis, "transform_applied_to_syncwb_current_pose", "x,y,z"
    )
    cross_point_identity = find_transform_row(
        cross_point_axis,
        "transform_applied_to_syncwb_current_root_centred_points",
        "x,y,z",
    )
    checkpoint_bounds = checkpoint_pose_bound_report(args.checkpoint, syncwb.poses)

    report = {
        "inputs": {
            "packed_root": str(Path(args.packed_root)),
            "sqlite": str(Path(args.sqlite)),
            "artifact_root": str(Path(args.artifact_root)),
            "subject": str(args.subject),
            "mapping_version": str(args.mapping_version),
            "window_size": int(args.window_size),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "mmfi_point_order": MMFI_POINT_ORDER.tolist(),
            "near_body_threshold_m": float(args.near_body_threshold_m),
            "clutter_threshold_m": float(args.clutter_threshold_m),
            "signal_range_edges_m": list(map(float, args.signal_range_edges)),
        },
        "domains": {
            mmfi.name: mmfi.metadata,
            syncwb.name: syncwb.metadata,
        },
        "axis_conventions": {
            "mmfi_best": mmfi_axis.iloc[0].to_dict(),
            "mmfi_current_identity": mmfi_current,
            "syncwb_best": sync_axis.iloc[0].to_dict(),
            "syncwb_current_x_z_minus_y": sync_current,
            "cross_domain_pose_best": cross_pose_axis.iloc[0].to_dict(),
            "cross_domain_pose_identity": cross_pose_identity,
            "cross_domain_points_best": cross_point_axis.iloc[0].to_dict(),
            "cross_domain_points_identity": cross_point_identity,
            "important_note": (
                "Axis searches use radar centroid motion and point-to-skeleton proximity; "
                "they are diagnostics rather than formal extrinsic calibration."
            ),
        },
        "reference_frame_hypotheses": frame_hypotheses.to_dict(orient="records"),
        "reference_frame_details": frame_hypothesis_details,
        "support_overlap": support_df.to_dict(orient="records"),
        "checkpoint_pose_bounds": checkpoint_bounds,
        "centroid_minus_pelvis_offset": {
            mmfi.name: summarise_offset(frame_df, mmfi.name),
            syncwb.name: summarise_offset(frame_df, syncwb.name),
        },
        "model_feature_gaps": model_gap.sort_values(
            "current_normalisation_quantile_gap_iqr_units", ascending=False
        ).to_dict(orient="records"),
        "model_feature_saturation": saturation,
        "channel_quantisation": quantisation_df.to_dict(orient="records"),
        "signal_proximity_summary": proximity_summary_df.to_dict(orient="records"),
        "signal_proximity_deciles": proximity_deciles_df.to_dict(orient="records"),
        "signal_proximity_strength_groups": strength_groups_df.to_dict(orient="records"),
        "signal_proximity_range_bins": range_bins_df.to_dict(orient="records"),
        "signal_normalisation_comparison": signal_norm_df.to_dict(orient="records"),
        "signal_warning": (
            "MM-Fi intensity and SyncWB SNR are compared as return-strength channels, "
            "not treated as the same calibrated physical quantity."
        ),
    }

    report_json = json_ready(report)
    (out / "report.json").write_text(
        json.dumps(report_json, indent=2), encoding="utf-8"
    )
    summary = build_summary_text(report_json)
    (out / "summary.txt").write_text(summary, encoding="utf-8")

    save_feature_quantile_plot(feature_df, plots_dir / "point_feature_quantiles.png")
    save_axis_search_plot(mmfi_axis, sync_axis, plots_dir / "axis_search_top12.png")
    save_pose_template_plot(mmfi, syncwb, plots_dir / "pose_templates_current_convention.png")
    save_signal_plot(mmfi_model_values, sync_model_values, plots_dir / "signal_normalisation.png")
    save_signal_body_proximity_plot(
        proximity_deciles_df, plots_dir / "signal_body_proximity.png"
    )
    save_signal_range_auc_plot(
        range_bins_df, plots_dir / "signal_range_stratified_auc.png"
    )
    save_signal_normalisation_candidates_plot(
        signal_norm_df, plots_dir / "signal_normalisation_candidates.png"
    )

    print(summary)
    print(f"Detailed outputs written to: {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Diagnose MM-Fi -> SyncWB radar/pose domain shift.",
    )
    parser.add_argument("--packed-root", required=True)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--mapping-version", required=True)
    parser.add_argument("--out", default="runs/mmfi_syncwb_domain_diagnostics")
    parser.add_argument("--person-index", type=int, default=0)
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Must match the model window size for exact model-input statistics",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained checkpoint; reports SyncWB pose coordinates outside SimCC bounds",
    )
    parser.add_argument("--syncwb-pitch-down-deg", type=float, default=30.0)
    parser.add_argument("--syncwb-sensor-height-m", type=float, default=1.76)
    parser.add_argument("--max-mmfi-frames", type=int, default=5000)
    parser.add_argument("--max-syncwb-frames", type=int, default=5000)
    parser.add_argument("--max-points", type=int, default=500000)
    parser.add_argument(
        "--axis-search-frames",
        type=int,
        default=500,
        help="Frames used for the slower point-to-skeleton part of each 48-way axis search",
    )
    parser.add_argument("--geometry-points-per-frame", type=int, default=128)
    parser.add_argument(
        "--signal-points-per-frame",
        type=int,
        default=128,
        help="Maximum points sampled per frame for signal/body-proximity analysis",
    )
    parser.add_argument(
        "--near-body-threshold-m",
        type=float,
        default=0.20,
        help="Point-to-skeleton distance treated as a near-body return",
    )
    parser.add_argument(
        "--clutter-threshold-m",
        type=float,
        default=0.40,
        help="Point-to-skeleton distance treated as likely clutter for AUROC",
    )
    parser.add_argument(
        "--signal-range-edges",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0],
        help="Range-bin edges in metres for stratified and range-matched signal tests",
    )
    parser.add_argument(
        "--filter-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
