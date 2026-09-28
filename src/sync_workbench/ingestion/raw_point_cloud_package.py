"""Validate the trusted ETL pickle/manifest pair before canonical writes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

SAMPLES_FILE = "radar_raw_samples.zst"
MANIFEST_FILE = "radar_raw_point_cloud_versions.json"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_hash(stream)


def stream_hash(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def integer(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or pd.isna(value) or int(value) != value:
        raise ValueError(f"{label} must be an integer")
    return int(value)


@dataclass
class RawCloudVersion:
    manifest: dict
    samples: pd.DataFrame
    payload_fingerprint: str


@dataclass
class RawCloudPackage:
    versions: list[RawCloudVersion]
    device_runs: pd.DataFrame
    package_metadata: dict

    @classmethod
    def read(cls, root: str | Path):
        root = Path(root)
        manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("schema") != "syncwb.raw_point_cloud_package.v1" or manifest.get("sample_file") != SAMPLES_FILE:
            raise ValueError("Unsupported raw point-cloud package schema or sample file")
        with (root / SAMPLES_FILE).open("rb") as stream:
            if stream_hash(stream) != manifest.get("sample_file_sha256"):
                raise ValueError("Raw point-cloud package checksum mismatch")
            stream.seek(0)
            samples = pd.read_pickle(stream, compression="zstd")
        if not isinstance(samples, pd.DataFrame):
            raise ValueError("Raw sample file must contain a dataframe")
        columns = {"subject_id", "run_id", "frame_number", "sample_kind", "estimated_wallclock_from_start_end",
                   "points", "point_count", "point_count_filtered", "point_cloud_version_id", "point_status"}
        if columns - set(samples.columns):
            raise ValueError(f"Missing raw sample columns: {sorted(columns - set(samples.columns))}")
        if len(samples) != manifest.get("sample_row_count") or samples.empty:
            raise ValueError("Package sample row count mismatch")
        if samples[["subject_id", "run_id", "point_cloud_version_id"]].isna().any().any():
            raise ValueError("Missing raw sample identity")
        versions, seen = [], set()
        for entry in manifest.get("versions", []):
            key = (entry["subject_id"], entry["run_id"], entry["point_cloud_version_id"])
            if key in seen or any(not isinstance(v, str) or not v.strip() for v in key):
                raise ValueError("Duplicate or invalid cloud version identity")
            seen.add(key)
            group = samples[(samples.subject_id == key[0]) & (samples.run_id == key[1]) &
                            (samples.point_cloud_version_id == key[2])].sort_values("frame_number").copy()
            versions.append(_validate_version(entry, group))
        if not versions or sum(len(v.samples) for v in versions) != len(samples):
            raise ValueError("Sample rows and manifest versions do not match")
        # The sibling acquisition table is mandatory; no generator imports needed.
        runs = pd.read_pickle(root / "device_runs.zst", compression="zstd")
        if not isinstance(runs, pd.DataFrame) or {"subject_id", "run_id", "device_type", "start_wallclock_est", "end_wallclock_est", "nominal_fps"} - set(runs.columns):
            raise ValueError("Invalid device_runs.zst")
        for version in versions:
            m = version.manifest
            matches = runs[(runs.subject_id == m["subject_id"]) & (runs.run_id == m["run_id"]) & (runs.device_type == "radar_raw")]
            if len(matches) != 1:
                raise ValueError("Each raw version must match exactly one DEVICE_RUN in the package")
            acquisition = m["acquisition"]
            for col in ("start_wallclock_est", "end_wallclock_est"):
                if str(matches.iloc[0][col]) != acquisition[col]:
                    raise ValueError(f"Raw acquisition {col} disagrees with device_runs.zst")
        metadata = {key: value for key, value in manifest.items() if key != "versions"}
        return cls(versions, runs, metadata)


def _validate_version(m: dict, group: pd.DataFrame) -> RawCloudVersion:
    if m.get("device_type") != "radar_raw" or m.get("tracking_enabled") is not True:
        raise ValueError("Only tracking-enabled radar_raw versions are supported")
    if m.get("hdf5_schema") != "iwr6843.raw_point_cloud_sequence.hdf5.v1":
        raise ValueError("Unsupported source HDF5 schema")
    if m.get("visualization_excluded_association_ids") != [253, 254, 255]:
        raise ValueError("Raw visualization filter must exclude exactly 253, 254, 255")
    if m.get("point_dtype") != "float32" or m.get("point_columns") != ["x", "y", "z", "radial_velocity", "snr", "association_id"]:
        raise ValueError("Unsupported raw point payload contract")
    base = m.get("source_frame_number_base")
    if type(base) is not int or base not in (0, 1):
        raise ValueError("Invalid source numbering convention")
    if not re.fullmatch(r"[0-9a-f]{64}", m.get("hdf5_sha256", "")):
        raise ValueError("Invalid HDF5 content hash")
    binding = {key: m[key] for key in ("hdf5_sha256", "subject_id", "run_id", "device_type", "source_frame_number_base")}
    if m.get("point_cloud_version_id") != "raw_" + json_hash(binding):
        raise ValueError("Cloud version ID does not match source binding")
    generator = m.get("generator_metadata", {})
    if generator.get("tracking_enabled") is not True or any(key not in generator for key in (
            "source_config_sha256", "native_executable_sha256", "native_version", "created_utc",
            "calibration_sources", "tracking_state_start_sequence_index")) or not isinstance(m.get("processing_cfg"), str):
        raise ValueError("Missing tracked generator provenance")
    n = integer(m["acquisition_frame_count"], "acquisition_frame_count")
    if n < 1 or len(group) != n:
        raise ValueError("Each version must carry the full acquisition index")
    frames = [integer(value, "frame_number") for value in group.frame_number]
    if frames != list(range(1, n + 1)) or not group.sample_kind.eq("frame").all():
        raise ValueError("Acquisition frame numbers must be consecutive 1..N with sample_kind=frame")
    timestamps = group.estimated_wallclock_from_start_end.tolist()
    parsed = [datetime.strptime(value, TIME_FORMAT) for value in timestamps]
    if any(b < a for a, b in zip(parsed, parsed[1:])):
        raise ValueError("Raw acquisition timestamps must be monotonic")
    a = m["acquisition"]
    if (a.get("subject_id"), a.get("run_id"), a.get("device_type"), a.get("frame_count")) != (m["subject_id"], m["run_id"], "radar_raw", n):
        raise ValueError("Acquisition provenance identity mismatch")
    timing_hash = json_hash({"subject_id": m["subject_id"], "run_id": m["run_id"], "frame_number": frames, "timestamps": timestamps})
    if timing_hash != a.get("acquisition_timeline_sha256") or timestamps[0] != a.get("start_wallclock_est") or timestamps[-1] != a.get("end_wallclock_est"):
        raise ValueError("Acquisition timeline fingerprint/endpoint mismatch")
    available = group.point_status.eq("available")
    if not group.point_status.isin(["available", "unprocessed"]).all():
        raise ValueError("Invalid point_status")
    source_ids = [integer(v, "source frame ID") for v in m["source_frame_ids"]]
    sequence = [integer(v, "sequence index") for v in m["source_sequence_indices"]]
    if (any(v < base for v in source_ids) or any(v < 0 for v in sequence)
            or any(b <= a for a, b in zip(sequence, sequence[1:]))
            or len(sequence) != len(source_ids)
            or [v + 1 - base for v in source_ids] != group.loc[available, "frame_number"].tolist()
            or int(available.sum()) != m["processed_frame_count"]):
        raise ValueError("Processed coverage does not match source frame identities")
    digest = hashlib.sha256()
    digest.update(json_hash(binding).encode())
    digest.update(timing_hash.encode())
    total = 0
    for row in group.itertuples(index=False):
        points = row.points
        if not isinstance(points, np.ndarray) or points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 6:
            raise ValueError("Raw points must be float32 arrays with six columns")
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite raw point values")
        ids = points[:, 5]
        if np.any((ids < 0) | (ids > 255) | (ids != np.floor(ids))):
            raise ValueError("Invalid GTRACK association code")
        if row.point_status == "unprocessed":
            if len(points) or not pd.isna(row.point_count) or not pd.isna(row.point_count_filtered):
                raise ValueError("Unprocessed frames must have empty payload and unavailable counts")
        else:
            count = integer(row.point_count, "point_count")
            filtered = integer(row.point_count_filtered, "point_count_filtered")
            if count != len(points) or filtered != np.count_nonzero(~np.isin(ids, [253, 254, 255])):
                raise ValueError("Raw point counts do not match payload/filter")
            total += count
        digest.update(f"{row.frame_number}:{row.point_status}:{len(points)};".encode())
        digest.update(points.astype("<f4", copy=False).tobytes(order="C"))
    if total != m["point_count"]:
        raise ValueError("Manifest point total mismatch")
    group["sample_index"] = np.asarray(frames, dtype=np.int64) - 1
    return RawCloudVersion(m, group, digest.hexdigest())
