"""Export one point-cloud interval with RGB and raw-radar frame links.

This is a deliberately small, single-use utility. Input frame numbers are the
original one-based ``frame_number`` values from the temporary ingestion files,
not canonical zero-based ``sample_index`` values.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sync_workbench.storage.artifact_store import ArtifactStore
from sync_workbench.storage.ragged_npz import RaggedNpzReader
from sync_workbench.storage.ragged_npz import RaggedNpzWriter
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


PC_TIMELINE = "radar_pc_linear_from_index"
RAW_TIMELINE = "radar_raw_wallclock_from_start_end"
SOURCE_FRAME_RE = re.compile(r"(?:^|;)source_frame_number=([^;]+)")
POINT_COLUMNS = ["x", "y", "z", "radial_velocity", "snr", "gtrack_target_id"]
POSE2D_COLUMNS = ["image_x", "image_y", "confidence"]
POSE3D_COLUMNS = ["x", "y", "z", "confidence"]
POSE2D_JOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip",
    "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "Head_Apex",
    "Neck", "Hip_Center", "L_BigToe", "R_BigToe", "L_SmallToe",
    "R_SmallToe", "L_Heel", "R_Heel",
]
POSE2D_LIMB_CONNECTIONS = [
    (0, 1), (0, 2), (2, 1), (2, 4), (1, 3), (4, 6), (3, 5), (5, 7),
    (6, 8), (7, 9), (8, 10), (19, 11), (19, 12), (11, 13), (12, 14),
    (13, 15), (14, 16), (15, 24), (16, 25), (15, 20), (15, 22),
    (16, 21), (16, 23), (19, 18), (18, 17), (18, 5), (18, 6),
]
POSE3D_JOINT_NAMES = [
    "pelvis", "spine - navel", "spine - chest", "neck", "left clavicle",
    "left shoulder", "left elbow", "left wrist", "left hand",
    "left handtip", "left thumb", "right clavicle", "right shoulder",
    "right elbow", "right wrist", "right hand", "right handtip",
    "right thumb", "left hip", "left knee", "left ankle", "left foot",
    "right hip", "right knee", "right ankle", "right foot", "head", "nose",
    "left eye", "left ear", "right eye", "right ear",
]
_POSE3D_NAME_TO_INDEX = {
    name: index for index, name in enumerate(POSE3D_JOINT_NAMES)
}
POSE3D_LIMB_CONNECTIONS = [
    (_POSE3D_NAME_TO_INDEX[parent], _POSE3D_NAME_TO_INDEX[child])
    for parent, child in [
        ("pelvis", "spine - navel"),
        ("spine - navel", "spine - chest"),
        ("spine - chest", "neck"),
        ("neck", "head"),
        ("spine - chest", "left clavicle"),
        ("left clavicle", "left shoulder"),
        ("left shoulder", "left elbow"),
        ("left elbow", "left wrist"),
        ("left wrist", "left hand"),
        ("left hand", "left handtip"),
        ("left hand", "left thumb"),
        ("spine - chest", "right clavicle"),
        ("right clavicle", "right shoulder"),
        ("right shoulder", "right elbow"),
        ("right elbow", "right wrist"),
        ("right wrist", "right hand"),
        ("right hand", "right handtip"),
        ("right hand", "right thumb"),
        ("pelvis", "left hip"),
        ("left hip", "left knee"),
        ("left knee", "left ankle"),
        ("left ankle", "left foot"),
        ("pelvis", "right hip"),
        ("right hip", "right knee"),
        ("right knee", "right ankle"),
        ("right ankle", "right foot"),
    ]
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export point clouds and their synchronized RGB/raw radar frame numbers."
    )
    parser.add_argument("--sqlite", default="workbench.sqlite")
    parser.add_argument("--artifact-root", default="artifact_store")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--mapping-version", required=True)
    parser.add_argument("--rgb-start-frame", required=True, type=int)
    parser.add_argument("--rgb-end-frame", required=True, type=int)
    parser.add_argument("--pc-start-frame", required=True, type=int)
    parser.add_argument("--pc-end-frame", required=True, type=int)
    return parser


def _mapping_context(store: SQLiteCoreStore, subject: str, version: str) -> pd.Series:
    versions = store.read_table("MAPPING_VERSION")
    rows = versions[
        (versions["subject_id"].astype(str) == subject)
        & (versions["mapping_version_id"].astype(str) == version)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one mapping version for {subject}/{version}, found {len(rows)}."
        )
    context = rows.iloc[0]
    if str(context.source_device_type) != "kinect_rgb" or str(context.target_device_type) != "radar_pc":
        raise ValueError("The export mapping must be kinect_rgb -> radar_pc.")
    return context


def _source_frame_number(notes: object) -> int:
    match = SOURCE_FRAME_RE.search(str(notes))
    if match is None:
        raise ValueError(f"RUN_SAMPLE notes do not contain source_frame_number: {notes!r}")
    value = float(match.group(1))
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"Invalid source frame number in RUN_SAMPLE notes: {notes!r}")
    return int(value)


def _run_samples(
    store: SQLiteCoreStore,
    subject: str,
    run_id: str,
    device_type: str,
) -> pd.DataFrame:
    samples = store.read_table("RUN_SAMPLE")
    rows = samples[
        (samples["subject_id"].astype(str) == subject)
        & (samples["run_id"].astype(str) == run_id)
        & (samples["device_type"].astype(str) == device_type)
    ].copy()
    if rows.empty:
        raise KeyError(f"No RUN_SAMPLE rows for {subject}/{run_id}/{device_type}.")
    rows["sample_index"] = pd.to_numeric(rows["sample_index"], errors="raise").astype(int)
    rows["frame_number"] = rows["notes"].map(_source_frame_number)
    return rows.sort_values("sample_index").reset_index(drop=True)


def _frame_range(rows: pd.DataFrame, start: int, end: int, label: str) -> pd.DataFrame:
    if end < start:
        raise ValueError(f"{label} end frame must not precede its start frame.")
    available = set(rows["frame_number"].astype(int))
    missing_endpoints = [frame for frame in (start, end) if frame not in available]
    if missing_endpoints:
        raise KeyError(f"{label} frame endpoints do not exist: {missing_endpoints}")
    selected = rows[
        rows["frame_number"].astype(int).between(start, end, inclusive="both")
    ].copy()
    if selected.empty:
        raise ValueError(f"No {label} frames selected.")
    return selected.sort_values("sample_index").reset_index(drop=True)


def _best_rgb_mapping_rows(
    store: SQLiteCoreStore,
    subject: str,
    version: str,
    rgb_sample_indices: np.ndarray,
    pc_run_id: str,
) -> pd.DataFrame:
    mappings = store.read_table("SAMPLE_MAPPING")
    rows = mappings[
        (mappings["subject_id"].astype(str) == subject)
        & (mappings["mapping_version_id"].astype(str) == version)
        & (mappings["target_run_id"].astype(str) == pc_run_id)
        & (mappings["target_device_type"].astype(str) == "radar_pc")
        & (pd.to_numeric(mappings["source_sample_index"], errors="coerce").isin(rgb_sample_indices))
    ].copy()
    if rows.empty:
        raise KeyError("No SAMPLE_MAPPING rows fall inside the requested RGB interval.")

    rows["source_sample_index"] = pd.to_numeric(rows["source_sample_index"], errors="raise").astype(int)
    rows["target_sample_index"] = pd.to_numeric(rows["target_sample_index"], errors="raise").astype(int)
    rows["__primary"] = rows["is_primary"].astype(str).str.lower().isin({"true", "1"})
    rows["__rank"] = pd.to_numeric(rows["rank"], errors="coerce").fillna(1_000_000)
    rows["__abs_delta"] = pd.to_numeric(
        rows["predicted_minus_estimated_ms"], errors="coerce"
    ).abs().fillna(1_000_000)
    rows = rows.sort_values(
        ["source_sample_index", "__primary", "__rank", "__abs_delta"],
        ascending=[True, False, True, True],
    ).drop_duplicates("source_sample_index", keep="first")

    # At 15 FPS RGB and 20 FPS radar, not every PC frame is a direct mapping
    # target. Keep one representative RGB row per mapped target; dense PC rows
    # are associated to the nearest such target below.
    return rows.sort_values(
        ["target_sample_index", "__primary", "__abs_delta", "source_sample_index"],
        ascending=[True, False, True, True],
    ).drop_duplicates("target_sample_index", keep="first").reset_index(drop=True)


def _nearest_positions(sorted_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    if len(sorted_values) == 0:
        raise ValueError("Cannot find nearest values in an empty array.")
    right = np.searchsorted(sorted_values, query_values, side="left")
    right = np.clip(right, 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    choose_left = np.abs(query_values - sorted_values[left]) <= np.abs(
        sorted_values[right] - query_values
    )
    return np.where(choose_left, left, right).astype(int)


def _timeline(
    store: SQLiteCoreStore,
    subject: str,
    run_id: str,
    device_type: str,
    model_id: str,
) -> pd.DataFrame:
    times = store.read_table("SAMPLE_TIME_ESTIMATE")
    rows = times[
        (times["subject_id"].astype(str) == subject)
        & (times["run_id"].astype(str) == run_id)
        & (times["device_type"].astype(str) == device_type)
        & (times["timeline_model_id"].astype(str) == model_id)
    ].copy()
    if rows.empty:
        raise KeyError(f"Timeline not found: {subject}/{run_id}/{device_type}/{model_id}")
    rows["sample_index"] = pd.to_numeric(rows["sample_index"], errors="raise").astype(int)
    rows["time"] = pd.to_datetime(rows["time_value_datetime"], errors="coerce")
    if rows["time"].isna().any():
        raise ValueError(f"Timeline {model_id} contains invalid datetime values.")
    rows["epoch_ns"] = rows["time"].astype("int64")
    return rows.sort_values("epoch_ns").reset_index(drop=True)


def _bundle_reader(
    store: SQLiteCoreStore,
    artifact_root: str | Path,
    subject: str,
    run_id: str,
    device_type: str,
    asset_role: str,
) -> RaggedNpzReader:
    assets = store.read_table("RUN_ASSET")
    rows = assets[
        (assets["subject_id"].astype(str) == subject)
        & (assets["run_id"].astype(str) == run_id)
        & (assets["device_type"].astype(str) == device_type)
        & (assets["asset_role"].astype(str) == asset_role)
    ]
    if len(rows) != 1:
        raise KeyError(
            f"Expected one {asset_role} asset for "
            f"{subject}/{run_id}/{device_type}, found {len(rows)}."
        )
    artifact_store = ArtifactStore(artifact_root)
    return RaggedNpzReader(
        artifact_store.path_for_ref(str(rows.iloc[0]["asset_ref"]))
    )


def export_sequence(args: argparse.Namespace) -> Path:
    store = SQLiteCoreStore(args.sqlite)
    context = _mapping_context(store, args.subject, args.mapping_version)
    rgb_run_id = str(context.source_run_id)
    pc_run_id = str(context.target_run_id)

    rgb_all = _run_samples(store, args.subject, rgb_run_id, "kinect_rgb")
    pc_all = _run_samples(store, args.subject, pc_run_id, "radar_pc")
    raw_all = _run_samples(store, args.subject, pc_run_id, "radar_raw")
    rgb = _frame_range(rgb_all, args.rgb_start_frame, args.rgb_end_frame, "RGB")
    pc = _frame_range(pc_all, args.pc_start_frame, args.pc_end_frame, "point-cloud")

    mapped = _best_rgb_mapping_rows(
        store,
        args.subject,
        args.mapping_version,
        rgb["sample_index"].to_numpy(dtype=int),
        pc_run_id,
    )
    mapped_targets = mapped["target_sample_index"].to_numpy(dtype=int)
    mapped_pos = _nearest_positions(
        mapped_targets,
        pc["sample_index"].to_numpy(dtype=int),
    )
    mapped_for_pc = mapped.iloc[mapped_pos].reset_index(drop=True)
    rgb_frame_by_sample = rgb_all.set_index("sample_index")["frame_number"]
    pc_frame_by_sample = pc_all.set_index("sample_index")["frame_number"]

    pc_times = _timeline(store, args.subject, pc_run_id, "radar_pc", PC_TIMELINE)
    pc_times = pc_times.set_index("sample_index").loc[pc["sample_index"]].reset_index()
    raw_times = _timeline(store, args.subject, pc_run_id, "radar_raw", RAW_TIMELINE)
    raw_pos = _nearest_positions(
        raw_times["epoch_ns"].to_numpy(dtype=np.int64),
        pc_times["epoch_ns"].to_numpy(dtype=np.int64),
    )
    raw_for_pc = raw_times.iloc[raw_pos].reset_index(drop=True)
    raw_frame_by_sample = raw_all.set_index("sample_index")["frame_number"]

    links = pd.DataFrame(
        {
            "sequence_index": np.arange(len(pc), dtype=np.int64),
            "subject_id": args.subject,
            "mapping_version_id": args.mapping_version,
            "rgb_run_id": rgb_run_id,
            "rgb_frame_number": mapped_for_pc["source_sample_index"].map(rgb_frame_by_sample).astype(int),
            "rgb_sample_index": mapped_for_pc["source_sample_index"].astype(int),
            "rgb_mapping_target_pc_frame_number": mapped_for_pc["target_sample_index"].map(pc_frame_by_sample).astype(int),
            "rgb_mapping_target_pc_sample_index": mapped_for_pc["target_sample_index"].astype(int),
            "pc_run_id": pc_run_id,
            "pc_frame_number": pc["frame_number"].astype(int),
            "pc_sample_index": pc["sample_index"].astype(int),
            "pc_minus_rgb_mapping_target_samples": (
                pc["sample_index"].to_numpy(dtype=int)
                - mapped_for_pc["target_sample_index"].to_numpy(dtype=int)
            ),
            "raw_run_id": pc_run_id,
            "raw_frame_number": raw_for_pc["sample_index"].map(raw_frame_by_sample).astype(int),
            "raw_sample_index": raw_for_pc["sample_index"].astype(int),
            "pc_time_estimate": pc_times["time_value_datetime"].astype(str),
            "raw_time_estimate": raw_for_pc["time_value_datetime"].astype(str),
            "raw_minus_pc_ms": (
                (raw_for_pc["epoch_ns"].to_numpy(dtype=np.int64)
                 - pc_times["epoch_ns"].to_numpy(dtype=np.int64))
                / 1_000_000.0
            ),
            "rgb_to_pc_predicted_minus_estimated_ms": pd.to_numeric(
                mapped_for_pc["predicted_minus_estimated_ms"], errors="coerce"
            ),
            "rgb_to_pc_support_status": mapped_for_pc["support_status"].astype(str),
        }
    )

    point_reader = _bundle_reader(
        store,
        args.artifact_root,
        args.subject,
        pc_run_id,
        "radar_pc",
        "radar_points_bundle",
    )
    point_frames: list[tuple[int, np.ndarray]] = []
    for sample_index in links["pc_sample_index"].astype(int):
        points = np.asarray(point_reader.get(sample_index), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 6:
            raise ValueError(
                f"PC sample {sample_index} has shape {points.shape}; expected (P, 6)."
            )
        point_frames.append((sample_index, points))

    pose2d_frames: list[tuple[int, np.ndarray]] = []
    pose3d_frames: list[tuple[int, np.ndarray]] = []
    pose2d_reader = _bundle_reader(
        store,
        args.artifact_root,
        args.subject,
        rgb_run_id,
        "kinect_rgb",
        "pose2d_bundle",
    )
    pose3d_reader = _bundle_reader(
        store,
        args.artifact_root,
        args.subject,
        rgb_run_id,
        "kinect_rgb",
        "pose3d_bundle",
    )
    for sample_index in links["rgb_sample_index"].drop_duplicates().astype(int):
        pose2d = np.asarray(pose2d_reader.get(sample_index), dtype=np.float64)
        pose3d = np.asarray(pose3d_reader.get(sample_index), dtype=np.float64)
        if pose2d.ndim != 3 or pose2d.shape[1:] != (26, 3):
            raise ValueError(
                f"RGB sample {sample_index} has pose2d shape {pose2d.shape}; "
                "expected (N, 26, 3)."
            )
        if pose3d.ndim != 3 or pose3d.shape[1:] != (32, 4):
            raise ValueError(
                f"RGB sample {sample_index} has pose3d shape {pose3d.shape}; "
                "expected (N, 32, 4)."
            )
        pose2d_frames.append((sample_index, pose2d))
        pose3d_frames.append((sample_index, pose3d))

    output = Path(args.output) / args.sequence_name
    output.mkdir(parents=True, exist_ok=True)
    RaggedNpzWriter.write(
        output / "point_clouds.npz",
        point_frames,
        tail_shape=(6,),
        dtype="float64",
    )
    RaggedNpzWriter.write(
        output / "rgb_pose2d.npz",
        pose2d_frames,
        tail_shape=(26, 3),
        dtype="float64",
    )
    RaggedNpzWriter.write(
        output / "rgb_pose3d.npz",
        pose3d_frames,
        tail_shape=(32, 4),
        dtype="float64",
    )
    links.to_csv(output / "frame_links.csv", index=False)

    metadata = {
        "format_version": 1,
        "sequence_name": args.sequence_name,
        "subject_id": args.subject,
        "mapping_version_id": args.mapping_version,
        "rgb_run_id": rgb_run_id,
        "pc_run_id": pc_run_id,
        "raw_run_id": pc_run_id,
        "ranges_are_inclusive": True,
        "requested_frame_ranges": {
            "rgb": [args.rgb_start_frame, args.rgb_end_frame],
            "radar_pc": [args.pc_start_frame, args.pc_end_frame],
        },
        "exported_point_cloud_frames": int(len(links)),
        "exported_unique_rgb_pose_frames": int(len(pose2d_frames)),
        "point_columns": POINT_COLUMNS,
        "pose2d_shape_per_person": [26, 3],
        "pose2d_columns": POSE2D_COLUMNS,
        "pose2d_joint_names": POSE2D_JOINT_NAMES,
        "pose2d_limb_connections": POSE2D_LIMB_CONNECTIONS,
        "pose2d_coordinate_frame": "Kinect low-quality RGB image",
        "pose2d_coordinate_units": "pixels",
        "pose3d_shape_per_person": [32, 4],
        "pose3d_columns": POSE3D_COLUMNS,
        "pose3d_joint_names": POSE3D_JOINT_NAMES,
        "pose3d_limb_connections": POSE3D_LIMB_CONNECTIONS,
        "pose3d_coordinate_frame": "Kinect camera",
        "pose3d_coordinate_units": "millimetres",
        "pc_timeline_model_id": PC_TIMELINE,
        "raw_timeline_model_id": RAW_TIMELINE,
        "raw_sync_method": "nearest estimated raw wallclock for each smoothed PC acquisition wallclock",
        "raw_minus_pc_ms": {
            "min": float(links["raw_minus_pc_ms"].min()),
            "median": float(links["raw_minus_pc_ms"].median()),
            "max": float(links["raw_minus_pc_ms"].max()),
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(_handoff_readme(), encoding="utf-8")
    return output


def _handoff_readme() -> str:
    return """# Exported radar sequence

`frame_links.csv` has one row per exported online point-cloud frame. It links
that frame to an RGB frame through the requested SyncWB mapping version and to
the nearest raw-radar frame by estimated acquisition wallclock. Because RGB is
15 FPS and radar is 20 FPS, not every exported PC frame is a direct mapping
target. `rgb_mapping_target_pc_sample_index` identifies the mapped PC frame;
`pc_minus_rgb_mapping_target_samples` records the small nearest-frame offset.

`point_clouds.npz` is a ragged NumPy bundle. Load frame `i` as follows:

```python
import numpy as np
import pandas as pd

links = pd.read_csv("frame_links.csv")
bundle = np.load("point_clouds.npz", allow_pickle=False)
assert np.array_equal(bundle["sample_index"], links["pc_sample_index"])

i = 0
points = bundle["values"][bundle["offsets"][i]:bundle["offsets"][i + 1]]
# points.shape == (P, 6): x, y, z, radial velocity, SNR, GTRACK target id
```

`rgb_pose2d.npz` and `rgb_pose3d.npz` use the same ragged layout, but are
keyed by the unique canonical `rgb_sample_index` values rather than PC sample
indices. Join through `frame_links.csv`. For example:

```python
pose3d_bundle = np.load("rgb_pose3d.npz", allow_pickle=False)
rgb_sample_index = int(links.iloc[i]["rgb_sample_index"])
pose_row = np.flatnonzero(pose3d_bundle["sample_index"] == rgb_sample_index)[0]
pose3d = pose3d_bundle["values"][
    pose3d_bundle["offsets"][pose_row]:pose3d_bundle["offsets"][pose_row + 1]
]
# pose3d.shape == (N, 32, 4): people, Kinect joints, (x, y, z, confidence)
# pose2d.shape == (N, 26, 3): people, Kinect joints, (image x, image y, confidence)
```

Use `raw_frame_number` to locate the corresponding parsed raw-radar frame. It
is the official one-based `frame_number` from `radar_raw_samples.zst`.
`raw_sample_index` is SyncWB's zero-based canonical index and, for these current
captures, is also `raw_frame_number - 1`. This distinction is included because
some older extracted files may have been named with zero-based counters.

The raw association is approximate: SyncWB compares the online point-cloud
`radar_pc_linear_from_index` timestamp with the raw radar's uniformly
interpolated `radar_raw_wallclock_from_start_end` timestamp and chooses the
nearest raw frame. Inspect `raw_minus_pc_ms` before using the association.
"""


def main() -> int:
    args = build_parser().parse_args()
    output = export_sequence(args)
    links = pd.read_csv(output / "frame_links.csv")
    print(f"Exported {len(links)} point-cloud frames to {output}")
    print(
        "Raw-minus-PC timing delta (ms): "
        f"min={links.raw_minus_pc_ms.min():.3f}, "
        f"median={links.raw_minus_pc_ms.median():.3f}, "
        f"max={links.raw_minus_pc_ms.max():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
