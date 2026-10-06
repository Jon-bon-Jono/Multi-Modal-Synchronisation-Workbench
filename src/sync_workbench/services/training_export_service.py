"""Read-only, homogeneous, modality-separated Living Lab HPE exports.

The manifest is the completion marker. Source SQLite connections use mode=ro;
payloads are read once per segment, and HDF5 arrays are written in frame batches.
"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import shutil
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from sync_workbench.core.geometry import (
    GEOMETRY_PROFILE_ID, KINECT_JOINT_NAMES, SENSOR_HEIGHT_M,
    SENSOR_PITCH_DOWN_DEG, points_sensor_to_world, pose3d_to_world,
    sensor_to_world_rotation_matrix,
)
from sync_workbench.core.time_utils import utc_now_str
from sync_workbench.ingestion.raw_point_cloud_package import file_hash, json_hash
from sync_workbench.storage.ragged_npz import RaggedNpzReader
from sync_workbench.sync.mapping import timeline_numeric_values

SCHEMA = "syncwb.training_export.v1"
RUN = "subject_id=? AND run_id=? AND device_type=?"


def _rows(conn, table, where, args):
    return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}" WHERE {where}', args)]


def _one(rows, label):
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {label}; found {len(rows)}")
    return rows[0]


def _path(root, ref):
    path = (root / ref).resolve()
    if not ref or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or non-portable artifact reference: {ref}")
    return path


def processing_recipe(manifest):
    """Conservative equality: identical cfg text/hash, binary, calibration and DSP contract."""
    g = manifest.get("generator_metadata", {})
    required = ("source_config_sha256", "native_executable_sha256", "native_version",
                "calibration_sources", "tracking_state_start_sequence_index")
    if (manifest.get("tracking_enabled") is not True or g.get("tracking_enabled") is not True
            or any(k not in g for k in required) or not manifest.get("processing_cfg")):
        raise ValueError("Missing processing recipe provenance; legacy clouds cannot be exported")
    for key in ("source_config_sha256", "native_executable_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(g[key])):
            raise ValueError(f"Invalid recipe {key}")
    calibration = []
    for source in g["calibration_sources"]:
        digest = source.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Calibration provenance needs content hashes")
        calibration.append(digest)
    # Paths, acquisition IDs, coverage, creation times and runtime statistics are
    # result provenance, not a shared processing recipe. Calibration order is kept.
    keys = ("native_version", "native_executable_sha256", "source_config_sha256",
            "tracking_enabled", "tracking_state_start_sequence_index", "adc_dtype",
            "adc_shape", "point_coordinate_frame", "point_units", "point_columns",
            "association_labels", "scenery_box_frame", "target_state_frame")
    recipe = {k: g.get(k) for k in keys}
    recipe.update(schema="syncwb.processing_recipe.v1", processing_cfg=manifest["processing_cfg"],
                  calibration_sha256=calibration, native_profile=g.get("native_run", {}).get("profile"),
                  point_columns=manifest["point_columns"], point_dtype=manifest["point_dtype"])
    if (g.get("point_coordinate_frame") != "sensor" or len(g.get("point_units", [])) != 5
            or g["point_units"][:3] != ["m"] * 3):
        raise ValueError("GUI geometry requires documented radar sensor XYZ in metres")
    if manifest["point_dtype"] != "float32":
        raise ValueError("Unsupported raw point dtype")
    if manifest["point_columns"] != ["x", "y", "z", "radial_velocity", "snr", "association_id"]:
        raise ValueError("Unsupported raw point columns")
    return recipe


def _timeline(conn, key, timeline_id):
    model = _one(_rows(conn, "RUN_TIMELINE_MODEL", RUN + " AND timeline_model_id=?",
                       (*key, timeline_id)), "timeline model")
    samples = _rows(conn, "RUN_SAMPLE", RUN, key)
    times = _rows(conn, "SAMPLE_TIME_ESTIMATE", RUN + " AND timeline_model_id=?", (*key, timeline_id))
    frame = pd.DataFrame(times)
    if frame.empty:
        raise ValueError("Missing timeline samples")
    frame["sample_index"] = pd.to_numeric(frame.sample_index, errors="raise").astype("int64")
    frame = frame.sort_values("sample_index").reset_index(drop=True)
    if frame.sample_index.duplicated().any() or set(frame.sample_index) != {int(r["sample_index"]) for r in samples}:
        raise ValueError("Timeline must cover every acquisition sample exactly once")
    values, kind = timeline_numeric_values(frame)
    t = np.asarray(values, dtype=np.float64)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Export timelines must have finite, strictly increasing timestamps")
    sample_rows = {int(r["sample_index"]): r for r in samples}
    numbers = []
    for idx in frame.sample_index:
        match = re.search(r"(?:^|;)source_frame_number=([^;]+)", str(sample_rows[int(idx)].get("notes", "")))
        numbers.append(int(float(match[1])) if match else -1)
    return {"indices": frame.sample_index.to_numpy(), "time": t, "kind": kind,
            "frame_number": np.asarray(numbers, dtype=np.int64), "model": model,
            "fingerprint": json_hash(times)}


def _predict(model, t):
    if model["model_type"] == "identity_time":
        return t.copy()
    if model["model_type"] != "piecewise_affine":
        raise ValueError("Export supports identity_time and piecewise_affine models")
    segments = json.loads(model["parameters_json"]).get("segments", [])
    if not segments:
        raise ValueError("Piecewise model has no segments")
    result = np.full(len(t), np.nan)
    previous = None
    for s in segments:
        left, right, slope, intercept = [float(s[k]) for k in ("source_left", "source_right", "slope", "intercept")]
        if not np.isfinite([left, right, slope, intercept]).all() or right <= left or slope <= 0:
            raise ValueError("Piecewise model must be finite and strictly increasing")
        if previous is not None and (left != previous[0] or not np.isclose(slope * left + intercept, previous[1], rtol=0, atol=1e-5)):
            raise ValueError("Piecewise model must be continuous")
        mask = (t >= left) & (t <= right)
        result[mask] = slope * t[mask] + intercept
        previous = (right, slope * right + intercept)
    return result  # No extrapolated labels, even if the navigation model allows them.


def _ranges(indices):
    """Compact inclusive canonical-index ranges, including singletons."""
    if not len(indices):
        return []
    groups = np.split(np.asarray(indices, dtype=np.int64), np.flatnonzero(np.diff(indices) != 1) + 1)
    return [[int(g[0]), int(g[-1])] for g in groups]


def _geometry():
    rotation = sensor_to_world_rotation_matrix()
    axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    return dict(profile_id=GEOMETRY_PROFILE_ID, coordinate_frame="approximate_floor_world",
                units="m", axes=["right", "forward", "up"], handedness="right",
                sensor_pitch_down_deg=SENSOR_PITCH_DOWN_DEG, sensor_height_m=SENSOR_HEIGHT_M,
                radar_rotation=rotation.tolist(), translation_m=[0, 0, SENSOR_HEIGHT_M],
                kinect_mm_to_sensor_m=(axis * 0.001).tolist(),
                kinect_mm_to_world_linear=(rotation @ axis * 0.001).tolist(),
                convention="column vectors: world = linear @ input + translation",
                calibration_status="historical GUI assumption; no calibrated Kinect-radar extrinsics",
                point_filter="none", temporal_accumulation="none")


def _prepare(conn, selections, gap_factor):
    segments, contexts, recipes, profiles = [], [], [], []
    selected_clouds, pairs = {}, set()
    claimed_radar, claimed_pose = {}, {}
    if not selections:
        raise ValueError("Select at least one mapping/cloud result")
    subjects = sorted({s["subject_id"] for s in selections})
    for selected in selections:
        subject = selected["subject_id"]
        mapping = _one(_rows(conn, "MAPPING_VERSION", "subject_id=? AND mapping_version_id=?",
                             (subject, selected["mapping_version_id"])), "mapping version")
        if (mapping["source_device_type"], mapping["target_device_type"]) != ("kinect_rgb", "radar_raw"):
            raise ValueError("Only Kinect-to-offline-raw mappings may be exported")
        rk = (subject, mapping["target_run_id"], "radar_raw")
        pk = (subject, mapping["source_run_id"], "kinect_rgb")
        if (rk, pk) in pairs:
            raise ValueError("Duplicate run pair / multiple mapping versions for the same pair")
        pairs.add((rk, pk))
        vid = selected["point_cloud_version_id"]
        if not vid.startswith("raw_") or vid == "raw_legacy":
            raise ValueError("Only provenance-backed offline raw clouds may be exported")
        if rk in selected_clouds and selected_clouds[rk] != vid:
            raise ValueError("Multiple point-cloud versions selected for the same DEVICE_RUN")
        selected_clouds[rk] = vid
        cloud = _one(_rows(conn, "POINT_CLOUD_VERSION", RUN + " AND point_cloud_version_id=?", (*rk, vid)), "cloud version")
        manifest = json.loads(cloud["provenance_json"])["version"]
        binding = {k: manifest[k] for k in ("hdf5_sha256", "subject_id", "run_id", "device_type", "source_frame_number_base")}
        if vid != "raw_" + json_hash(binding) or (binding["subject_id"], binding["run_id"], binding["device_type"]) != rk:
            raise ValueError("Cloud result binding disagrees with acquisition")
        recipes.append(processing_recipe(manifest))
        if not re.fullmatch(r"[0-9a-f]{64}", str(cloud.get("artifact_sha256", ""))):
            raise ValueError("Selected raw cloud requires a valid bundle checksum")
        model = _one(_rows(conn, "SYNC_MODEL", "subject_id=? AND sync_model_id=?",
                           (subject, mapping["source_sync_model_id"])), "sync model")
        for side in ("source", "target"):
            if any(model[f"{side}_{f}"] != mapping[f"{side}_{f}"] for f in ("run_id", "device_type")):
                raise ValueError("Sync model and mapping acquisition identities disagree")
        rt = _timeline(conn, rk, model["target_timeline_model_id"])
        pt = _timeline(conn, pk, model["source_timeline_model_id"])
        if rt["kind"] != pt["kind"]:
            raise ValueError("Incompatible timeline coordinates")
        params = json.loads(mapping["parameters_json"])
        if rt["kind"] == "numeric" and model["model_type"] == "identity_time" and params.get("allow_numeric_identity") is not True:
            raise ValueError("Numeric identity timelines require explicit compatibility in the mapping")
        for side in ("source", "target"):
            if params.get(f"{side}_timeline_model_id") != model[f"{side}_timeline_model_id"]:
                raise ValueError("Mapping timeline selection differs from sync model")
        tolerance = min(float(params["weak_support_threshold_ms"]), float(params["max_allowed_delta_ms"]))
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Invalid correspondence tolerance")
        profile = {"model_type": model["model_type"], "mapping_method": mapping["mapping_method"],
                   "parameters": {k: v for k, v in params.items() if k != "parent_mapping_version_id"},
                   "extrapolation_policy": model["extrapolation_policy"],
                   "timeline_types": [pt["model"]["timeline_model_type"], rt["model"]["timeline_model_type"]],
                   "timeline_time_bases": [pt["model"].get("source_time_basis"), rt["model"].get("source_time_basis")],
                   "correspondence": "nearest_predicted_kinect_time_v1", "tolerance_ms": tolerance,
                   "support_policy": "supported_only_no_extrapolation", "gap_factor": gap_factor}
        for k in ("primary_policy", "source_window_policy", "extrapolation_policy"):
            if k in profile["parameters"]:
                profile["parameters"][k] = str(profile["parameters"][k]).replace("-", "_")
        profiles.append(profile)
        rr = _one(_rows(conn, "DEVICE_RUN", RUN, rk), "raw run")
        pr = _one(_rows(conn, "DEVICE_RUN", RUN, pk), "Kinect run")
        periods = [1 / float(r["nominal_fps"]) for r in (rr, pr)]
        if not np.isfinite(periods).all() or min(periods) <= 0:
            raise ValueError("Runs require positive nominal frame rates")
        predicted = _predict(model, pt["time"])
        usable = np.flatnonzero(np.isfinite(predicted))
        if len(usable) == 0 or np.any(np.diff(predicted[usable]) <= 0):
            raise ValueError("No monotonic supported Kinect timeline")
        context = dict(selection=selected, mapping=mapping, sync_model=model, cloud=cloud,
                       radar_timeline=rt, pose_timeline=pt, predicted=predicted,
                       radar_key=rk, pose_key=pk, periods=periods, tolerance_ms=tolerance,
                       segments=[], anchors=_rows(conn, "MODEL_ANCHOR", "subject_id=? AND sync_model_id=?",
                                                 (subject, model["sync_model_id"])))
        contexts.append(context)
        lo = max(rt["time"][0], predicted[usable[0]])
        hi = min(rt["time"][-1], predicted[usable[-1]])
        r = np.flatnonzero((rt["time"] >= lo) & (rt["time"] <= hi))
        p = np.flatnonzero(np.isfinite(predicted) & (predicted >= lo) & (predicted <= hi))
        if not len(r) or not len(p):
            raise ValueError(f"Selected mapping has no supported overlap: {selected}")
        # Files correspond to run pairs. Cadence gaps create logical sequence
        # boundaries inside each file, without discarding frames in the overlap.
        boundaries = []
        for timeline, positions, aligned, period in ((rt, r, rt["time"], periods[0]),
                                                      (pt, p, predicted, periods[1])):
            breaks = np.flatnonzero((np.diff(timeline["indices"][positions]) != 1)
                        | (np.diff(timeline["time"][positions]) > period * gap_factor))
            boundaries.extend((aligned[positions[breaks]] + aligned[positions[breaks + 1]]) / 2)
        # Never duplicate a native frame across overlapping run pairs.
        for claimed, key, ids in ((claimed_radar, rk, rt["indices"][r]), (claimed_pose, pk, pt["indices"][p])):
            seen = claimed.setdefault(key, set())
            if seen.intersection(map(int, ids)):
                raise ValueError("Ambiguous overlapping segments would duplicate acquisition frames")
            seen.update(map(int, ids))
        sid = "segment_" + json_hash({**selected, "radar_first": int(rt["indices"][r[0]]),
                                     "pose_first": int(pt["indices"][p[0]])})[:20]
        segment = dict(segment_id=sid, context=context, radar_rows=r, pose_rows=p, origin=lo,
                       sequence_boundaries=sorted(set(boundaries)))
        segments.append(segment)
        context["segments"].append(sid)
    if len({json_hash(r) for r in recipes}) != 1:
        raise ValueError("Mixed processing recipes: settings, generator or calibration differ")
    if len({json_hash(p) for p in profiles}) != 1:
        raise ValueError("Mixed mapping profiles/methods are not allowed")
    # Subject selection is complete for raw acquisitions; no implicit skipped run.
    for subject in subjects:
        raw_runs = _rows(conn, "DEVICE_RUN", "subject_id=? AND device_type='radar_raw'", (subject,))
        for run in raw_runs:
            if (subject, run["run_id"], "radar_raw") not in selected_clouds:
                raise ValueError(f"Missing selection for raw acquisition {subject}/{run['run_id']}")
        # Every registered Kinect/raw pair needs an explicit release selection.
        for m in _rows(conn, "MAPPING_VERSION", "subject_id=? AND source_device_type='kinect_rgb' AND target_device_type='radar_raw'", (subject,)):
            pair = ((subject, m["target_run_id"], "radar_raw"), (subject, m["source_run_id"], "kinect_rgb"))
            if pair not in pairs:
                raise ValueError(f"Missing mapping selection for run pair {pair}")
        # A new Kinect restart without any mapping row must not vanish silently.
        for run in _rows(conn, "DEVICE_RUN", "subject_id=? AND device_type='kinect_rgb'", (subject,)):
            key = (subject, run["run_id"], "kinect_rgb")
            if any(c["pose_key"] == key for c in contexts):
                continue
            t = _timeline(conn, key, contexts[0]["sync_model"]["source_timeline_model_id"])
            possible_overlap = profiles[0]["model_type"] != "identity_time"
            for c in contexts:
                if c["radar_key"][0] == subject:
                    r = c["radar_timeline"]
                    possible_overlap |= (t["kind"] != r["kind"] or
                        max(t["time"][0], r["time"][0]) <= min(t["time"][-1], r["time"][-1]))
            if possible_overlap:
                raise ValueError(f"Missing mapping selection for Kinect acquisition {key}")
    return segments, contexts, recipes[0], profiles[0]


def _dataset(group, name, values):
    a = np.asarray(values)
    return group.create_dataset(name, data=a, **({"compression": "gzip", "shuffle": True} if a.size else {}))


def _nearest(radar, pose, tolerance_ms):
    right = np.searchsorted(pose, radar).clip(0, len(pose) - 1)
    left = (right - 1).clip(0)
    # Earlier frame wins exact ties.
    chosen = np.where(np.abs(pose[left] - radar) <= np.abs(pose[right] - radar), left, right)
    residual = (pose[chosen] - radar) * 1000
    chosen[np.abs(residual) > tolerance_ms + 1e-6] = -1
    return chosen, residual


def _bundle(conn, root, key, role, version, cache):
    rows = _rows(conn, "SAMPLE_ARTIFACT", RUN + " AND artifact_role=? AND point_cloud_version_id=?",
                 (*key, role, version))
    by_index = {}
    for row in rows:
        idx = int(row["sample_index"])
        if idx in by_index or row["artifact_format"] != "ragged_npz" or row["storage_key"] != "artifact_store":
            raise ValueError("Invalid or duplicate sample artifact")
        by_index[idx] = row
    readers = {}
    for ref in {r["artifact_ref"] for r in rows}:
        path = _path(root, ref)
        if path not in cache:
            cache[path] = file_hash(path)
        reader = RaggedNpzReader(path)
        if reader.validate():
            raise ValueError(f"Invalid ragged bundle: {ref}")
        readers[ref] = reader
    return by_index, readers


def _write_segment(conn, root, folder, segment, recipe_id, profile_id, geometry, hashes):
    import h5py

    c = segment["context"]
    sid, rr, pp = segment["segment_id"], segment["radar_rows"], segment["pose_rows"]
    rt, pt = c["radar_timeline"], c["pose_timeline"]
    rid, pid = rt["indices"][rr], pt["indices"][pp]
    radar_time, pose_time = rt["time"][rr] - segment["origin"], c["predicted"][pp] - segment["origin"]
    nearest, residual = _nearest(radar_time, pose_time, c["tolerance_ms"])
    boundaries = np.asarray(segment["sequence_boundaries"]) - segment["origin"]
    radar_blocks = np.searchsorted(boundaries, radar_time, side="right")
    pose_blocks = np.searchsorted(boundaries, pose_time, side="right")
    nearest[radar_blocks != pose_blocks[nearest.clip(0)]] = -1
    common = dict(schema=SCHEMA, segment_id=sid, subject_id=c["radar_key"][0],
                  radar_run_id=c["radar_key"][1], pose_run_id=c["pose_key"][1],
                  processing_recipe_id=recipe_id, mapping_profile_id=profile_id,
                  point_cloud_version_id=c["selection"]["point_cloud_version_id"],
                  mapping_version_id=c["mapping"]["mapping_version_id"],
                  aligned_origin=segment["origin"], time_coordinate_kind=rt["kind"],
                  geometry_json=json.dumps(geometry, sort_keys=True))
    result = dict(segment_id=sid, subject_id=c["radar_key"][0], radar_run_id=c["radar_key"][1],
                  pose_run_id=c["pose_key"][1], radar_frames=len(rid), pose_frames=len(pid),
                  unmatched_radar_frames=int(np.count_nonzero(nearest < 0)),
                  aligned_origin=segment["origin"], sequence_boundaries_s=boundaries.tolist(), files={})
    pose_counts = None
    for modality, key, ids, timeline, positions, role, version in (
        ("pose", c["pose_key"], pid, pt, pp, "pose3d", ""),
        ("radar", c["radar_key"], rid, rt, rr, "radar_points", common["point_cloud_version_id"]),
    ):
        metadata, readers = _bundle(conn, root, key, role, version, hashes)
        summaries = _rows(conn, "SAMPLE_SUMMARY", RUN + " AND point_cloud_version_id=?", (*key, version))
        if len({int(r["sample_index"]) for r in summaries}) != len(summaries):
            raise ValueError("Duplicate sample summaries")
        summary = {int(r["sample_index"]): r for r in summaries}
        if modality == "radar":
            cloud_path = _path(root, c["cloud"]["artifact_ref"])
            if cloud_path not in hashes:
                hashes[cloud_path] = file_hash(cloud_path)
            if hashes[cloud_path] != c["cloud"]["artifact_sha256"]:
                raise ValueError("Selected raw bundle checksum mismatch")
            if any(row["artifact_ref"] != c["cloud"]["artifact_ref"] for row in metadata.values()):
                raise ValueError("Raw sample artifacts disagree with selected cloud bundle")
        filename = sid + f".{modality}.h5"
        lengths, available, counts = [], [], []
        for idx in ids:
            row = metadata.get(int(idx))
            sm = summary.get(int(idx), {})
            status = sm.get("point_status") if modality == "radar" else ("available" if row else "missing")
            if modality == "radar" and (status not in {"available", "unprocessed"} or (status == "available") != (row is not None)):
                raise ValueError("Raw processing status and artifacts disagree")
            arr = readers[row["artifact_ref"]].get(int(idx)) if row else None
            n = len(arr) if arr is not None else 0
            if arr is not None and arr.shape[1:] != ((6,) if modality == "radar" else (32, 4)):
                raise ValueError("Unexpected point/pose payload shape")
            if modality == "radar" and row and int(sm["point_count"]) != n:
                raise ValueError("Raw point count differs from payload")
            lengths.append(n)
            available.append(row is not None)
            value = sm.get("num_people")
            counts.append(-1 if value is None else int(value))
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        with h5py.File(folder / filename, "w") as h:
            for k, v in {**common, "modality": modality,
                         "timeline_model_id": timeline["model"]["timeline_model_id"],
                         "timeline_origin": float(timeline["time"][0]),
                         "nominal_sample_period_s": c["periods"][0 if modality == "radar" else 1]}.items():
                h.attrs[k] = v
            frames = h.create_group("frames")
            _dataset(frames, "sample_index", ids)
            _dataset(frames, "frame_number", timeline["frame_number"][positions])
            _dataset(frames, "timeline_time_s", timeline["time"][positions] - timeline["time"][0])
            _dataset(frames, "aligned_time_s", radar_time if modality == "radar" else pose_time)
            _dataset(frames, "sequence_block", radar_blocks if modality == "radar" else pose_blocks)
            _dataset(frames, "payload_available", np.asarray(available, dtype=np.bool_))
            arrays = h.create_group("points" if modality == "radar" else "poses")
            _dataset(arrays, "offsets", offsets)
            shape = (int(offsets[-1]), 6) if modality == "radar" else (int(offsets[-1]), 32, 3)
            values = arrays.create_dataset("values" if modality == "radar" else "xyz", shape=shape, dtype="float32",
                                          **({"compression": "gzip", "shuffle": True} if shape[0] else {}))
            if modality == "pose":
                confidence = arrays.create_dataset("confidence", shape=(shape[0], 32), dtype="float32")
                valid = arrays.create_dataset("finite_xyz", shape=(shape[0], 32), dtype="bool")
                arrays.attrs["joint_names_json"] = json.dumps(KINECT_JOINT_NAMES)
                _dataset(frames, "num_people", np.asarray(counts, dtype=np.int64))
                _dataset(frames, "pose_person_count", np.asarray(lengths, dtype=np.int64))
                pose_counts = np.asarray(lengths)
            else:
                arrays.attrs["columns_json"] = json.dumps(["x", "y", "z", "radial_velocity", "snr", "association_id"])
                arrays.attrs["units_json"] = json.dumps(["m", "m", "m", *json.loads(c["cloud"]["provenance_json"])["version"]["generator_metadata"]["point_units"][3:], "identifier"])
                _dataset(frames, "point_count", np.where(available, lengths, -1))
                _dataset(frames, "point_status", np.asarray([b"available" if a else b"unprocessed" for a in available], dtype="S11"))
                link = h.create_group("correspondence")
                link.attrs["pose_file"] = sid + ".pose.h5"
                link.attrs["residual_definition"] = "predicted Kinect time minus radar timeline time"
                link.attrs["tolerance_ms"] = c["tolerance_ms"]
                _dataset(link, "pose_row", nearest)
                _dataset(link, "pose_sample_index", np.where(nearest >= 0, pid[nearest.clip(0)], -1))
                _dataset(link, "matched", nearest >= 0)
                _dataset(link, "nearest_residual_ms", residual)
                _dataset(link, "pose_payload_available", (nearest >= 0) & (pose_counts[nearest.clip(0)] > 0))
            # Bound transformed arrays; source NPZ format still decompresses per bundle.
            for start in range(0, len(ids), 256):
                stop = min(start + 256, len(ids))
                chunks = [readers[metadata[int(idx)]["artifact_ref"]].get(int(idx)) for idx in ids[start:stop] if int(idx) in metadata]
                if not chunks or offsets[stop] == offsets[start]:
                    continue
                arr = np.concatenate(chunks)
                dest = slice(int(offsets[start]), int(offsets[stop]))
                if modality == "radar":
                    if not np.isfinite(arr).all():
                        raise ValueError("Nonfinite raw points")
                    values[dest] = points_sensor_to_world(arr)
                else:
                    xyz = pose3d_to_world(arr)
                    values[dest] = xyz
                    confidence[dest] = arr[..., 3]
                    valid[dest] = np.isfinite(xyz).all(axis=-1)
        result["files"][modality] = {"path": filename, "sha256": file_hash(folder / filename)}
        result[f"{modality}_missing_payload_frames"] = int(np.count_nonzero(~np.asarray(available)))
        del readers
    return result


def export_training_data(*, sqlite_path, artifact_root, output, selections=None,
                         subjects=None, mapping_version_id=None, point_cloud_version_id=None,
                         point_cloud_label=None, read_only_roots=(), gap_factor=3.0, dry_run=False):
    """Export explicit pair selections, or resolve one mapping/cloud per subject.

    dry_run validates selection, recipes, timeline coverage and output safety;
    bundle contents/checksums are verified during the real export.
    """
    source, root, output = Path(sqlite_path).resolve(), Path(artifact_root).resolve(), Path(output).resolve()
    if not source.is_file() or not root.is_dir():
        raise ValueError("Source database and artifact root must exist")
    protected = [source, root, *(Path(p).resolve() for p in read_only_roots)]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("Output overlaps an input or read-only root")
    if output.exists():
        raise FileExistsError("Training export requires a new output directory")
    if not np.isfinite(gap_factor) or gap_factor <= 1:
        raise ValueError("gap_factor must be greater than one")
    if selections is not None:
        required = {"subject_id", "mapping_version_id", "point_cloud_version_id"}
        if not isinstance(selections, list) or not selections or any(
                not isinstance(s, dict) or set(s) != required or
                any(not isinstance(v, str) or not v.strip() for v in s.values()) for s in selections):
            raise ValueError("Selection must be a nonempty list of subject_id/mapping_version_id/point_cloud_version_id objects")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")  # One consistent read snapshot; never migrate/update source.
        if selections is None:
            if not subjects or not mapping_version_id or bool(point_cloud_version_id) == bool(point_cloud_label):
                raise ValueError("Specify subjects, mapping version, and exactly one cloud ID or exact label")
            selections = []
            for subject in subjects:
                m = _one(_rows(conn, "MAPPING_VERSION", "subject_id=? AND mapping_version_id=?",
                               (subject, mapping_version_id)), "selected mapping")
                key = (subject, m["target_run_id"], m["target_device_type"])
                column = "point_cloud_version_id" if point_cloud_version_id else "readable_label"
                v = _one(_rows(conn, "POINT_CLOUD_VERSION", RUN + f" AND {column}=?",
                               (*key, point_cloud_version_id or point_cloud_label)), "selected cloud")
                selections.append(dict(subject_id=subject, mapping_version_id=mapping_version_id,
                                       point_cloud_version_id=v["point_cloud_version_id"]))
        selections = sorted(selections, key=lambda s: (s["subject_id"], s["mapping_version_id"], s["point_cloud_version_id"]))
        segments, contexts, recipe, profile = _prepare(conn, selections, gap_factor)
        recipe_id, profile_id = "recipe_" + json_hash(recipe), "mapping_profile_" + json_hash(profile)
        geometry = _geometry()
        bindings, coverage = [], []
        for c in contexts:
            bindings.append(dict(selection=c["selection"], mapping=c["mapping"], sync_model=c["sync_model"],
                                 cloud=c["cloud"], model_anchors=c["anchors"],
                                 timelines={kind: {"model": c[kind]["model"], "fingerprint": c[kind]["fingerprint"]}
                                            for kind in ("radar_timeline", "pose_timeline")}))
            for kind in ("radar", "pose"):
                timeline = c[f"{kind}_timeline"]
                used = set()
                for seg in segments:
                    if seg["context"] is c:
                        used.update(map(int, seg[f"{kind}_rows"]))
                excluded = [int(idx) for i, idx in enumerate(timeline["indices"]) if i not in used]
                coverage.append(dict(subject_id=c[f"{kind}_key"][0], run_id=c[f"{kind}_key"][1],
                                     modality=kind, mapping_version_id=c["mapping"]["mapping_version_id"],
                                     total_frames=len(timeline["indices"]), exported_frames=len(used),
                                     excluded_sample_ranges=_ranges(excluded),
                                     reason="outside selected supported overlap"))
        release = "sync_release_" + json_hash([{k: b[k] for k in ("selection", "mapping", "sync_model", "timelines", "model_anchors")} for b in bindings])
        for subject in sorted({s["subject_id"] for s in selections}):
            selected_pose = {c["pose_key"] for c in contexts}
            for run in _rows(conn, "DEVICE_RUN", "subject_id=? AND device_type='kinect_rgb'", (subject,)):
                key = (subject, run["run_id"], "kinect_rgb")
                if key not in selected_pose:
                    ids = sorted(int(r["sample_index"]) for r in _rows(conn, "RUN_SAMPLE", RUN, key))
                    coverage.append(dict(subject_id=subject, run_id=key[1], modality="pose", total_frames=len(ids),
                                         exported_frames=0, excluded_sample_ranges=_ranges(ids),
                                         reason="no initial-timeline overlap with selected raw acquisitions"))
        manifest = dict(schema=SCHEMA, subjects=sorted({s["subject_id"] for s in selections}),
                        processing_recipe_id=recipe_id, processing_recipe=recipe,
                        mapping_profile_id=profile_id, mapping_profile=profile,
                        synchronization_release_id=release, geometry=geometry, selections=bindings,
                        coverage=coverage, scope="supported overlapping segments; 3D Kinect poses and offline raw clouds",
                        timestamp_policy="run-local timeline_time_s; common segment-local aligned_time_s in radar timeline; datetime origins are UTC-like naive coordinates, not verified UTC",
                        missing_policy="unprocessed/missing payloads retained; unavailable counts and references = -1",
                        person_policy="original per-frame ordering; no identity tracking or single-person filtering")
        if dry_run:
            return dict(dry_run=True, segment_count=len(segments), payload_file_count=2 * len(segments),
                        processing_recipe_id=recipe_id, mapping_profile_id=profile_id,
                        selections=selections, coverage=coverage)
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('Install the export dependency: python -m pip install -e ".[hpe-export]"') from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".syncwb-training-", dir=output.parent) as temp:
            staging = Path(temp)
            hashes = {}
            manifest["segments"] = [_write_segment(conn, root, staging, s, recipe_id, profile_id, geometry, hashes) for s in segments]
            # Detect changes during export instead of publishing inconsistent payloads.
            for path, digest in hashes.items():
                if file_hash(path) != digest:
                    raise ValueError("Source artifact changed during export")
            manifest["source_artifacts"] = [{"artifact_ref": p.relative_to(root).as_posix(), "sha256": d} for p, d in sorted(hashes.items())]
            manifest["exporter_sha256"] = file_hash(Path(__file__))
            manifest["geometry_implementation_sha256"] = file_hash(Path(__file__).parents[1] / "core" / "geometry.py")
            manifest["dataset_fingerprint"] = json_hash(manifest)
            manifest["created_at"] = utc_now_str()
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            (staging / "README.txt").write_text(
                "SyncWB training export v1. manifest.json is the completion marker.\n"
                "Each segment has independent radar/pose HDF5 files; windows must stay within segment_id + frames/sequence_block.\n"
                "Both use approximate GUI world geometry in metres; do not transform again.\n"
                "Ragged values use offsets[i]:offsets[i+1]. Pose ordering is not track identity.\n"
                "Radar correspondence/pose_row references its paired pose file, with -1 for no match.\n"
                "Use frames/aligned_time_s for cross-modal relative times; preserve missing-frame gaps.\n"
                "Inspect manifest provenance and confidence before using Kinect estimates as supervision.\n",
                encoding="utf-8")
            output.mkdir()  # Exclusive reservation; never replace an existing destination.
            try:
                for file in sorted(staging.iterdir(), key=lambda p: p.name == "manifest.json"):
                    file.rename(output / file.name)
            except Exception:
                # This exact directory was created above, and output safety was checked.
                shutil.rmtree(output)
                raise
        return dict(output=str(output), dataset_fingerprint=manifest["dataset_fingerprint"],
                    segment_count=len(segments), payload_file_count=2 * len(segments))
