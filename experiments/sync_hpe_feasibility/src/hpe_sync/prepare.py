from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from sync_workbench.storage.ragged_npz import RaggedNpzReader
from sync_workbench.sync.piecewise_affine import AnchorPair, fit_piecewise_affine
from .common import (CONDITIONS, REPO, TASKS, digest, file_hash, inverse_map, nearest,
                     provenance, quantiles, safe_output, split_blocks, write_json)
from .targets import pose_targets, radar_features


def rows(conn, table, where, args):
    return [dict(r) for r in conn.execute(f'SELECT * FROM "{table}" WHERE {where}', args)]


def bundle(conn, key, role, version, hashes):
    rr = rows(conn, "SAMPLE_ARTIFACT",
              "subject_id=? AND run_id=? AND device_type=? AND artifact_role=? AND point_cloud_version_id=?",
              (*key, role, version))
    refs = {r["artifact_ref"] for r in rr}
    if len(refs) != 1 or any(r["storage_key"] != "artifact_store" for r in rr):
        raise ValueError(f"Expected one artifact-store bundle for {role}: {refs}")
    root = (REPO / "artifact_store").resolve()
    path = (root / refs.pop()).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes artifact_store")
    hashes[str(path.relative_to(REPO))] = file_hash(path)
    b = RaggedNpzReader(path)
    if b.validate():
        raise ValueError(f"Invalid bundle: {b.validate()}")
    return b


def timeline(conn, key, model, indices):
    rr = rows(conn, "SAMPLE_TIME_ESTIMATE",
              "subject_id=? AND run_id=? AND device_type=? AND timeline_model_id=?", (*key, model))
    lookup = {int(r["sample_index"]): r for r in rr}
    if len(lookup) != len(rr):
        raise ValueError("Duplicate timeline rows")
    rr = [lookup[int(i)] for i in indices]
    if all(r["time_value_datetime"] for r in rr):
        parsed = pd.to_datetime([r["time_value_datetime"] for r in rr], utc=True).tz_localize(None)
        # pandas may choose microsecond resolution; explicitly convert to ns.
        t = parsed.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
    else:
        t = np.array([float(r["time_value_sec"]) for r in rr])
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timeline is not finite and strictly increasing")
    return t


def make_queries(reference, radar_times, anchors, config):
    """Shifts are in source/pose seconds: positive selects a later pose."""
    queries = {"reference": reference, "shift_m200": reference - .2,
               "shift_p200": reference + .2}
    queries["jitter_100"] = reference + np.random.default_rng(410).normal(0, .1, len(reference))
    keep = sorted(set(range(0, len(anchors), 2)) | {len(anchors) - 1})
    queries["sparse_2"] = inverse_map(radar_times, [anchors[i] for i in keep])
    # Negative control remains in the same split/block, at least 10 seconds away.
    size, margin = config["block_seconds"], config["boundary_buffer_s"]
    start = np.floor(reference / size) * size
    queries["unrelated"] = start + margin + ((reference - start - margin + 10) % (size - 2 * margin))
    return queries


def prepare(config):
    started = time.monotonic()
    out = safe_output(config["output"])
    if (out / "prepared.npz").exists() or (out / "manifest.json").exists():
        raise FileExistsError("Prepared output exists. Use a new output in a copied configuration.")
    subject = config["subject"]
    db = REPO / "workbench.sqlite"
    hashes = {}
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        mv = rows(conn, "MAPPING_VERSION", "subject_id=? AND mapping_version_id=?",
                  (subject, config["mapping_version"]))
        if len(mv) != 1:
            raise ValueError("Mapping version was not uniquely identified")
        mv = mv[0]
        if mv["target_device_type"] != "radar_pc":
            raise ValueError("Online radar mapping required")
        model = rows(conn, "SYNC_MODEL", "subject_id=? AND sync_model_id=?",
                     (subject, mv["source_sync_model_id"]))[0]
        if model["model_type"] != "piecewise_affine":
            raise ValueError("Expected the saved piecewise-affine model")
        params = json.loads(model["parameters_json"])
        pose_key = (subject, mv["source_run_id"], mv["source_device_type"])
        radar_key = (subject, mv["target_run_id"], mv["target_device_type"])
        pose = bundle(conn, pose_key, "pose2d", "", hashes)
        bbox = bundle(conn, pose_key, "conf2d", "", hashes)
        cloud = bundle(conn, radar_key, "radar_points", config["point_cloud_version"], hashes)
        pose_ids, radar_ids = pose.sample_index.copy(), cloud.sample_index.copy()
        if cloud.values.ndim != 2 or cloud.values.shape[1] != 6:
            raise ValueError("Expected online [x,y,z,Doppler,SNR,target_id]")
        pt = timeline(conn, pose_key, model["source_timeline_model_id"], pose.sample_index)
        rt = timeline(conn, radar_key, model["target_timeline_model_id"], cloud.sample_index)
        origin = float(pt[0])
        pt, rt = pt - origin, rt - origin
        anchors = [dict(a, source_time=a["source_time"] - origin,
                        target_time=a["target_time"] - origin) for a in params["anchor_pairs"]]
        members = rows(conn, "ANCHOR_MEMBER", "subject_id=?", (subject,))
        for a in anchors:
            for key, index in ((pose_key, a["source_sample_index"]), (radar_key, a["target_sample_index"])):
                match = [r for r in members if r["anchor_id"] == a["anchor_id"]
                         and r["run_id"] == key[1] and r["device_type"] == key[2]]
                if len(match) != 1 or int(match[0]["sample_index"]) != index:
                    raise ValueError("Saved model is stale relative to current anchor members")
        summary = rows(conn, "SAMPLE_SUMMARY",
                       "subject_id=? AND run_id=? AND device_type=? AND point_cloud_version_id=''", pose_key)
        summary_lookup = {int(r["sample_index"]): r for r in summary}
        expected = np.array([float(summary_lookup[int(i)]["num_people"])
                             if summary_lookup[int(i)]["num_people"] is not None else np.nan
                             for i in pose.sample_index])
        hashes["selected_database_metadata"] = digest({"mapping": mv, "model": model, "summary": summary,
                                                       "pose_times": pt.tolist(), "radar_times": rt.tolist()})

    print("Building 2D targets (person order is ignored)...", flush=True)
    target = {t: np.zeros((len(pt), 104), np.float32) for t in TASKS}
    masks = {t: np.zeros((len(pt), 104), bool) for t in TASKS}
    reasons = Counter()
    for pos, idx in enumerate(pose.sample_index):
        values, reason = pose_targets(pose.get(idx), expected[pos], bbox.get(idx), config)
        reasons[reason] += 1
        for t in TASKS:
            target[t][pos], masks[t][pos] = values[t]
    # Optional reviewed bad-label intervals are in seconds from the RGB timeline origin.
    excluded = np.zeros(len(pt), bool)
    for interval in config.get("excluded_pose_intervals", []):
        excluded |= (pt >= interval["start_s"]) & (pt <= interval["end_s"])
    for t in TASKS:
        masks[t][excluded] = False

    print("Extracting radar features without pose-based crops...", flush=True)
    n, cap = len(rt), config["max_points"]
    points = np.zeros((n, cap, 5), np.float32)
    point_mask = np.zeros((n, cap), bool)
    stats = np.zeros((n, 27), np.float32)
    counts = np.zeros(n, np.int32)
    rng = np.random.default_rng(512)
    for i, idx in enumerate(cloud.sample_index):
        stats[i], valid = radar_features(cloud.get(idx))
        counts[i] = len(valid)
        if len(valid) > cap:
            valid = valid[rng.choice(len(valid), cap, replace=False)]
        points[i, :len(valid)] = valid
        point_mask[i, :len(valid)] = True
        if i and i % 15000 == 0:
            print(f"  {i}/{n} radar frames", flush=True)
    del cloud, pose, bbox

    centers = np.arange(config["max_context_frames"] - 1, n, config["sample_stride"])
    rt_center = rt[centers]
    reference = inverse_map(rt_center, anchors)
    finite_ref = np.isfinite(reference)
    # Avoid integer casts of NaN for unsupported queries; their validity remains false.
    reference_safe = np.where(finite_ref, reference, 0)
    blocks, splits, buffered = split_blocks(reference_safe, config)
    queries = make_queries(reference_safe, rt_center, anchors, config)
    condition_rows, condition_valid, actual_offset = [], [], []
    ref_row, ref_match, _ = nearest(pt, reference, config["pose_match_tolerance_s"])
    for name in CONDITIONS:
        idx, valid, _ = nearest(pt, queries[name], config["pose_match_tolerance_s"])
        label_block, label_split, label_buffer = split_blocks(pt[idx], config)
        valid &= (finite_ref & ref_match & buffered & label_buffer
                  & (label_block == blocks) & (label_split == splits)
                  & (expected[idx] == expected[ref_row]))
        condition_rows.append(idx)
        condition_valid.append(valid)
        actual_offset.append(pt[idx] - pt[ref_row])
    label_rows = np.asarray(condition_rows)
    map_valid = np.asarray(condition_valid)

    # The widest context defines one shared cohort, even for the single-frame model.
    start = centers - config["max_context_frames"] + 1
    context_valid = np.ones(len(centers), bool)
    for k in range(config["max_context_frames"]):
        rr = start + k
        rpose = inverse_map(rt[rr], anchors)
        rb, rs, rv = split_blocks(np.nan_to_num(rpose, nan=-10000), config)
        context_valid &= np.isfinite(rpose) & rv & (rb == blocks) & (rs == splits)
        if k:
            context_valid &= (rt[rr] - rt[rr - 1] <= config["radar_gap_limit_s"])
    # Empty returns are kept (explicit padding/masks); they are reported, not silently filtered.
    arrays = {"points": points, "point_mask": point_mask, "stats": stats, "counts": counts,
              "radar_time": rt, "radar_sample_index": radar_ids}
    arrays.update(centers=centers, reference_time=reference, block=blocks, split=splits,
                  pose_time=pt, label_rows=label_rows, map_valid=map_valid,
                  actual_offset=np.asarray(actual_offset), expected_people=expected,
                  pose_sample_index=pose_ids)
    eligible_counts = {}
    for t in TASKS:
        # Same scalar targets in every condition, as well as the same rows.
        common_mask = np.logical_and.reduce(masks[t][label_rows], axis=0)
        cohort = (context_valid & map_valid.all(axis=0)
                  & (common_mask.sum(axis=1) >= 2 * config["minimum_valid_joints"]))
        arrays[f"target_{t}"] = target[t]
        arrays[f"mask_{t}"] = masks[t]
        arrays[f"common_mask_{t}"] = common_mask
        arrays[f"cohort_{t}"] = cohort
        eligible_counts[t] = {
            "pose_frames_eligible": int(masks[t].any(axis=1).sum()),
            "reference_matched_centers": int((context_valid & map_valid[0] & masks[t][ref_row].any(axis=1)).sum()),
            "common_centers": int(cohort.sum()),
            "splits": {name: int((cohort & (splits == i)).sum())
                       for i, name in enumerate(("train", "validation", "test"))},
            "per_condition_eligible": {name: int((context_valid & map_valid[i]
                                                  & masks[t][label_rows[i]].any(axis=1)).sum())
                                       for i, name in enumerate(CONDITIONS)}
        }
        if any(v < 30 for v in eligible_counts[t]["splits"].values()):
            raise ValueError(f"Too little eligible data for {t}: {eligible_counts[t]}")

    fit = fit_piecewise_affine([AnchorPair(**a) for a in anchors])
    loo = fit.leave_one_out_errors().to_dict("records")
    interior = [r["error_ms"] for r in loo if r["region"] == "interpolation" and r["status"] == "ok"]
    counts2d = np.array([float(summary_lookup[int(i)]["num_2d"]) for i in pose_ids])
    counts3d = np.array([float(summary_lookup[int(i)]["num_3d"]) for i in pose_ids])
    audit = {
        "anchors": len(anchors), "segments": len(fit.segments),
        "anchor_gap_seconds": quantiles(np.diff([a["source_time"] for a in anchors])),
        "segment_slopes": quantiles([s.slope for s in fit.segments]),
        "interior_loao_absolute_ms": quantiles(np.abs(interior)),
        "rgb_interval_seconds": quantiles(np.diff(pt)), "radar_interval_seconds": quantiles(np.diff(rt)),
        "rgb_frames": len(pt), "radar_frames": n,
        "count_valid_2d": int(((counts2d == expected) & (expected > 0)).sum()),
        "count_valid_3d": int(((counts3d == expected) & (expected > 0)).sum()),
        "target_rejection_counts": dict(reasons), "reviewed_excluded_frames": int(excluded.sum()),
        "finite_radar_point_count": quantiles(counts), "empty_radar_frames": int((counts == 0).sum()),
        "coverage": eligible_counts,
        "mapping_actual_offset_ms": {name: quantiles(1000 * np.asarray(actual_offset)[i, map_valid[i]])
                                     for i, name in enumerate(CONDITIONS)},
        "split_blocks": {name: sorted(np.unique(blocks[splits == i]).tolist())
                         for i, name in enumerate(("train", "validation", "test"))},
        "reference_status": "Existing manual-anchor fit; independent timing accuracy has not been established.",
        "label_status": "Automatic count/confidence screening; no new visual label review performed.",
        "spatial_status": "Native radar features and 2D labels; no extrinsic calibration used.",
        "human_return_status": "Counts describe all finite returns, not returns belonging to people."
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "prepared.npz", **arrays)
    pd.DataFrame(loo).to_csv(out / "anchor_leave_one_out.csv", index=False)
    manifest = {"config": config, "conditions": CONDITIONS, "input_hashes": hashes,
                "origin_epoch_s": origin, "anchors": anchors, "audit": audit,
                "provenance": provenance(), "prepare_seconds": time.monotonic() - started}
    manifest["prepared_sha256"] = file_hash(out / "prepared.npz")
    manifest["preparation_id"] = digest({"inputs": hashes, "config": config,
                                       "prepared": manifest["prepared_sha256"],
                                       "code": manifest["provenance"]["code_sha256"]})
    write_json(out / "manifest.json", manifest)
    write_json(out / "audit.json", audit)
    print(f"Prepared {out} in {time.monotonic() - started:.1f}s", flush=True)
    return out
