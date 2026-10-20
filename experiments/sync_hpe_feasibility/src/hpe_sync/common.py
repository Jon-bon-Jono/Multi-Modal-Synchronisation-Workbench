from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

STUDY = Path(__file__).resolve().parents[2]
REPO = STUDY.parents[1]
CONDITIONS = ("reference", "shift_m200", "shift_p200", "jitter_100", "sparse_2", "unrelated")
TASKS = ("image", "shape")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    files = sorted((STUDY / "src").rglob("*.py"))
    files += [STUDY / "run_study.py", REPO / "src/sync_workbench/sync/piecewise_affine.py"]
    return digest({str(p.relative_to(REPO)): file_hash(p) for p in files})


def safe_output(relative):
    p = (STUDY / relative).resolve()
    if not p.is_relative_to(STUDY) or p == STUDY:
        raise ValueError("Output must be a subdirectory of the study directory")
    return p


def load_config(path=None):
    path = Path(path) if path else STUDY / "configs/19_MM.json"
    c = json.loads(path.read_text(encoding="utf-8"))
    safe_output(c["output"])
    if c["point_cloud_version"] != "online_original":
        raise ValueError("This study is explicitly scoped to the online cloud")
    if c["max_context_frames"] < 1 or c["sample_stride"] < 1:
        raise ValueError("Context and stride must be positive")
    if c["boundary_buffer_s"] < 1:
        raise ValueError("A buffer of at least one second is required")
    return c


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {"git_commit": commit, "code_sha256": code_hash(), "created_unix_s": time.time()}


def quantiles(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return None if not len(a) else dict(zip(("min", "median", "p90", "p95", "max"),
                                          np.quantile(a, [0, .5, .9, .95, 1]).tolist()))


def nearest(times, query, tolerance):
    times, query = np.asarray(times), np.asarray(query)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Timeline must be nonempty and strictly increasing")
    right = np.searchsorted(times, query).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    idx = np.where(np.abs(times[left] - query) <= np.abs(times[right] - query), left, right)
    residual = times[idx] - query
    valid = (np.isfinite(query) & (query >= times[0]) & (query <= times[-1])
             & (np.abs(residual) <= tolerance))
    return idx, valid, residual


def inverse_map(radar_time, anchors):
    source = np.array([a["source_time"] for a in anchors])
    target = np.array([a["target_time"] for a in anchors])
    if len(source) < 2 or np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0):
        raise ValueError("Anchor coordinates must be strictly monotonic")
    return np.interp(radar_time, target, source, left=np.nan, right=np.nan)


def split_blocks(times, config):
    block = np.floor(times / config["block_seconds"]).astype(int)
    pattern = {"train": 0, "validation": 1, "test": 2}
    labels = np.array([pattern[x] for x in config["split_pattern"]])
    split = labels[block % len(labels)]
    within = times - block * config["block_seconds"]
    buffer = config["boundary_buffer_s"]
    valid = (within >= buffer) & (within <= config["block_seconds"] - buffer)
    return block, split, valid


def standardize_fit(values, mask=None):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) if mask is None else np.asarray(mask) & np.isfinite(values)
    n = valid.sum(axis=0)
    mean = np.where(valid, values, 0).sum(axis=0) / np.maximum(n, 1)
    var = np.where(valid, (values - mean) ** 2, 0).sum(axis=0) / np.maximum(n, 1)
    scale = np.sqrt(var)
    active = (n >= 8) & (scale > 1e-6)
    return mean.astype("float32"), np.where(active, scale, 1).astype("float32"), active


def block_bootstrap_difference(a, b, blocks, repeats=2000, seed=304):
    """CI for paired episode/block means; never resample individual frames."""
    a, b, blocks = map(np.asarray, (a, b, blocks))
    valid = np.isfinite(a) & np.isfinite(b)
    groups = np.unique(blocks[valid])
    deltas = np.array([np.mean((a - b)[valid & (blocks == g)]) for g in groups])
    if not len(deltas):
        return {"blocks": 0, "mean": None, "ci95": None}
    result = {"blocks": len(groups), "mean": float(deltas.mean()), "ci95": None}
    if len(groups) >= 3:
        rng = np.random.default_rng(seed)
        means = deltas[rng.integers(len(deltas), size=(repeats, len(deltas)))].mean(axis=1)
        result["ci95"] = np.quantile(means, [.025, .975]).tolist()
    return result
