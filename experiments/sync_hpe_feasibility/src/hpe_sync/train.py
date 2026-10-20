from __future__ import annotations

import json
import random
import time

import numpy as np

from .common import (CONDITIONS, TASKS, code_hash, file_hash, nearest, safe_output,
                     split_blocks, standardize_fit, write_json)


def job_id(job):
    return f"{job['task']}__{job['model']}__w{job['context']}__{job['condition']}__s{job['seed']}"


def make_plan(config, suite):
    jobs = []
    if suite == "smoke":
        specs = [(task, model, 1, "reference", config["seeds"][0])
                 for task in TASKS for model in ("pointnet", "pose_control")]
        specs += [("image", "stats_mlp", 1, "reference", config["seeds"][0]),
                  ("shape", "pointnet", config["max_context_frames"], "shift_p200", config["seeds"][0]),
                  ("shape", "pose_control", 1, "shift_p200", config["seeds"][0])]
    elif suite == "core":
        specs = [(task, model, 1, condition, seed) for task in TASKS
                 for model in ("pointnet", "pose_control")
                 for condition in CONDITIONS for seed in config["seeds"]]
        specs += [(task, "stats_mlp", 1, "reference", seed)
                  for task in TASKS for seed in config["seeds"]]
    elif suite == "temporal":
        specs = [(task, "pointnet", config["max_context_frames"], condition, seed)
                 for task in TASKS for condition in ("reference", "shift_m200", "shift_p200")
                 for seed in config["seeds"]]
    else:
        raise ValueError("Unknown suite")
    for task, model, context, condition, seed in specs:
        jobs.append(dict(task=task, model=model, context=context, condition=condition,
                         seed=seed, smoke=suite == "smoke"))
    return jobs


def load_prepared(config):
    out = safe_output(config["output"])
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Configuration differs from prepared data. Prepare a new output directory.")
    path = out / "prepared.npz"
    if file_hash(path) != manifest["prepared_sha256"]:
        raise ValueError("Prepared data checksum mismatch")
    with np.load(path, allow_pickle=False) as b:
        data = {k: b[k] for k in b.files}
    return data, manifest


def build_model(torch, kind, context, output_dim):
    nn = torch.nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "pointnet":
                self.point_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 96), nn.ReLU())
                features = 96 * 2 + 2
            else:
                features = 104 * 2 if kind == "pose_control" else context * 28
            self.head = nn.Sequential(nn.Linear(features, 128), nn.ReLU(), nn.Dropout(.1),
                                      nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_dim))

        def forward(self, x, mask=None, extra=None):
            if kind == "pointnet":
                h = self.point_encoder(x)
                w = mask[..., None].to(h.dtype)
                mean = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1)
                maximum = h.masked_fill(~mask[..., None], -1e6).max(dim=1).values
                maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
                x = torch.cat((mean, maximum, extra), dim=1)
            return self.head(x)

    return Model()


class Inputs:
    """One fixed input tensor per sample, independent of training-label condition."""
    def __init__(self, data, task, context, train_ids, kind, ymean, yscale, active):
        self.d, self.kind, self.context = data, kind, context
        windows = data["centers"][train_ids, None] + np.arange(1 - context, 1)
        frame_ids = np.unique(windows)
        self.point_mean, self.point_scale, _ = standardize_fit(
            data["points"][frame_ids][data["point_mask"][frame_ids]])
        self.stat_mean, self.stat_scale, _ = standardize_fit(data["stats"][frame_ids])
        self.ymean, self.yscale, self.active = ymean, yscale, active
        self.task = task

    def batch(self, ids, torch, device):
        d, kind = self.d, self.kind
        win = d["centers"][ids, None] + np.arange(1 - self.context, 1)
        dt = d["radar_time"][win] - d["radar_time"][d["centers"][ids, None]]
        tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
        if kind == "pose_control":
            row = d["label_rows"][0, ids]
            mask = d[f"mask_{self.task}"][row] & self.active
            x = np.where(mask, (d[f"target_{self.task}"][row] - self.ymean) / self.yscale, 0)
            return (tensor(np.concatenate((x, mask.astype("float32")), axis=1)),)
        if kind == "stats_mlp":
            x = (d["stats"][win] - self.stat_mean) / self.stat_scale
            return (tensor(np.concatenate((x, dt[..., None]), axis=-1).reshape(len(ids), -1)),)
        p = (d["points"][win] - self.point_mean) / self.point_scale
        mask = d["point_mask"][win]
        p = np.where(mask[..., None], p, 0)
        dt = np.broadcast_to(dt[..., None, None], (*p.shape[:-1], 1))
        x = np.concatenate((p, dt), axis=-1).reshape(len(ids), -1, 6)
        extra = np.stack((np.log1p(d["counts"][d["centers"][ids]]) / 6,
                          mask.mean(axis=(1, 2))), axis=1)
        return tensor(x), tensor(mask.reshape(len(ids), -1), torch.bool), tensor(extra)


def train_job(data, manifest, config, job):
    import torch
    started = time.monotonic()
    root = safe_output(config["output"]) / ("smoke_runs" if job["smoke"] else "runs")
    folder = root / job_id(job)
    stamp = {"preparation_id": manifest["preparation_id"], "code_sha256": code_hash(), "job": job}
    if (folder / "result.json").exists():
        old = json.loads((folder / "result.json").read_text())
        if old["stamp"] != stamp:
            raise ValueError(f"Stale results at {folder}; use a new output directory")
        print(f"SKIP completed {job_id(job)}", flush=True)
        return old
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(job["seed"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = job["task"]
    cohort = data[f"cohort_{task}"]
    ids = [np.flatnonzero(cohort & (data["split"] == i)) for i in range(3)]
    if job["smoke"]:
        ids = [v[:min(len(v), 256)] for v in ids]
    tr, va, te = ids
    if not all(len(v) >= 16 for v in ids):
        raise ValueError("Not enough samples in each split")
    ref_rows = data["label_rows"][0]
    target = data[f"target_{task}"]
    common_mask = data[f"common_mask_{task}"]
    mean, scale, active = standardize_fit(target[ref_rows[tr]], common_mask[tr])
    mask = common_mask & active
    # Every condition must have at least one active target per retained row.
    ids = [v[mask[v].any(axis=1)] for v in ids]
    tr, va, te = ids
    if not all(len(v) >= 16 for v in ids):
        raise ValueError("Too few samples after training-only active-feature selection")
    inputs = Inputs(data, task, job["context"], tr, job["model"], mean, scale, active)
    model = build_model(torch, job["model"], job["context"], target.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    condition_index = CONDITIONS.index(job["condition"])
    train_y = (target[data["label_rows"][condition_index]] - mean) / scale
    clean_y = (target[ref_rows] - mean) / scale
    batch_size = config["batch_size"]

    @torch.no_grad()
    def predict(which):
        model.eval()
        return np.concatenate([model(*inputs.batch(which[k:k + batch_size], torch, device)).cpu().numpy()
                               for k in range(0, len(which), batch_size)])

    def mse(pred, yy, mm):
        return (np.where(mm, (pred - yy) ** 2, 0).sum(axis=1) / mm.sum(axis=1).clip(1))

    epochs = 2 if job["smoke"] else config["epochs"]
    best, patience, history, best_state = float("inf"), 0, [], None
    rng = np.random.default_rng(job["seed"])
    print(f"TRAIN {job_id(job)} on {device}: {len(tr)}/{len(va)}/{len(te)} samples", flush=True)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(tr)
        losses = []
        for k in range(0, len(order), batch_size):
            ix = order[k:k + batch_size]
            pred = model(*inputs.batch(ix, torch, device))
            yy = torch.as_tensor(train_y[ix], dtype=torch.float32, device=device)
            mm = torch.as_tensor(mask[ix], dtype=torch.float32, device=device)
            loss = (((pred - yy) ** 2 * mm).sum(dim=1) / mm.sum(dim=1).clamp_min(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = float(mse(predict(va), clean_y[va], mask[va]).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_mse": val})
        if val < best - 1e-5:
            best, patience = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1}: validation standardized MSE={val:.4f}", flush=True)
        if patience >= config["patience"]:
            break
    model.load_state_dict(best_state)
    pred = predict(te)
    errors = mse(pred, clean_y[te], mask[te])
    mean_errors = mse(np.zeros_like(pred), clean_y[te], mask[te])
    raw_pred = pred * scale + mean
    raw_error = np.where(mask[te], np.abs(raw_pred - target[ref_rows[te]]), 0).sum(axis=1) / mask[te].sum(axis=1)
    np.savez_compressed(folder / "test_predictions.npz", sample_id=te, prediction=raw_pred,
                        mask=mask[te], standardized_mse=errors, mean_baseline_mse=mean_errors,
                        raw_mae=raw_error, block=data["block"][te],
                        people=data["expected_people"][ref_rows[te]],
                        total_points=data["counts"][data["centers"][te]])
    torch.save(best_state, folder / "weights.pt")
    np.savez(folder / "normalization.npz", target_mean=mean, target_scale=scale, active=active,
             point_mean=inputs.point_mean, point_scale=inputs.point_scale,
             stats_mean=inputs.stat_mean, stats_scale=inputs.stat_scale)
    write_json(folder / "history.json", history)

    # Freeze the learned predictor and vary evaluation pairing. This measures label
    # sensitivity, not an independent estimate of true sensor lag.
    sweeps = []
    offsets = [-.5, -.3, -.2, -.1, -.05, 0., .05, .1, .2, .3, .5]
    sweep_rows, sweep_masks = [], []
    for offset in offsets:
        row, match, _ = nearest(data["pose_time"], data["reference_time"][te] + offset,
                                config["pose_match_tolerance_s"])
        block, split, buffered = split_blocks(data["pose_time"][row], config)
        valid = (match & buffered & (block == data["block"][te]) & (split == 2)
                 & (data["expected_people"][row] == data["expected_people"][ref_rows[te]]))
        sweep_rows.append(row)
        sweep_masks.append(data[f"mask_{task}"][row] & mask[te] & valid[:, None])
    fixed = np.logical_and.reduce(sweep_masks)
    informative = fixed.sum(axis=1) >= 2 * config["minimum_valid_joints"]
    for offset, row in zip(offsets, sweep_rows):
        yy = (target[row] - mean) / scale
        ee = mse(pred, yy, fixed)
        dd = mse(clean_y[te], yy, fixed)
        sweeps.append({"requested_offset_ms": int(round(offset * 1000)),
                       "samples": int(informative.sum()),
                       "model_mse": float(ee[informative].mean()) if informative.any() else None,
                       "label_displacement_mse": float(dd[informative].mean()) if informative.any() else None,
                       "unchanged_pose_row_fraction": float((row[informative] == ref_rows[te[informative]]).mean())
                       if informative.any() else None})
    # Validation-defined low/high total-return groups; they are not human-return counts.
    thresholds = np.quantile(data["counts"][data["centers"][tr]], [.25, .75])
    subsets = {"one_person": data["expected_people"][ref_rows[te]] == 1,
               "multiple_people": data["expected_people"][ref_rows[te]] > 1,
               "low_total_returns": data["counts"][data["centers"][te]] <= thresholds[0],
               "high_total_returns": data["counts"][data["centers"][te]] >= thresholds[1]}
    strata = {key: {"n": int(v.sum()), "mse": float(errors[v].mean()) if v.any() else None}
              for key, v in subsets.items()}
    result = {"stamp": stamp, "device": device, "torch_version": torch.__version__,
              "seconds": time.monotonic() - started, "epochs": len(history),
              "best_validation_mse": best,
              "test_mse": float(errors.mean()), "mean_baseline_mse": float(mean_errors.mean()),
              "raw_mae": float(raw_error.mean()), "raw_units": "pixels" if task == "image" else "torso lengths",
              "train_samples": len(tr), "validation_samples": len(va), "test_samples": len(te),
              "test_blocks": np.unique(data["block"][te]).tolist(),
              "strata": strata, "frozen_label_sweep": sweeps,
              "warning": "Smoke run only; excluded from study conclusions." if job["smoke"] else None}
    write_json(folder / "result.json", result)  # Completion marker written last.
    print(f"DONE {job_id(job)}: {result['seconds']:.1f}s", flush=True)
    return result
