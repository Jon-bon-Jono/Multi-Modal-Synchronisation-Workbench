from __future__ import annotations

from collections import defaultdict
import json
import math

import numpy as np
import pandas as pd

from .common import (CONDITIONS, TASKS, block_bootstrap_difference, code_hash, nearest,
                     safe_output, split_blocks, standardize_fit, write_json)
from .train import job_id, make_plan


def label_diagnostics(data, config):
    result = {}
    for task in TASKS:
        target, mask = data[f"target_{task}"], data[f"common_mask_{task}"]
        cohort = data[f"cohort_{task}"]
        tr = np.flatnonzero(cohort & (data["split"] == 0))
        te = np.flatnonzero(cohort & (data["split"] == 2))
        ref = data["label_rows"][0]
        mean, scale, active = standardize_fit(target[ref[tr]], mask[tr])
        common = mask[te] & active
        records = []
        for i, name in enumerate(CONDITIONS):
            delta = target[data["label_rows"][i, te]] - target[ref[te]]
            denom = common.sum(axis=1)
            good = denom > 0
            mse = np.where(common, (delta / scale) ** 2, 0).sum(axis=1) / denom.clip(1)
            mae = np.where(common, np.abs(delta), 0).sum(axis=1) / denom.clip(1)
            records.append({"condition": name, "samples": int(good.sum()),
                            "standardized_displacement_mse": float(mse[good].mean()),
                            "raw_displacement_mae": float(mae[good].mean()),
                            "same_pose_row_fraction": float((data["label_rows"][i, te] == ref[te]).mean())})
        result[task] = records
    return result


def fmt(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "unavailable"
    return f"{value:.{digits}f}"


def report(config, data=None, manifest=None):
    out = safe_output(config["output"])
    if manifest is None:
        manifest = json.loads((out / "manifest.json").read_text())
    audit = manifest["audit"]
    if data is not None:
        displacement = label_diagnostics(data, config)
        write_json(out / "label_displacement.json", displacement)
    elif (out / "label_displacement.json").exists():
        displacement = json.loads((out / "label_displacement.json").read_text())
    else:
        displacement = {}
    results, stale = [], []
    for file in sorted((out / "runs").glob("*/result.json")):
        r = json.loads(file.read_text())
        if (r["stamp"]["preparation_id"] != manifest["preparation_id"]
                or r["stamp"]["code_sha256"] != code_hash()):
            stale.append(str(file))
            continue
        results.append((file.parent, r))
    expected = make_plan(config, "core")
    completed = {job_id(r["stamp"]["job"]) for _, r in results}
    missing = [job_id(j) for j in expected if job_id(j) not in completed]
    groups = defaultdict(list)
    for folder, r in results:
        j = r["stamp"]["job"]
        groups[(j["task"], j["model"], j["context"], j["condition"])].append((folder, r))
    comparisons = []
    for key, group in sorted(groups.items()):
        task, model, context, condition = key
        baseline = groups.get((task, model, context, "reference"), [])
        by_seed = {r["stamp"]["job"]["seed"]: (f, r) for f, r in baseline}
        paired = []
        for folder, r in group:
            seed = r["stamp"]["job"]["seed"]
            if seed not in by_seed:
                continue
            other_folder, _ = by_seed[seed]
            with np.load(folder / "test_predictions.npz") as a, np.load(other_folder / "test_predictions.npz") as b:
                if not np.array_equal(a["sample_id"], b["sample_id"]) or not np.array_equal(a["mask"], b["mask"]):
                    raise ValueError("Cannot compare results with different test cohorts or target masks")
                paired.append((a["standardized_mse"], b["standardized_mse"], a["block"]))
        ci = None
        if paired:
            # Average seeds within each block before resampling blocks; seeds do not
            # create new independent observations of the recording.
            if any(not np.array_equal(p[2], paired[0][2]) for p in paired):
                raise ValueError("Seed test-block mismatch")
            ci = block_bootstrap_difference(np.mean([p[0] for p in paired], axis=0),
                                           np.mean([p[1] for p in paired], axis=0), paired[0][2],
                                           config["bootstrap_repeats"])
        comparisons.append({"task": task, "model": model, "context": context,
                            "condition": condition, "seeds": len(group),
                            "mse": float(np.mean([r["test_mse"] for _, r in group])),
                            "mean_baseline_mse": float(np.mean([r["mean_baseline_mse"] for _, r in group])),
                            "paired_difference": ci})
    write_json(out / "comparisons.json", comparisons)
    lines = [
        "# 19_MM synchronization feasibility study",
        "",
        f"Status: {'core training complete' if not missing else 'preliminary; training incomplete'}. "
        f"{len(expected) - len(missing)}/{len(expected)} core runs complete. Smoke runs are excluded.",
        "",
        "## Recommendation",
        "",
        "The available evidence does not establish that temporal synchronization makes this dataset "
        "incapable of radar-based pose estimation. The saved anchor fit is an uncertain reference, "
        "not independently verified timing ground truth. This report tests sensitivity to changed "
        "pairings; it cannot turn that reference into a measurement of the true residual error.",
        "",
    ]
    if missing:
        lines += ["Complete the listed training runs before interpreting the learned tasks. "
                  "No feasibility conclusion is drawn from the smoke tests.", ""]
    else:
        for task in TASKS:
            ref = next((x for x in comparisons if x["task"] == task and x["model"] == "pointnet"
                        and x["context"] == 1 and x["condition"] == "reference"), None)
            if ref:
                gain = 1 - ref["mse"] / max(ref["mean_baseline_mse"], 1e-9)
                lines += [f"For the {task} task, the reference radar model changes standardized MSE "
                          f"by {100 * gain:.1f}% relative to the training-mean predictor "
                          "(positive means improvement)."]
                if gain <= 0:
                    lines += ["This radar task does not demonstrate useful prediction beyond the mean. "
                              "It cannot diagnose synchronization failure: weak human returns, limited "
                              "training coverage, label errors, or model limitations remain explanations."]
                else:
                    lines += ["This is evidence of learnable correspondence for this descriptor and "
                              "recording. It is not full-pose accuracy or a cross-subject result."]
                changed = [x for x in comparisons if x["task"] == task and x["model"] == "pointnet"
                           and x["context"] == 1 and x["condition"] != "reference"]
                for x in changed:
                    ci = x["paired_difference"]
                    if ci and ci["ci95"]:
                        direction = ("higher" if ci["ci95"][0] > 0 else
                                     "lower" if ci["ci95"][1] < 0 else "not resolved as different")
                        lines += [f"With {x['condition']} labels, error is {direction} than reference: "
                                  f"paired block-mean difference {fmt(ci['mean'])}, "
                                  f"95% interval [{fmt(ci['ci95'][0])}, {fmt(ci['ci95'][1])}]."]
                lines += [""]
        lines += [
            "A substantial penalty under added shifts means timing matters at those tested magnitudes. "
            "It does not show that the saved mapping contains those errors. Similar performance under "
            "corruption is also inconclusive if the target or model is insensitive to that corruption.",
            "",
            "Use the pose-input controls to check whether training-label changes cause a measurable "
            "penalty with informative input. They deliberately use 2D pose as input and are not radar "
            "models. Good control performance cannot establish that the online cloud contains enough "
            "human information.",
            "",
            "Before recommending more anchoring, compare independently reviewed timing checkpoints "
            "against the sensitivity range shown here. Do not reject the dataset on this pilot alone.",
            "",
        ]
    lines += [
        "## Data and eligibility",
        "",
        f"- {audit['rgb_frames']} RGB samples and {audit['radar_frames']} online radar samples.",
        f"- {audit['anchors']} anchors; largest gap {audit['anchor_gap_seconds']['max']:.1f} s.",
        f"- Interior leave-one-anchor-out absolute error: median "
        f"{audit['interior_loao_absolute_ms']['median']:.1f} ms; p95 "
        f"{audit['interior_loao_absolute_ms']['p95']:.1f} ms. These are withheld-anchor errors.",
        f"- {audit['count_valid_2d']} RGB frames have the expected nonzero number of 2D detections; "
        f"{audit['count_valid_3d']} have the expected number of 3D detections.",
        "- A pose-count mismatch excludes that frame. Per-joint and bounding-box scores are also "
        "screened. Detector scores are thresholds, not calibrated probabilities.",
        "- Missing 3D poses do not exclude 2D training. No 3D poses supervise these experiments.",
        "- Person order is ignored. No cross-frame tracking or radar-to-person assignment is used.",
        f"- {audit['reviewed_excluded_frames']} frames excluded by supplied review intervals. "
        "Automatic confidence screening does not establish label correctness.",
        "- All finite radar returns are retained before fixed-seed point subsampling. No pose crops "
        "or extrinsic transforms are used. Total-return counts are not counts near a person.",
        "",
        "| Target | Eligible pose frames | Reference matched centers | Common train | Common validation | Common test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, x in audit["coverage"].items():
        lines.append(f"| {task} | {x['pose_frames_eligible']} | {x['reference_matched_centers']} | "
                     f"{x['splits']['train']} | {x['splits']['validation']} | {x['splits']['test']} |")
    lines += [
        "",
        "The common cohort requires every training condition to have the same person count, valid "
        "timing support, and sufficient common target components. This is stricter than operational "
        "coverage. Per-condition eligible counts are in audit.json. Results apply to this retained subset.",
        "",
        "## Experiments",
        "",
        "The image task predicts the mean and spread of each joint's pixel coordinates over the people "
        "in one frame. The shape task first centers each pose on its hips and divides by its shoulder-to-hip "
        "distance, then predicts mean and spread over people. Spread is excluded for single-person frames. "
        "These summaries discard identity and some multiperson information; they are pose-related proxies.",
        "",
        "PointNet uses a shared point MLP, masked mean/max pooling, and an MLP regression head. "
        "The statistics MLP is a simpler radar baseline. The pose-input MLP is an explicitly privileged "
        "control. All transformations and target scales are fitted on training data only. "
        "The optional temporal model uses past frames only.",
        "",
        "Each condition changes training-label times while preserving radar inputs, splits, scalar masks, "
        "initialization seeds, and training budget. Validation and test labels always use the saved "
        "reference. The reference fit uses manual anchors from the recording, including test intervals; "
        "this is an offline alignment experiment, not a test of unseen-session synchronization.",
        "",
        "| Condition | Meaning |",
        "|---|---|",
        "| reference | Current 27-anchor fit; not an independently established optimum |",
        "| shift_m200 / shift_p200 | Pose labels 200 ms earlier/later in the pose clock |",
        "| jitter_100 | Independent Gaussian label-time perturbations, 100 ms standard deviation |",
        "| sparse_2 | Refit using every second current anchor, retaining endpoints |",
        "| unrelated | 10-second within-block cyclic label shift; repeated/static poses may remain similar |",
        "",
        "The jitter magnitude is a controlled stress test, not an estimate of actual timing noise. "
        "The reduced-anchor condition does not simulate a uniform annotation spacing. "
        "Nearest observed poses are used within a 50 ms tolerance; no pose interpolation fills gaps.",
        "",
        "## Label changes without learning",
        "",
        "| Target | Condition | Standardized displacement MSE | Raw MAE | Same pose row |",
        "|---|---|---:|---:|---:|",
    ]
    for task, records in displacement.items():
        for r in records:
            lines.append(f"| {task} | {r['condition']} | {fmt(r['standardized_displacement_mse'])} | "
                         f"{fmt(r['raw_displacement_mae'])} | {100*r['same_pose_row_fraction']:.1f}% |")
    lines += [
        "",
        "Raw units are pixels for image summaries and torso lengths for shape summaries. "
        "These distances need no person association, but averaging people can conceal changes. "
        "They measure disagreement with the saved mapping, not absolute synchronization error.",
        "",
        "## Learned results",
        "",
        "| Task | Model | Frames | Condition | Seeds | Test standardized MSE | Mean-pose MSE | Paired difference [95% interval] |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for r in comparisons:
        ci = r["paired_difference"]
        uncertainty = "unavailable"
        if ci:
            uncertainty = fmt(ci["mean"])
            if ci["ci95"]:
                uncertainty += f" [{fmt(ci['ci95'][0])}, {fmt(ci['ci95'][1])}]"
        lines.append(f"| {r['task']} | {r['model']} | {r['context']} | {r['condition']} | "
                     f"{r['seeds']} | {fmt(r['mse'])} | {fmt(r['mean_baseline_mse'])} | {uncertainty} |")
    if not comparisons:
        lines.append("| Pending | — | — | — | — | — | — | — |")
    lines += [
        "",
        "Lower MSE is better. Components are standardized using reference training data; constant "
        "components are excluded. This is not MPJPE. Confidence intervals average seeds first and "
        "resample paired 120-second block means. Repeated activities can still correlate across blocks; "
        "one recording cannot support population-level inference.",
        "",
        "Per-run result.json files include one-person/multiperson and low/high total-return errors, "
        "raw-unit MAE, and frozen-predictor offset sweeps on a common sweep mask. A frozen sweep changes "
        "the label pairing, not the input; its best offset is not an independent sensor-lag estimate.",
        "",
        "## Limits and next decision",
        "",
        "- No claim of dataset incapability is justified by failure on this online point cloud.",
        "- Image-task success can reflect room position. Shape-task results are needed to support a "
        "claim about articulation; these pooled targets still do not establish complete HPE.",
        "- Bed and doorway conditions are retained whenever 2D labels pass the checks, but have not "
        "been manually certified or separately annotated in this study.",
        "- Missing/low-confidence labels can disproportionately exclude hard activities. Review the "
        "coverage before applying results to the full recording.",
        "- Better radar processing could improve returns. Its benefit is not measured by these tests.",
        "- A consistent unknown offset in the reference remains possible. More anchors do not "
        "automatically remove systematic annotation error.",
        f"- The configured {100*config['material_relative_error']:.0f}% material-error margin is "
        f"{config['threshold_status']}. It is not used to issue an acceptance verdict.",
        "",
        "If informative radar and pose-control models both deteriorate under errors comparable to "
        "independently established residual errors, improve timing before full HPE training. "
        "If radar models fail even with reference labels while pose controls learn, investigate the "
        "radar representation and available human returns. If controlled shifts barely change labels "
        "or predictions, use more dynamic intervals before judging timing.",
        "",
        "## Reproducibility",
        "",
        f"Preparation ID: `{manifest['preparation_id']}`.",
        f"Current code SHA-256: `{code_hash()}`.",
        "The manifest records input hashes, anchor coordinates, split settings, configuration, and "
        "code provenance. All writes are confined to this study directory. The master database is read-only.",
        "",
        f"Remaining core runs: {len(missing)}. Stale results excluded: {len(stale)}.",
    ]
    text = "\n".join(lines) + "\n"
    # Markdown is a generated experiment report, alongside machine-readable results.
    (out / "report.md").write_text(text, encoding="utf-8")
    from .presentation import plot_results, render_html
    figures = plot_results(out, displacement, comparisons)
    (out / "report.html").write_text(render_html(text, figures), encoding="utf-8")
    if comparisons:
        pd.DataFrame([{k: v for k, v in r.items() if k != "paired_difference"} for r in comparisons]).to_csv(
            out / "metrics.csv", index=False)
    print(f"Report: {out / 'report.html'} ({len(missing)} core runs pending)", flush=True)
    return out / "report.html"
