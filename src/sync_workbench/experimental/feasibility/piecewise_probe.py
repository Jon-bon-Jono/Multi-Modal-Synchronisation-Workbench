"""Run synthetic piecewise-affine feasibility probes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sync_workbench.experimental.feasibility.plots import write_case_plots
from sync_workbench.experimental.feasibility.synthetic_cases import SyntheticCase, case_to_frames, make_synthetic_cases
from sync_workbench.sync.piecewise_affine import AnchorPair, fit_piecewise_affine


def run_synthetic_report(output_dir: str | Path) -> pd.DataFrame:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for case in make_synthetic_cases():
        summary_rows.append(run_case(case, root / case.name))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "summary.csv", index=False)
    (root / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_case(case: SyntheticCase, case_dir: str | Path) -> dict[str, object]:
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    source, target, anchors = case_to_frames(case)
    pairs = [
        AnchorPair(str(row.anchor_id), -1, -1, float(row.source_time), float(row.target_time))
        for row in anchors.itertuples(index=False)
    ]
    model = fit_piecewise_affine(pairs, extrapolation_policy="allow_linear", require_monotonic_target=False)
    predictions = model.predict_many(case.source_times)
    predictions["truth_target_time"] = case.truth_target_times
    predictions["error_sec"] = predictions["target_time"] - predictions["truth_target_time"]
    predictions["nearest_target_sample_index"] = [int(np.argmin(np.abs(target["time_sec"].to_numpy() - t))) for t in predictions["target_time"]]
    predictions["nearest_target_time"] = [float(target.iloc[i]["time_sec"]) for i in predictions["nearest_target_sample_index"]]
    predictions["nearest_selection_error_sec"] = predictions["nearest_target_time"] - predictions["truth_target_time"]
    residuals = model.anchor_residuals()
    loo = model.leave_one_out_errors()

    source.to_csv(case_dir / "source_timeline.csv", index=False)
    target.to_csv(case_dir / "target_timeline.csv", index=False)
    anchors.to_csv(case_dir / "anchors.csv", index=False)
    predictions.to_csv(case_dir / "source_target_mapping.csv", index=False)
    residuals.to_csv(case_dir / "anchor_residuals.csv", index=False)
    loo.to_csv(case_dir / "leave_one_out_errors.csv", index=False)
    diagnostics = {
        "case": case.name,
        "description": case.description,
        "n_source": int(len(source)),
        "n_target": int(len(target)),
        "n_anchors": int(len(anchors)),
        "abs_mapping_error_ms_median": float(np.nanmedian(np.abs(predictions["error_sec"]) * 1000.0)),
        "abs_mapping_error_ms_max": float(np.nanmax(np.abs(predictions["error_sec"]) * 1000.0)),
        "abs_nearest_selection_error_ms_median": float(np.nanmedian(np.abs(predictions["nearest_selection_error_sec"]) * 1000.0)),
        "abs_nearest_selection_error_ms_max": float(np.nanmax(np.abs(predictions["nearest_selection_error_sec"]) * 1000.0)),
    }
    (case_dir / "case_summary.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    write_case_plots(case_dir, case, source, target, anchors, predictions)
    return diagnostics
