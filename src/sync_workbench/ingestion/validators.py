"""Validation for temporary ingestion inputs."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class InputValidationIssue:
    severity: str
    source: str
    message: str


REQUIRED_DEVICE_RUNS = (
    "subject_id",
    "run_id",
    "device_type",
    "start_wallclock_est",
    "end_wallclock_est",
    "nominal_fps",
    "notes",
)

REQUIRED_RGB_SAMPLES = (
    "frame_number",
    "pts_sec",
    "subject_id",
    "run_id",
    "sample_kind",
    "video_ref",
    "wallclock_est",
)

REQUIRED_RADAR_PC_SAMPLES = (
    "subject_id",
    "run_id",
    "frame_number",
    "sample_kind",
    "observed_wallclock",
)


def _missing(df: pd.DataFrame, cols: tuple[str, ...]) -> list[str]:
    return [c for c in cols if c not in df.columns]


def validate_temp_inputs(
    device_runs: pd.DataFrame,
    rgb_samples: pd.DataFrame | None,
    radar_pc_samples: pd.DataFrame | None,
) -> list[InputValidationIssue]:
    issues: list[InputValidationIssue] = []
    miss = _missing(device_runs, REQUIRED_DEVICE_RUNS)
    if miss:
        issues.append(InputValidationIssue("error", "device_runs.zst", f"Missing required columns: {miss}"))

    if rgb_samples is None:
        issues.append(InputValidationIssue("warning", "rgb_samples.zst", "File not found; RGB ingestion skipped."))
    else:
        miss = _missing(rgb_samples, REQUIRED_RGB_SAMPLES)
        if miss:
            issues.append(InputValidationIssue("error", "rgb_samples.zst", f"Missing required columns: {miss}"))

    if radar_pc_samples is None:
        issues.append(InputValidationIssue("warning", "radar_pc_samples.zst", "File not found; radar PC ingestion skipped."))
    else:
        miss = _missing(radar_pc_samples, REQUIRED_RADAR_PC_SAMPLES)
        if miss:
            issues.append(InputValidationIssue("error", "radar_pc_samples.zst", f"Missing required columns: {miss}"))

    return issues


def temp_issues_to_frame(issues: list[InputValidationIssue]) -> pd.DataFrame:
    return pd.DataFrame([issue.__dict__ for issue in issues], columns=["severity", "source", "message"])
