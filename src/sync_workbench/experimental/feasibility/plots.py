"""Plot helpers for the experimental feasibility sandbox."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_case_plots(case_dir: str | Path, case, source: pd.DataFrame, target: pd.DataFrame, anchors: pd.DataFrame, predictions: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    case_dir = Path(case_dir)

    fig = plt.figure()
    plt.plot(source["time_sec"], predictions["target_time"], label="fitted")
    plt.plot(source["time_sec"], predictions["truth_target_time"], label="truth")
    plt.scatter(anchors["source_time"], anchors["target_time"], label="anchors")
    plt.xlabel("source time (s)")
    plt.ylabel("target time (s)")
    plt.title(case.name)
    plt.legend()
    fig.savefig(case_dir / "timeline_plot.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(source["time_sec"], predictions["error_sec"] * 1000.0)
    plt.xlabel("source time (s)")
    plt.ylabel("mapping error (ms)")
    plt.title(f"{case.name} mapping error")
    fig.savefig(case_dir / "mapping_error_plot.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(source["time_sec"], predictions["nearest_selection_error_sec"] * 1000.0)
    plt.xlabel("source time (s)")
    plt.ylabel("nearest target selection error (ms)")
    plt.title(f"{case.name} nearest-frame selection")
    fig.savefig(case_dir / "residual_plot.png", bbox_inches="tight")
    plt.close(fig)
