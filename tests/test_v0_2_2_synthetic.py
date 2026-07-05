from pathlib import Path

from sync_workbench.experimental.feasibility.reports import build_piecewise_synthetic_report


def test_synthetic_piecewise_report_smoke(tmp_path: Path):
    out = tmp_path / "synthetic"
    summary = build_piecewise_synthetic_report(out)
    assert len(summary) >= 10
    assert (out / "summary.csv").exists()
    assert (out / "identity" / "source_target_mapping.csv").exists()
    identity = summary[summary["case"] == "identity"].iloc[0]
    assert float(identity["abs_mapping_error_ms_max"]) < 1e-6
