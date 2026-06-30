"""Application service for temporary-package ingestion."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sync_workbench.core.validation import issues_to_frame, validate_canonical_tables
from sync_workbench.ingestion.temp_package import TempPackage
from sync_workbench.ingestion.temp_to_canonical import TransformResult, TempToCanonicalTransformer
from sync_workbench.ingestion.validators import temp_issues_to_frame, validate_temp_inputs
from sync_workbench.storage.parquet_export import export_tables
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


class IngestionService:
    def ingest_temp_package(
        self,
        input_dir: str | Path,
        sqlite_path: str | Path,
        *,
        parquet_dir: str | Path | None = None,
        reports_dir: str | Path | None = None,
    ) -> TransformResult:
        package = TempPackage.read(input_dir)
        input_issues = validate_temp_inputs(package.device_runs, package.rgb_samples, package.radar_pc_samples)
        if any(issue.severity == "error" for issue in input_issues):
            raise ValueError(temp_issues_to_frame(input_issues).to_string(index=False))

        result = TempToCanonicalTransformer(package).transform()
        canonical_issues = validate_canonical_tables(result.tables)
        result.reports["input_validation_issues"] = temp_issues_to_frame(input_issues)
        result.reports["canonical_validation_issues"] = issues_to_frame(canonical_issues)

        store = SQLiteCoreStore(sqlite_path)
        store.initialise_empty()
        for name, df in result.tables.items():
            store.write_table(name, df, if_exists="replace")

        if parquet_dir is not None:
            export_tables(result.tables, parquet_dir)

        if reports_dir is not None:
            self.write_reports(result.reports, reports_dir)

        return result

    @staticmethod
    def write_reports(reports: dict[str, pd.DataFrame], reports_dir: str | Path) -> None:
        output = Path(reports_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, df in reports.items():
            safe = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            df.to_csv(output / f"{safe}.csv", index=False)
        _write_markdown_summary(reports, output / "ingestion_report.md")


def _write_markdown_summary(reports: dict[str, pd.DataFrame], path: Path) -> None:
    lines = ["# Sync Workbench v0.1 ingestion report", ""]
    if "table_counts" in reports:
        lines += ["## Table counts", "", reports["table_counts"].to_markdown(index=False), ""]
    if "canonical_validation_issues" in reports:
        issues = reports["canonical_validation_issues"]
        lines += ["## Canonical validation issues", ""]
        lines.append("No issues." if issues.empty else issues.to_markdown(index=False))
        lines.append("")
    if "asset_path_warnings" in reports:
        warnings = reports["asset_path_warnings"]
        lines += ["## Asset path warnings", ""]
        lines.append("No warnings." if warnings.empty else warnings.to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
