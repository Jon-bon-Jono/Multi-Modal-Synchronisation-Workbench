"""Application service for temporary-package ingestion."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sync_workbench.core.validation import issues_to_frame, validate_canonical_tables
from sync_workbench.ingestion.temp_package import TempPackage
from sync_workbench.ingestion.temp_to_canonical import TransformResult, TempToCanonicalTransformer
from sync_workbench.ingestion.validators import temp_issues_to_frame, validate_temp_inputs
from sync_workbench.storage.parquet_export import export_tables
from sync_workbench.storage.point_cloud_migration import migrate_point_cloud_versions
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
        if Path(sqlite_path).exists():
            versions = SQLiteCoreStore(sqlite_path).read_table("POINT_CLOUD_VERSION")
            if not versions.empty and versions.device_type.eq("radar_raw").any():
                raise ValueError("Legacy ingestion cannot reset acquisitions with registered raw cloud versions")
        package = TempPackage.read(input_dir)
        input_issues = validate_temp_inputs(
            package.device_runs,
            package.rgb_samples,
            package.radar_pc_samples,
            package.radar_raw_samples,
        )
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

        migrate_point_cloud_versions(sqlite_path)
        result.tables["POINT_CLOUD_VERSION"] = store.read_table("POINT_CLOUD_VERSION")
        result.reports["table_counts"] = pd.DataFrame(
            [{"table": name, "rows": len(frame)} for name, frame in result.tables.items()]
        ).sort_values("table")

        if parquet_dir is not None:
            export_tables(result.tables, parquet_dir)

        if reports_dir is not None:
            self.write_reports(result.reports, reports_dir)

        return result

    def ingest_radar_raw_samples(
        self,
        input_dir: str | Path,
        sqlite_path: str | Path,
        *,
        reports_dir: str | Path | None = None,
    ) -> TransformResult:
        """Add or replace canonical raw-radar rows without resetting the store."""
        sqlite_path = Path(sqlite_path)
        if not sqlite_path.exists():
            raise FileNotFoundError(
                f"Canonical SQLite store does not exist: {sqlite_path}. "
                "Run ingest-temp first."
            )

        if Path(sqlite_path).exists():
            versions = SQLiteCoreStore(sqlite_path).read_table("POINT_CLOUD_VERSION")
            if not versions.empty and versions.device_type.eq("radar_raw").any():
                raise ValueError("Legacy ingestion cannot reset acquisitions with registered raw cloud versions")
        package = TempPackage.read(input_dir)
        issues = validate_temp_inputs(
            package.device_runs,
            package.rgb_samples,
            package.radar_pc_samples,
            package.radar_raw_samples,
        )
        raw_errors = [
            issue
            for issue in issues
            if issue.severity == "error"
            and issue.source in {"device_runs.zst", "radar_raw_samples.zst"}
        ]
        if raw_errors:
            raise ValueError(temp_issues_to_frame(raw_errors).to_string(index=False))

        result = TempToCanonicalTransformer(package).transform_radar_raw()
        canonical_issues = validate_canonical_tables(result.tables)
        if any(issue.severity == "error" for issue in canonical_issues):
            raise ValueError(issues_to_frame(canonical_issues).to_string(index=False))

        result.reports["input_validation_issues"] = temp_issues_to_frame(
            [issue for issue in issues if issue.source in {"device_runs.zst", "radar_raw_samples.zst"}]
        )
        result.reports["canonical_validation_issues"] = issues_to_frame(canonical_issues)

        store = SQLiteCoreStore(sqlite_path)
        for name, frame in result.tables.items():
            run_keys = frame[["subject_id", "run_id", "device_type"]].drop_duplicates()
            for row in run_keys.itertuples(index=False):
                store.delete_where(
                    name,
                    {
                        "subject_id": str(row.subject_id),
                        "run_id": str(row.run_id),
                        "device_type": str(row.device_type),
                    },
                )
            store.write_table(name, frame, if_exists="append")

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
