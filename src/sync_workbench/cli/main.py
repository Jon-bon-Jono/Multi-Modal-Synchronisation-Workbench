"""Command-line interface for Sync Workbench v0.1."""
from __future__ import annotations

import argparse
from pathlib import Path

from sync_workbench.services.ingestion_service import IngestionService
from sync_workbench.services.mapping_service import MappingService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import TimelineSelection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syncwb", description="Multi-modal Synchronisation Workbench v0.1 backend CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest-temp", help="Convert temporary zst ingestion files into a canonical SQLite store")
    ingest.add_argument("--input", required=True, help="Folder containing device_runs.zst, rgb_samples.zst, radar_pc_samples.zst")
    ingest.add_argument("--sqlite", required=True, help="Output SQLite store path")
    ingest.add_argument("--parquet", default=None, help="Optional output directory for canonical parquet/csv exports")
    ingest.add_argument("--reports", default=None, help="Optional output directory for ingestion reports")

    summary = sub.add_parser("summary", help="Print row counts for a canonical SQLite store")
    summary.add_argument("--sqlite", required=True, help="SQLite store path")

    mapn = sub.add_parser("map-nearest", help="Generate a crude nearest-time mapping intended for anchor-placement/navigation")
    mapn.add_argument("--sqlite", required=True, help="SQLite store path")
    mapn.add_argument("--subject", required=True)
    mapn.add_argument("--source-run", required=True)
    mapn.add_argument("--source-device", default="kinect_rgb")
    mapn.add_argument("--source-timeline", required=True)
    mapn.add_argument("--target-run", required=True)
    mapn.add_argument("--target-device", default="radar_pc")
    mapn.add_argument("--target-timeline", required=True)
    mapn.add_argument("--mapping-version", required=True)
    mapn.add_argument("--sync-model-id", default=None)
    mapn.add_argument("--top-k", type=int, default=3)
    mapn.add_argument("--weak-support-threshold-ms", type=float, default=75.0)
    mapn.add_argument("--max-allowed-delta-ms", type=float, default=200.0)
    mapn.add_argument("--extrapolation-policy", default="disallow")
    mapn.add_argument("--allow-numeric-identity", action="store_true")
    mapn.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing mapping version and delete its existing SAMPLE_MAPPING rows before writing new rows.",
    )
    mapn.add_argument("--diagnostics-csv", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "ingest-temp":
        result = IngestionService().ingest_temp_package(
            args.input,
            args.sqlite,
            parquet_dir=args.parquet,
            reports_dir=args.reports,
        )
        counts = result.reports.get("table_counts")
        print("Ingestion complete.")
        if counts is not None:
            print(counts.to_string(index=False))
        return 0

    if args.command == "summary":
        store = SQLiteCoreStore(args.sqlite)
        rows = []
        for name in store.list_tables():
            rows.append((name, len(store.read_table(name))))
        if not rows:
            print("No tables found.")
        else:
            width = max(len(name) for name, _ in rows)
            for name, count in rows:
                print(f"{name:<{width}}  {count}")
        return 0

    if args.command == "map-nearest":
        service = MappingService(args.sqlite)
        source = TimelineSelection(args.subject, args.source_run, args.source_device, args.source_timeline)
        target = TimelineSelection(args.subject, args.target_run, args.target_device, args.target_timeline)
        try:
            result = service.generate_nearest_mapping(
                source,
                target,
                mapping_version_id=args.mapping_version,
                sync_model_id=args.sync_model_id,
                top_k=args.top_k,
                weak_support_threshold_ms=args.weak_support_threshold_ms,
                max_allowed_delta_ms=args.max_allowed_delta_ms,
                extrapolation_policy=args.extrapolation_policy,
                allow_numeric_identity=args.allow_numeric_identity,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return 2
        print("Mapping generated.")
        print(result.diagnostics.to_string(index=False))
        if args.diagnostics_csv:
            Path(args.diagnostics_csv).parent.mkdir(parents=True, exist_ok=True)
            result.diagnostics.to_csv(args.diagnostics_csv, index=False)
        return 0

    raise AssertionError(f"Unhandled command {args.command}")

if __name__ == "__main__":
    raise SystemExit(main())
