"""Command-line interface for Sync Workbench v0.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sync_workbench.services.artifact_audit_service import ArtifactAuditService
from sync_workbench.services.artifact_build_service import ArtifactBuildService
from sync_workbench.services.ingestion_service import IngestionService
from sync_workbench.services.anchor_service import AnchorEndpoint, AnchorService
from sync_workbench.services.piecewise_sync_service import PiecewiseSyncService
from sync_workbench.experimental.feasibility.reports import build_piecewise_synthetic_report
from sync_workbench.services.mapping_service import MappingService
from sync_workbench.services.pair_inspection_service import PairInspectionService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import TimelineSelection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syncwb", description="Multi-modal Synchronisation Workbench backend CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest-temp", help="Convert temporary zst ingestion files into a canonical SQLite store")
    ingest.add_argument("--input", required=True, help="Folder containing device_runs.zst, rgb_samples.zst, radar_pc_samples.zst")
    ingest.add_argument("--sqlite", required=True, help="Output SQLite store path")
    ingest.add_argument("--parquet", default=None, help="Optional output directory for canonical parquet/csv exports")
    ingest.add_argument("--reports", default=None, help="Optional output directory for ingestion reports")

    summary = sub.add_parser("summary", help="Print row counts for a canonical SQLite store")
    summary.add_argument("--sqlite", required=True, help="SQLite store path")

    build_artifacts = sub.add_parser("build-artifacts", help="Build v0.2.1 artifact bundles from temporary payload columns")
    build_artifacts.add_argument("--input-temp", required=True, help="Folder containing temporary ingestion .zst files")
    build_artifacts.add_argument("--sqlite", required=True, help="Canonical SQLite store path created by ingest-temp")
    build_artifacts.add_argument("--artifact-root", required=True, help="Artifact store root directory")
    build_artifacts.add_argument("--subject", default=None, help="Optional subject_id filter")
    build_artifacts.add_argument("--devices", nargs="+", choices=["kinect_rgb", "radar_pc"], default=None)
    build_artifacts.add_argument("--overwrite", action="store_true", help="Replace existing artifact bundle files and metadata rows")

    audit_artifacts = sub.add_parser("audit-artifacts", help="Audit v0.2.1 artifact metadata and bundle files")
    audit_artifacts.add_argument("--sqlite", required=True, help="Canonical SQLite store path")
    audit_artifacts.add_argument("--artifact-root", required=True, help="Artifact store root directory")
    audit_artifacts.add_argument("--issues-csv", default=None, help="Optional CSV path for audit issues")

    inspect_pair = sub.add_parser("inspect-pair", help="Inspect one mapped source-target pair and its available payloads")
    inspect_pair.add_argument("--sqlite", required=True, help="Canonical SQLite store path")
    inspect_pair.add_argument("--artifact-root", required=True, help="Artifact store root directory")
    inspect_pair.add_argument("--subject", default=None, help="Subject id. Required if mapping-version is not globally unique.")
    inspect_pair.add_argument("--mapping-version", required=True)
    inspect_pair.add_argument("--source-sample", required=True, type=int)
    inspect_pair.add_argument("--include-secondary", action="store_true", help="Allow a non-primary/secondary candidate if it sorts first by rank")
    inspect_pair.add_argument("--json", action="store_true", help="Print machine-readable JSON")

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
        "--source-window-policy",
        choices=["target-overlap", "all"],
        default="target-overlap",
        help="Which source samples to generate mappings for. Default maps only source samples inside target timeline coverage.",
    )
    mapn.add_argument("--source-margin-ms", type=float, default=0.0)
    mapn.add_argument(
        "--primary-policy",
        choices=["supported-only", "within-max-delta", "nearest-any"],
        default="supported-only",
    )
    mapn.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing mapping version and delete its existing SAMPLE_MAPPING rows before writing new rows.",
    )
    mapn.add_argument("--diagnostics-csv", default=None)

    mapall = sub.add_parser(
        "map-nearest-all",
        help="Generate nearest-time mappings for all overlapping source-target run pairs",
    )
    mapall.add_argument("--sqlite", required=True, help="SQLite store path")
    mapall.add_argument("--source-device", default="kinect_rgb")
    mapall.add_argument("--source-timeline", default="rgb_wallclock_from_pts")
    mapall.add_argument("--target-device", default="radar_pc")
    mapall.add_argument("--target-timeline", default="radar_pc_linear_from_index")
    mapall.add_argument("--mapping-version-prefix", required=True)
    mapall.add_argument("--top-k", type=int, default=3)
    mapall.add_argument("--min-overlap-sec", type=float, default=5.0)
    mapall.add_argument("--weak-support-threshold-ms", type=float, default=75.0)
    mapall.add_argument("--max-allowed-delta-ms", type=float, default=200.0)
    mapall.add_argument("--extrapolation-policy", default="disallow")
    mapall.add_argument("--allow-numeric-identity", action="store_true")
    mapall.add_argument(
        "--source-window-policy",
        choices=["target-overlap", "all"],
        default="target-overlap",
    )
    mapall.add_argument("--source-margin-ms", type=float, default=0.0)
    mapall.add_argument(
        "--primary-policy",
        choices=["supported-only", "within-max-delta", "nearest-any"],
        default="supported-only",
    )
    mapall.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mapping versions generated with the same deterministic IDs.",
    )
    mapall.add_argument("--pair-report-csv", default=None)


    fit_piecewise = sub.add_parser("fit-piecewise", help="Fit a piecewise-affine sync model from canonical anchors and generate a mapping version")
    fit_piecewise.add_argument("--sqlite", required=True, help="SQLite store path")
    fit_piecewise.add_argument("--subject", required=True)
    fit_piecewise.add_argument("--source-run", required=True)
    fit_piecewise.add_argument("--source-device", default="kinect_rgb")
    fit_piecewise.add_argument("--source-timeline", required=True)
    fit_piecewise.add_argument("--target-run", required=True)
    fit_piecewise.add_argument("--target-device", default="radar_pc")
    fit_piecewise.add_argument("--target-timeline", required=True)
    fit_piecewise.add_argument("--sync-model", required=True)
    fit_piecewise.add_argument("--mapping-version", required=True)
    fit_piecewise.add_argument("--parent-mapping-version", default="")
    fit_piecewise.add_argument("--top-k", type=int, default=3)
    fit_piecewise.add_argument("--weak-support-threshold-ms", type=float, default=75.0)
    fit_piecewise.add_argument("--max-allowed-delta-ms", type=float, default=200.0)
    fit_piecewise.add_argument("--extrapolation-policy", choices=["disallow", "allow-linear", "allow_linear"], default="disallow")
    fit_piecewise.add_argument("--primary-policy", choices=["supported-only", "within-max-delta", "nearest-any"], default="supported-only")
    fit_piecewise.add_argument("--overwrite", action="store_true")
    fit_piecewise.add_argument("--allow-nonmonotonic-target", action="store_true", help="Allow target anchor times that are not strictly increasing")
    fit_piecewise.add_argument("--diagnostics-csv", default=None)

    export_anchors = sub.add_parser("export-anchors", help="Export pair anchors as JSON")
    export_anchors.add_argument("--sqlite", required=True)
    export_anchors.add_argument("--output", required=True)
    export_anchors.add_argument("--subject", required=True)
    export_anchors.add_argument("--source-run", required=True)
    export_anchors.add_argument("--source-device", default="kinect_rgb")
    export_anchors.add_argument("--target-run", required=True)
    export_anchors.add_argument("--target-device", default="radar_pc")
    export_anchors.add_argument("--annotator-id", default="")
    export_anchors.add_argument("--initial-mapping-version", default="")

    import_anchors = sub.add_parser("import-anchors", help="Import anchors from JSON")
    import_anchors.add_argument("--sqlite", required=True)
    import_anchors.add_argument("--input", required=True)
    import_anchors.add_argument("--overwrite", action="store_true")

    synthetic = sub.add_parser("piecewise-synthetic-report", help="Generate experimental synthetic piecewise-affine feasibility reports")
    synthetic.add_argument("--output", required=True)

    gui = sub.add_parser("anchoring-gui", help="Launch the experimental v0.2.2 anchoring GUI")
    gui.add_argument("--sqlite", required=True)
    gui.add_argument("--artifact-root", required=True)
    gui.add_argument("--rgb-root", required=True)
    gui.add_argument("--subject", required=True)
    gui.add_argument("--mapping-version", required=True)
    gui.add_argument("--annotator-id", default="")

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


    if args.command == "build-artifacts":
        try:
            result = ArtifactBuildService().build_from_temp_package(
                args.input_temp,
                args.sqlite,
                args.artifact_root,
                overwrite=args.overwrite,
                subject_id=args.subject,
                devices=args.devices,
            )
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            print(f"Error: {exc}")
            return 2
        print("Artifact build complete.")
        print(result.reports["table_counts"].to_string(index=False))
        if not result.reports["artifact_build_summary"].empty:
            print(result.reports["artifact_build_summary"].to_string(index=False))
        return 0

    if args.command == "audit-artifacts":
        issues = ArtifactAuditService(args.sqlite, args.artifact_root).audit()
        if args.issues_csv:
            Path(args.issues_csv).parent.mkdir(parents=True, exist_ok=True)
            issues.to_csv(args.issues_csv, index=False)
        if issues.empty:
            print("Artifact audit passed: no issues found.")
            return 0
        print(issues.to_string(index=False))
        return 1 if "error" in set(issues["severity"].astype(str)) else 0

    if args.command == "inspect-pair":
        try:
            result = PairInspectionService(args.sqlite, args.artifact_root).inspect_pair(
                args.mapping_version,
                args.source_sample,
                subject_id=args.subject,
                primary_only=not args.include_secondary,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}")
            return 2
        if args.json:
            print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        else:
            print("Mapping:")
            for key, value in result["mapping"].items():
                print(f"  {key}: {value}")
            print("Source:")
            _print_endpoint(result["source"], result["payload_shapes"]["source"])
            print("Target:")
            _print_endpoint(result["target"], result["payload_shapes"]["target"])
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
                source_window_policy=args.source_window_policy,
                source_margin_ms=args.source_margin_ms,
                primary_policy=args.primary_policy,
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

    if args.command == "map-nearest-all":
        service = MappingService(args.sqlite)
        result = service.generate_nearest_mappings_for_overlaps(
            source_device=args.source_device,
            source_timeline=args.source_timeline,
            target_device=args.target_device,
            target_timeline=args.target_timeline,
            mapping_version_prefix=args.mapping_version_prefix,
            top_k=args.top_k,
            min_overlap_sec=args.min_overlap_sec,
            source_window_policy=args.source_window_policy,
            source_margin_ms=args.source_margin_ms,
            weak_support_threshold_ms=args.weak_support_threshold_ms,
            max_allowed_delta_ms=args.max_allowed_delta_ms,
            extrapolation_policy=args.extrapolation_policy,
            allow_numeric_identity=args.allow_numeric_identity,
            primary_policy=args.primary_policy,
            overwrite=args.overwrite,
        )

        if result.pair_report.empty:
            print("No source-target run pairs found.")
        else:
            print(result.pair_report.to_string(index=False))

        if args.pair_report_csv:
            Path(args.pair_report_csv).parent.mkdir(parents=True, exist_ok=True)
            result.pair_report.to_csv(args.pair_report_csv, index=False)

        return 0



    if args.command == "fit-piecewise":
        source = TimelineSelection(args.subject, args.source_run, args.source_device, args.source_timeline)
        target = TimelineSelection(args.subject, args.target_run, args.target_device, args.target_timeline)
        try:
            result = PiecewiseSyncService(args.sqlite).fit_piecewise_and_generate_mapping(
                source,
                target,
                sync_model_id=args.sync_model,
                mapping_version_id=args.mapping_version,
                parent_mapping_version_id=args.parent_mapping_version,
                top_k=args.top_k,
                weak_support_threshold_ms=args.weak_support_threshold_ms,
                max_allowed_delta_ms=args.max_allowed_delta_ms,
                extrapolation_policy=args.extrapolation_policy.replace("-", "_"),
                primary_policy=args.primary_policy.replace("-", "_"),
                overwrite=args.overwrite,
                require_monotonic_target=not args.allow_nonmonotonic_target,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return 2
        print("Piecewise affine sync model fitted and mapping generated.")
        print(result.diagnostics.to_string(index=False))
        if args.diagnostics_csv:
            Path(args.diagnostics_csv).parent.mkdir(parents=True, exist_ok=True)
            result.diagnostics.to_csv(args.diagnostics_csv, index=False)
        return 0

    if args.command == "export-anchors":
        payload = AnchorService(args.sqlite).export_pair_anchors_json(
            args.output,
            subject_id=args.subject,
            source_run_id=args.source_run,
            source_device_type=args.source_device,
            target_run_id=args.target_run,
            target_device_type=args.target_device,
            session_metadata={
                "annotator_id": args.annotator_id,
                "initial_mapping_version_id": args.initial_mapping_version,
            },
        )
        print(f"Exported {len(payload.get('ANCHOR', []))} anchors to {args.output}")
        return 0

    if args.command == "import-anchors":
        counts = AnchorService(args.sqlite).import_anchors_json(args.input, overwrite=args.overwrite)
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0

    if args.command == "piecewise-synthetic-report":
        summary = build_piecewise_synthetic_report(args.output)
        print(summary.to_string(index=False))
        return 0

    if args.command == "anchoring-gui":
        try:
            from sync_workbench.experimental.anchoring_gui.app import run_anchoring_gui
            return run_anchoring_gui(
                sqlite_path=args.sqlite,
                artifact_root=args.artifact_root,
                rgb_root=args.rgb_root,
                subject_id=args.subject,
                mapping_version_id=args.mapping_version,
                annotator_id=args.annotator_id,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 2

    raise AssertionError(f"Unhandled command {args.command}")


def _print_endpoint(endpoint: dict, payload_shapes: dict) -> None:
    print(f"  {endpoint['subject_id']} / {endpoint['run_id']} / {endpoint['device_type']} / sample {endpoint['sample_index']}")
    print(f"  payload_roles: {', '.join(endpoint.get('payload_roles', [])) or '(none)'}")
    if endpoint.get("summary"):
        print("  summary:")
        for key, value in endpoint["summary"].items():
            if key in {"subject_id", "run_id", "device_type", "sample_index", "notes"}:
                continue
            print(f"    {key}: {value}")
    if payload_shapes:
        print("  payload_shapes:")
        for role, desc in payload_shapes.items():
            print(f"    {role}: {desc}")


def _json_safe(value):
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value

if __name__ == "__main__":
    raise SystemExit(main())
