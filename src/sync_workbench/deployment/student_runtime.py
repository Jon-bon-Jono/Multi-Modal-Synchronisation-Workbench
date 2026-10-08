"""Validate a relocated student package and launch its assigned session."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import re
import shutil
import sys
import sqlite3

from sync_workbench.deployment.package_layout import local_path, read_manifest, sha256, verify_package


def _verify_runtime(root, manifest):
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Student packages require Python 3.11; use the package setup script")
    installed = Path(__file__).resolve().parents[1]
    prefix = "application/src/sync_workbench/"
    for entry in manifest["files"]:
        if entry["path"].startswith(prefix):
            file = local_path(installed, entry["path"][len(prefix):])
            if not file.is_file() or sha256(file) != entry["sha256"]:
                raise ValueError("Installed SyncWB code differs from this package. Run this package's setup script.")


def saved_annotator(root):
    root = Path(root).resolve()
    path = local_path(root, "work/annotator.json")
    if not path.exists():
        return None
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("package_id") != read_manifest(root)["package_id"]:
        raise ValueError("Annotator identity belongs to another package")
    return _annotator_id(identity.get("annotator_id", ""))


def _annotator_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("Enter an assigned annotator ID: 1-64 letters, numbers, dots, underscores or hyphens")
    return value


def _database_marker(manifest):
    template_sha = next(f["sha256"] for f in manifest["files"] if f["path"] == "database/template.sqlite")
    return manifest["package_id"], manifest["manifest_sha256"], template_sha


def _verify_database_identity(db, manifest, *, working, sqlite_integrity_check=False):
    """Check a few identity rows, never fingerprint the acquisition tables."""
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        if sqlite_integrity_check:
            if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError(f"Database failed its SQLite integrity check: {db.name}; retain it for recovery")
        if working:
            try:
                rows = conn.execute("SELECT package_id,manifest_sha256,template_sha256 FROM STUDENT_PACKAGE LIMIT 2").fetchall()
            except sqlite3.OperationalError as exc:
                raise ValueError("Working database has no package identity; retain it and contact the coordinator") from exc
            # Older working copies may also have core_sha256. It is intentionally ignored.
            if rows != [_database_marker(manifest)]:
                raise ValueError("Working database belongs to a different package; it was not replaced")
        assignment = manifest["assignment"]
        rows = conn.execute(
            "SELECT source_run_id,source_device_type,target_run_id,target_device_type FROM MAPPING_VERSION "
            "WHERE subject_id=? AND mapping_version_id=? LIMIT 2",
            (assignment["subject_id"], assignment["mapping_version_id"])).fetchall()
        expected = tuple(assignment[k] for k in ("source_run_id", "source_device_type", "target_run_id", "target_device_type"))
        if rows != [expected]:
            raise ValueError("Database assignment differs from the package's RGB/radar pair or mapping")
        rows = conn.execute(
            "SELECT payload_fingerprint,acquisition_timeline_sha256 FROM POINT_CLOUD_VERSION "
            "WHERE subject_id=? AND run_id=? AND device_type=? AND point_cloud_version_id=? LIMIT 2",
            tuple(assignment[k] for k in ("subject_id", "target_run_id", "target_device_type", "point_cloud_version_id"))).fetchall()
        cloud = manifest["cloud_provenance"]
        expected = tuple(cloud.get(k) or "" for k in ("payload_fingerprint", "acquisition_timeline_sha256"))
        if len(rows) != 1 or tuple(v or "" for v in rows[0]) != expected:
            raise ValueError("Database point-cloud identity differs from the package assignment")


def verify_student_runtime(root, *, full_checksums=False, sqlite_integrity_check=False):
    """Read-only verification; optional expensive checks never create student work."""
    root = Path(root).resolve()
    result = verify_package(root, full=full_checksums)
    manifest = read_manifest(root)
    _verify_runtime(root, manifest)
    if manifest.get("spatial_calibration") is not None:
        from sync_workbench.core.geometry import load_spatial_calibration
        load_spatial_calibration(local_path(root, manifest["spatial_calibration"]["path"]))
    checked = []
    for ref, working in (("database/template.sqlite", False), ("work/workbench.sqlite", True)):
        db = local_path(root, ref)
        if working and not db.exists():
            continue
        _verify_database_identity(db, manifest, working=working, sqlite_integrity_check=sqlite_integrity_check)
        if sqlite_integrity_check:
            checked.append(ref)
    result["sqlite_integrity_checked"] = checked
    return result


def prepare_student_launch(root, annotator_id, *, full_checksums=False, sqlite_integrity_check=False):
    """Automatable launch preparation; never opens a GUI or modifies input assets."""
    root = Path(root).resolve()
    annotator_id = _annotator_id(annotator_id)
    verify_student_runtime(root, full_checksums=full_checksums, sqlite_integrity_check=sqlite_integrity_check)
    manifest = read_manifest(root)
    existing = saved_annotator(root)
    if existing and existing != annotator_id:
        raise ValueError("This working copy belongs to another annotator. Use a fresh package copy for a different person.")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    db = local_path(root, config["paths"]["working_database"])
    template = local_path(root, config["paths"]["database_template"])
    marker = _database_marker(manifest)
    template_sha = marker[2]
    db.parent.mkdir(parents=True, exist_ok=True)
    if not db.exists():
        # Exclusive creation never overwrites an existing student's annotations.
        with db.open("xb") as dst, template.open("rb") as src:
            shutil.copyfileobj(src, dst)
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute("CREATE TABLE STUDENT_PACKAGE (package_id TEXT, manifest_sha256 TEXT, template_sha256 TEXT)")
            conn.execute("INSERT INTO STUDENT_PACKAGE VALUES (?,?,?)", marker)
        _verify_database_identity(db, manifest, working=True, sqlite_integrity_check=sqlite_integrity_check)
    identity = local_path(root, "work/annotator.json")
    if not identity.exists():
        with identity.open("x", encoding="utf-8") as stream:
            json.dump({"package_id": manifest["package_id"], "annotator_id": annotator_id}, stream, indent=2)
    assignment = config["assignment"]
    return {"sqlite_path": db, "artifact_root": local_path(root, config["paths"]["artifact_root"]),
            "rgb_root": local_path(root, config["paths"]["rgb_root"]), "subject_id": assignment["subject_id"],
            "mapping_version_id": assignment["mapping_version_id"], "point_cloud_version_id": assignment["point_cloud_version_id"],
            "annotator_id": annotator_id,
            "spatial_calibration_path": (local_path(root, config["spatial_calibration"]["path"])
                                         if config.get("spatial_calibration") is not None else None),
            "package_provenance": {"schema": manifest["schema"], "package_id": manifest["package_id"],
                                   "manifest_sha256": manifest["manifest_sha256"], "template_sha256": template_sha,
                                   "runtime_id": manifest["runtime_id"], "assignment": assignment}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open a portable SyncWB student assignment")
    parser.add_argument("--package", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Run selected checks without launching or creating student work")
    parser.add_argument("--full-checksums", action="store_true", help="Read and checksum every packaged asset and the database template")
    parser.add_argument("--sqlite-integrity-check", action="store_true", help="Run SQLite integrity_check on the template and existing working database")
    parser.add_argument("--annotator-id", default=None)
    args = parser.parse_args(argv)
    try:
        if args.full_checksums:
            print("Full package checksums requested: reading every packaged file...", flush=True)
        if args.sqlite_integrity_check:
            print("SQLite integrity checks requested: scanning the database(s)...", flush=True)
        if args.verify_only:
            result = verify_student_runtime(args.package, full_checksums=args.full_checksums,
                                            sqlite_integrity_check=args.sqlite_integrity_check)
            print(json.dumps(result, indent=2))
            return 0
        annotator = args.annotator_id or saved_annotator(args.package)
        if not annotator:
            from PySide6.QtWidgets import QApplication, QInputDialog
            app = QApplication.instance() or QApplication([])
            annotator, accepted = QInputDialog.getText(None, "SyncWB student assignment", "Your assigned annotator ID:")
            if not accepted:
                return 0
        options = prepare_student_launch(args.package, annotator.strip(), full_checksums=args.full_checksums,
                                         sqlite_integrity_check=args.sqlite_integrity_check)
        from sync_workbench.experimental.anchoring_gui.app import run_anchoring_gui
        return run_anchoring_gui(**options)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"Cannot open student package: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
