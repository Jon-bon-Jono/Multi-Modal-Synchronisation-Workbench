"""Export a clean, portable one-pair student assignment from canonical data."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import uuid

from sync_workbench.core.tables import TABLE_SPECS
from sync_workbench.core.geometry import load_spatial_calibration
from sync_workbench.services.anchor_transfer import acquisition_digest
from sync_workbench.deployment.package_layout import SCHEMA, json_digest, local_path, sha256, verify_package, write_json
from sync_workbench.experimental.anchoring_gui.session_selection import resolve_session
from sync_workbench.storage.point_cloud_migration import require_version_schema

REQUIREMENTS = """numpy>=1.26,<3
pandas>=2.2,<4
PyYAML>=6,<7
zstandard>=0.22,<1
tabulate>=0.9,<1
PySide6>=6.6,<7
pyqtgraph>=0.13,<0.15
PyOpenGL>=3.1,<4
opencv-python>=4.8,<5
"""


def _rows(conn, table, where, values):
    return [dict(r) for r in conn.execute(f'SELECT * FROM "{table}" WHERE {where}', values)]


def _one(rows, label):
    if len(rows) != 1:
        raise ValueError(f"Expected one {label}, found {len(rows)}")
    return rows[0]


def _subset(conn, selection):
    s = selection
    key = (s.subject_id, s.mapping_version_id)
    tables = {name: [] for name in TABLE_SPECS}
    tables["SUBJECT"] = [_one(_rows(conn, "SUBJECT", "subject_id=?", key[:1]), "subject")]
    mapping = _one(_rows(conn, "MAPPING_VERSION", "subject_id=? AND mapping_version_id=?", key), "mapping")
    model = _one(_rows(conn, "SYNC_MODEL", "subject_id=? AND sync_model_id=?", (s.subject_id, mapping["source_sync_model_id"])), "navigation model")
    # Students get clean annotations and an initial navigation aid, not a fitted model with hidden anchor dependencies.
    if mapping["mapping_method"] != "initial_nearest_for_anchoring" or model["model_type"] != "identity_time" or mapping.get("parent_mapping_version_id"):
        raise ValueError("Student packages require an initial nearest-time navigation mapping without a parent")
    tables["MAPPING_VERSION"], tables["SYNC_MODEL"] = [mapping], [model]
    tables["SAMPLE_MAPPING"] = _rows(conn, "SAMPLE_MAPPING", "subject_id=? AND mapping_version_id=?", key)
    run_where = "subject_id=? AND ((run_id=? AND device_type='kinect_rgb') OR (run_id=? AND device_type=?))"
    runs = (s.subject_id, s.source_run_id, s.target_run_id, s.target_device_type)
    for name in ("DEVICE_RUN", "RUN_SAMPLE", "RUN_TIMELINE_MODEL", "SAMPLE_TIME_ESTIMATE"):
        tables[name] = _rows(conn, name, run_where, runs)
    payload_where = "subject_id=? AND ((run_id=? AND device_type='kinect_rgb' AND COALESCE(point_cloud_version_id,'')='') OR (run_id=? AND device_type=? AND point_cloud_version_id=?))"
    for name in ("SAMPLE_ARTIFACT", "SAMPLE_SUMMARY"):
        tables[name] = _rows(conn, name, payload_where, (*runs, s.point_cloud_version_id))
    tables["POINT_CLOUD_VERSION"] = [_one(_rows(conn, "POINT_CLOUD_VERSION", "subject_id=? AND run_id=? AND device_type=? AND point_cloud_version_id=?",
        (s.subject_id, s.target_run_id, s.target_device_type, s.point_cloud_version_id)), "cloud version")]
    referenced = {(r["storage_key"], r["artifact_ref"]) for r in tables["SAMPLE_ARTIFACT"]}
    tables["RUN_ASSET"] = [r for r in _rows(conn, "RUN_ASSET", payload_where, (*runs, s.point_cloud_version_id))
        if (r["storage_key"], r["asset_ref"]) in referenced or r["asset_role"] == "rgb_video"]
    _one([r for r in tables["RUN_ASSET"] if r["asset_role"] == "rgb_video"], "RGB video asset")
    _validate_subset(tables, s)
    return tables


def _validate_subset(tables, s):
    pair = {(s.source_run_id, "kinect_rgb"), (s.target_run_id, s.target_device_type)}
    if {(r["run_id"], r["device_type"]) for r in tables["DEVICE_RUN"]} != pair:
        raise ValueError("Incomplete acquisition pair")
    samples = {(r["run_id"], r["device_type"], int(r["sample_index"])) for r in tables["RUN_SAMPLE"]}
    if not tables["SAMPLE_MAPPING"]:
        raise ValueError("Empty navigation mapping")
    for row in tables["SAMPLE_MAPPING"]:
        for side in ("source", "target"):
            key = (row[f"{side}_run_id"], row[f"{side}_device_type"], int(row[f"{side}_sample_index"]))
            expected = (s.source_run_id, "kinect_rgb") if side == "source" else (s.target_run_id, s.target_device_type)
            if key[:2] != expected or key not in samples:
                raise ValueError("Mapping references an unavailable acquisition sample")
    for row in tables["SAMPLE_ARTIFACT"] + tables["SAMPLE_SUMMARY"] + tables["SAMPLE_TIME_ESTIMATE"]:
        if (row["run_id"], row["device_type"], int(row["sample_index"])) not in samples:
            raise ValueError("Metadata references a missing captured sample")
    timelines = {(r["run_id"], r["device_type"], r["timeline_model_id"]) for r in tables["RUN_TIMELINE_MODEL"]}
    model = tables["SYNC_MODEL"][0]
    for side in ("source", "target"):
        expected = (s.source_run_id, "kinect_rgb") if side == "source" else (s.target_run_id, s.target_device_type)
        if (model[f"{side}_run_id"], model[f"{side}_device_type"]) != expected:
            raise ValueError("Navigation model acquisition differs from assignment")
        timeline = (model[f"{side}_run_id"], model[f"{side}_device_type"], model[f"{side}_timeline_model_id"])
        if timeline not in timelines or not any((r["run_id"], r["device_type"], r["timeline_model_id"]) == timeline for r in tables["SAMPLE_TIME_ESTIMATE"]):
            raise ValueError("Navigation model timeline is missing")


def _write_database(path, source, tables):
    with closing(sqlite3.connect(path)) as target, target:
        for name, spec in TABLE_SPECS.items():
            columns = {r[1]: r[2] or "TEXT" for r in source.execute(f'PRAGMA table_info("{name}")')}
            if set(spec.columns) - columns.keys():
                raise ValueError(f"Source schema requires migration: {name}")
            definitions = ",".join(f'"{c}" {columns[c]}' for c in spec.columns)
            target.execute(f'CREATE TABLE "{name}" ({definitions})')
            column_sql = ",".join(f'"{c}"' for c in spec.columns)
            target.executemany(f'INSERT INTO "{name}" ({column_sql}) VALUES ({",".join("?" for _ in spec.columns)})',
                               (tuple(row.get(c) for c in spec.columns) for row in tables[name]))
            keys = ",".join(f'"{c}"' for c in spec.key)
            target.execute(f'CREATE UNIQUE INDEX "uq_{name}" ON "{name}" ({keys})')


def _copy_assets(root, tables, artifact_root, rgb_root):
    roots = {"artifact_store": Path(artifact_root), "rgb": Path(rgb_root)}
    folder = {"artifact_store": "assets/artifacts", "rgb": "assets/rgb"}
    references = {(r["storage_key"], r["asset_ref"]) for r in tables["RUN_ASSET"]}
    references |= {(r["storage_key"], r["artifact_ref"]) for r in tables["SAMPLE_ARTIFACT"]}
    registry = tables["POINT_CLOUD_VERSION"][0]
    references.add(("artifact_store", registry["artifact_ref"]))
    mapped, assets = {}, {}
    for storage, ref in sorted(references):
        if storage not in roots:
            raise ValueError(f"Unsupported asset storage root: {storage}")
        source = local_path(roots[storage], ref)
        if not source.is_file():
            raise FileNotFoundError(f"Required asset missing: {source}")
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        digest = sha256(source)
        new_ref = digest + source.suffix.lower()
        target = root / folder[storage] / new_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256(target) != digest or before != (source.stat().st_size, source.stat().st_mtime_ns):
            raise ValueError("Asset changed while packaging")
        mapped[(storage, ref)] = new_ref
        assets[target.relative_to(root).as_posix()] = {"path": target.relative_to(root).as_posix(), "storage_key": storage, "sha256": digest, "bytes": target.stat().st_size}
    old_ref = registry["artifact_ref"]
    new_ref = mapped[("artifact_store", old_ref)]
    digest = new_ref.split(".", 1)[0]
    if registry.get("artifact_sha256") and registry["artifact_sha256"] != digest:
        raise ValueError("Selected cloud bundle differs from its registered checksum")
    registry["artifact_ref"], registry["artifact_sha256"] = new_ref, digest
    for name, column in (("RUN_ASSET", "asset_ref"), ("SAMPLE_ARTIFACT", "artifact_ref")):
        for row in tables[name]:
            row[column] = mapped[(row["storage_key"], row[column])]
    return list(assets.values())


def _copy_application(root, application_root):
    source = Path(application_root).resolve()
    for name in ("pyproject.toml", "README.md"):
        if not (source / name).is_file():
            raise ValueError("Application checkout required; provide --application-root")
        destination = root / "application" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    files = sorted((source / "src/sync_workbench").rglob("*.py"))
    if not files:
        raise ValueError("Application source is missing")
    for file in files:
        destination = root / "application" / file.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
    for name in ("spatial_calibration.md", "student_package.md", "training_export.md"):
        file = source / "docs" / name
        if file.is_file():
            destination = root / "application/docs" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, destination)
    (root / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    code = [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p)} for p in sorted((root / "application").rglob("*")) if p.is_file()]
    return json_digest({"files": code, "requirements": REQUIREMENTS, "python": "3.11"})[:20]


def _write_launchers(root, runtime_id):
    from sync_workbench.deployment.conda_launchers import write_conda_launchers
    write_conda_launchers(root, runtime_id)


def bundle_spatial_calibration(root, path, target_device_type):
    """Copy an optional validated calibration into the immutable portable package."""
    calibration = load_spatial_calibration(path)
    if calibration is None:
        return None
    if target_device_type != "radar_raw":
        raise ValueError("Spatial calibration requires an offline radar_raw assignment")
    ref = "calibration/kinect_radar/" + Path(path).name
    destination = local_path(root, ref)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    if sha256(destination) != calibration.sha256:
        raise ValueError("Spatial calibration changed while packaging")
    return {"path": ref, "sha256": calibration.sha256}


def export_student_package(*, sqlite_path, artifact_root, rgb_root, output, subject_id, mapping_version_id,
                           point_cloud_version_id, application_root=None, read_only_roots=(), spatial_calibration_path=None):
    output = Path(output).resolve()
    source = Path(sqlite_path).resolve()
    protected = [Path(p).resolve() for p in (artifact_root, rgb_root, *read_only_roots)]
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("Package output must be outside input/read-only asset roots")
    if output.exists():
        raise FileExistsError("Choose a new package output folder; existing work is never overwritten")
    selection = resolve_session(source, artifact_root, subject_id, mapping_version_id, point_cloud_version_id)
    if spatial_calibration_path is not None:
        calibration_path = Path(spatial_calibration_path).resolve()
        if output == calibration_path or output.is_relative_to(calibration_path) or calibration_path.is_relative_to(output):
            raise ValueError("Package output overlaps the spatial calibration input")
    output.parent.mkdir(parents=True, exist_ok=True)
    package_id = "student_" + uuid.uuid4().hex
    with TemporaryDirectory(prefix=".syncwb-package-", dir=output.parent) as staging:
        root = Path(staging) / "package"
        (root / "database").mkdir(parents=True)
        calibration = bundle_spatial_calibration(root, spatial_calibration_path, selection.target_device_type)
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            require_version_schema(conn)
            tables = _subset(conn, selection)
            assets = _copy_assets(root, tables, artifact_root, rgb_root)
            _write_database(root / "database/template.sqlite", conn, tables)
        runtime_id = _copy_application(root, application_root or Path(__file__).resolve().parents[3])
        _write_launchers(root, runtime_id)
        assignment = {"subject_id": subject_id, "mapping_version_id": mapping_version_id,
                      "source_run_id": selection.source_run_id, "source_device_type": "kinect_rgb",
                      "target_run_id": selection.target_run_id, "target_device_type": selection.target_device_type,
                      "point_cloud_version_id": point_cloud_version_id}
        with closing(sqlite3.connect(root / "database/template.sqlite")) as conn:
            capture_digest = acquisition_digest(conn,assignment)
        config = {"schema": SCHEMA, "package_id": package_id, "assignment": assignment,
                  "paths": {"database_template": "database/template.sqlite", "working_database": "work/workbench.sqlite",
                            "artifact_root": "assets/artifacts", "rgb_root": "assets/rgb"}}
        if calibration is not None:
            config["spatial_calibration"] = calibration
        write_json(root / "config.json", config)
        files = []
        for p in sorted(root.rglob("*")):
            if p.is_file():
                ref = p.relative_to(root).as_posix()
                role = "asset" if ref.startswith("assets/") else "database_template" if ref.startswith("database/") else "calibration" if ref.startswith("calibration/") else "runtime"
                files.append({"path": ref, "bytes": p.stat().st_size, "sha256": sha256(p), "role": role})
        manifest = {"schema": SCHEMA, "package_id": package_id, "created_at": datetime.now(timezone.utc).isoformat(),
                    "assignment": assignment, "runtime_id": runtime_id, "python": "3.11", "acquisition_sha256": capture_digest,
                    "cloud_provenance": selection.cloud_provenance(), "table_counts": {t: len(rows) for t, rows in tables.items()},
                    "pose_roles": sorted({r["artifact_role"] for r in tables["SAMPLE_ARTIFACT"] if r["device_type"] == "kinect_rgb"}),
                    "assets": assets, "files": files}
        if calibration is not None:
            manifest["spatial_calibration"] = calibration
        manifest["manifest_sha256"] = json_digest(manifest)
        write_json(root / "manifest.json", manifest)
        verify_package(root)
        root.rename(output)
    return {"package_id": package_id, "manifest_sha256": manifest["manifest_sha256"], "output": str(output),
            "bytes": sum(f["bytes"] for f in files), "assets": len(assets), "table_counts": manifest["table_counts"]}
