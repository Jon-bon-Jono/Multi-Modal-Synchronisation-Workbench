"""Add complete Kinect runs and immutable raw versions without resetting a workbench.

Build in isolation, compare identities/content, then insert missing metadata in one
transaction. Existing files/rows are never replaced. Dry runs use a read-only master
connection and temporary scratch files only. Interrupted file publication can leave
orphans; these are explicitly rejected, not adopted or overwritten on the next run.
"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path, PureWindowsPath
import shutil
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from sync_workbench.core.tables import TABLE_SPECS
from sync_workbench.core.validation import validate_canonical_tables
from sync_workbench.ingestion.raw_point_cloud_package import RawCloudPackage, file_hash
from sync_workbench.ingestion.temp_package import TempPackage, _read_pickle_lenient
from sync_workbench.ingestion.temp_to_canonical import TempToCanonicalTransformer
from sync_workbench.ingestion.validators import validate_temp_inputs
from sync_workbench.services.artifact_build_service import ArtifactBuildService
from sync_workbench.services.raw_point_cloud_import_service import RawPointCloudImportService, insert_rows
from sync_workbench.storage.point_cloud_migration import migrate_point_cloud_versions, require_version_schema
from sync_workbench.storage.sqlite_store import SQLiteCoreStore

TABLES = ("SUBJECT", "DEVICE_RUN", "RUN_SAMPLE", "RUN_TIMELINE_MODEL", "SAMPLE_TIME_ESTIMATE",
          "POINT_CLOUD_VERSION", "RUN_ASSET", "SAMPLE_ARTIFACT", "SAMPLE_SUMMARY")
RUN_KEY = ("subject_id", "run_id", "device_type")
NUMERIC = {"sample_index", "nominal_fps", "time_value_sec", "residual_ms", "payload_bytes",
           "point_count", "point_count_filtered", "num_people", "num_2d", "num_3d"}
COSMETIC = {"created_at", "notes", "timeline_model_name", "readable_label"}


def _safe_ref(root, ref):
    ref = str(ref).replace("\\", "/")
    target = (root / ref).resolve()
    if PureWindowsPath(ref).drive or not ref or not target.is_relative_to(root) or target == root:
        raise ValueError(f"Unsafe asset reference: {ref}")
    return target


def _value(column, value):
    if value is None or value == "":
        return None
    if column in NUMERIC or column.startswith("has_"):
        return float(value)
    if column.endswith("_json") or column in {"payload_shape", "artifact_member_key"}:
        return json.loads(value)
    return value


def _equal_row(table, old, new):
    if table == "SUBJECT":
        return True  # Catalogue imports do not edit manually maintained subject metadata.
    for column in TABLE_SPECS[table].columns:
        identity_note = column == "notes" and (table == "RUN_SAMPLE" or
            (table == "RUN_ASSET" and new["asset_role"] == "rgb_video_integrity"))
        if column in COSMETIC and not identity_note:
            continue
        if table == "POINT_CLOUD_VERSION" and column == "artifact_sha256":
            continue  # Verify the registered bundle below; payload fingerprint binds staged content.
        left, right = _value(column, old[column]), _value(column, new[column])
        if table == "POINT_CLOUD_VERSION" and column == "provenance_json":
            mutable = {"readable_label", "source_hdf5_name", "source_hdf5_path_at_packaging"}
            left = {k: v for k, v in left["version"].items() if k not in mutable}
            right = {k: v for k, v in right["version"].items() if k not in mutable}
        if left != right:
            return False
    return True


def _key(table, row):
    return tuple(_value(c, row[c]) for c in TABLE_SPECS[table].key)


def _same_file(a, b):
    """Compare payload meaning, not NPZ ZIP timestamps or generated manifest timestamps."""
    if file_hash(a) == file_hash(b):
        return True
    if a.suffix == ".npz":
        with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
            if set(x.files) != set(y.files):
                return False
            for key in x.files:
                left, right = x[key], y[key]
                if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right, equal_nan=True):
                    return False
            return True
    if a.name.startswith("sample_payload_manifest."):
        read = pd.read_parquet if a.suffix == ".parquet" else pd.read_csv
        x, y = read(a), read(b)
        cols = [c for c in x.columns if c not in COSMETIC]
        return set(x.columns) == set(y.columns) and x[cols].equals(y[cols])
    return file_hash(a) == file_hash(b)


def _read_package(input_dir, rgb_root):
    root = Path(input_dir).resolve()
    if (root / "radar_pc_samples.zst").exists():
        raise ValueError("import-package accepts Kinect and versioned offline radar only; omit radar_pc_samples.zst")
    raw_files = [(root / n).exists() for n in ("radar_raw_samples.zst", "radar_raw_point_cloud_versions.json")]
    if any(raw_files) and not all(raw_files):
        raise ValueError("Raw samples and their version manifest must both be present")
    raw = RawCloudPackage.read(root) if all(raw_files) else None
    runs = raw.device_runs if raw else _read_pickle_lenient(root / "device_runs.zst")
    rgb = _read_pickle_lenient(root / "rgb_samples.zst") if (root / "rgb_samples.zst").exists() else None
    errors = [i.message for i in validate_temp_inputs(runs, rgb, None) if i.severity == "error"]
    if errors:
        raise ValueError("; ".join(errors))
    selected = set()
    if rgb is not None:
        if rgb.empty:
            raise ValueError("Empty Kinect sample file")
        if rgb_root is None:
            raise ValueError("Kinect import requires --rgb-root to validate video references")
        required = {"pose2d", "conf2d", "pose3d", "activity", "num_people"}
        if required - set(rgb.columns):
            raise ValueError(f"Kinect run package is missing payload columns: {sorted(required - set(rgb.columns))}")
        for (subject, run), group in rgb.groupby(["subject_id", "run_id"], dropna=False):
            ids = sorted(group.frame_number.tolist())
            if ids != list(range(1, len(group) + 1)) or not group.sample_kind.eq("frame").all():
                raise ValueError("Kinect package must contain complete one-based native frame identities; subsets/rebasing are unsupported")
            if group.video_ref.nunique(dropna=False) != 1:
                raise ValueError("A Kinect acquisition must reference exactly one RGB video")
            path = _safe_ref(Path(rgb_root).resolve(), group.video_ref.iloc[0])
            if not path.is_file():
                raise FileNotFoundError(path)
            selected.add((subject, run, "kinect_rgb"))
    if raw:
        selected.update((v.manifest["subject_id"], v.manifest["run_id"], "radar_raw") for v in raw.versions)
    if not selected:
        raise ValueError("Package has no Kinect samples or offline radar versions")
    for key in selected:
        if any(not isinstance(v, str) or not v.strip() or v in {".", ".."}
               or any(c in v for c in "/\\:") for v in key):
            raise ValueError(f"Invalid acquisition identity: {key}")
    scoped = runs.loc[[tuple(row) in selected for row in runs[list(RUN_KEY)].itertuples(index=False, name=None)]].copy()
    if len(scoped) != len(selected) or scoped.duplicated(list(RUN_KEY)).any():
        raise ValueError("Each selected acquisition requires exactly one catalogue row")
    return TempPackage(root, scoped, rgb_samples=rgb), raw, sorted(selected)


def _stage(package, raw, directory, rgb_root):
    database, artifacts = directory / "staging.sqlite", directory / "artifacts"
    tables = TempToCanonicalTransformer(package).transform().tables
    # Store a separate integrity record, preserving legacy RGB asset annotations.
    # Existing acquisitions establish this baseline on their first additive import.
    video_integrity = []
    for asset in tables["RUN_ASSET"].to_dict("records"):
        if asset["asset_role"] == "rgb_video":
            video_integrity.append({**asset, "asset_id": asset["asset_id"] + "__integrity_sha256",
                "asset_role": "rgb_video_integrity",
                "notes": "sha256=" + file_hash(_safe_ref(Path(rgb_root).resolve(), asset["asset_ref"]))})
    if video_integrity:
        tables["RUN_ASSET"] = pd.concat([tables["RUN_ASSET"], pd.DataFrame(video_integrity)], ignore_index=True)
    errors = [i for i in validate_canonical_tables(tables) if i.severity == "error"]
    if errors:
        raise ValueError(str(errors))
    store = SQLiteCoreStore(database)
    store.initialise_empty()
    for table, frame in tables.items():
        if not frame.empty:
            store.write_table(table, frame, if_exists="append")
    migrate_point_cloud_versions(database)
    if package.rgb_samples is not None:
        ArtifactBuildService().build_from_package(package, database, artifacts, devices=["kinect_rgb"])
    if raw:
        RawPointCloudImportService().import_validated_package(raw, package.root, database, artifacts)
    return database, artifacts


def _plan_files(master, staged, staged_root, artifact_root):
    additions = []
    rows = list(staged.execute("SELECT * FROM RUN_ASSET WHERE storage_key='artifact_store'"))
    for row in rows:
        source = _safe_ref(staged_root, row["asset_ref"])
        target = _safe_ref(artifact_root, row["asset_ref"])
        registered = master.execute("SELECT 1 FROM RUN_ASSET WHERE storage_key='artifact_store' AND asset_ref=?",
                                    (row["asset_ref"],)).fetchone()
        if registered:
            if not target.is_file() or not _same_file(source, target):
                raise ValueError(f"Existing artifact missing or conflicting: {row['asset_ref']}")
        elif target.exists():
            raise ValueError(f"Unregistered artifact destination already exists: {target}; inspect the orphan before retrying")
        else:
            additions.append((source, target))
    for row in staged.execute("SELECT * FROM POINT_CLOUD_VERSION"):
        prior = master.execute("SELECT * FROM POINT_CLOUD_VERSION WHERE subject_id=? AND run_id=? AND device_type=? AND point_cloud_version_id=?",
                               tuple(row[c] for c in TABLE_SPECS["POINT_CLOUD_VERSION"].key)).fetchone()
        if prior and file_hash(_safe_ref(artifact_root, prior["artifact_ref"])) != prior["artifact_sha256"]:
            raise ValueError("Existing raw cloud bundle checksum changed")
    return additions


def _check_acquisitions(master, staged, runs):
    for key in runs:
        where = "subject_id=? AND run_id=? AND device_type=?"
        old_count = master.execute("SELECT COUNT(*) FROM RUN_SAMPLE WHERE " + where, key).fetchone()[0]
        new_count = staged.execute("SELECT COUNT(*) FROM RUN_SAMPLE WHERE " + where, key).fetchone()[0]
        if old_count and old_count != new_count:
            raise ValueError(f"Existing acquisition frame coverage differs: {key}; refusing renumbering or replacement")


def _merge_table(master, staged, table, subjects, dry_run):
    added = unchanged = 0
    for subject in subjects:
        old = {}
        for row in master.execute(f'SELECT * FROM "{table}" WHERE subject_id=?', (subject,)):
            key = _key(table, row)
            if key in old:
                raise ValueError(f"Duplicate canonical identity in {table}: {key}")
            old[key] = row
        pending = []
        seen = set()
        for row in staged.execute(f'SELECT * FROM "{table}" WHERE subject_id=?', (subject,)):
            key = _key(table, row)
            if key in seen:
                raise ValueError(f"Duplicate incoming identity in {table}: {key}")
            seen.add(key)
            if key in old:
                if not _equal_row(table, old[key], row):
                    raise ValueError(f"Conflicting {table} identity: {key}; existing data was not replaced")
                unchanged += 1
            else:
                added += 1
                if not dry_run:
                    pending.append(dict(row))
                    if len(pending) >= 1000:
                        insert_rows(master, table, pending)
                        pending.clear()
        if pending:
            insert_rows(master, table, pending)
    return {"added": added, "already_present": unchanged}


class PackageImportService:
    def import_package(self, input_dir, sqlite_path, artifact_root, *, rgb_root=None, dry_run=False):
        source, database, root = Path(input_dir).resolve(), Path(sqlite_path).resolve(), Path(artifact_root).resolve()
        if not database.is_file():
            raise FileNotFoundError("An initialized canonical database is required")
        protected = [source, database, *([Path(rgb_root).resolve()] if rgb_root else [])]
        if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
            raise ValueError("Artifact root must not overlap package, database or RGB inputs")
        package, raw, runs = _read_package(source, rgb_root)
        with TemporaryDirectory(prefix="syncwb-import-") as temporary:
            staged_db, staged_root = _stage(package, raw, Path(temporary), rgb_root)
            del package, raw
            with closing(sqlite3.connect(database.as_uri() + ("?mode=ro" if dry_run else "?mode=rw"), uri=True)) as master, \
                    closing(sqlite3.connect(staged_db)) as staged:
                master.row_factory = staged.row_factory = sqlite3.Row
                require_version_schema(master)
                installed = []
                try:
                    master.execute("BEGIN" if dry_run else "BEGIN IMMEDIATE")
                    _check_acquisitions(master, staged, runs)
                    files = _plan_files(master, staged, staged_root, root)
                    counts = {table: _merge_table(master, staged, table, sorted({r[0] for r in runs}), dry_run)
                              for table in TABLES}
                    if not dry_run:
                        for source_file, target in files:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with source_file.open("rb") as src, target.open("xb") as dst:
                                installed.append(target)
                                shutil.copyfileobj(src, dst)
                        master.commit()
                    else:
                        master.rollback()
                except Exception:
                    master.rollback()
                    for path in reversed(installed):
                        path.unlink(missing_ok=True)
                    raise
        return {"dry_run": dry_run, "runs": [dict(zip(RUN_KEY, run)) for run in runs],
                "tables": counts, "artifact_files_added": len(files),
                "rows_added": sum(v["added"] for v in counts.values()),
                "status": "validated" if dry_run else ("imported" if any(v["added"] for v in counts.values()) else "already_imported")}
