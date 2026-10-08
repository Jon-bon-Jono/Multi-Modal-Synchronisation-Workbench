"""Add immutable cloud payload versions without replacing acquisition identities."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import pandas as pd

from sync_workbench.core.tables import TABLE_SPECS
from sync_workbench.core.time_utils import utc_now_str
from sync_workbench.ingestion.raw_point_cloud_package import RawCloudPackage, file_hash
from sync_workbench.ingestion.temp_package import TempPackage
from sync_workbench.ingestion.temp_to_canonical import TempToCanonicalTransformer
from sync_workbench.storage.point_cloud_migration import connect_existing, require_version_schema
from sync_workbench.storage.ragged_npz import RaggedNpzWriter

RUN_WHERE = "subject_id=? AND run_id=? AND device_type=?"
VERSION_WHERE = RUN_WHERE + " AND point_cloud_version_id=?"
TIMELINE = "radar_raw_wallclock_from_start_end"


def insert_rows(conn, table, rows):
    columns = TABLE_SPECS[table].columns
    sql = f'INSERT INTO "{table}" (' + ",".join(f'"{c}"' for c in columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")"
    def values(row):
        for col in columns:
            value = row.get(col)
            if value is None or pd.isna(value):
                yield None
            else:
                yield value.item() if hasattr(value, "item") else value
    conn.executemany(sql, (tuple(values(row)) for row in rows))


class RawPointCloudImportService:
    def import_package(self, input_dir, sqlite_path, artifact_root):
        package = RawCloudPackage.read(input_dir)
        return self.import_validated_package(package, input_dir, sqlite_path, artifact_root)

    def import_validated_package(self, package: RawCloudPackage, input_dir, sqlite_path, artifact_root):
        """Install a package returned by RawCloudPackage.read; permits shared import staging."""
        root = Path(artifact_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        installed = []
        results = []
        with closing(connect_existing(sqlite_path)) as conn, TemporaryDirectory(prefix=".raw-import-", dir=root) as staging:
            try:
                require_version_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                for version in package.versions:
                    m = version.manifest
                    key = (m["subject_id"], m["run_id"], "radar_raw")
                    vkey = (*key, m["point_cloud_version_id"])
                    self._ensure_acquisition(conn, package, version, Path(input_dir))
                    prior = conn.execute("SELECT * FROM POINT_CLOUD_VERSION WHERE " + VERSION_WHERE, vkey).fetchone()
                    if prior is not None:
                        if prior["payload_fingerprint"] != version.payload_fingerprint:
                            raise ValueError("Existing cloud version has different payload/coverage; refusing replacement")
                        old_manifest = json.loads(prior["provenance_json"])["version"]
                        mutable_labels = {"readable_label", "source_hdf5_name", "source_hdf5_path_at_packaging"}
                        stable = lambda manifest: {k: v for k, v in manifest.items() if k not in mutable_labels}
                        if stable(old_manifest) != stable(m):
                            raise ValueError("Existing cloud version has conflicting processing/acquisition provenance")
                        path = root / prior["artifact_ref"]
                        if not path.is_file() or file_hash(path) != prior["artifact_sha256"]:
                            raise ValueError("Existing cloud version bundle is missing or changed")
                        results.append({"point_cloud_version_id": vkey[-1], "status": "already_imported"})
                        continue
                    # The validated content-derived ID binds subject/run and is safe on all platforms.
                    ref = f"point_cloud_versions/{vkey[-1]}/points.npz"
                    staged = Path(staging) / (vkey[-1] + ".npz")
                    available = version.samples[version.samples.point_status == "available"]
                    infos = RaggedNpzWriter.write(staged, zip(available.sample_index, available.points), tail_shape=(6,), dtype="float32")
                    digest = file_hash(staged)
                    target = root / ref
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Exclusive creation prevents replacement, including orphaned bundles after a crash.
                    with staged.open("rb") as src, target.open("xb") as dst:
                        installed.append(target)
                        shutil.copyfileobj(src, dst)
                    created = utc_now_str()
                    identity = dict(zip(TABLE_SPECS["POINT_CLOUD_VERSION"].key, vkey))
                    insert_rows(conn, "POINT_CLOUD_VERSION", [{**identity, "readable_label": m["readable_label"],
                        "payload_fingerprint": version.payload_fingerprint,
                        "acquisition_timeline_sha256": m["acquisition"]["acquisition_timeline_sha256"],
                        "artifact_ref": ref, "artifact_sha256": digest,
                        "provenance_json": json.dumps({"version": m, "package": package.package_metadata}, sort_keys=True),
                        "created_at": created}])
                    insert_rows(conn, "RUN_ASSET", [{**identity, "asset_id": vkey[-1] + "_points", "asset_role": "radar_points",
                        "storage_key": "artifact_store", "asset_ref": ref, "notes": "Immutable offline cloud bundle"}])
                    insert_rows(conn, "SAMPLE_ARTIFACT", ({**identity, "sample_index": info.sample_index,
                        "artifact_role": "radar_points", "artifact_id": vkey[-1] + "_points", "storage_key": "artifact_store",
                        "artifact_ref": ref, "artifact_member_key": json.dumps({"sample_index": info.sample_index}),
                        "artifact_format": "ragged_npz", "payload_shape": json.dumps(info.shape), "payload_dtype": info.dtype,
                        "payload_bytes": info.nbytes, "created_at": created, "notes": ""} for info in infos))
                    insert_rows(conn, "SAMPLE_SUMMARY", ({**identity, "sample_index": int(row.sample_index),
                        "point_count": row.point_count, "point_count_filtered": row.point_count_filtered,
                        "point_status": row.point_status, "has_points": row.point_status == "available" and len(row.points) > 0,
                        "created_at": created, "notes": ""} for row in version.samples.itertuples(index=False)))
                    results.append({"point_cloud_version_id": vkey[-1], "status": "imported", "frames": len(version.samples),
                                    "available_frames": len(available), "points": m["point_count"], "artifact_ref": ref})
                conn.commit()
            except Exception:
                conn.rollback()
                for path in installed:
                    path.unlink(missing_ok=True)
                    if not any(path.parent.iterdir()):
                        path.parent.rmdir()
                raise
        return {"versions": results}

    @staticmethod
    def _ensure_acquisition(conn, package, version, input_dir):
        m, samples = version.manifest, version.samples
        key = (m["subject_id"], m["run_id"], "radar_raw")
        if conn.execute("SELECT 1 FROM SUBJECT WHERE subject_id=?", key[:1]).fetchone() is None:
            raise ValueError("Subject is not registered; ingest acquisition metadata first")
        runs = conn.execute("SELECT * FROM DEVICE_RUN WHERE " + RUN_WHERE, key).fetchall()
        rows = conn.execute("SELECT sample_index,sample_kind,notes FROM RUN_SAMPLE WHERE " + RUN_WHERE + " ORDER BY CAST(sample_index AS INTEGER)", key).fetchall()
        times = conn.execute("SELECT sample_index,time_value_datetime FROM SAMPLE_TIME_ESTIMATE WHERE " + RUN_WHERE + " AND timeline_model_id=? ORDER BY CAST(sample_index AS INTEGER)", (*key, TIMELINE)).fetchall()
        models = conn.execute("SELECT 1 FROM RUN_TIMELINE_MODEL WHERE " + RUN_WHERE + " AND timeline_model_id=?", (*key, TIMELINE)).fetchall()
        if rows or times or models:
            if len(runs) != 1 or len(models) != 1 or len(rows) != len(samples) or len(times) != len(samples):
                raise ValueError("Existing raw acquisition is incomplete or has different frame coverage")
            for i, row in enumerate(rows):
                match = re.search(r"(?:^|;)\s*source_frame_number=([0-9]+(?:\.0+)?)\s*(?:;|$)", row["notes"] or "")
                if int(row["sample_index"]) != i or row["sample_kind"] != "frame" or match is None or float(match[1]) != i + 1:
                    raise ValueError("Existing raw sample identity differs from package; refusing renumbering")
            if [(int(r[0]), r[1]) for r in times] != list(zip(range(len(samples)), samples.estimated_wallclock_from_start_end)):
                raise ValueError("Existing acquisition timeline differs from package")
        else:
            temp = TempPackage(input_dir, package.device_runs, radar_raw_samples=samples.drop(columns="sample_index"))
            tables = TempToCanonicalTransformer(temp).transform_radar_raw().tables
            for name, frame in tables.items():
                if name == "DEVICE_RUN" and runs:
                    continue
                insert_rows(conn, name, frame.to_dict("records"))
        if len(runs) > 1:
            raise ValueError("Duplicate DEVICE_RUN identity")
        if runs:
            for col in ("start_wallclock_est", "end_wallclock_est"):
                if runs[0][col] != m["acquisition"][col]:
                    raise ValueError(f"Existing acquisition {col} differs from package")
