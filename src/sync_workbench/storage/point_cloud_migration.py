"""Explicit, transactional metadata migration; acquisition tables are untouched."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from sync_workbench.core.tables import TABLE_SPECS
from sync_workbench.core.time_utils import utc_now_str


def connect_existing(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def require_version_schema(conn: sqlite3.Connection) -> None:
    for table in ("POINT_CLOUD_VERSION", "RUN_ASSET", "SAMPLE_ARTIFACT", "SAMPLE_SUMMARY"):
        columns = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        if "point_cloud_version_id" not in columns:
            raise ValueError("Database requires migrate-point-clouds before versioned import")
    if "point_status" not in {r[1] for r in conn.execute('PRAGMA table_info("SAMPLE_SUMMARY")')}:
        raise ValueError("Database requires point-cloud status migration")


def migrate_point_cloud_versions(path: str | Path) -> dict:
    """Idempotently migrate an explicitly selected database in one transaction."""
    with closing(connect_existing(path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("RUN_ASSET", "SAMPLE_ARTIFACT", "SAMPLE_SUMMARY"):
            if table not in existing:
                raise ValueError(f"Not a canonical database: missing {table}")
            cols = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
            if "point_cloud_version_id" not in cols:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN point_cloud_version_id TEXT NOT NULL DEFAULT \'\'')
            conn.execute(f'''UPDATE "{table}" SET point_cloud_version_id = CASE device_type
                WHEN 'radar_pc' THEN 'online_original' WHEN 'radar_raw' THEN 'raw_legacy' ELSE '' END
                WHERE point_cloud_version_id IS NULL OR point_cloud_version_id = '' ''')
            if table == "SAMPLE_SUMMARY":
                if "point_status" not in cols:
                    conn.execute('ALTER TABLE SAMPLE_SUMMARY ADD COLUMN point_status TEXT NOT NULL DEFAULT \'\'')
                conn.execute("UPDATE SAMPLE_SUMMARY SET point_status='available' WHERE device_type IN ('radar_pc','radar_raw') AND COALESCE(point_status,'')=''")
        spec = TABLE_SPECS["POINT_CLOUD_VERSION"]
        columns = ",".join(f'"{column}" TEXT NOT NULL DEFAULT \'\'' for column in spec.columns)
        conn.execute(f'CREATE TABLE IF NOT EXISTS POINT_CLOUD_VERSION ({columns})')
        for table in ("POINT_CLOUD_VERSION", "SAMPLE_ARTIFACT", "SAMPLE_SUMMARY"):
            keys = ",".join(f'"{key}"' for key in TABLE_SPECS[table].key)
            conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "uq_{table}_cloud_version" ON "{table}" ({keys})')
        # Register online acquisitions, including runs whose assets are not built
        # yet. The empty artifact_ref explicitly represents that condition.
        pairs = conn.execute("SELECT subject_id,run_id,'radar_pc' AS device_type,'online_original' AS version FROM DEVICE_RUN WHERE device_type='radar_pc' UNION SELECT subject_id,run_id,device_type,point_cloud_version_id FROM SAMPLE_ARTIFACT WHERE device_type IN ('radar_pc','radar_raw') AND artifact_role='radar_points'").fetchall()
        added = 0
        for subject, run, device, version in pairs:
            if version not in {"online_original", "raw_legacy"}:
                continue
            refs = [r[0] for r in conn.execute("SELECT DISTINCT artifact_ref FROM SAMPLE_ARTIFACT WHERE subject_id=? AND run_id=? AND device_type=? AND point_cloud_version_id=? AND artifact_role='radar_points'", (subject, run, device, version))]
            if len(refs) > 1:
                raise ValueError(f"Multiple legacy cloud bundles for {subject}/{run}/{version}")
            before = conn.total_changes
            conn.execute('''INSERT OR IGNORE INTO POINT_CLOUD_VERSION
                (subject_id,run_id,device_type,point_cloud_version_id,readable_label,artifact_ref,provenance_json,created_at)
                VALUES (?,?,?,?,?,?,?,?)''',
                (subject, run, device, version, "Original online cloud" if device == "radar_pc" else "Legacy raw payload",
                 refs[0] if refs else "", json.dumps({"origin": "legacy_metadata_migration", "processing_provenance_available": False}), utc_now_str()))
            added += conn.total_changes - before
            if refs:
                conn.execute("UPDATE POINT_CLOUD_VERSION SET artifact_ref=? WHERE subject_id=? AND run_id=? AND device_type=? AND point_cloud_version_id=? AND artifact_ref=''", (refs[0], subject, run, device, version))
        return {"registered_versions": added, "total_versions": conn.execute("SELECT COUNT(*) FROM POINT_CLOUD_VERSION").fetchone()[0]}


def migrate_database_copy(source: str | Path, output: str | Path) -> dict:
    """Back up a read-only SQLite connection, then migrate the new copy."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError("Migration output must be a NEW database path")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation prevents accidental overwrite, even across callers.
    with output.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src, closing(sqlite3.connect(output)) as dst:
            src.backup(dst)
        return {"output": str(output), **migrate_point_cloud_versions(output)}
    except Exception:
        # Leave the copy for diagnosis; the source was opened read-only.
        raise
