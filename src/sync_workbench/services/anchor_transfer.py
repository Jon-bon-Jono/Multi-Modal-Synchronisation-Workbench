"""Validated anchor exchange and atomic JSON/recovery files."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from sync_workbench.core.tables import TABLE_SPECS
from sync_workbench.deployment.package_layout import json_digest

SCHEMA = 'syncwb.anchors.v1'
PAIR_KEYS = ('subject_id', 'source_run_id', 'source_device_type', 'target_run_id', 'target_device_type')


def atomic_json(path, payload):
    path = Path(path)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def index(value):
    if isinstance(value, bool):
        raise ValueError('Sample indices must be nonnegative integers')
    try:
        result = int(value)
        if result < 0 or float(value) != result:
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ValueError('Sample indices must be nonnegative integers') from None
    return result


def normalize_row(table, row):
    spec = TABLE_SPECS[table]
    if not isinstance(row, dict) or set(row) - set(spec.columns):
        raise ValueError(f'Unexpected fields in {table}')
    result = {}
    for col in spec.columns:
        value = row.get(col)
        if col == 'sample_index':
            value = index(value)
        elif col == 'confidence':
            if value is None or value == '':
                value = None
            else:
                value = float(value)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Confidence must be between 0 and 1')
        else:
            value = '' if value is None else value
            if not isinstance(value, str):
                raise ValueError(f'{table}.{col} must be text')
        if col in spec.required and value in (None, ''):
            raise ValueError(f'Missing required {table}.{col}')
        result[col] = value
    return result


def normalized_rows(table, rows):
    return [normalize_row(table, row) for row in rows]


def acquisition_digest(conn, pair):
    """Capture identity only: independent of clouds, navigation and fitted models."""
    digest = hashlib.sha256()
    for side in ('source', 'target'):
        keys = (pair['subject_id'], pair[side+'_run_id'], pair[side+'_device_type'])
        run = conn.execute('SELECT start_wallclock_est,end_wallclock_est,nominal_fps FROM DEVICE_RUN WHERE subject_id=? AND run_id=? AND device_type=?', keys).fetchall()
        if len(run) != 1:
            raise ValueError(f'Acquisition unavailable or ambiguous: {keys}')
        start, end, fps = run[0]
        digest.update(json.dumps([*keys, start or '', end or '', float(fps) if fps not in (None, '') else None]).encode())
        count = 0
        for sample, kind in conn.execute('SELECT sample_index,sample_kind FROM RUN_SAMPLE WHERE subject_id=? AND run_id=? AND device_type=? ORDER BY CAST(sample_index AS INTEGER)', keys):
            digest.update(json.dumps([index(sample), kind]).encode())
            count += 1
        if not count:
            raise ValueError(f'Acquisition has no samples: {keys}')
    return digest.hexdigest()


def validate_payload(payload, conn, expected_manifest=None):
    if not isinstance(payload, dict) or payload.get('schema') not in (None, SCHEMA):
        raise ValueError('Unsupported anchor export format')
    if payload.get('schema') == SCHEMA:
        expected = json_digest({k:v for k,v in payload.items() if k != 'content_sha256'})
        if payload.get('content_sha256') != expected:
            raise ValueError('Anchor export checksum mismatch')
    pair = payload.get('pair', {})
    if not isinstance(pair,dict) or set(pair) != set(PAIR_KEYS) or any(not isinstance(v,str) or not v for v in pair.values()):
        raise ValueError('Export must identify one acquisition pair')
    if (pair['source_run_id'],pair['source_device_type']) == (pair['target_run_id'],pair['target_device_type']):
        raise ValueError('Anchor endpoints must belong to distinct streams')
    capture = acquisition_digest(conn, pair)
    if payload.get('schema') == SCHEMA and payload.get('acquisition_sha256') != capture:
        raise ValueError('Export acquisition differs from destination database')
    session = payload.get('session', {})
    if not isinstance(session,dict):
        raise ValueError('Invalid export session')
    package = session.get('package')
    if package is not None and not isinstance(package,dict):
        raise ValueError('Invalid package provenance')
    if package:
        if expected_manifest is None:
            raise ValueError('Student returns require the original --package-manifest')
        manifest = json.loads(Path(expected_manifest).read_text(encoding='utf-8'))
        if manifest.get('manifest_sha256') != json_digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'}):
            raise ValueError('Original package manifest checksum mismatch')
        for key in ('schema','package_id','manifest_sha256','runtime_id','assignment'):
            if package.get(key) != manifest.get(key):
                raise ValueError(f'Export package {key} differs from original manifest')
        template_hash = next((f['sha256'] for f in manifest['files'] if f['path']=='database/template.sqlite'), None)
        if package.get('template_sha256') != template_hash:
            raise ValueError('Export template identity differs from original package')
        if manifest.get('acquisition_sha256') != capture:
            raise ValueError('Original package acquisition differs from destination')
        if any(manifest['assignment'].get(k) != pair[k] for k in PAIR_KEYS):
            raise ValueError('Export pair differs from original assignment')
        if not session.get('annotator_id'):
            raise ValueError('Student return has no annotator identity')
    elif expected_manifest is not None:
        raise ValueError('Expected a student return with package provenance')
    for table in ('ANCHOR','ANCHOR_MEMBER'):
        if not isinstance(payload.get(table),list):
            raise ValueError(f'Missing {table} rows')
    anchors = normalized_rows('ANCHOR',payload['ANCHOR'])
    members = normalized_rows('ANCHOR_MEMBER',payload['ANCHOR_MEMBER'])
    grouped = defaultdict(list)
    ids = set()
    for row in anchors:
        key = (row['subject_id'],row['anchor_id'])
        if key in ids or row['subject_id'] != pair['subject_id']:
            raise ValueError('Duplicate anchor ID or subject mismatch')
        ids.add(key)
        if package:
            try:
                provenance = json.loads(row['notes'])['provenance']
                version = provenance['point_cloud']['point_cloud_version_id']
                if (provenance['package'] != package or provenance['annotator_id'] != session['annotator_id']
                    or version != manifest['assignment']['point_cloud_version_id']):
                    raise ValueError()
            except (ValueError,KeyError,TypeError):
                raise ValueError('Anchor provenance differs from assigned package/annotator/cloud') from None
    for row in members:
        key = (row['subject_id'],row['anchor_id'])
        if key not in ids:
            raise ValueError('Orphan anchor member')
        grouped[key].append(row)
        found = conn.execute('SELECT 1 FROM RUN_SAMPLE WHERE subject_id=? AND run_id=? AND device_type=? AND sample_index=?',
                             (row['subject_id'],row['run_id'],row['device_type'],row['sample_index'])).fetchone()
        if found is None:
            raise ValueError(f"Anchor {row['anchor_id']} references an unavailable sample")
    for key in ids:
        rows = grouped[key]
        if len(rows) != 2 or {r['member_role'] for r in rows} != {'source','target'}:
            raise ValueError('Each anchor must have exactly one source and one target')
        for row in rows:
            side = row['member_role']
            if (row['run_id'],row['device_type']) != (pair[side+'_run_id'],pair[side+'_device_type']):
                raise ValueError('Anchor member differs from assigned acquisition pair')
    return anchors, grouped


def insert_rows(conn, table, rows):
    columns = TABLE_SPECS[table].columns
    names = ','.join('"'+c+'"' for c in columns)
    placeholders = ','.join('?' for _ in columns)
    conn.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
                     [tuple(row[c] for c in columns) for row in rows])


def equivalent(left, right):
    # Numeric SQLite affinity and JSON formatting must not turn retries into conflicts.
    def stable(row):
        row = dict(row)
        if row.get('notes'):
            try:
                row['notes'] = json.loads(row['notes'])
            except (TypeError,ValueError):
                pass
        return json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)
    return sorted(map(stable,left)) == sorted(map(stable,right))
