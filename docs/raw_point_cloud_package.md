# Versioned offline raw point-cloud temporary package

Status: the ETL producer and SyncWB version-aware backend are implemented.
For combined Kinect/offline packages and progressive subject additions, use
[`import-package`](additive_ingestion.md). It accepts the files together, stages
both modalities, and adds missing data without resetting the master database.
Use `migrate-point-clouds` and `import-raw-point-clouds` below. The GUI now
selects a single cloud before launch and records that selection in anchor/export
provenance.
The legacy `ingest-temp`, `ingest-raw-temp`, and `build-artifacts` commands reject
this package so they cannot discard its version/status contract.

## Agreed scope

- Every processing result is a version of payloads for an existing `radar_raw`
  acquisition. Processing settings do not create new captured frames or runs.
- Tracking-enabled HDF5 results only. Tracking-disabled results are rejected.
- Raw point arrays are float32, with six columns: x, y, z, radial velocity, SNR,
  and GTRACK association ID. The producer appends the HDF5 association vector;
  it does not change the physical point values or filter out points.
- `point_count_filtered` excludes exactly GTRACK IDs 253, 254, and 255. IDs 250,
  251, and 252 remain. This is a visualization policy, not a detector-quality or
  people-count assertion. `point_count` counts every stored point.
- Package import does not transform geometry. Optional [spatial calibration](spatial_calibration.md)
  is applied later by the GUI and training exporter, with its own geometry provenance.
  Original generator metadata/config are retained for provenance only. Geometry
  corrections are deferred.
- One source/version is selected BEFORE an anchoring session and remains fixed
  for that session. There will be no source/version selector during a session.
- Migration explicitly registers existing online clouds as `online_original`
  throughout canonical artifact metadata. Their existing
  artifact files need not be rewritten merely to establish version identity.

## Producer and execution

In the UNSW-PANOPTES-ETL-Pipeline repository:

- `upetl/syncwb_raw_packaging.py` is the standalone producer.
- The final section of `sync_workbench_packaging.ipynb` is the user entry point.
- `requirements-syncwb-packaging.txt` lists its small dependency set.

Run only the final notebook code cell; it does not need earlier notebook cells,
the old placeholder pickle, or the radar DSP/processing module. Earlier cells
retain their legacy behavior, including writing the old placeholder package.
Do not rerun the legacy raw-catalogue cell after generating the new package.

The final cell selects sources explicitly and regenerates their FULL acquisition
indices from the connected recordings, then attaches each selected cloud version.
The old dataset-wide `radar_raw_samples.zst` can be replaced. The replacement
contains only the selected acquisition/version combinations, not automatically
every subject previously present in that file. Add sources to the list to package
more runs/versions. Re-running replaces the output; it does not append it.

Input recordings are read-only. Only log/config text, capture sizes, and the first
HSI header are read to count acquisition frames. The producer does not parse ADC
frames, regenerate clouds, save raw timing tables, move the HDF5, or update SQLite.
Output inside the configured recordings root is rejected. HDF5 can remain in the
offline generator's existing build directory.

## Output files

Both files belong together in the temporary ingestion directory:

1. `radar_raw_samples.zst`: zstd-compressed pandas pickle (protocol 5).
2. `radar_raw_point_cloud_versions.json`: package manifest and version registry.

Other package files (`device_runs.zst`, `rgb_samples.zst`, and
`radar_pc_samples.zst`) are untouched. The importer verifies the selected run identities against `device_runs.zst`.
Subjects must already be registered. Missing raw acquisition rows can be added;
existing acquisition rows must agree exactly.

| Sample column | Contract |
|---|---|
| `subject_id`, `run_id` | Explicit acquisition identity; run ID is the original session folder name |
| `frame_number` | int64, official one-based ACQUISITION frame number |
| `sample_kind` | Always `frame` |
| `estimated_wallclock_from_start_end` | Original timezone-unspecified wallclock, `%Y-%m-%dT%H:%M:%S.%f` |
| `points` | `(P, 6)` float32, all points; `(0, 6)` for an available empty frame or unprocessed frame |
| `point_count` | Nullable Int64; zero for an available empty frame, unavailable for an unprocessed frame |
| `point_count_filtered` | Nullable Int64; count excluding only 253/254/255, unavailable when unprocessed |
| `point_cloud_version_id` | Required reference to a manifest version |
| `point_status` | `available` or `unprocessed`; required to distinguish no detections from no processing |

Row key: `(subject_id, run_id, point_cloud_version_id, frame_number)`.
The one-based dataframe `frame_id` index is only package-local and MUST NOT be
used as a captured-frame identity. Multiple versions repeat acquisition rows;
canonical acquisition samples and timeline estimates are checked for agreement
and reused, not assigned new sample indices for each version.

## Frame identities and time

HDF5 `sample_index` contains the numerical filename ID, not necessarily a SyncWB
sample index. Every selected source must explicitly declare `frame_number_base`:

- 0: legacy bulk parser; official `frame_number = HDF5 sample_index + 1`.
- 1: official one-based extraction; official `frame_number = HDF5 sample_index`.

HDF5 local row and `sequence_index` are separate references, never time bases.
The version manifest preserves source IDs and sequence indices. IDs must be
strictly increasing, unique, and inside the acquisition frame range.

The full acquisition index is rebuilt from complete ADC frame counts and logged
start/end timestamps, using the same interpolation as the existing ETL catalogue.
The original capture layout cfg, not the cloud-processing cfg, determines frame
size. The producer supports the existing 16-bit complex, header-enabled ADC-only
layout. Split capture files must be consecutive from zero. Trailing partial-frame
bytes are excluded and their count is recorded.

All acquisition frames are retained for each selected version. A partial processing
selection does not shorten/rebase the run or stretch its timestamps. Frames outside
HDF5 coverage are `unprocessed`; available empty clouds retain zero counts.
Incomplete HDF5 output is rejected even when partial coverage was intentional:
a successfully completed subset is different from an interrupted processing job.

For the verified 19_MM example, HDF5 IDs 0..61786 map to acquisition frames
1..61787 in `Session-2024-January-15 09-43-27-126274`. Its full timestamps are
2024-01-15T09:44:08.000000 through 2024-01-15T10:35:37.000000. All 61,787
timestamps were checked against the old catalogue during implementation.
SyncWB canonical indices for this full raw acquisition remain
`sample_index = frame_number - 1`, independently of payload version/coverage.

## Version identity and provenance

Manifest schema: `syncwb.raw_point_cloud_package.v1`.
Supported HDF5 schema: `iwr6843.raw_point_cloud_sequence.hdf5.v1`.

The producer assigns `raw_<sha256>` at packaging. The digest hashes canonical
JSON (sorted keys, compact separators, UTF-8, ensure_ascii=False) with these fields:

```
hdf5_sha256, subject_id, run_id, device_type, source_frame_number_base
```

This identifies the exact completed HDF5 result AND its acquisition/numbering
binding. Moving the unchanged file or changing its readable label preserves the
ID. Changing the file's contents or its interpretation changes the ID. Equivalent
processing reruns may have different IDs because generation metadata changes;
this is result identity, not a claim of semantic recipe equivalence.

Each version records the readable label, HDF5 checksum/schema, original generator
metadata, embedded processing cfg, source frame IDs/sequence indices, coverage,
payload contract, visualization filter, and acquisition timing provenance.
Generator metadata includes executable version/hash, config hash, calibration
file hashes, creation time, and tracker starting position. A config hash alone is
not a sufficient version ID.

Acquisition provenance includes capture segment names/sizes, capture log/config
hashes, layout cfg text, frame count, start/end time, and a deterministic hash of
the COMPLETE acquisition frame-number/timestamp catalogue. Raw ADC contents are
not fully hashed; paths/sizes do not prove raw-data content identity. Existing
generator metadata is preserved without inventing missing source-code revisions.

The package also records packager version/source hash, creation time, and the
SHA-256 of the output sample pickle. The importer checks this checksum
before accepting the sample file/manifest pair, and rejects conflicting content
for an existing version. File publication uses staged files and
per-file atomic replacement, not a multi-file transaction; the checksum detects
a mismatched pair after an interrupted publication.

### Comparing settings across recordings

For cross-recording settings comparisons, use the retained `processing_cfg` and
`generator_metadata`, stored after import under `POINT_CLOUD_VERSION.provenance_json.version`.
Compare the config, executable/version, calibration sources and tracking setup;
the config hash alone is insufficient. Result IDs and readable labels do not
establish settings equivalence. The canonical registry does not assign a shared
`processing_recipe_id`. The [training exporter](training_export.md) computes a
conservative recipe fingerprint for export validation; it does not attempt
automatic semantic equivalence between different configurations.
See [the data-model comparison fields](02_data_model_and_tables.md#121-comparing-processing-settings-across-recordings).

## Anchor and mapping compatibility

Anchor endpoints remain captured-sample identities. Record the displayed
`point_cloud_version_id` in anchor provenance, not in endpoint keys. Raw version 1
anchors can be viewed/used with raw version 2 of the same acquisition, and a sync
mapping fitted from them remains applicable. Validate shared acquisition identity
and timing; coverage determines whether a cloud can be displayed, not whether
the captured sample or mapping exists. This does not transfer anchors/mappings
between `radar_pc` and `radar_raw`, which are distinct acquisition streams.

Backend registration, version-aware artifact keys/paths and summaries, additive
import, payload lookup, fixed GUI session selection, and anchor/export display
provenance are implemented. Legacy anchors retain unknown display provenance;
the GUI shows "not recorded" rather than guessing their original cloud.

## Backend workflow

Run from the SyncWB environment. Use a NEW output path for the database copy:

```powershell
syncwb migrate-point-clouds --source-sqlite workbench.sqlite --output-sqlite workbench_raw.sqlite
syncwb import-raw-point-clouds --input "<temporary-package-directory>" --sqlite workbench_raw.sqlite --artifact-root "<artifact-store-root>"
syncwb list-point-cloud-versions --sqlite workbench_raw.sqlite --subject 19_MM --run "Session-2024-January-15 09-43-27-126274" --device radar_raw
syncwb audit-artifacts --sqlite workbench_raw.sqlite --artifact-root "<artifact-store-root>"
```

Migration opens the source read-only and uses SQLite backup; it never replaces an
existing output file. The Python `migrate_point_cloud_versions(path)` function is
an explicit in-place migration for applications/tests; the CLI always creates a
copy. Migration changes payload metadata and registers online acquisitions,
including acquisitions without built assets. It leaves original artifact files,
sample identities, timestamps, anchors, synchronization models, and mappings alone.

For an isolated trial, copy the original artifact store too and use that copy as
`--artifact-root`; its existing relative references must still resolve. Do not
point output at the read-only recordings drive. A database copy alone does not
copy the external artifact files.

`import-raw-point-clouds` reads only the raw pickle, its manifest, and the sibling
`device_runs.zst`. It does not rerun radar processing, reopen HDF5, read raw capture
files, or ingest RGB/online siblings. It validates the full package before writes.
Raw samples retain `sample_index = frame_number - 1`. If existing frame coverage,
source frame numbers, or timestamps differ, import fails without renumbering or
replacing them. All versions in a package commit in one metadata transaction.

The importer stores all six float32 columns, including IDs 253/254/255; only the
summary's visualization count excludes them. Available empty frames have a real
empty bundle member and zero counts. Unprocessed frames have null counts and no
artifact member, but still retain their captured sample and timestamp.

Bundles use portable relative paths:
`point_cloud_versions/<point_cloud_version_id>/points.npz`.
The registry retains the complete version manifest and package provenance JSON,
plus the bundle SHA-256 and a payload/coverage fingerprint. A repeated import is
a no-op after consistency/hash checks. Conflicting payload or processing
provenance is rejected. Moving/relabeling the same source does not create a new
version; a repeat import retains its initially registered label/provenance.

Failures roll back SQL and remove files created by that import. Files and SQLite
are not a single crash-atomic transaction: an abrupt process/power failure can
leave an unregistered bundle. An existing unregistered destination is never
silently overwritten. Investigate and move that orphan out of the artifact store
before retrying. Re-import does not repair manually damaged metadata; use audit.

`PayloadService` accepts `point_cloud_version_id` for payloads, summaries and
windows. Versioned raw requires an explicit selection even with one version;
online defaults to its registered `online_original`. Non-radar summaries/artifacts
use an empty version ID. The old `raw_legacy` sentinel only preserves existing
unversioned raw artifacts and does not assert generator provenance.

`get_mapped_pair_payloads` and `PairInspectionService.inspect_pair` accept
`source_point_cloud_version_id` / `target_point_cloud_version_id`; `inspect-pair`
exposes `--source-point-cloud-version` / `--target-point-cloud-version`. Choosing a
payload version does not change the mapping. Missing cloud members raise on
single-payload access; sample summaries expose `unprocessed`. Raw window reads
raise if any in-acquisition frame in the window is unprocessed.

`audit-artifacts` checks version registration and joins summaries by version,
then verifies raw bundle hashes, coverage, member metadata, dtype, and counts.
`ArtifactAuditService.audit_cloud_versions()` runs just these version checks.
No live database has to be migrated to inspect a legacy online cloud: legacy
read normalization remains compatible, while explicit migration persists IDs.

## Validation and practical limits

The producer rejects unsupported/incomplete HDF5, tracking-disabled data,
nonfinite point values, malformed offsets, duplicate/out-of-range IDs, mismatched
association arrays, and capture-file gaps. It retains empty processed frames.
Synthetic tests exercise round-trip packaging, both numbering conventions,
version stability, shared acquisition timing, filtering, and protected inputs.

Packaging retains point arrays in the dataframe until writing the pickle. The
current 21.8-million-point recording needs approximately 500 MiB for six-column
float32 payloads alone, plus dataframe/compression overhead. HDF5 is read in
256-frame chunks to avoid an additional whole-session input array. Enough RAM
must be available; this is not a fully streaming pickle format.

The canonical NPZ writer and reader also materialize a whole version in memory.
This milestone keeps the existing bundle format; it does not add streaming or
random-access HDF5 serving. Call `PayloadService.clear_cache()` when releasing a
selected version. The GUI fixes its selection at startup and clears payload/video
caches when the session closes.


## GUI sessions and anchor display provenance

Start `syncwb anchoring-gui` with the database, artifact root, RGB root, and
subject. The launch dialog offers compatible **point-cloud source**, **version**,
and **synchronization mapping** choices before the anchoring window opens.
`--mapping-version` alone preselects a mapping; add `--point-cloud-version` to
select an exact session directly without the dialog. Online uses
`online_original`; raw requires an explicitly selected tracked version.
Canceling this dialog does not create a session or write to the database.

Only mappings from `kinect_rgb` to the chosen radar acquisition are offered.
A raw version can reuse another raw version's mapping for the same acquisition.
An RGB-to-`radar_pc` mapping does not become an RGB-to-`radar_raw` mapping merely
by selecting raw payloads. Create an initial raw navigation mapping using
`rgb_wallclock_from_pts` and `radar_raw_wallclock_from_start_end` if one does not
already exist. Navigation mappings are estimates, not anchor-fitted alignment.

The anchoring window shows the fixed selection in a banner and its full ID in a
tooltip. It has no source/version switching control. Opening a different version
requires closing the session and starting another. It starts on the first mapped
source/target pair; the displayed frames and sample boxes agree. The default geometry
and optional RGB projection are shared across online/raw clouds. Raw sessions can
add `--spatial-calibration PATH` for [calibrated Kinect alignment](spatial_calibration.md).
Existing pose
prediction files are indexed to online samples and are rejected for raw sessions;
omit `--pose-predictions` when using raw until version-bound raw predictions are
implemented.

A missing/unprocessed frame or incomplete point window clears the previous 3D
cloud, displays an availability message, and disables anchor placement. A processed
empty frame is explicitly labeled and remains a valid captured sample. Visualization
filtering still excludes GTRACK IDs 253, 254, and 255; IDs 250-252 remain. Raw payloads
are not filtered in storage. Closing a session releases its loaded bundles and video.

Each new anchor stores the following within the existing JSON `ANCHOR.notes`
`provenance` object (no schema migration or endpoint-key change is required):

- `point_cloud`: subject, run, device type, version ID, readable label, payload
  fingerprint, and acquisition timeline fingerprint when available.
- `session_id`, `session_started_at`, `annotator_id`, and
  `initial_mapping_version_id`.
- `display`: target sample, point-window radius, excluded association IDs (empty
  if the filter was off), and point availability status.

Exports include the current selection in `session.point_cloud`. Each exported
anchor retains its own original `notes.provenance.point_cloud`, even if created
while viewing another raw version. The export session is not a claim that every
anchor was placed using its current selection. JSON import preserves this
provenance. All anchors for the acquisition pair remain visible; the table's
"viewed cloud" column identifies their recorded label and exposes the ID via a
tooltip. Anchors and derived synchronization mappings remain version-independent.
