# Additive subject ingestion

Use one master database and separate, immutable package directories per subject or
processing revision. `import-package` is the normal path for new Kinect runs and
offline radar results. It does not rebuild existing tables or touch mappings and
anchors. `ingest-temp` now requires a new database path and remains a bootstrap
tool for legacy packages.

```powershell
syncwb import-package --input "<09_SY_initial>" --sqlite workbench.sqlite --artifact-root artifact_store --rgb-root "D:/smart_cup_recordings/Kinect" --dry-run
syncwb import-package --input "<09_SY_initial>" --sqlite workbench.sqlite --artifact-root artifact_store --rgb-root "D:/smart_cup_recordings/Kinect"
```

The database must already have the versioned-cloud schema. Existing legacy stores
can use `migrate-point-clouds` to create a migrated copy. Back up the master using
SQLite backup before an operational import; keep its existing artifact store.

## Package contents

- `device_runs.zst`: original catalogue rows for selected subject/run/device keys.
- Optional `rgb_samples.zst`: complete Kinect acquisitions, with native frame
  numbers 1..N, poses (2D/confidence/3D), activities, people counts and RGB references.
- Optional `radar_raw_samples.zst` plus `radar_raw_point_cloud_versions.json`:
  the existing versioned offline-radar contract. Both must be present together.

At least one sample modality is required. RGB references must resolve under
`--rgb-root`. The importer does not copy videos, read ADC or HDF5 source files, or
rerun processing. Online-radar packages are deliberately rejected by this command.
Extra catalogue rows are ignored; only runs selected by payloads are imported.
Kinect-only and raw-only deliveries are supported. Missing modalities do not mean
deletion. A later raw-only package can add another version of an existing run.

## Identity and conflicts

Acquisition identity is `(subject_id, run_id, device_type)`. Frame identity adds
the original frame number, translated to the canonical zero-based sample index.
Full native Kinect numbering is required; cropped/rebased packages are rejected.
Existing acquisitions must retain the same full frame coverage and timings.
Package-local dataframe indices do not identify captured frames.

New rows are inserted. Identical rows are retained, ignoring generated creation
timestamps and descriptive labels/notes (except source-frame and video-integrity
notes). Existing subject annotations are preserved. Metadata-only registered runs
can gain their missing samples, timelines and payloads. Different content under an
existing identity is an error, never an implicit replacement. Raw result IDs remain
immutable; different results are separate versions. There is no blanket overwrite
flag. Correcting frame identity/timing or replacing Kinect payloads needs a separate,
explicit correction workflow.

NPZ comparisons check array indices, shape, dtype and values, including NaNs;
compression and ZIP timestamps do not define a different payload. Existing raw
bundles are also checked against their registered checksum. RGB videos receive a
`rgb_video_integrity` run-asset record with a SHA-256 in its notes; subsequent imports
reject changed video content. Legacy videos establish their integrity baseline on
their first additive import, since their historical bytes were not recorded.

## Transactions, dry runs and recovery

The importer uses the existing transformers/artifact builders in a temporary
database and artifact directory. Dry runs perform this full staging and comparison
work, including reading arrays and video hashes, but open the master read-only and
never publish files. Allow temporary disk space and memory for one package.

Real imports hold one SQLite write transaction for conflict checks and additions.
Artifact files are published by exclusive creation after comparisons pass, then
metadata commits. Ordinary failures roll back SQL and remove files created by that
attempt. A hard process/power failure can leave an unregistered file; the next
attempt rejects that destination. Inspect and move aside an orphan before retrying.
Missing/damaged registered assets are also rejected instead of silently repaired.
SQLite and the external artifact filesystem are not jointly crash-atomic.

Import reports list added and already-present rows per table. Reimporting the same
package produces `already_imported` with zero added rows/files. Navigation mappings
are a separate explicit step (`map-nearest`); import does not automatically fit or
change synchronization. Student packages and training exports are created later.
