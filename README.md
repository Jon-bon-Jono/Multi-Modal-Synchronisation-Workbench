# Multi-modal Synchronisation Workbench v0.2.1

Backend-first implementation of the Multi-modal Synchronisation Workbench.

v0.2.1 keeps the v0.1 canonical ingestion and nearest-time mapping baseline, then adds an artifact and payload access layer for RGB pose/activity payloads and radar point-cloud payloads. The GUI-facing boundary is now service-based: frontends should call payload/pair-inspection services rather than reading temporary `.zst` files or artifact bundle internals directly.

## What the backend does

- Reads temporary ingestion files:
  - `device_runs.zst`
  - `rgb_samples.zst`
  - `radar_pc_samples.zst`
- Creates canonical tables:
  - `SUBJECT`
  - `DEVICE_RUN`
  - `RUN_SAMPLE`
  - `RUN_ASSET`
  - `SAMPLE_ARTIFACT`
  - `SAMPLE_SUMMARY`
  - `RUN_TIMELINE_MODEL`
  - `SAMPLE_TIME_ESTIMATE`
  - plus empty anchor/sync/mapping tables as needed
- Builds baseline timelines:
  - `rgb_wallclock_from_pts`
  - `rgb_pts_elapsed`
  - `rgb_kinect_elapsed_observed`
  - `rgb_kinect_elapsed_from_pts`
  - `rgb_smartcup_os_observed`
  - `rgb_smartcup_os_from_pts`
  - `radar_pc_observed_wallclock`
  - `radar_pc_linear_from_index`
- Generates nearest-time RGB-to-radar mappings with top-k candidates.
- Stores mapping provenance through `SYNC_MODEL -> MAPPING_VERSION -> SAMPLE_MAPPING`.
- Exports reports and optional Parquet/CSV canonical tables.
- Builds run-level artifact bundles for `pose2d`, `conf2d`, `pose3d`, `activity`, and radar `points`.
- Provides services and CLI commands for payload retrieval, artifact auditing, and mapped-pair inspection.

## Install for development

```bash
conda env create -f environment.yaml
conda activate syncwb
python -m pip install -e .
```

For Parquet export, install pyarrow:

```bash
python -m pip install -e ".[parquet]"
```

Without pyarrow, export falls back to CSV and writes an error note next to the
failed Parquet target.

## Ingest temporary files

```bash
syncwb ingest-temp \
  --input path/to/temp_ingestion_folder \
  --sqlite workbench.sqlite \
  --parquet canonical_export \
  --reports reports
```
* running this consecutive times will wipe all tables

## Inspect store summary

```bash
syncwb summary --sqlite workbench.sqlite
```

## Build v0.2.1 payload artifacts

Run this after `ingest-temp`. It reads payload columns from the temporary ingestion package, writes run-level artifact bundles, and upserts `RUN_ASSET`, `SAMPLE_ARTIFACT`, and `SAMPLE_SUMMARY` metadata into the SQLite store.

```bash
syncwb build-artifacts \
  --input-temp path/to/temp_ingestion_folder \
  --sqlite workbench.sqlite \
  --artifact-root artifact_store \
  --overwrite
```

The default artifact layout is:

```text
artifact_store/
  subjects/
    <subject_id>/
      kinect_rgb/
        <run_id>/
          pose2d.npz
          conf2d.npz
          pose3d.npz
          activity.jsonl
          sample_payload_manifest.parquet  # or .csv fallback when pyarrow is unavailable
      radar_pc/
        <run_id>/
          points.npz
          sample_payload_manifest.parquet  # or .csv fallback when pyarrow is unavailable
```

The `.npz` files use a ragged bundle layout with `sample_index`, `offsets`, and concatenated `values`. This avoids one file per frame and keeps payloads outside the canonical core tables.

## Audit payload artifacts

```bash
syncwb audit-artifacts \
  --sqlite workbench.sqlite \
  --artifact-root artifact_store \
  --issues-csv reports/artifact_audit.csv
```

The audit checks artifact file existence, ragged NPZ offset consistency, JSONL member readability, and basic `SAMPLE_ARTIFACT`/`SAMPLE_SUMMARY` consistency.

## Inspect a mapped pair and available payloads

```bash
syncwb inspect-pair \
  --sqlite workbench.sqlite \
  --artifact-root artifact_store \
  --subject P001 \
  --mapping-version rgb_to_pc_initial_v001 \
  --source-sample 123 \
  --json
```

This uses `SAMPLE_MAPPING` to find the target candidate and reports source/target summaries plus available payload roles and shapes. It is the service/CLI path the experimental GUI should build on.

## Generate an initial RGB-to-radar mapping for one source/target run pair

```bash
syncwb map-nearest \
  --sqlite workbench.sqlite \
  --subject P001 \
  --source-run "Session-..." \
  --source-device kinect_rgb \
  --source-timeline rgb_wallclock_from_pts \
  --target-run "Session-..." \
  --target-device radar_pc \
  --target-timeline radar_pc_linear_from_index \
  --mapping-version rgb_to_pc_initial_v001 \
  --top-k 3 \
  --source-window-policy target-overlap \
  --source-margin-ms 100 \
  --primary-policy supported-only
```

The command writes:

- one `SYNC_MODEL` row with `model_type = identity_time`,
- one `MAPPING_VERSION` row with mapping parameters in `parameters_json`,
- top-k `SAMPLE_MAPPING` candidate rows.

By default, the command refuses to reuse an existing mapping_version_id. Use --overwrite only when you intentionally want to replace an existing mapping version. When --overwrite is used, existing SAMPLE_MAPPING rows for that mapping version are deleted before regenerated rows are written.

## Generate mappings for all overlapping RGB/radar run pairs

```bash
syncwb map-nearest-all \
  --sqlite workbench.sqlite \
  --source-device kinect_rgb \
  --source-timeline rgb_wallclock_from_pts \
  --target-device radar_pc \
  --target-timeline radar_pc_linear_from_index \
  --mapping-version-prefix initial_rgb_to_pc_v001 \
  --top-k 3 \
  --min-overlap-sec 5 \
  --source-window-policy target-overlap \
  --source-margin-ms 100 \
  --primary-policy supported-only \
  --pair-report-csv reports/initial_rgb_to_pc_v001_pair_report.csv
```

This command searches for source-target run pairs from the same subject whose selected timelines overlap by at least `--min-overlap-sec`, generates deterministic mapping version IDs using the prefix, and writes a pair report.


## Important design choices

### Option A for mapping provenance

Every generated mapping references a `SYNC_MODEL`, even when the sync model is a
trivial `identity_time` relation between selected timelines. This prevents
anonymous direct joins and keeps the model/mapping separation intact.

### Derived Kinect/SmartCup timelines do not overwrite source data

Missing `kinect_internal_elapsed_sec` and `smartcup_os_time` values are not
filled in the temporary dataframes. Instead, v0.1 creates derived timeline models
such as `rgb_kinect_elapsed_from_pts` and `rgb_smartcup_os_from_pts` and records
residuals where observed values exist.

### Large arrays stay out of the canonical core

Pose arrays, activity dictionaries, and point arrays are not embedded directly in the canonical core tables. v0.2.1 writes them to run-level artifact bundles and references those bundles from `SAMPLE_ARTIFACT`. Scalar preview/filter fields live in `SAMPLE_SUMMARY`.

### Datetime and numeric time values are stored separately

The written `SAMPLE_TIME_ESTIMATE` table uses:

- `time_value_datetime`
- `time_value_sec`

This is a practical v0.1 adjustment to avoid representing relative seconds such
as PTS or Kinect elapsed time as fake datetimes.

### Primary mapping policy

The default `primary_policy` is `supported-only`.

This means `is_primary=True` is assigned only to the rank-1 candidate when `support_status=supported`. Weakly supported candidates remain in `SAMPLE_MAPPING`, but they are not primary unless a less conservative policy such as `within-max-delta` or `nearest-any` is explicitly selected.

### Source-window policy

The default `source_window_policy` is `target-overlap`.

This means source samples are mapped only when their predicted target time falls inside the target timeline coverage, optionally expanded by `source_margin_ms`. Use `--source-window-policy all` to generate mappings for all source samples, including those outside the target run.

## Run tests

```bash
python -m pytest
```
