# Multi-modal Synchronisation Workbench v0.1

Backend-first initial implementation of the Multi-modal Synchronisation Workbench.

v0.1 is intentionally narrow. It turns the temporary ingestion package into a
canonical local workbench store, builds initial timeline models, and generates
crude RGB-to-radar nearest-time mappings through a recorded `identity_time`
`SYNC_MODEL`.

## What v0.1 does

- Reads temporary ingestion files:
  - `device_runs.zst`
  - `rgb_samples.zst`
  - `radar_pc_samples.zst`
- Creates canonical tables:
  - `SUBJECT`
  - `DEVICE_RUN`
  - `RUN_SAMPLE`
  - `RUN_ASSET`
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


## Important v0.1 design choices

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

Pose arrays, activity dictionaries, and point arrays are not embedded in the
canonical core tables. v0.1 preserves scalar metadata and timing structure. Large
payload artifact export can be added later.

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
