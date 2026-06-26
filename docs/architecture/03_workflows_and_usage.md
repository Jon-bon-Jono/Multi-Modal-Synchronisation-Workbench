# 05 — Workflows and Usage

**Status:** Operational usage note  
**Purpose of this file:** Describe how the workbench should be used in practice. This file complements the main context file and the schema file.

---

## 1. What this file covers

This file is about **procedure**, not theory.

It explains how to:

- organise a new recording unit,
- register assets without hard-coding machine-specific paths,
- ingest sample metadata,
- estimate timelines,
- create anchors,
- fit synchronisation models,
- generate sample mappings,
- inspect and revise results,
- share subsets of the dataset with other users.

---

## 2. Working assumptions

The workbench assumes the following:

- a subject may have multiple device runs,
- a device restart creates a new run,
- timestamps may be incomplete, noisy, or derived,
- manual anchors are expensive to create and should be reusable,
- canonical metadata should stay portable across machines,
- large payloads should remain in artifacts rather than core tables.

---

## 3. Recommended usage pattern

For each subject or recording unit, the normal order is:

1. create the subject entry,
2. register device runs,
3. register run-level assets,
4. ingest canonical sample rows,
5. ingest raw or observed timing signals,
6. build one or more run timeline models,
7. place anchors across runs,
8. fit a sync model for a chosen source–target pair,
9. generate a mapping version,
10. inspect diagnostics and iterate if needed.

Do not start by forcing everything into one shared timestamp column. That removes information too early.

---

## 4. Workflow A — Create the recording unit

Create one `SUBJECT` row for the top-level unit you want to manage.

In the current project this may correspond to a participant, or to a practical recording bundle if that is how the data are packaged.

At this stage, only store stable metadata. Keep it light.

Typical actions:

- assign `subject_id`,
- record experiment date or label,
- add brief notes if needed.

Do not try to encode device-run structure here. That belongs in `DEVICE_RUN`.

---

## 5. Workflow B — Register device runs

Create one `DEVICE_RUN` row for each uninterrupted acquisition segment.

Examples:

- Kinect RGB recording,
- Kinect depth recording,
- radar point-cloud stream,
- radar raw stream,
- IMU stream.

If a stream stops and resumes, create a new run. Do not stretch one run across a discontinuity.

Typical fields to populate early:

- `subject_id`,
- `run_id`,
- `device_type`,
- best available run start and end wallclock estimates,
- nominal FPS or sample rate,
- notes on interruptions or acquisition issues.

This is the first place where the data become operationally useful. Most downstream work is run-based, not subject-wide.

---

## 6. Workflow C — Register assets

Register files and folders in `RUN_ASSET` before trying to resolve every path on every machine.

Examples:

- RGB video,
- depth video,
- point-cloud export,
- raw radar capture folder,
- calibration files,
- temporary ingestion package,
- derived summaries.

### 6.1 Canonical rule

The canonical metadata should store a **portable asset reference**, not a machine-specific absolute path.

Good examples:

- `subject_07/kinect/rgb/video.mp4`
- `subject_07/radar_raw/session_a/`
- `derived/subject_07/radar_pc/pc_frames.parquet`

Bad examples:

- `C:\Users\Jonathan\Desktop\...`
- `/mnt/hpc_share/project_xyz/...`

### 6.2 Local resolution strategy

Actual path resolution should happen through configuration outside the canonical tables.

A practical pattern is:

- define one or more root directories per modality or storage class,
- resolve `storage_key + asset_ref` against those roots,
- allow per-user or per-machine overrides,
- keep the canonical metadata unchanged.

This is important because the same project may be used from:

- a Windows workstation,
- an external drive,
- OneDrive,
- a Linux/HPC machine,
- a research assistant’s local subset.

---

## 7. Workflow D — Ingest canonical sample rows

After runs are known, create `RUN_SAMPLE` rows.

Each row represents one canonical sample within a run. Usually this will be one frame.

Populate at least:

- `subject_id`,
- `run_id`,
- `sample_index`,
- `sample_kind`.

### 7.1 Important rule

`sample_index` is the canonical index within a run.

Original frame numbers, packet IDs, or vendor frame counters can be stored separately if needed, but they should not replace `sample_index` unless they are guaranteed to behave canonically.

### 7.2 Large payloads

Do not store large arrays directly in the core sample table.

Put large per-sample objects in artifacts, for example:

- pose arrays,
- point clouds,
- radar tensors,
- per-frame dictionaries,
- image crops,
- debug visualisations.

Reference those through `SAMPLE_ARTIFACT` or run-level assets.

---

## 8. Workflow E — Ingest timing evidence

The workbench should preserve raw timing evidence rather than collapsing it immediately.

Examples of timing evidence include:

- Kinect PTS,
- Kinect device elapsed time,
- host OS timestamps,
- parser completion time,
- inferred uniform-cadence times for raw radar.

The key idea is simple: observed timing evidence should remain inspectable.

Do not overwrite one timing signal with another just because one seems more convenient.

### 8.1 Temporary ingestion package

A temporary wide-format ingestion package is acceptable as a staging format.

Its purpose is to make it easy to gather messy source metadata into one place before converting into the canonical tables.

That staging package is not the long-term data contract. It is only a bridge into the canonical model.

---

## 9. Workflow F — Build run timeline models

Once timing evidence exists, create one or more `RUN_TIMELINE_MODEL` entries for a run.

A timeline model assigns a usable time axis to samples in that run.

Examples:

- direct use of PTS,
- direct use of per-frame OS timestamps
- regression from device time to wallclock,
- uniform interpolation between logged start and end times,
- piecewise estimation around known discontinuities.

The output of this step is `SAMPLE_TIME_ESTIMATE`.

### 9.1 Why this step exists

This step isolates modality-specific timing problems before cross-modal synchronisation.

That keeps the sync model cleaner. It should not need to also explain every pathology of every raw time source.

### 9.2 Multiple timeline models are allowed

A run may have several timeline models.

That is useful when comparing alternatives such as:

- naive observed times,
- corrected wallclock estimates,
- a revised model after excluding bad segments.

Do not overwrite old models. Version them.

---

## 10. Workflow G — Place anchors

Anchors are manual cross-run correspondences.

Each anchor should capture a judged alignment event such as:

- a distinctive body transition,
- a clear movement onset,
- a peak pose,
- another visually identifiable event.

Use `ANCHOR` for the anchor itself and `ANCHOR_MEMBER` for the participating samples.

### 10.1 Good anchoring practice

Prefer anchors that are:

- visually distinctive,
- temporally localised,
- easy to identify in multiple modalities,
- spread across the run rather than clustered in one region.

### 10.2 Bad anchoring practice

Avoid relying heavily on:

- ambiguous repetitive motion,
- long static periods,
- anchors concentrated only at the beginning,
- anchors that cross a run restart or discontinuity without acknowledging it.

### 10.3 Important rule

Anchors should remain reusable. They are not owned by one sync model.

A sync model may select a subset of anchors through `MODEL_ANCHOR`, but the anchor data themselves should stay independent.

---

## 11. Workflow H — Fit a sync model

After timeline models and anchors exist, create a `SYNC_MODEL` for a chosen source run and target run.

Examples:

- Kinect RGB to radar point cloud,
- Kinect RGB to radar raw,
- depth to RGB,
- radar raw to radar point cloud.

A sync model should specify:

- source run,
- target run,
- optional source and target timeline models,
- model type,
- extrapolation policy,
- parameter payload,
- notes.

### 11.1 What the sync model should do

Its job is to map one run timeline into another.

That is all.

Do not bury user-interface decisions, file-resolution logic, or final one-to-one sample-selection policies inside the sync model definition.

### 11.2 Extrapolation policy

The model should state how it behaves outside supported anchor regions.

Typical regimes:

- supported interpolation,
- weak extrapolation near the boundary,
- unsupported far outside anchor coverage.

Make this explicit. Do not silently treat all mapped samples as equally trustworthy.

---

## 12. Workflow I — Generate a mapping version

A `MAPPING_VERSION` converts a sync model into concrete sample correspondences.

This is where the system answers questions such as:

> For this source sample, which target sample is the best match?

The mapping version may include rules for:

- nearest-neighbour selection in target time,
- candidate ranking,
- support-region labels,
- primary versus secondary candidates,
- confidence scoring.

The detailed row-level output lives in `SAMPLE_MAPPING`.

### 12.1 Keep this separate from the sync model

This separation matters because the same sync model may support different matching policies.

For example, one mapping version may keep the top 3 candidates, while another may keep only the primary candidate.

---

## 13. Workflow J — Inspect diagnostics

Every mapping pass should be inspected before being treated as final.

Useful checks include:

- anchor residuals,
- time residual patterns over the run,
- boundary behaviour near the first and last anchor,
- duplicated target assignments,
- large gaps,
- behaviour around dropped-frame regions,
- behaviour around sensor restarts.

### 13.1 What to do if results look bad

Revise the smallest layer that plausibly explains the problem:

- bad asset registration: fix assets,
- bad raw timing evidence: fix ingestion,
- bad within-run time axis: revise timeline model,
- bad cross-run relation: revise sync model,
- bad candidate selection policy: revise mapping version.

Do not immediately throw away anchors or redesign the schema unless the problem really requires it.

---

## 14. Workflow K — Iterate without losing provenance

When you revise the work, create a new versioned object rather than mutating the old one in place.

Typical examples:

- new timeline model after excluding a bad segment,
- new sync model using a different anchor subset,
- new mapping version with a stricter primary-selection policy.

The point is to preserve the lineage of results.

This matters for debugging and for later comparison.

---

## 15. Workflow L — Share subsets with other users

The workbench should support partial local copies.

A research assistant should not need the entire dataset to work on one subject or one modality.

A sensible sharing pattern is:

- send the canonical metadata,
- send only the relevant assets or a small subset,
- let the recipient configure local roots or overrides,
- keep `asset_ref` unchanged.

That way the same metadata can function across different machines and storage layouts.

---