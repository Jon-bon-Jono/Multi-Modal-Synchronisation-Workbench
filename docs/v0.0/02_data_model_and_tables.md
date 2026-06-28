# 03 — Data Model and Tables

**Status:** Canonical schema note for the current design stage  
**Purpose of this file:** Define the concrete tables, columns, keys, and storage conventions for the Multi-Modal Synchronisation Workbench.

This file should be read **after** `01_main_context_file.md`.

---

## 1. Scope of this schema

This schema is intended to support:

- subjects / top-level recording instances,
- uninterrupted device runs,
- samples within runs,
- run-level and sample-level assets,
- multiple time estimates per sample,
- manual anchors,
- run-specific timeline models,
- cross-run synchronisation models,
- versioned mapping outputs,
- portable asset references across different machines and users.

The schema is deliberately split into two layers:

1. **canonical core tables** for durable project metadata and synchronisation logic,
2. **optional ingestion or convenience tables/files** for modality-specific working data.

The canonical core should stay relatively stable. Convenience tables may evolve more freely.

---

## 2. Relationship to the current ER diagram

This file is aligned with the current canonical ER diagram in `sync_workbench_er_V0.1.mmd`.

The current canonical entities are:

- `SUBJECT`
- `DEVICE_RUN`
- `RUN_SAMPLE`
- `RUN_ASSET`
- `SAMPLE_ARTIFACT`
- `RUN_TIMELINE_MODEL`
- `SAMPLE_TIME_ESTIMATE`
- `ANCHOR`
- `ANCHOR_MEMBER`
- `SYNC_MODEL`
- `MODEL_ANCHOR`
- `MAPPING_VERSION`
- `SAMPLE_MAPPING`

---

## 3. General design rules

### 3.1 Stable IDs first

Use stable identifiers in the canonical tables. Avoid using file paths as keys.

### 3.2 Runs are uninterrupted acquisition segments

A `run_id` identifies one uninterrupted device acquisition segment for one modality/device stream. If a sensor restarts, that is a new run.

### 3.3 `sample_index` is canonical within a run

Each sample in a run gets a canonical integer `sample_index`, typically zero-based and strictly increasing.

Original frame numbers may also be stored if needed, but they are not the canonical join key unless explicitly identical.

### 3.4 Observed time and estimated time are separate

Raw timestamps, inferred timestamps, and timeline-model-assigned times must remain separate concepts.

### 3.5 Large arrays should not live in core table columns

Large arrays or structured payloads such as:

- 2D poses,
- 3D poses,
- point clouds,
- raw radar frame tensors,
- complex activity dictionaries,
- large JSON blobs,

should usually be stored as artifacts and referenced via `RUN_ASSET` or `SAMPLE_ARTIFACT`, rather than embedded directly into core table columns.

Scalar summaries and diagnostics are fine in tables.

### 3.6 Asset refs should be portable

Store canonical asset references in a portable form:

- logical refs,
- storage keys,
- relative paths,
- POSIX-style separators (`/`),
- no machine-specific absolute Windows or Linux paths in canonical metadata.

Filesystem resolution should happen through configuration.

### 3.7 Use one consistent datetime representation

Use one consistent timestamp format across the project: "%Y-%m-%dT%H:%M:%S.%f".

---

## 4. Recommended key conventions

### 4.1 Primary-key style

The ER currently uses mostly composite keys. That is acceptable and keeps the relationships explicit.

In practice, implementations may still choose to add surrogate IDs internally, but the **logical** keys below should remain valid.

### 4.2 Foreign-key style

All tables tied to a specific subject/run/sample should carry the relevant foreign-key columns explicitly, even if they could theoretically be inferred.

This improves inspectability and simplifies table-based workflows.

### 4.3 ID naming

Use the following naming pattern consistently:

- `subject_id`
- `run_id`
- `sample_index`
- `asset_id`
- `timeline_model_id`
- `anchor_id`
- `sync_model_id`
- `mapping_version_id`

---

## 5. Canonical core tables

## 5.1 `SUBJECT`

Represents the top-level recording unit.

In the current project, this may function as either:

- a true subject identifier, or
- a practical top-level recording/session unit,

depending on how the dataset is packaged.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | Canonical top-level identifier |
| `experiment_date` | string or date | no | Human-readable date or experiment label |
| `notes` | string | no | Free-text notes |

**Key:** `subject_id`

---

## 5.2 `DEVICE_RUN`

Represents one uninterrupted acquisition segment for one device/modality.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `run_id` | string | yes | Canonical run identifier within subject |
| `device_type` | string | yes | Device/modality type |
| `start_wallclock_est` | datetime | no | Best available run start time in wallclock coordinates |
| `end_wallclock_est` | datetime | no | Best available run end time in wallclock coordinates |
| `nominal_fps` | float | no | Nominal frame/sample rate |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `run_id`, `device_type`)

**Recommended controlled values for `device_type`:**

- `kinect_rgb`
- `kinect_depth`
- `radar_pc`
- `radar_raw`
- `imu_accel`
- `imu_gyro`
- `other`

**Important rule:** a sensor restart creates a new `DEVICE_RUN` row.

---

## 5.3 `RUN_SAMPLE`

Represents one canonical sample in one run.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `run_id` | string | yes | FK to `DEVICE_RUN` |
| `device_type` | string | yes | FK to `DEVICE_RUN` |
| `sample_index` | int | yes | Canonical sample index within run |
| `sample_kind` | string | yes | Type of sample |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `run_id`, `device_type`, `sample_index`)

**Recommended controlled values for `sample_kind`:**

- `frame`
- `pose_frame`
- `packet`
- `window`
- `event`
- `other`

**Rule:** `sample_index` should be monotonic and unique within a run.

---

## 5.4 `RUN_ASSET`

Represents assets attached to a whole run.

Typical examples:

- RGB video file,
- depth video file,
- point-cloud file,
- raw radar capture folder.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `asset_id` | string | yes | Canonical asset identifier |
| `subject_id` | string | yes | FK to `SUBJECT` |
| `run_id` | string | yes | FK to `DEVICE_RUN` |
| `device_type` | string | yes | FK to `DEVICE_RUN` |
| `asset_role` | string | yes | Role of the asset |
| `storage_key` | string | no | Logical storage namespace / root group |
| `asset_ref` | string | yes | Portable reference to the asset |
| `notes` | string | no | Free-text notes |

**Recommended implementation key:** `asset_id`  
**Relationship key:** (`subject_id`, `run_id`, `device_type`) identifies the owning run.

**Recommended controlled values for `asset_role`:**

- `rgb_video`
- `depth_video`
- `pc_file`
- `pc_folder`
- `raw_capture_folder`
- `preview_bundle`

**Path policy:** `asset_ref` should usually be a portable relative or logical path, not an absolute machine path.

---

## 5.5 `SAMPLE_ARTIFACT`

Represents assets attached to an individual sample.

Typical examples:

- RGB preview image,
- per-frame point-cloud file,
- raw frame file,
- per-frame pose JSON,
- derived diagnostic plot.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `run_id` | string | yes | FK to `DEVICE_RUN` |
| `device_type` | string | yes | FK to `DEVICE_RUN` |
| `sample_index` | int | yes | FK to `RUN_SAMPLE` |
| `artifact_role` | string | yes | Role of this sample artifact |
| `artifact_ref` | string | yes | Portable reference to the artifact |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `run_id`, `device_type`, `sample_index`, `artifact_role`)

**Recommended controlled values for `artifact_role`:**

- `preview_image`
- `pose2d_json`
- `pose3d_json`
- `pc_frame_file`
- `raw_frame_file`
- `activity_json`
- `diagnostic_plot`
- `other`

---

## 5.6 `RUN_TIMELINE_MODEL`

Represents a run-specific model that assigns or refines times for samples in one run.

This is the **within-run** timing layer.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `run_id` | string | yes | FK to `DEVICE_RUN` |
| `device_type` | string | yes | FK to `DEVICE_RUN` |
| `timeline_model_id` | string | yes | Canonical timeline model identifier |
| `timeline_model_name` | string | no | Human-readable name |
| `timeline_model_type` | string | yes | Model family/type |
| `source_time_basis` | string | no | Which observed time source or basis was used |
| `parameters_json` | string / json | no | Serialized parameters/configuration |
| `created_at` | datetime | no | Creation timestamp |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `run_id`, `device_type`, `timeline_model_id`)

**Recommended controlled values for `timeline_model_type`:**

- `identity_observed`
- `linear_from_index`
- `start_end_uniform`
- `regressed_observed_vs_index`
- `piecewise`
- `spline`
- `manual`
- `other`

**Examples:**
- point-cloud latent timeline estimated by regressing observed timestamps on frame index,
- raw-radar timeline derived uniformly from start/end times,
- Kinect timeline using observed PTS directly.

---

## 5.7 `SAMPLE_TIME_ESTIMATE`

Represents one assigned time for one sample under one timeline model.

This table is important because it prevents raw timestamps and modelled timestamps from being collapsed together.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `run_id` | string | yes | FK to `DEVICE_RUN` |
| `device_type` | string | yes | FK to `DEVICE_RUN` |
| `timeline_model_id` | string | yes | FK to `RUN_TIMELINE_MODEL` |
| `sample_index` | int | yes | FK to `RUN_SAMPLE` |
| `time_value` | datetime | yes | Assigned time under this model |
| `time_kind` | string | yes | What kind of time this is |
| `reference_time` | datetime | no | Optional reference origin or anchor time |
| `residual_ms` | float | no | Residual or timing error diagnostic |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `run_id`, `device_run`, `timeline_model_id`, `sample_index`)

**Recommended controlled values for `time_kind`:** 

- `observed_wallclock`
- `device_elapsed`
- `pts_based`
- `estimated_wallclock`
- `latent_acquisition_time`
- `sync_target_time`
- `other`

**Important rule:** one sample may have multiple rows here under different timeline models.

---

## 5.8 `ANCHOR`

Represents one manually defined correspondence event.

The anchor itself is intentionally model-independent.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `anchor_id` | string | yes | Canonical anchor identifier |
| `anchor_type` | string | yes | Type/category of anchor |
| `label` | string | no | Human-readable label |
| `confidence` | float | no | Human confidence score |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `anchor_id`)

**Recommended controlled values for `anchor_type`:**

- `manual_correspondence`
- `event_peak`
- `activity_transition`
- `pose_match`
- `sync_marker`
- `other`

---

## 5.9 `ANCHOR_MEMBER`

Represents the participation of a specific sample in a specific anchor.

This allows one anchor to include multiple modalities/runs/samples.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `anchor_id` | string | yes | FK to `ANCHOR` |
| `run_id` | string | yes | FK to `DEVICE_RUN` |
| `device_type` | string | yes | FK to `DEVICE_RUN` |
| `sample_index` | int | yes | FK to `RUN_SAMPLE` |
| `member_role` | string | no | Optional role within the anchor |
| `confidence` | float | no | Confidence for this member assignment |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `anchor_id`, `run_id`, `device_type`, `sample_index`)

**Recommended uses of `member_role`:**

- `reference`
- `supporting`
- `kinect_member`
- `radar_member`
- `other`

---

## 5.10 `SYNC_MODEL`

Represents a model that maps one run timeline to another run timeline.

This is the **between-run** timing layer.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `sync_model_id` | string | yes | Canonical sync model identifier |
| `source_run_id` | string | yes | Source run |
| `source_device_type` | string | yes | Source device type |
| `target_run_id` | string | yes | Target run |
| `target_device_type` | string | yes | Target device type |
| `source_timeline_model_id` | string | no | Source-side timeline model used |
| `target_timeline_model_id` | string | no | Target-side timeline model used |
| `model_name` | string | no | Human-readable name |
| `model_type` | string | yes | Sync model family |
| `extrapolation_policy` | string | no | Policy outside anchor-supported regions |
| `parameters_json` | string / json | no | Serialized parameters/configuration |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `sync_model_id`)

**Recommended controlled values for `model_type`:**

- `piecewise_affine`
- `global_affine`
- `piecewise_linear`
- `spline`
- `manual`
- `other`

**Recommended controlled values for `extrapolation_policy`:**

- `disallow`
- `allow_nearest_only`
- `allow_linear`
- `allow_with_penalty`
- `manual`

---

## 5.11 `MODEL_ANCHOR`

Associates a sync model with the anchors it uses.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `sync_model_id` | string | yes | FK to `SYNC_MODEL` |
| `anchor_id` | string | yes | FK to `ANCHOR` |

**Logical key:** (`subject_id`, `sync_model_id`, `anchor_id`)

This table exists so that anchors remain durable, reusable objects while sync models explicitly declare which anchors they used.

---

## 5.12 `MAPPING_VERSION`

Represents one generated mapping output set.

A mapping version is derived from some method and often based on a sync model, but it is not the same thing as the sync model itself.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `mapping_version_id` | string | yes | Canonical mapping version identifier |
| `source_run_id` | string | yes | Source run |
| `source_device_type` | string | yes | Source device type |
| `target_run_id` | string | yes | Target run |
| `target_device_type` | string | yes | Target device type |
| `mapping_name` | string | no | Human-readable name |
| `mapping_method` | string | yes | Method used to generate mappings |
| `created_at` | datetime | no | Creation timestamp |
| `parent_mapping_version_id` | string | no | Parent mapping version if derived/refined |
| `source_sync_model_id` | string | no | Sync model used as the basis |
| `notes` | string | no | Free-text notes |

**Logical key:** (`subject_id`, `mapping_version_id`)

**Recommended controlled values for `mapping_method`:**

- `nearest_predicted_time`
- `nearest_observed_wallclock`
- `windowed_nearest`
- `top_k_candidates`
- `manual`
- `other`

---

## 5.13 `SAMPLE_MAPPING`

Represents one candidate or selected mapping pair between one source sample and one target sample.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `subject_id` | string | yes | FK to `SUBJECT` |
| `mapping_version_id` | string | yes | FK to `MAPPING_VERSION` |
| `source_run_id` | string | yes | Source run |
| `source_device_type` | string | yes | Source device type |
| `source_sample_index` | int | yes | Source sample |
| `target_run_id` | string | yes | Target run |
| `target_device_type` | string | yes | Target device type |
| `target_sample_index` | int | yes | Target sample |
| `predicted_minus_estimated_ms` | float | no | Timing residual for this mapping pair |
| `rank` | int | no | Candidate ranking within source sample |
| `is_primary` | bool | no | Whether this is the selected primary match |
| `mapping_region_type` | string | no | Interpolation / extrapolation / unsupported region type |
| `support_status` | string | no | Support/confidence category |
| `confidence_score` | float | no | Optional confidence score |

**Logical key:** (`subject_id`, `mapping_version_id`, `source_run_id`, `source_device_type`, `source_sample_index`, `target_run_id`, `target_device_type`, `target_sample_index`)

**Recommended controlled values for `mapping_region_type`:**

- `interpolation`
- `left_extrapolation`
- `right_extrapolation`
- `unsupported`
- `manual`

**Recommended controlled values for `support_status`:**

- `supported`
- `weak_support`
- `extrapolated`
- `outside_run`
- `missing_target`
- `manual_override`

---

## 6. Relationship summary

The core intended relationships are:

- one `SUBJECT` has many `DEVICE_RUN`s,
- one `DEVICE_RUN` has many `RUN_SAMPLE`s,
- one `DEVICE_RUN` has many `RUN_ASSET`s,
- one `RUN_SAMPLE` can have many `SAMPLE_ARTIFACT`s,
- one `RUN_TIMELINE_MODEL` assigns many `SAMPLE_TIME_ESTIMATE`s,
- one `ANCHOR` has many `ANCHOR_MEMBER`s,
- one `SYNC_MODEL` may use many anchors through `MODEL_ANCHOR`,
- one `MAPPING_VERSION` contains many `SAMPLE_MAPPING`s.

This is the operational relational core of the workbench.

---

## 7. Recommended storage conventions for paths and assets

### 7.1 Do not store absolute machine paths canonically

Avoid values like:

- `C:\Users\Jonathan\...`
- `/mnt/datasets/...`

in canonical tables unless there is a very specific temporary reason.

### 7.2 Preferred pattern

Use combinations such as:

- `storage_key = radar_pc_local`
- `asset_ref = subject_001/session_a/radar_pc/frame_000123.npy`

or

- `storage_key = onedrive_main`
- `asset_ref = subject_001/rgb/video.mp4`

### 7.3 Use POSIX-style separators in canonical refs

Even on Windows, prefer `/` inside canonical metadata fields.

Path resolution code can convert to the local OS format when needed.

### 7.4 Multiple roots are expected

The project already assumes that:

- different modalities may live under different roots,
- different users may register different roots,
- assistants may only possess a subset of the assets.

That root-resolution logic should live in configuration and code, not in the core relational schema.

---

## 8. Recommended observed time sources by modality

This section does **not** create extra tables. It simply clarifies what kinds of observed or modelled times are expected to appear in `SAMPLE_TIME_ESTIMATE` and related ingestion data.

### 8.1 Kinect RGB / depth

Possible sources include:

- video PTS,
- Kinect internal elapsed timestamp,
- SmartCup OS timestamp,
- inferred wallclock estimate,
- future latent timeline estimate if ever needed.

### 8.2 Radar point cloud

Possible sources include:

- observed OS timestamp captured after parser completion,
- latent acquisition-time estimate from regression vs frame index.

### 8.3 Radar raw

Possible sources include:

- run start wallclock,
- run end wallclock,
- inferred per-frame times under start/end constrained interpolation,
- any future improved latent timeline estimate.

---

## 9. Temporary ingestion files

Rather than exposing the full raw dataset, the data necessary for this synchronisation workbench has been compiled into pandas dataframes and pickled into temporary zst files which will be ingested by some preprocessing scripts and transformed to meet the contract established by the entity relation diagram.

A first-pass ingestion package contains the files:

- `device_runs.zst`
- `rgb_samples.zst`
- `radar_pc_samples.zst`

Later, the ingestion package may also contain the files:

- `radar_raw_samples.zst`
- `run_assets.zst`

In the past, we discussed the temporary files: `initial_mapping_rgb_to_pc.zst` and `initial_mapping_rgb_to_raw.zst`, though this initial mapping can be derived from existing zst files. 

### 9.1 `device_runs.zst`

Temporary file containing serialised pandas data frame corresponding largely to `DEVICE_RUN`. One row per uninterrupted device run.

| Column | DType | Description | NA |
| ------ | ----- | ----------- | -- |
| `subject_id` | Object (str) | Unique for each subject involved in experiments. | Never |
| `run_id` | Object (str) | Unique for each execution of SmartCup. This is essentially a datetime string with prefix "Session". For each SmartCup execution, data is saved in a folder with `run_id` as the name. Radar pc and radar raw data streams share the same `run_id` because they were captured on Host 2 in the same SmartCup session/execution. | Never | 
| `device_type` | Object (str) | Options: "kinect_rgb", "kinect_depth", "radar_pc", "radar_raw", "imu_accel". | Never |
| `start_wallclock_est` | Object (str) | Estimated start datetime for the device run. This is a string formatted using the official sync workbench datetime format "%Y-%m-%dT%H:%M:%S.%f". Each unique `device_type` had its own unique protocol for deriving this, although for some problematic device runs we had to improvise and derive this using irregular and potentially less accurate means. | Never |
| `end_wallclock_est` | Object (str) | Estimated end datetime for the device run. This is a string formatted using the official sync workbench datetime format "%Y-%m-%dT%H:%M:%S.%f". Each unique `device_type` had its own unique protocol for deriving this, although for some problematic device runs we had to improvise and derive this using irregular and potentially less accurate means. | Never |
| `nominal_fps` | int64 | Expected frame rate which was set for the device during this run. Should be constant for each unique `device_type` | Never |
| `notes` | Object (str) | Some info if there is anything irregular/unique about this device run, otherwise this is just empty string "". | Never |
| `source_host_id` | Object (str) | Name of host system which ran SmartCup for this device run. Options: "host1", "host2", "host3" | Never |
| `session_folder_ref` | Object (str) | File path stem formatted like `f"{subject_id}/{run_id}"` | Never |

#### Primary Key
Each entry in device_runs.zst can be uniquely identified by `run_id` + `device_type`. Device types 'radar_pc' and 'radar_raw' are always captured together on Host 2 in the same SmartCup session. 

#### Wallclock derivations for device type `kinect_rgb`
* For each 3D pose detection sent from Kinect to Host 1, SmartCup saved the current Windows OS datetime along with the elapsed time since the start of the recording (which Kinect maintained using its own internal clock).
* `start_wallclock_est` was derived by taking the first 3D pose's OS datetime and subtracting the Kinect's elapsed time
* `end_wallclock_est` was derived from the modification time of the raw video file (checking Windows OS logs to see when the original raw video file was last modified). 

#### Wallclock derivations for device type `radar_pc`
* For every radar pc frame sent from MMWAVEICBOOST to Host 2, SmartCup saved the current Windows OS datetime
* `start_wallclock_est` is the Windows OS datetime for the first frame in the device run
* `end_wallclock_est` is the Windows OS datetime for the last frame in the device run

#### Wallclock derivations for device type `radar_raw`
* The DCA1000_CLI application used for collecting raw radar frames logs the start and end times of the recording using Host 2's Windows OS datetime so `start_wallclock_est` and `end_wallclock_est` are taken from this. 
* The device run with `run_id`=="Session-2023-November-09 15-41-28-000000" and `device-type`== "radar_raw" was missing the DCA1000_CLI log file. I'm pretty sure this device run's `start_wallclock_est` and `end_wallclock_est` were inferred from the DCA1000_CLI global log file. Not sure how these datetime stamps differ from the local log file, so this might not be be exactly consistent/reliable compared with other raw radar runs.

#### Wallclock derivations for device type `imu_accel`
* SmartCup running on Host 3 logs the Windows OS datetimes each time it receives some packets from the wearables. I think packets might contain multiple wearbale sensor frames taken at consecutive points in time, so the resolution isn't great. 
* `start_wallclock_est` and `end_wallclock_est` were taken from the Os datetimes of the first and last of these frames. 
* There is some other timestamp attached to these frames but I can't make sense of it. We wont be using `imu_accel` data for quite a while, so we'll circle back to this and improve these timestamps

#### Missing Runs
The device run with `run_id`=="Session-2024-January-18 10-19-47-431085" and `device_type`=="radar_raw" was completely lost, so there is no raw radar frames only point cloud radar frames. This was quite a short session though. 

### 9.2 `rgb_samples.zst`

Temporary file containing serialised pandas data frame for Kinect/RGB-specific metadata. One row per Kinect RGB frame. The dataframe index name is 'frame_id' and represents a unique rgb frame identifier over the entire dataset starting from 1. 

| Column | DType | Description | NA |
| ------ | ----- | ----------- | -- |
| `frame_number` | int64 | RGB frame number for the current device run. | Never |
| `pts_sec` | float64 | Native RGB PTS from the raw Kinect RGB video recording (in seconds). | Never |
| `subject_id` | Object (str) | Unique for each subject involved in experiments | Never |
| `run_id` | Object (str) | Unique for each execution of SmartCup. This is essentially a datetime string with prefix "Session". For each SmartCup execution, data is saved in a folder with `run_id` as the name. Radar pc and radar raw data streams share the same `run_id` because they were captured on Host 2 in the same SmartCup session/execution. | Never | 
| `sample_kind` | Object (str) | This is always "frame". | Never |
| `video_ref` | Object (str) | Path stem to mp4 video file formatted like `f"{subject_id}/{run_id}/kinect_camera_recording_rgb_lq.mp4"`. | Never |
| `num_people` | int64 | Official number of people in each frame as counted by a human. | Never |
| `wallclock_est` | Object (str) | Best-effort absolute wallclock timestamp derived from the current `pts_sec` added to the device run's `start_wallclock_est`. This is a string formatted using the official sync workbench datetime format "%Y-%m-%dT%H:%M:%S.%f". Every row has a `wallclock_est`. | Never |
| `pose3d` | Object (Numpy array with dtype float64) | 3D pose estimates for current frame as a numpy array `X` with shape (`M`, 32, 4) where `M` is the number of detections made by the Kinect, each with 32 keypoints. `X[0,:,-1]` is the confidence score. `np.empty((0, len(KINECT_JOINT_NAMES), 4), dtype=float)` is used when no 3D detections were made in the frame. | Never |
| `num_3d` | int32 | Number of 3D pose detections made by the Kinect. | Never |
| `kinect_internal_elapsed_sec` | float64 | Seconds elapsed since start of Kinect device run. This is maintained by the Kinect's internal clock. Not available when either the Kinect didn't detect any people or the initial detection captured by SmartCup was lost. | NaN used when this data isn't available. |
| `smartcup_os_time` | Object (str) | Host1's Windows OS wallclock datetime stamp logged each time it received a 3D pose estimate from the Kinect. Not available when either the Kinect didn't detect any people or the initial detection captured by SmartCup was lost. | Never, empty string used when this data isn't available |
| `pose2d` | Object (Numpy array with dtype float64) | 2D poses estimated for the current framee as a numpy array `Y` with shape (`N`, 26, 3) where `N` is the number of detections made by RTMPose, each with 26 keypoints. `Y[0,:,-1]` is the confidence score. `np.empty((0, len(BODY8_JOINT_NAMES), 3), dtype=float)` is used when no 2D detections were made in the frame. | Never | 
| `conf2d` | Object (Numpy array with dtype float64) | Overall detection-confidence/'objectness' score for each 2D pose detection as a numpy array with shape (`N`,). `np.empty((0), dtype=float)` is used when no 2D detections were made in the frame. | Never |
| `num_2d` | int64 | Number of 2D pose detections made by RTMPose. | Never |
| `activity` | object (Dict) | Dictionary with keys: ['01-PhysicalState', '01-Activity', '01-Objects', '01-Notes', '02-PhysicalState', '02-Activity', '02-Objects']. Each dictionary item contains a list of strings and lists are allowed to be empty. | Never |
  
#### `kinect_internal_elapsed_sec` and `smartcup_os_time` availability
This data is not available when either the Kinect didn't detect any people or the initial detection was lost. SmartCup only saved this information when it received a 3D pose estimate from the Kinect. SmartCup also didn't save 3D poses correctly and they were sometimes lost entirely. The poses themselves were recovered retrospectively by feeding the raw RGB+Depth+IR video into the Kinect Body Tracking SDK, though the associated `kinect_internal_elapsed_sec` and `smartcup_os_time` for these poses couldn't be recovered and is unavailable. 

### 9.3 `radar_pc_samples.zst`

Temporary file containing serialised pandas data frame for point-cloud radar. One row per point-cloud radar frame. The dataframe index name is 'frame_id' and represents a unique radar point cloud frame identifier over the entire dataset starting from 1.

| Column | DType | Description | NA |
| ------ | ----- | ----------- | -- |
| `subject_id` | Object (str) | Unique for each subject involved in experiments | Never |
| `run_id` | Object (str) | Unique for each execution of SmartCup. This is essentially a datetime string with prefix "Session". For each SmartCup execution, data is saved in a folder with `run_id` as the name. Radar pc and radar raw data streams share the same `run_id` because they were captured on Host 2 in the same SmartCup session/execution. | Never | 
| `frame_number` | float64 | Radar point cloud frame number for the current device run. | Never |
| `sample_kind` | Object (str) | This is always "frame". | Never |
| `observed_wallclock` | object (str) | The Windows OS timestamp recorded by SmartCup when parsing of the frame is completed. This is a string formatted using the official sync workbench datetime format "%Y-%m-%dT%H:%M:%S.%f". | Never |
| `points` | Object (Numpy Array) | Numpy array containing the radar point cloud frame with shape  (`P`, 6) where `P` is the number of cloud points. The dimensions are x, y, z, radial velocity, SNR and GTRACK target identifier. | Never |
| `point_count` | int64 | Number of radar cloud point in the frame. | Never |
| `point_count_filtered` | int32 | Number of radar cloud point in the frame after removing noise using GTRACK. | Never |

## 10. Current bottom line

The schema should stay centred on a **normalised canonical core** that makes the following explicit:

- uninterrupted runs,
- canonical sample indices,
- run/sample asset references,
- multiple time estimates per sample,
- model-independent anchors,
- run-specific timeline models,
- cross-run sync models,
- versioned mapping outputs.

The temporary zst files used during ingestion should map into this core cleanly.

The existing ER diagram remains acceptable as the current compact relational view of this design.
