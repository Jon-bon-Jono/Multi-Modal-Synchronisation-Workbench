# Living Lab training exports

`export-training-data` writes one radar HDF5 file and one Kinect 2D/3D-pose HDF5
file for each selected overlapping run pair. It also writes `manifest.json`
and `README.txt`. A Kinect restart during a continuous radar recording therefore
produces two pairs of modality files. Timing gaps within a run pair are marked
inside the files; they do not create hundreds of small files.

The exporter opens the canonical SQLite database read-only and reads existing
artifact bundles. It does not read ADC captures, RGB videos or temporary pickles,
reprocess radar, migrate the database, or write new canonical mappings. The
radar-centred correspondence is a derived export table tied to the selected
mapping/model snapshot. The HPE framework is not modified.

## Export 09_SY and 19_MM together

From the SyncWB repository root, install the optional HDF5 dependency in the
environment used by the launcher (the current `syncwb` environment needs it):

```powershell
& "$env:USERPROFILE\anaconda3\envs\syncwb\python.exe" -m pip install 'h5py>=3.7'
.\scripts\syncwb\export_training_data_living_lab.bat --dry-run
.\scripts\syncwb\export_training_data_living_lab.bat
```

The launcher reads `scripts/syncwb/living_lab_09_SY_19_MM_selection.json`, which
pins both subjects' different raw cloud IDs and their `initial_rgb_to_raw_v001`
mappings. It embeds `calibration/kinect_radar/2026-10-06-desk/desk_all.json` and
each Kinect recording's `kinect_camera_recording_calibration.json` from
`D:\smart_cup_recordings\Kinect`. Set `SYNCWB_KINECT_ROOT` to relocate the latter.
Supplying a Kinect root requires a valid recording calibration for every selected
run; omitting `--kinect-root` in a custom CLI call records null camera calibration.
The camera documents are read once and embedded with SHA-256 hashes. No video is
opened or copied.

Default output:
`%USERPROFILE%\Documents\SyncWB\training_exports\living_lab_09_SY_19_MM_initial_v001_v3`.
Override with `--output "C:\path\to\new_export"` or `SYNCWB_EXPORT_OUTPUT`.
The directory must be new. `SYNCWB_SOURCE_ENV` and `SYNCWB_CONDA_EXE` override
the launcher's environment and Conda executable. `--dry-run` needs no HDF5
dependency and does not create the destination.

The subset consists of both subjects' entire supported temporal overlap; it does
not crop chosen activities, filter people, or construct training windows/splits.
The preset mapping is the initial timestamp alignment, not a claim of manually
verified synchronization. Change the selected mapping IDs after refining timing.
The selected spatial calibration is shared with the visualization launchers;
using it for both subjects assumes the relevant sensor geometry applies to both.

The new schema is `syncwb.training_export.v3`. It preserves the v2 radar/3D
datasets and adds 2D poses, labels, counts, PTS and recording calibration. Old
export directories are never changed. Framework adapters must recognize v3.

A read-only preflight on 2026-10-08 found the following exported-overlap counts:

| Subject | Radar frames | Kinect frames (shared by 2D/3D/labels) |
|---|---:|---:|
| 09_SY | 68,909 | 47,205 |
| 19_MM | 57,037 | 33,992 |

For 09_SY it excludes raw indices 0–762 and 69,672–69,843; all Kinect frames
are in overlap. For 19_MM it excludes raw indices 0–4,749 and Kinect indices
33,992–34,285. These are frame catalogue counts, including missing/empty payloads,
not usable supervised example counts. The actual study export was left for the
user to run; preflight does not validate all payload contents.

## Export the prepared 19_MM subset

For the current checkout's database/artifacts and bundled spatial calibration,
use `scripts/syncwb/export_training_data_19_MM.bat --dry-run`, then run the same
script without `--dry-run` when ready. Calibration and output path are configured
inside that script. See [the calibrated presets](spatial_calibration.md).
The explicit commands below remain available for the separate backend-validation
database and custom selections; add calibration explicitly for those commands.

Activate your SyncWB environment and run these commands from the repository root.
Install the optional export dependency once:

```powershell
conda activate syncwb
python -m pip install -e ".[hpe-export]"
```

The requested label `dense_static_far a55b95bbd5a2 calibrated` currently resolves
in the prepared database to:

```text
raw_d3f274e06a775b757e89dc86f8f57dfb12ab4a198e66912af21dd176d39620f3
```

Use this exact ID to freeze the selection. Set a new output directory outside
the source recordings and artifact store. The following PowerShell argument array
is reusable for the preflight and actual export:

```powershell
$exportArgs = @(
  "export-training-data",
  "--sqlite", "$env:USERPROFILE/Documents/SyncWB/backend_validation/workbench_raw_validation.sqlite",
  "--artifact-root", "$env:USERPROFILE/Documents/SyncWB/backend_validation/artifact_store",
  "--subject", "19_MM",
  "--mapping-version", "initial_rgb_to_raw_v001",
  "--point-cloud-version", "raw_d3f274e06a775b757e89dc86f8f57dfb12ab4a198e66912af21dd176d39620f3",
  "--read-only-root", "D:/smart_cup_recordings",
  "--output", "$env:USERPROFILE/Documents/SyncWB/training_exports/19_MM_dense_static_far_initial_v001"
)
python -m sync_workbench @exportArgs --dry-run
```

Preflight validates selection, processing/mapping consistency, timeline coverage
and output-path safety without loading point/pose arrays or creating output.
It is not a payload-integrity check. When ready, run the export yourself:

```powershell
python -m sync_workbench @exportArgs
```

The output must not already exist. Repeated `--read-only-root` options protect
additional directories. Source database and artifact paths are protected
automatically; output inside or enclosing a protected path is rejected.
An exact, unique `--point-cloud-label` may replace `--point-cloud-version`, but
labels are only selectors and do not establish processing equivalence.

A read-only preflight on the prepared database during implementation found one
overlapping run pair: 57,037 radar frames and 33,992 pose frames. It excluded
radar sample indices 0–4,749 before overlap and pose indices 33,992–34,285 after
overlap. These counts describe that database snapshot, not a completed export.
The real 19_MM export was not run during implementation.

## Selection and homogeneity

For multiple subjects with the same mapping ID and cloud label, repeat `--subject`.
For multiple run pairs or different run-specific IDs, use `--selection selection.json`
instead of subject/mapping/cloud arguments. The JSON is a nonempty list:

```json
[
  {"subject_id": "S1", "mapping_version_id": "initial_pair_1", "point_cloud_version_id": "raw_<result-1-hash>"},
  {"subject_id": "S1", "mapping_version_id": "initial_pair_2", "point_cloud_version_id": "raw_<result-2-hash>"},
  {"subject_id": "S2", "mapping_version_id": "initial_pair_1", "point_cloud_version_id": "raw_<result-3-hash>"}
]
```

The exporter requires a selection for every raw acquisition of the requested
subjects and every registered Kinect/raw run pair. Duplicate pair selections,
overlapping assignments of the same native frame, and different cloud results
for the same raw run are errors. Unselected Kinect runs with initial-timeline
overlap are errors even when no mapping has been registered yet. For piecewise
exports, unselected Kinect runs are conservatively rejected because their aligned
overlap cannot be established without a selected model. No-overlap selections
also fail rather than disappearing silently.

Only provenance-backed `radar_raw` results are accepted. `radar_pc` and
`raw_legacy` cannot enter this export. Across different raw runs, result IDs may
differ but their `syncwb.processing_compatibility.v2` profiles must match.
The profile compares cfg text/hash, native executable hash/version/profile,
tracker starting sequence index, tracking flag, ADC layout, point schema/units/frame,
association labels and scenery/target-state frame. Calibration is per acquisition:
the input frame hashes may differ, while calibration enabled state, frame count
and procedure must match. An explicit generator `calibration_procedure` record is
compared when present. For legacy manifests, the documented identity basis is the
shared executable/config/native profile; input frames are acquisition-specific.
This permits subjects such as 09_SY and 19_MM to share an export without claiming
their calibration results are numerically identical. Different cfg or algorithms
still fail compatibility checks.

Every selection retains its unchanged cloud provenance and its exact v1 recipe
(including ordered calibration hashes) as `exact_processing_recipe` and
`exact_processing_recipe_id`. The top-level `processing_compatibility` and its ID
describe shared method compatibility. Since `syncwb.training_export.v2`,
there is no top-level assertion of one exact recipe. Each modality HDF5 retains
its own exact `processing_recipe_id` and the shared `processing_compatibility_id`.
Consumers of v1's top-level recipe fields must use these explicit v2/v3 fields.
Existing exported files and cloud IDs are not rewritten. Paths, subjects, coverage,
creation time and runtime statistics remain result provenance, not method-equality criteria.

Mapping profiles compare method, model type, mapping parameters (excluding the
parent mapping ID), selected timeline IDs/types, extrapolation policy and export
correspondence/gap policy. Per-run fitted coefficients and anchors may differ.
Initial `identity_time` and fitted `piecewise_affine` models are supported, but
cannot be mixed within an export. Recipe/profile/release IDs are stored in the
manifest; they do not introduce new canonical database tables.

## Files and array contract

```text
export/
  manifest.json
  README.txt
  segment_<id>.radar.h5
  segment_<id>.pose.h5
```

File attributes include schema, modality, subject/run/segment identities, cloud
and mapping IDs, recipe/profile IDs, geometry JSON, nominal sample period, original
timeline origin and shared aligned origin. Each modality can be read independently.

| Dataset | Files | Meaning |
|---|---|---|
| `frames/sample_index` | Both | Original zero-based canonical acquisition index; never rebased to hide gaps |
| `frames/frame_number` | Both | Original frame number from sample provenance, or -1 when unavailable |
| `frames/timeline_time_s` | Both | Original selected timeline time minus its own run origin |
| `frames/aligned_time_s` | Both | Time in the shared radar coordinate minus the common segment origin |
| `frames/sequence_block` | Both | Common continuity partition; windows must stay within `(segment_id, sequence_block)` |
| `frames/payload_available` | Both | Distinguishes available empty data from missing/unprocessed data |
| `frames/point_count` | Radar | Count including all association IDs; -1 when unprocessed |
| `frames/point_status` | Radar | Fixed-width ASCII `available` or `unprocessed` |
| `points/offsets` | Radar | int64 offsets, length frame count + 1 |
| `points/values` | Radar | float32 `[total_points, 6]`: world XYZ, radial velocity, SNR, association ID |
| `frames/num_people` | Pose | Original summary count; -1 if unavailable |
| `frames/pose_person_count` | Pose | Actual number of stored poses in this frame |
| `frames/num_2d`, `frames/num_3d` | Pose | Original summary detection counts, int64; -1 if unknown, independent of annotated `num_people` |
| `frames/pose2d_person_count` | Pose | Actual stored 2D detection count, int64; zero for empty or missing payload, distinguished by availability |
| `frames/pose2d_payload_available`, `frames/conf2d_payload_available` | Pose | Boolean source payload presence, including present empty arrays |
| `frames/rgb_pts_s` | Pose | float64 original RGB presentation timestamp in seconds; NaN if absent |
| `frames/activity_json` | Pose | UTF-8 JSON string per frame: original activity payload or `null` when missing |
| `frames/activity_payload_available` | Pose | Boolean activity source payload presence; an available empty label set is not missing |
| `poses/offsets` | Pose | int64 offsets into the concatenated person arrays |
| `poses/xyz` | Pose | float32 `[total_people, 32, 3]` in GUI world metres |
| `poses/confidence` | Pose | float32 `[total_people, 32]`, original confidence values |
| `poses/finite_xyz` | Pose | Coordinate finiteness mask; does not claim confidence/quality validity |
| `poses2d/offsets` | Pose | int64 offsets of length Kinect frame count + 1, independent of 3D offsets |
| `poses2d/xy` | Pose | float64 `[total_2d_people, 26, 2]`, source prediction-image pixel X/Y, unchanged |
| `poses2d/confidence` | Pose | float64 `[total_2d_people, 26]`, original detector keypoint scores |
| `poses2d/bbox_confidence` | Pose | float64 `[total_2d_people]`, original detector bounding-box scores (`conf2d`); NaN if missing |
| `poses2d/finite_xy` | Pose | Boolean `[total_2d_people, 26]`; both coordinates finite, not a quality threshold |
| `correspondence/pose_row` | Radar | Row in the paired pose file, or -1 when unmatched |
| `correspondence/pose_sample_index` | Radar | Original Kinect sample index, or -1 |
| `correspondence/matched` | Radar | A supported temporal match exists |
| `correspondence/nearest_residual_ms` | Radar | Predicted Kinect time minus radar time for the nearest candidate, even if rejected |
| `correspondence/pose_payload_available` | Radar | Match exists and its pose frame contains at least one stored **3D** pose; use matched pose-row fields to check 2D availability |

Point columns/units and Kinect joint names are HDF5 attributes. Pose person
ordering is preserved independently for each frame; it is not persistent identity.
`poses/finite_xyz` has shape `[total_3d_people, 32]`; all three coordinates must be
finite. Common frame indices, frame numbers and sequence blocks are int64,
times are float64 seconds, and availability fields are boolean. Point counts and
correspondence row/index fields are int64; residuals are float64 milliseconds.

Confidence values are not normalized or thresholded by this exporter. Kinect
3D confidence retains its native body-tracking levels; 2D scores have a separate
detector meaning. Do not apply one threshold to both. Nonfinite pose coordinates
are retained with a false finiteness mask. Available empty radar frames have
zero points; unprocessed frames have no points and an unavailable status.

For frame row `i`, slice values using `offsets[i]:offsets[i+1]`. No fixed-point
padding, point filtering, pose resampling, multi-frame accumulation, single-person
filtering, or RGB video copying is performed. Activity and 2D payloads reflect
what was already imported; the ETL pipeline previously filtered 2D detections.
Bulk numerical arrays use lossless
gzip compression. The source NPZ format requires decompressing a bundle into
memory; transformed output is written in batches of 256 frames. Allow RAM for
one run's decompressed source bundles, rather than the whole subject collection.

`num_people` preserves the imported annotated presence count. It may disagree
with detected 2D/3D counts. There are no exported bounding-box coordinates,
persistent Kinect body IDs, or cross-modal person identities because the imported
artifacts do not contain them. The radar association ID is a radar tracker output,
not a Kinect person ID. No new confidence/person filtering is applied.

`activity_json` preserves arbitrary source keys and lists. The current Living Lab
payload keys are `01-Activity`, `01-Notes`, `01-Objects`, `01-PhysicalState`,
`02-Activity`, `02-Objects`, `02-PhysicalState`. Empty lists mean no recorded label
in that tier. Tier prefixes must not be equated with pose detection-array rows.

## HDF5 attributes and skeletons

Both files have these attributes: `schema`, `segment_id`, `subject_id`,
`radar_run_id`, `pose_run_id`, `processing_recipe_id`,
`processing_compatibility_id`, `mapping_profile_id`, `point_cloud_version_id`,
`mapping_version_id`, `aligned_origin`, `time_coordinate_kind`, `geometry_json`,
`modality`, `timeline_model_id`, `timeline_origin`, `nominal_sample_period_s`.
IDs bind the arrays to the manifest selection. Origins reconstruct original
time coordinates; `geometry_json` repeats the complete world-geometry metadata.

Group attributes:

* `points`: `columns_json` = `[x,y,z,radial_velocity,snr,association_id]`;
  `units_json` = metres for XYZ, generator-declared Doppler/SNR units, identifier
  for association ID (currently m/s and linear power ratio).
* `poses`: `joint_names_json` gives Kinect's native 32-joint order.
* `poses2d`: `joint_names_json`, `schema` (`Body8-Halpe26`), `coordinate_frame`,
  `image_size_status`, `person_correspondence`. These explicitly record source
  image pixels, unknown image dimensions/scaling, and independent person order.
* `correspondence`: `pose_file`, `residual_definition`, `tolerance_ms`.

Kinect 3D joint order (zero-based): pelvis, spine - navel, spine - chest, neck,
left clavicle, left shoulder, left elbow, left wrist, left hand, left handtip,
left thumb, right clavicle, right shoulder, right elbow, right wrist, right hand,
right handtip, right thumb, left hip, left knee, left ankle, left foot, right hip,
right knee, right ankle, right foot, head, nose, left eye, left ear, right eye,
right ear.

2D order: Nose, L_Eye, R_Eye, L_Ear, R_Ear, L_Shoulder, R_Shoulder, L_Elbow,
R_Elbow, L_Wrist, R_Wrist, L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle,
Head_Apex, Neck, Hip_Center, L_BigToe, R_BigToe, L_SmallToe, R_SmallToe,
L_Heel, R_Heel. It is not Kinect's skeleton projected to 2D.

## Manifest fields

`manifest.json` is a self-contained metadata/provenance snapshot:

| Field | Contents |
|---|---|
| `schema`, `subjects` | Format version and selected subject IDs |
| `processing_compatibility_id`, `processing_compatibility` | Shared generator/settings/calibration-procedure contract; acquisition-specific calibration inputs may differ |
| `mapping_profile_id`, `mapping_profile` | Common timing method, timeline types, tolerance and gap policy |
| `synchronization_release_id` | Hash-derived identity of the selected synchronization snapshots |
| `geometry` | World axes/units, rotations/translations, height/pitch assumptions, native-Kinect-to-world linear transform, selected spatial calibration's name/hash/full document |
| `selections` | Per-run selection and source records, detailed below |
| `coverage` | Per run: subject/run/modality/mapping IDs, total/exported frames, excluded canonical-index ranges and reason |
| `scope`, `timestamp_policy`, `missing_policy`, `person_policy`, `pose2d_policy` | Explicit interpretation and missing-data rules |
| `segments` | Per run pair: identities, frame counts, unmatched radar count, common origin, sequence boundaries, modality paths/checksums, missing-payload counts |
| `source_artifacts` | Each consumed NPZ/JSONL artifact's relative source reference and SHA-256 |
| `exporter_sha256`, `geometry_implementation_sha256` | Hashes of the export and geometry implementations |
| `dataset_fingerprint` | Hash of the manifest before adding this field and `created_at`; includes payload hashes |
| `created_at` | Export creation timestamp |

Each `selections[]` entry contains:

* `selection`: `subject_id`, `mapping_version_id`, `point_cloud_version_id`.
* `mapping`, `sync_model`, `model_anchors`: complete canonical source records;
  JSON-valued database fields remain JSON strings, preserving method parameters,
  coefficients, references and notes. They are provenance, not extra training rows.
* `cloud`: complete registered point-cloud-version record, including artifact
  hash and `provenance_json`. Decode that JSON string to obtain `version`, its
  generator metadata, original HDF5 binding, cfg, processing/tracking details,
  per-acquisition calibration source frame references and hashes.
* `exact_processing_recipe_id`, `exact_processing_recipe`: exact per-run recipe
  including calibration source hashes. These can differ across subjects while
  their compatibility ID is the same.
* `timelines`: `radar_timeline` and `pose_timeline`, each with complete selected
  `model` record and timeline `fingerprint`.
* `kinect`: `video_assets` (registered RGB video and any integrity records),
  `video_included` (false), `camera_calibration` (null if no root supplied).
  Camera calibration contains `source_ref`, `sha256`, `document`,
  `interpretation`. The original document is embedded without reinterpreting
  its vendor-specific fields: `CalibrationInformation.Cameras` includes
  intrinsics, distortion model/parameters, sensor dimensions and `Rt` transforms;
  `InertialSensors` and `Metadata` are retained too. Factory sensor dimensions
  are not the 2D prediction image dimensions. Capture-mode calibration and
  resize/crop history are still required for correct RGB projection.

Each segment's `files.radar` and `files.pose` contains `path` (relative) and
`sha256`. `radar_missing_payload_frames` and `pose_missing_payload_frames` count
missing radar and 3D payloads respectively; derive separate 2D missing counts
from `frames/pose2d_payload_available`. The six-file two-subject package consists
of the manifest, brief `README.txt`, and two radar/pose HDF5 pairs.

All inherited source-record fields are retained as provenance. Paths in those
records describe original inputs and are not required to load the exported
arrays; only the explicit external video references require the recordings.

## Time and correspondence

Pose predictions use the selected sync model on the selected Kinect timeline.
Radar aligned timestamps use its selected acquisition timeline. Both subtract
the same segment origin; their first frames need not both have time zero.
Original timeline coordinates can be reconstructed by adding the stored origin.
Datetime origins use SyncWB's UTC-like numeric interpretation of naive datetimes;
they do not certify a UTC acquisition clock. Raw timing remains an estimate.

Every radar frame is matched to the nearest predicted Kinect time, with earlier
frames winning exact ties. The tolerance is the minimum of the selected mapping's
weak-support threshold and maximum delta. Only supported matches are exported:
there is no extrapolation, even if the navigation mapping allowed it. Piecewise
models must be continuous and strictly increasing. The original RGB-centred
`SAMPLE_MAPPING` rows are not inverted and the source database is not changed.

Files contain all samples within the supported overlapping interval, including
unprocessed radar frames and missing poses. The manifest reports excluded
canonical-index ranges outside overlap, per run pair. Cadence gaps exceeding
`--gap-factor` times the nominal period (default 3), or missing canonical indices,
create shared sequence boundaries at gap midpoints. Correspondences cannot cross
those boundaries. Missing pose payloads and unprocessed radar frames remain
explicit; training must apply its own availability/quality policy.

## Geometry

GUI and exporter share `core/geometry.py`. Without `--spatial-calibration`, the
profile is `syncwb.gui_world.v1`. For column vectors:

```text
R = [[1, 0, 0], [0, cos(30°), sin(30°)], [0, -sin(30°), cos(30°)]]
t = [0, 0, 1.76] metres
A = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
radar_world_m = R @ radar_sensor_m + t
kinect_world_m = R @ (0.001 * A @ kinect_mm) + t
```

World axes are right/forward/up, right-handed; the origin is the assumed floor
point beneath the sensor. This is the historical approximate GUI alignment.
It does not add a calibrated Kinect-to-radar translation/rotation. The cloud
label's word `calibrated` refers to its generator provenance and does not change
this geometry contract. Doppler remains sensor radial velocity; SNR and
association IDs are unchanged. The manifest stores the actual matrices and
geometry implementation hash. Consumers must not apply the world transform a
second time.

To use measured Kinect/raw-radar extrinsics, pass `--spatial-calibration PATH`
to both GUI and export commands. The native Kinect-to-radar matrix replaces the
axis-only conversion above before the shared floor transform. Each export uses
one calibration, embeds its complete JSON and hash in both modality files and the
manifest, and receives a calibration-specific geometry profile. See
[spatial calibration usage and conventions](spatial_calibration.md), including
the prepared 19_MM launcher and how to extend `$exportArgs` above.

## Reproducibility and the future Living Lab adapter

The manifest freezes exact cloud provenance, mapping/model parameters, model-anchor
references, timeline fingerprints, source artifact hashes, file checksums,
exporter/geometry source hashes, coverage, geometry and a dataset fingerprint.
It is written last as the completion marker. Existing destinations are never
overwritten. Failed generation cleans up its staging data and publishes no
complete export; interrupted publication may leave a directory without a manifest.

A future `LivingLabAdapter` can build records from the frame catalogues and read
only the selected point/pose slices. It should use subject grouping for splits,
`(segment_id, sequence_block)` for sequences, and aligned timestamps for temporal
features. Unprocessed frames must remain absent from the usable radar index
without compressing time. A single-person policy should check `num_people`, actual
pose count, confidence and coordinate validity; temporal models may need this
across the entire window. `Pose3DTarget` requires finite coordinates even for
masked joints, so the adapter must replace invalid coordinates with a documented
finite placeholder while preserving the false validity mask. Joint-schema
conversion, cropping, augmentation and pelvis-relative encoding belong in the
training framework. No adapter implementation is included here.

For the current `VIP_mmWave_HPE_demo/ml` contracts, use a dataset descriptor with
the manifest schema/fingerprint, sequence records grouped by subject, and atomic
radar-frame records pointing to HDF5 row numbers. `load_frame` reads radar slices;
`load_target` follows `correspondence/pose_row` and applies an explicit person and
joint policy. Both modalities are already in the common world frame. If the
framework's target convention requires radar sensor coordinates, invert the
stored radar-to-world transform for **both** points and 3D targets before passing
them to the framework; do not merely relabel world coordinates as sensor-local.

Minimal reading example (no filtering, skeleton conversion or target selection):

```python
import json
from pathlib import Path
import h5py

root = Path(r"C:\path\to\export")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
segment = manifest["segments"][0]
with h5py.File(root / segment["files"]["radar"]["path"], "r") as radar, \
     h5py.File(root / segment["files"]["pose"]["path"], "r") as pose:
    row = 100  # HDF5 row, not necessarily the native sample index
    lo, hi = radar["points/offsets"][row:row + 2]
    points = radar["points/values"][lo:hi]
    pose_row = int(radar["correspondence/pose_row"][row])
    if pose_row >= 0:
        lo3, hi3 = pose["poses/offsets"][pose_row:pose_row + 2]
        lo2, hi2 = pose["poses2d/offsets"][pose_row:pose_row + 2]
        xyz = pose["poses/xyz"][lo3:hi3]       # independently ordered 3D people
        xy = pose["poses2d/xy"][lo2:hi2]       # independently ordered 2D people
        num_people = int(pose["frames/num_people"][pose_row])
        activity = json.loads(pose["frames/activity_json"][pose_row])
```

Check availability and masks before treating empty slices as usable examples.
Split subjects before constructing overlapping training windows to avoid leakage.
