# Living Lab training exports

`export-training-data` writes one radar HDF5 file and one Kinect 3D-pose HDF5
file for each selected overlapping run pair. It also writes `manifest.json`
and `README.txt`. A Kinect restart during a continuous radar recording therefore
produces two pairs of modality files. Timing gaps within a run pair are marked
inside the files; they do not create hundreds of small files.

The exporter opens the canonical SQLite database read-only and reads existing
artifact bundles. It does not read ADC captures, RGB videos or temporary pickles,
reprocess radar, migrate the database, or write new canonical mappings. The
radar-centred correspondence is a derived export table tied to the selected
mapping/model snapshot. The HPE framework is not modified.

## Export the prepared 19_MM subset

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
differ but processing recipes must match exactly. The v1 recipe compares cfg
text and hash, native executable hash/version/profile, ordered calibration
content hashes, tracker starting sequence index, tracking flag, ADC layout,
point schema/units/frame, association labels and scenery/target-state frame.
Paths, source subject/run IDs, processing coverage, creation time and execution
statistics are retained as provenance but excluded from recipe equality.
This is intentionally conservative: even text-only cfg changes or a different
calibration set require a separate export.

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
| `poses/offsets` | Pose | int64 offsets into the concatenated person arrays |
| `poses/xyz` | Pose | float32 `[total_people, 32, 3]` in GUI world metres |
| `poses/confidence` | Pose | float32 `[total_people, 32]`, original confidence values |
| `poses/finite_xyz` | Pose | Coordinate finiteness mask; does not claim confidence/quality validity |
| `correspondence/pose_row` | Radar | Row in the paired pose file, or -1 when unmatched |
| `correspondence/pose_sample_index` | Radar | Original Kinect sample index, or -1 |
| `correspondence/matched` | Radar | A supported temporal match exists |
| `correspondence/nearest_residual_ms` | Radar | Predicted Kinect time minus radar time for the nearest candidate, even if rejected |
| `correspondence/pose_payload_available` | Radar | Match exists and its pose frame contains at least one stored pose |

Point columns/units and Kinect joint names are HDF5 attributes. Pose person
ordering is preserved independently for each frame; it is not persistent identity.
Confidence values are not normalized or thresholded. Nonfinite pose coordinates
are retained with a false finiteness mask. Available empty radar frames have
zero points; unprocessed frames have no points and an unavailable status.

For frame row `i`, slice values using `offsets[i]:offsets[i+1]`. No fixed-point
padding, point filtering, pose resampling, multi-frame accumulation, single-person
filtering, or activity/RGB/2D-pose export is performed. Bulk arrays use lossless
gzip compression. The source NPZ format requires decompressing a bundle into
memory; transformed output is written in batches of 256 frames. Allow RAM for
one run's decompressed source bundles, rather than the whole subject collection.

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

GUI and exporter share `core/geometry.py`, profile `syncwb.gui_world.v1`.
For column vectors:

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
geometry implementation hash. Future sensor calibration requires a new geometry
profile/export. Consumers must not apply the world transform a second time.

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
