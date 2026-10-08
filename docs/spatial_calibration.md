# Kinect/raw-radar spatial calibration

The prepared 19_MM raw GUI and training-export scripts automatically apply
`calibration/kinect_radar/2026-10-06-desk/desk_all.json` from this checkout.
The path is configured inside each `.bat` script; no calibration argument is
needed when launching it. Portable student packages use their bundled configuration.

## GUI usage

From the repository root in PowerShell:

```powershell
.\scripts\syncwb\anchoring_gui_versioned_19_MM.bat
```

This launcher selects `initial_rgb_to_raw_v001` and the prepared
`dense_static_far a55b95bbd5a2 calibrated` point-cloud version. To select another
calibration, edit its `SYNCWB_SPATIAL_CALIBRATION` line, then restart. `%~dp0`
makes the path relative to the script, independent of the current directory.
The other three `anchoring_gui_19_MM*.bat` presets select online radar and explicitly
clear this setting: this calibration supports offline `radar_raw` only. Ingestion,
mapping and raw-processing scripts do not apply display geometry.

Enable the 3D Kinect pose overlay to inspect alignment. The radar
projection onto RGB also uses the inverse calibration, followed by the existing
Kinect depth-to-colour camera projection. That camera projection's parameters
are not re-estimated by this desk calibration.

The session Details/tooltip includes the selected filename and SHA-256. The
calibration is loaded once and fixed for the session. Restart with a different
path to compare calibrations; editing the file does not change an open session.

## Training exports

The matching preset reads `workbench.sqlite` and `artifact_store` from the
checkout, just like the GUI launcher. It writes to the new directory set in its
`SYNCWB_EXPORT_OUTPUT` line. Edit the output and calibration settings inside
`scripts/syncwb/export_training_data_19_MM.bat` as needed:

```powershell
.\scripts\syncwb\export_training_data_19_MM.bat --dry-run
# When ready:
.\scripts\syncwb\export_training_data_19_MM.bat
```

Choose a new output directory if an earlier export already exists. Preflight
validates the JSON and reports geometry without writing payloads. One calibration
applies to every selected subject and run pair in an export. Only combine runs
that share that physical sensor setup; different sensor arrangements require
separate exports with their respective calibrations. SyncWB does not infer
calibration applicability from subject or processing labels.

The manifest and each modality's HDF5 `geometry_json` attribute retain the full
calibration document, source filename, SHA-256 of the source bytes, and composed
transforms. The geometry profile is `syncwb.calibrated_gui_world.v1_<sha256>`;
the dataset fingerprint includes geometry. GUI anchor provenance and anchor
export session metadata also record it. The JSON is read once, so provenance
and applied matrices stay consistent if the source file changes during use.

## Direct CLI and portable packages

For custom workflows, the Python CLI still accepts `--spatial-calibration PATH`
on `anchoring-gui`, `export-training-data` and `export-student-package`. A direct
GUI/training-export command without it uses the historical axis-only alignment.
If using the `$exportArgs` array in [training_export.md](training_export.md), add
`'--spatial-calibration', 'calibration/kinect_radar/2026-10-06-desk/desk_all.json'`
to the array. Use the same file for GUI and export geometry.

The student-package builder copies the selected JSON into `calibration/kinect_radar/`
and records its relative path and hash in `config.json` and `manifest.json`.
Both Windows and macOS launchers load it automatically, without calibration
arguments or references to the coordinator's checkout. Calibration is checksum
verified even during normal lightweight startup. See [student packages](student_package.md)
for packaging and revision handling. Changing the calibration requires a new
verified package revision; students should not edit packaged files.

## Supported JSON and transformation

The supported schema is `radar-kinect-desk-calibration-v2`, with `units: "m"`.
The three result files `desk_all.json`, `desk_suffix_1.json`, and
`desk_accepted_only_sensitivity.json` in the 2026-10-06-desk directory are
supported. The prepared presets choose `desk_all.json`; this does not imply it
has been selected by an automated quality comparison. `input_audit.json` is an
audit, not a calibration, and is rejected.
Malformed matrices, unsupported units/schema/axes, non-rigid transforms, and
inconsistent forward/inverse transforms fail explicitly.

Native Kinect pose XYZ is in millimetres. The producer's
`native_kinect_to_radar_4x4` already includes the Kinect-to-radar axis conversion
`(x, y, z) -> (x, z, -y)`. SyncWB uses it directly after converting millimetres
to metres. It does not swap axes again or apply any reflector-cap offset to
body joints. With column vectors:

```text
C = native_kinect_to_radar_4x4
R = [[1, 0, 0], [0, cos(30°), sin(30°)], [0, -sin(30°), cos(30°)]]
t = [0, 0, 1.76] metres

kinect_radar_m = C[:3, :3] @ (kinect_native_mm / 1000) + C[:3, 3]
kinect_world_m = R @ kinect_radar_m + t
radar_world_m = R @ radar_sensor_m + t
```

Thus both modalities remain in the GUI's right-handed, right/forward/up world
frame, in metres. Calibration corrects the relative sensor alignment; the
30-degree floor pitch and 1.76-metre radar height remain historical assumptions,
not newly calibrated floor geometry. Radar XYZ receives only its existing world
transform. Pose confidence, person ordering/counts, radar non-spatial columns,
timestamps and temporal correspondence are unchanged.

The metadata records `kinect_mm_to_world_linear` and
`kinect_world_translation_m`, which fully specify the composed Kinect transform.
Exported coordinates have already been transformed; consumers must not apply
the calibration or floor transform again. Source artifacts, calibration JSONs,
point-cloud versions and temporal mappings are not rewritten. This spatial
calibration is separate from the radar generator calibration recorded in a
point-cloud processing recipe or its `calibrated` label.
