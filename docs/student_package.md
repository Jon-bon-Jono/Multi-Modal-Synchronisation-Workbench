# Portable student packages — WP1 and WP2

WP1 implements package generation, integrity/assignment checks, local working
databases, remembered annotator identity, and Windows/macOS setup/launch scripts.
Package and annotator identity accompany GUI anchor/export provenance. WP2 adds
transactional saves/returns, recovery snapshots, persistent save/export feedback
and frame-load safeguards; see [the return/import contract](anchor_returns.md).
Separate-computer dynamic testing is WP3; the final student guide is WP4.
The source database and recordings are never modified by the exporter.

## Generate a package

Use a migrated canonical database with built assets and an **initial nearest-time
navigation mapping**. WP1 requires `initial_nearest_for_anchoring` backed by an
`identity_time` model without a parent mapping. Fitted mappings are intentionally
rejected because they can depend on master anchors excluded from the clean student
database. Generate an initial navigation mapping first if necessary.

```powershell
syncwb export-student-package --sqlite "<prepared.sqlite>" --artifact-root "<artifact-root>" --rgb-root "D:/smart_cup_recordings/Kinect" --read-only-root "D:/smart_cup_recordings" --subject 19_MM --mapping-version initial_rgb_to_raw_v001 --point-cloud-version "<raw_version_id>" --application-root "<SyncWB-checkout>" --output "<new-package-folder>"
syncwb verify-student-package --package "<new-package-folder>"
```

The output must be new and outside input/protected roots. Failure leaves no
published package. Copy the entire output folder to a student's local computer.
No symlinks to recordings, master paths or environment activation paths are used
to locate package data. Historical provenance may retain original paths as text;
they are not operational asset references.

For an offline raw-radar assignment, add
`--spatial-calibration calibration/kinect_radar/2026-10-06-desk/desk_all.json`
to the **package-generation** command. The exporter validates and copies the
JSON into the package and binds its relative path/hash in config and manifest.
Windows and macOS student launch scripts then apply it automatically; students
do not pass a calibration argument. The calibration is checked on every launch,
including lightweight verification, and appears in GUI/anchor geometry provenance.
Packages without this optional setting preserve their previous geometry. Online
assignments reject raw-radar calibration. See [spatial calibration](spatial_calibration.md).

One export assigns one subject, one RGB/radar pair, one navigation mapping and
one cloud version. Export additional assignments as separate packages. The
exporter copies full acquisition runs (including unprocessed raw frames), all
their timelines, the selected mapping/model, source pose/activity payloads and
RGB video, and only the assigned cloud's artifacts/summaries/version metadata.
Optional source payload roles are listed in the manifest. Assets referenced by
metadata must exist. Unrelated subjects/runs, alternate clouds, fitted mappings,
and all master anchors/anchor members/model-anchor links are excluded.

Payload files are copied unchanged and renamed by content checksum within the
package. Database references are rewritten accordingly; canonical acquisition
identities, sample indices, timestamps, payload member keys and mapping IDs are
preserved. The raw cloud's registered bundle checksum is checked before publication.

## Layout and identity

```text
manifest.json                  immutable package ID, assignment, hashes, inventory
config.json                    immutable assignment and relative operational paths
database/template.sqlite       clean immutable local-database seed
assets/artifacts/<hash>.*       assigned cloud and source pose/activity bundles
assets/rgb/<hash>.mp4           copied RGB video
calibration/kinect_radar/*.json optional immutable Kinect/raw-radar calibration
application/                   exact SyncWB Python source snapshot and build metadata
requirements.txt               dependency bounds for Python 3.11
setup_windows.cmd / launch_windows.cmd
setup_macos.command / launch_macos.command
SETUP.txt                      installation essentials, not the final student guide
work/workbench.sqlite          created at first student launch; annotations live here
work/annotator.json             created after a valid first-launch annotator ID
work/recovery/                  retained JSON snapshots after saves and around deletions
work/exports/                   default destination for student return JSON files
```

The manifest format is `syncwb.student_package.v1`. Each export gets a fresh
`student_<uuid>` package ID, even for identical assignments. Its manifest digest
covers assignment, acquisition digest, cloud provenance, runtime ID, table counts and immutable file
inventory. File entries include relative path, byte count and SHA-256. These
checks detect corruption; they are not cryptographic publisher signatures.

The launcher checks manifest/configuration identity, safe relative paths, file existence
and sizes, and the small application/configuration file checksums. Normal startup
**does not hash media or the template, scan SQLite integrity, or fingerprint database
contents**. Same-size asset corruption requires explicit full verification to detect.

On first launch the template is copied to `work/workbench.sqlite`. A marker binds
it to the package ID, manifest digest and template checksum stored in the manifest.
Launch checks that marker and the assigned mapping's RGB/radar endpoints and cloud
version/provenance identity using a few metadata rows. It does not recompute stored
cloud fingerprints. Older markers containing `core_sha256` remain readable; that
column is ignored. Annotation/other database rows are not serialized for startup
validation. Existing work is never replaced when identity validation fails.

Both launchers accept independent optional flags:

- `--full-checksums`: read and hash every immutable packaged file, including media
  and the template. Mutable student work is excluded.
- `--sqlite-integrity-check`: run `PRAGMA integrity_check` on the template and an
  existing working database. This checks SQLite consistency, not synchronization quality.
- `--verify-only`: perform the selected checks without opening the GUI, prompting for
  an annotator or creating/modifying work. Alone, this runs lightweight checks.

For example:

```text
launch_windows.cmd --verify-only --full-checksums --sqlite-integrity-check
bash launch_macos.command --verify-only --full-checksums --sqlite-integrity-check
```

Omit `--verify-only` to launch after verification. Setup explicitly requests both
expensive checks; normal launch requests neither. The coordinator's
`syncwb verify-student-package` command retains its existing full-checksum default.


An assigned annotator ID is entered on first launch and remembered per working
copy. IDs use 1–64 letters/numbers/periods/underscores/hyphens. A different annotator
cannot silently take over that working copy: provide a fresh package copy instead.
The GUI retains package and annotator identity in Details, and new anchors/exports
retain it alongside cloud/version/session provenance. Existing non-package GUI use
remains supported without inventing package identities.

## Installation and portability

Use an existing **Anaconda, Miniconda or Miniforge** installation and internet
access for initial setup. The scripts create a dedicated Python **3.11** Conda
environment; no separately installed Python or `py` launcher is needed.

- Windows: `setup_windows.cmd`, then `launch_windows.cmd`.
- macOS Terminal, in the package folder: `bash setup_macos.command`, then
  `bash launch_macos.command`. No `conda activate` or `conda init` is needed.

Setup keeps dependencies in a dedicated Conda environment outside the package:
`%LOCALAPPDATA%/SyncWB/conda-environments/syncwb-py311` on Windows and
`~/Library/Application Support/SyncWB/conda-environments/syncwb-py311` on macOS.
The environment helpers also reuse the two previously distributed Conda environments
(`d361c810a58bc24a9d9b` and `869708dace69f4d913fa`) in place if the stable environment
does not exist. Setup and launch use the same selection logic.

**Code/GUI-only updates do not need setup again:** extract the new package into a
fresh folder and run its launcher. Setup and launch set `PYTHONPATH` to the bundled
`application/src`, so each package uses its own checked source snapshot even when
sharing dependencies. The runtime ID remains a fingerprint of code, requirements
and Python for provenance; it no longer determines the environment directory.
Working databases and package identities remain separate. Moving/renaming the
whole package on the same computer remains valid.

For first installation, missing dependencies, or changed dependency requirements,
run setup. It reuses the selected environment, skips Conda solving when Python 3.11
and pip are healthy, and runs pip against `requirements.txt` without `--upgrade` or
`--force-reinstall`. Satisfying installed dependencies remain installed; missing or
incompatible ones are installed/updated. Broken Python/pip is repaired with Conda
in place. Setup never deletes or recreates an existing environment. Future
incompatible Python/dependency generations should use a new environment name.

Setup announces each step and streams Conda/pip output. Failures stop subsequent
steps. Rerunning setup retries in the same environment. The Conda base environment
and student work are not modified. No application installation is needed; launch
checks the bundled code against the package manifest as before. A new computer
needs setup unless it already has a compatible SyncWB environment.

The shipped `find_conda_windows.cmd` and `find_conda_macos.sh` helpers search
`CONDA_EXE`, PATH, and common Anaconda/Miniconda/Miniforge locations. If discovery
fails, run from Anaconda Prompt on Windows or set `SYNCWB_CONDA_EXE` to the full
Conda executable path before running the scripts. Do not edit packaged scripts.
For a custom environment location, set `SYNCWB_ENV_ROOT` consistently for setup
and launch; the environment is selected below that directory. Windows automation
may set `SYNCWB_NO_PAUSE=1`. Both launchers accept `--verify-only` for a GUI-free check.

Conda uses the user's configured channels; any channel/account requirements must
be resolved using Conda's reported instructions before retrying setup. Setup does
not accept channel terms automatically or replace the user's Conda configuration.
Extract a newly issued package into a fresh folder. Keep any previous working copy
with annotations; its database is bound to that earlier package identity.

Dependency ranges bound supported major versions; this is not an offline installer
or a fully pinned cross-platform lockfile. No Python/native binaries from the
maintainer's Windows environment are copied to students. Actual macOS installation,
OpenGL rendering and the two-anchor round trip require the planned WP3 dynamic test.
Students must retain/copy the whole `work` folder when relocating active work.

## Automated WP1 verification

Tests cover database isolation, absence of master annotations, payload retrieval,
relative references, relocation with source recordings unavailable, identity
persistence, preserving existing work, corruption/missing-file rejection,
assignment mismatch, input-root protection, and package provenance reaching
anchor/export records. No GUI window or separate-computer installation is needed
for these automated checks. Package verification never starts a GUI or creates
student work; use the module entry point when validating the installed runtime:

```text
python -m sync_workbench.deployment.student_runtime --package <folder> --verify-only --full-checksums --sqlite-integrity-check
```

## Returning anchors (WP2)

Keep the original `manifest.json` when handing out each package. Student return
imports require it via `--package-manifest`; use `--dry-run` first to inspect the
new/unchanged counts. Identical anchor IDs/content are no-ops; differing content
rejects the entire batch. See [anchor returns and recovery](anchor_returns.md)
for exact behavior and recovery commands. WP1-only packages must be regenerated
to include the WP2 runtime and acquisition digest; do not edit a manifest in place.

The 7 October 2026 calibrated revision of
`deployment_data/packages/student_d9c3400a0c1a4fe0891d89c1b5b31029` retains the
assignment/package ID but has a new manifest digest and runtime snapshot. Its
previous ZIP, manifest, checksum and guides are retained in `previous_revisions/`.
Use the manifest matching each student's return, including the archived manifest
for work made with the previous release. Extract the updated ZIP into a fresh
folder and keep existing `work` folders intact; do not copy an old working
database into the revision. The package's `CALIBRATION_UPDATE.md` records the
revision and chosen calibration.
