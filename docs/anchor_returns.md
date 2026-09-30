# Anchor saves, recovery and return imports — WP2

Students place anchors in their local working database; the coordinator imports
return JSON files and generates synchronization mappings. No student-side fitting
or manual synchronization validation is required for this deployment phase.

## Save and export contract

Creating or deleting an anchor is one SQLite transaction, including its members.
A failed write rolls back the whole operation. Existing indexes remain intact.
An anchor already referenced by a synchronization model cannot be deleted or
overwritten, because doing so would invalidate that model's provenance.

After every successful GUI save/deletion, SyncWB writes a uniquely named JSON
snapshot in `work/recovery/` for student packages. It also snapshots before deletion;
if that pre-deletion snapshot fails, deletion is cancelled. Standalone GUI sessions
use `<database-name>_recovery/` beside their database. Snapshots are retained, not
rotated away. They include the entire pair's current anchor set and provenance.

The GUI reports database-save success and the recovery path. If the database commit
succeeds but the subsequent snapshot fails, it explicitly reports **saved in the
database, recovery copy failed**. The user can retry Export anchors. Export success
includes the anchor count; its full destination is available in Details and the
status tooltip. The main window uses a single-line sample/anchor summary, plus a
single-line notice only for errors, fallback or save/export feedback. Long messages
are elided rather than wrapped, and full messages remain available in Details/tooltips.
Errors/success messages remain visible through frame navigation until another
anchor/export action replaces them. Run/package IDs and technical mapping diagnostics
are in Details instead of consuming the main window's vertical space.

Export anchors defaults to `work/exports/<package-id>_<annotator-id>.json`.
JSON is written to a temporary sibling, flushed and atomically replaced, so a failed
replacement preserves the previous export. Package configuration, working database,
source asset folders and recovery snapshots are protected from GUI export overwrite.
Recovery files on the same disk do not protect against losing that disk: retain
the working folder and send/copy a return file to the coordinator.

## Coordinator import

Retain the **original manifest.json** for every distributed package. Import into
the prepared/master database containing the same acquisitions, not an empty database:

```text
syncwb import-anchors --sqlite <master.sqlite> --input <student-return.json> --package-manifest <original-manifest.json> --dry-run
syncwb import-anchors --sqlite <master.sqlite> --input <student-return.json> --package-manifest <original-manifest.json>
```

Dry run performs validation/conflict checks and reports proposed new anchors and
unchanged anchors without writing them. The actual import repeats validation inside
the same transaction as its inserts. A conflict or insertion failure rolls back the
entire batch. Success reports `anchors`, `members`, and `unchanged` counts.

| Situation | Policy |
| --- | --- |
| New anchor ID and valid endpoints | Insert anchor and both members together |
| Same ID, identical anchor and members | No-op; count as unchanged |
| Same ID, different contents or members | Reject the entire file; name conflicting IDs |
| Missing sample, different acquisition, malformed pair or orphan member | Reject the entire file |
| Wrong package, annotator inconsistency or cloud provenance mismatch | Reject the entire file |
| Anchor absent from a later return | Keep the existing master anchor |

Identical repeat imports remain no-ops even when a later export has a different
export timestamp. SQLite numeric affinities and JSON note formatting are normalized
for comparisons. Import does not silently merge conflicting records or assign new
IDs; `--overwrite` is rejected. To correct an already accepted annotation, the
coordinator must review it explicitly. Student-side deletion does not propagate to
the master, and importing an older recovery snapshot may restore a deleted anchor.

`syncwb.anchors.v1` exports include a content checksum, acquisition digest, pair,
session identity, anchors and members. Student returns additionally carry package,
runtime, template, annotator, cloud-version and display provenance. Their package
identity must match the original manifest and every returned anchor. These checks
detect corruption and assignment mistakes; they are not digital signatures.

The acquisition digest covers both run identities, captured sample indices/kinds,
run start/end estimates and nominal frame rate. It excludes asset paths, cloud
versions, navigation mappings and fitted models. Changing run metadata requires
coordinator reconciliation/repackaging, but using another cloud version or fitting
a new mapping does not change anchor endpoint identity. Import does not require the
displayed cloud version to be installed in the destination. Existing non-package
exports retain pair/sample validation without requiring a package manifest.

To recover lost local work, retain the damaged working folder, prepare a fresh copy
of the same package using the same annotator ID, and import a selected recovery JSON
into its new `work/workbench.sqlite` with the original manifest. The coordinator
should select the snapshot deliberately, since older snapshots can contain anchors
subsequently deleted. Never replace the original working database before preserving it.

## Display safeguards

Anchor placement requires a successfully loaded RGB frame and cloud/window for the
current sample indices. This also applies when the RGB rendering layer is hidden.
Enabled pose overlays must load before the GUI allows placement. Both the button and
its action handler enforce readiness; the controller rechecks RGB/cloud availability.
Failed frame loads clear old images/points and keep the error visible until a successful
load. Successfully processed empty clouds remain valid and are labelled as empty;
missing/unprocessed/corrupt clouds disable anchoring.

Fallback from linked mapped playback to nominal-rate playback is displayed explicitly,
including its reason. If mapping is unavailable when playback starts, that playback
continues at nominal rates until restarted. Both-stream stepping also labels its
frame/rate fallback. This navigation aid does not create a synchronization mapping.

## Verification and deployment

Automated tests exercise transactional rollback, duplicate/conflict policy, malformed
and mismatched returns, package/annotator provenance, atomic file failures, recovery,
RGB/cloud failures, valid empty clouds, persistent GUI status and visible fallback.
A synthetic two-anchor return is imported and used to fit a synchronization mapping.
Qt tests use offscreen widgets; no hands-on GUI or separate-computer test is performed.

Rebuild WP1 packages with the WP2 runtime and acquisition digest before distribution.
WP3 remains the official separate-computer installation/two-anchor round trip,
including an actual macOS check for Mac deployment. WP4 is the final student guide.
