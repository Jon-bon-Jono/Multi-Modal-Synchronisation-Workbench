# Synchronization feasibility study: 19_MM online radar

This study asks whether timing changes materially damage pose-related learning.
It does not assume that failure on a sparse, poorly configured online cloud
proves that the underlying dataset cannot support HPE.

All new code, tests, configurations, prepared arrays, checkpoints and reports
stay in this directory. The master SQLite database and artifacts are read-only.
Nothing in the installed SyncWB package imports this study.

## Run it yourself

From the repository root in PowerShell:

```powershell
# Light checks and preparation (already performed if outputs/19_MM_v1 exists):
.\experiments\sync_hpe_feasibility\run.ps1 test
.\experiments\sync_hpe_feasibility\run.ps1 prepare
.\experiments\sync_hpe_feasibility\run.ps1 smoke

# List jobs before starting longer computation:
.\experiments\sync_hpe_feasibility\run.ps1 plan

# Main study. Safe to interrupt and invoke again; completed jobs are skipped.
.\experiments\sync_hpe_feasibility\run.ps1 core

# Optional longer radar context after inspecting the core results:
.\experiments\sync_hpe_feasibility\run.ps1 temporal

# Rebuild the report at any point, without training:
.\experiments\sync_hpe_feasibility\run.ps1 report
```

The launcher uses `%USERPROFILE%\anaconda3\envs\syncwb\python.exe` when present;
otherwise it uses `python` on PATH. PyTorch is an optional study dependency, not
a new SyncWB dependency. The configured syncwb environment already has a CUDA
PyTorch installation on the development machine.

To run just the first unfinished job, or an explicit zero-based job:

```powershell
.\experiments\sync_hpe_feasibility\run.ps1 core -MaxJobs 1
.\experiments\sync_hpe_feasibility\run.ps1 core -Job 12
```

There are 78 core runs (two targets, three seeds, six training-label conditions,
two main models, plus six statistics baselines), and 18 optional temporal runs.
Their total duration can exceed ten minutes; launch these yourself. Each run
prints progress and elapsed seconds. A smoke suite has seven two-epoch runs on
at most 256 training examples, checks execution only, and is never included in
study conclusions. Use the first complete core run to estimate runtime; smoke
timings do not estimate convergence.

Completed jobs resume at the **job boundary**, not mid-epoch. Interrupted jobs
restart from the same seed. Do not run two processes on the same output directory.
Each successful core/temporal job refreshes the report. A result.json file is the
completion marker; partial weight/prediction files are not treated as completed.

Outputs:

```text
outputs/19_MM_v1/
    manifest.json                  # inputs, hashes, settings, anchors
    audit.json                     # counts and coverage
    anchor_leave_one_out.csv
    prepared.npz
    label_displacement.json
    smoke_runs/                    # verification only
    runs/<task-model-condition-seed>/
        weights.pt
        normalization.npz
        history.json
        test_predictions.npz
        result.json
    comparisons.json
    metrics.csv                    # appears after a complete study run
    report.md
    report.html
```

Configuration is `configs/19_MM.json`. For sensitivity analysis, copy it inside
`configs/`, choose a new `output` subdirectory, and pass `-Config PATH`.
Preparation refuses to overwrite existing data. Changing configuration requires
new preparation. Changed code or prepared data cannot silently reuse completed
training results.

## Targets without identity association

Only 2D poses supervise the study. The actual number of detections must equal
the independently recorded `num_people`, and that number must be positive.
A mismatch invalidates the entire target frame; selecting the best-looking
person does not rescue it. All expected detections must have adequate bounding-box
scores. Each joint statistic requires a confident, finite joint in every person.
At least 12 such joints are required. Missing 3D poses never exclude valid 2D.

Two small regression tasks use instantaneous, person-order-invariant targets:

1. **Image joint distribution:** mean x/y and standard deviation x/y over all
   people for each Halpe26 joint. Units are source pose pixels. No image dimensions
   are inferred from the preview video. Standard deviations are masked for one
   person so constant zero targets do not make the task look artificially easy.
2. **Body-relative shape:** normalize each detected skeleton within that frame
   using its hip center and shoulder-to-hip distance, then compute the same
   per-joint statistics over people. No person is matched across frames. This
   task removes overall image position and size but retains articulation and
   body orientation in the image.

These descriptors discard some multiperson information. Neither measures
person-specific tracking or complete multiperson HPE. A person swap does not
invalidate them; two different pose sets can nevertheless have similar moments.

Native joint scores can exceed one. They are treated as detector scores, not
probabilities. Defaults (joint 0.5, bounding box 0.5) are provisional. Review
representative high-confidence labels and run a stricter confidence sensitivity
configuration before final application claims. To exclude reviewed bad labels,
add `excluded_pose_intervals` to a copied config:

```json
[
  {"start_s": 123.0, "end_s": 127.0, "reason": "visually incorrect 2D pose"}
]
```

Times are seconds from the start of the selected RGB timeline. Do not tune these
exclusions by model test error. Bed and doorway labels are included when eligible;
the study does not currently have independent region annotations proving their
coverage.

## Inputs and controls

- **PointNet:** shared point MLP, masked mean/max pooling, two-layer regression
  head. Inputs are native radar x/y/z, Doppler, SNR and relative frame time.
  It sees all finite returns, including stationary and unassociated returns.
  Fixed-seed uniform subsampling caps points at 96 per frame. Padding is masked.
  Target IDs are not used as identities.
- **Statistics MLP:** predicts from radar feature means, spreads, quantiles,
  counts and moving-return fraction. A cheaper baseline for the point encoder.
- **Pose-input control:** predicts the same descriptors from the reference
  frame's 2D descriptors. It is deliberately given informative input. Only its
  training labels are corrupted. It tests susceptibility to changed supervision
  when radar sparsity and calibration are removed. This is **not radar HPE**
  and is not evidence that sparse radar contains those descriptors.
- **Training-mean predictor:** reports a no-radar baseline on the identical
  test samples and scalar mask.

Body-relative normalization uses the pose only to construct the target or the
explicit pose-input control. Radar inputs never use pose crops, pose centers,
identity, activity labels, absolute timestamps or extrinsic calibration.

Single-frame models are primary because long context can compensate for timing
error. The optional nine-frame model sees only current/past radar. Its relative
timestamps preserve the temporal support; future radar frames are never used.
Whole-body 3D reconstruction, Doppler-versus-3D radial velocity, and derivatives
of tracked joints are intentionally outside this study.

## Controlled label changes

The saved piecewise map is inverted at fixed radar times; the nearest observed
pose is accepted only within 50 ms. This is a fresh radar-centered pairing,
not a reverse lookup of the existing RGB-centered nearest-neighbor table.
Actual timestamps are used. No interpolation crosses the RGB gaps.

Conditions are reference, ±200 ms label shifts, Gaussian jitter with standard
deviation 100 ms, every-second-anchor refitting, and a 10-second within-block
cyclic shift. Positive shift means a later pose. The last control is named
`unrelated` in files, but it can still have a similar pose during a long static
activity; check measured label displacement before using it as a negative control.

The current 27-anchor fit is not independently verified timing truth. Every-second
anchoring is not a uniform sparse schedule. Leave-one-anchor-out errors are not
the actual distribution of errors between anchors. Gaussian jitter is an imposed
stress condition, not a measurement of clock error. No experiment automatically
corrects the saved mapping by optimizing test performance.

All conditions share the same sample IDs and per-component masks. The clean
reference supplies validation and test labels for every model. This isolates
changes in training supervision. A separate frozen-predictor sweep changes test
label times by ±50–500 ms on a common sweep mask; it measures pairing sensitivity,
not independent synchronization accuracy.

## Splits, metrics and limits

120-second chronological blocks repeat train, train, validation, train, train,
test. Two-second edge buffers and widest-context checks prevent input windows
and shifted labels from crossing split boundaries. The 10-second control wraps
within its own block. Label person counts must agree across conditions.
Every fourth radar frame is a center; duplicate nearest RGB targets can remain
within a split but never justify independent-sample confidence intervals.

No scaler or active-target selection is fitted on test data. Three seeds share
initialization conventions and budgets across conditions. Early stopping always
uses the same reference validation labels. Test metrics average valid components
per sample. Standardized MSE is the primary regression metric; raw MAE is in
pixels or torso lengths. These are not joint localization MPJPE metrics.

The report averages seeds before bootstrapping paired 120-second block means.
Only a few test blocks may remain; repetitive activities can correlate even across
blocks. Results concern this recording, not unseen people. The reference mapping
itself uses manual anchors throughout the recording: this is not a prospective
synchronization generalization test.

Report coverage alongside error. Low/high radar-support strata mean low/high
**total finite returns**, not measured points near a person. Without spatial
calibration, this study cannot certify human-return coverage. Count/confidence
screening does not establish that every remaining label is correct.

The report states whether controlled timing perturbations harm the learned
proxies and whether useful radar prediction was demonstrated. It must not call
the dataset incapable just because an online-radar model fails. Establishing
whether the **actual** residual errors are intolerable also requires independent
timing checkpoints. If those are unavailable, the final recommendation remains
conditional and names that missing evidence.

The configured 10% material-error margin is provisional and is not used to issue
an automatic pass/fail. Choose application tolerances independently of final test
results if a formal acceptance decision is needed.
