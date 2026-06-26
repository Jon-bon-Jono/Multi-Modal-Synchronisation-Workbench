# Multi-Modal Synchronisation Workbench — Main Context and Design Summary

**Purpose of this file:** This is the primary context file for the project. Read this file first before reading the detailed schema, ER diagram, workflow notes, or open-question log.

---

## 1. What this project is

The **Multi-Modal Synchronisation Workbench** is a design for a tool and data model that supports post hoc synchronisation of multimodal recordings collected from multiple devices and multiple host machines.

The motivating use case is a dataset containing approximately 50-minute recording sessions per subject, where different modalities were captured on different systems with different clocks and different timestamp mechanisms. The workbench is intended to make it possible to:

- ingest and organise these recordings in a consistent way,
- represent device runs and their samples explicitly,
- define manual cross-modal anchor correspondences,
- estimate or refine latent timelines for each device stream,
- fit synchronisation models between modalities,
- generate candidate sample mappings between source and target streams,
- keep track of multiple mapping/model versions without losing provenance,
- support flexible asset storage across multiple machines and users.

This project is not only about a single synchronisation formula. It is about creating a reproducible structure for managing uncertain timing information, human anchoring decisions, derived timelines, and resulting mappings.

---

## 2. Primary motivating dataset and timing problem

The current motivating dataset includes multimodal recordings collected during approximately 50-minute experiments per subject using:

- **Kinect RGB camera** at 15 FPS, captured by **host system 1**
- **Kinect depth camera** at 15 FPS, captured by **host system 1**
- **mmWave radar point-cloud stream** at 20 FPS, captured via **MMWAVEICBOOST** on **host system 2**
- **mmWave raw radar stream** at 20 FPS, captured via **DCA1000EVM** on **host system 2**
- **3x wearable IMUs** at 20 FPS, captured on **host system 3**

Each stream has timestamps derived from a different mechanism. Integration of the IMU stream is an after-thought which hasn't explicitly been covered in the design so far and isn't mentioned again in the documentation. We will most likely integrate this into the system in the future. 

### 2.1 Kinect-related time signals

For Kinect RGB and depth, relevant timing signals can include:

- video PTS for each video frame,
- Kinect internal elapsed timestamps for frames with a 3D pose estimate,
- host-system-1 OS timestamps logged by the SmartCup pipeline for frames with a 3D pose estimate.

### 2.2 Radar point-cloud timing

For point-cloud radar, timestamps are host-system-2 OS timestamps captured when frame parsing completes in the application pipeline. These are not necessarily true acquisition timestamps.

### 2.3 Raw radar timing

For raw radar, all individual frame timestamps are not directly available. Instead, the run will only have:

- recording start time (OS),
- recording end time (OS),

and individual frame times are inferred assuming approximately uniform frame cadence.

### 2.4 Why this is difficult

The modalities do **not** share a single hardware clock. They were captured on different machines, with different software stacks, different timing mechanisms, buffering effects, and scheduling delays. Therefore the timestamps are not perfectly aligned even when the sensors were active during the same physical event.

The core problem is to estimate useful cross-modal timing relationships despite:

- clock offsets,
- clock drift,
- parser/buffer jitter,
- run boundaries and restarts,
- uncertain or inferred timestamps,
- different notions of “time” within each modality.

---

## 3. Overall objective

The objective is to construct a system that can map from one modality’s sample timeline to another modality’s sample timeline in a way that is:

- explicit,
- inspectable,
- versioned,
- robust to uncertainty,
- tolerant of imperfect timing metadata,
- independent of any one particular synchronisation formula.

In the current use case, the main practical question is often:

> Given a Kinect frame, which radar frame is the best corresponding frame?

But the design should remain general enough to support other source/target modality pairs and multiple synchronisation/model variants.

---

## 4. Scope and non-goals

### 4.1 In scope

This project is intended to cover:

- representation of subjects, runs, samples, assets, anchors, timeline models, sync models, and mapping versions,
- support for interrupted runs and multiple device restarts,
- support for multiple timestamp types per sample,
- support for latent time estimates distinct from raw observed timestamps,
- support for manual anchoring,
- support for multiple mapping/model versions,
- support for diagnostics such as residuals, support status, and confidence,
- support for flexible asset references across machines and users.

### 4.2 Not the main focus

This project is **not** primarily about:

- writing one final production-grade synchronisation algorithm,
- forcing all modalities into one single timestamp column,
- assuming that observed timestamps are ground truth,
- hard-coding machine-specific absolute file paths into canonical tables,
- losing provenance by overwriting old mappings or models.

---

## 5. Core design principles

These principles are central and should be preserved when discussing or extending the project.

### 5.1 Anchors are first-class and should be model-independent

A key design choice is that **manual anchors are more fundamental than any specific synchronisation model**.

An anchor records that a human judged two or more samples across runs/modalities to correspond to the same physical event or moment. The anchor itself should not depend on whether the downstream sync model is affine, piecewise affine, spline-based, or something else.

This matters because:

- anchors are expensive human input,
- sync models may change over time,
- multiple sync models may reuse the same anchors,
- improving the modelling approach should not require redoing the anchoring work.

So the conceptual rule is:

> Anchors are durable human correspondences; sync models are derived interpretations of those correspondences.

### 5.2 Separate runs from samples

The workbench should explicitly represent a **device run** as one uninterrupted acquisition segment for one device/modality. This is important because interpolation across a restart or inactive interval is generally invalid.

A run is not just “all data for a subject for a modality”. A subject may have multiple runs per modality if the device was restarted, crashed, paused, or reacquired.

This means:

- synchronisation is performed between specific runs,
- samples belong to runs,
- timeline estimation is run-specific,
- support/interpolation regions are run-specific.

### 5.3 Separate observed timestamps from estimated timelines

Observed timestamps and estimated times should not be collapsed into one concept.

A sample may have:

- native device time,
- OS time,
- inferred wallclock time,
- one or more timeline-model-based estimated times.

These should be represented as distinct concepts because the observed timestamps may be noisy, delayed, incomplete, or derived from different clocks.

### 5.4 Separate timeline models from cross-modal sync models

A very important design choice is to split the synchronisation problem into two layers:

1. **within-run timeline modelling**  
   Estimate a useful time axis for samples of one run.

2. **between-run synchronisation modelling**  
   Map one run’s timeline to another run’s timeline.

This separation avoids forcing a single sync model to also explain modality-specific timestamp pathology.

### 5.5 Separate sync models from mapping outputs

A synchronisation model is not the same thing as the final list of matched samples.

A sync model might define how source time predicts target time. A mapping version then uses that model plus additional rules to generate candidate matches, ranks, support labels, and primary selections.

This separation allows:

- multiple mapping versions from one sync model,
- different pairing strategies on top of the same model,
- reproducible comparison of mapping policies.

### 5.6 Preserve provenance instead of overwriting

The system should support versioning of:

- timeline models,
- sync models,
- mapping outputs.

Old results should not be silently overwritten. The project should make it possible to trace how a mapping was generated and what assumptions or anchors it used.

### 5.7 Canonical IDs over brittle file assumptions

Canonical tables should use stable identifiers and structured references rather than relying on one machine’s local directory layout.

Absolute paths are too brittle for a project that spans:

- a personal workstation,
- external drives,
- OneDrive or shared folders,
- HPC or Linux machines,
- research assistants’ local subsets.

### 5.8 Asset references should be flexible and relocatable

Another important design choice discussed for this project is that asset paths should be managed in a way that supports different machines and users without forcing the canonical metadata to change.

The preferred idea is:

- store canonical asset references in a portable form,
- maintain configurable root paths,
- allow multiple candidate roots,
- allow modality-specific roots,
- allow per-user or per-machine overrides.

In practice, this means the canonical metadata should store something like a relative or logical asset reference, while resolution to an actual filesystem path happens through configuration.

This is preferable to embedding hard-coded absolute paths in metadata/tables.

### 5.9 Root-path configuration should be modality-aware

The root-path design should not assume that all assets live under one common root.

A more suitable approach is to allow roots to be associated with specific modalities or asset roles. For example:

- radar data may primarily live on one local disk,
- Kinect data may primarily live on an external drive,
- a smaller subset may exist in a shared folder,
- assistants may register an override root only for the modality they need.

This keeps the system flexible without making the canonical metadata machine-specific.

### 5.10 OS compatibility matters for stored references

Where path-like strings are stored in metadata/tables, the stored representation should be portable across operating systems as far as practical. That generally argues for:

- using logical storage keys or POSIX-style relative paths in canonical metadata,
- resolving them to OS-native paths only at runtime,
- avoiding dependence on one platform’s absolute path syntax.

### 5.11 One canonical sample index per run

Each run should have a canonical sample ordering, usually represented by `sample_index`. The original frame number may also be stored if needed, but it should remain distinct from the canonical index when the two are not guaranteed to be identical.

This helps keep mapping and anchoring consistent even when source files or frame numbering conventions differ.

---

## 6. Main conceptual entities

This file does not define the full schema in detail, but the following entities are central.

### 6.1 Subject

Represents the experimental subject or top-level recording instance.

### 6.2 Device run

Represents one uninterrupted capture segment for one device/modality.

Typical examples include:

- one Kinect RGB run,
- one Kinect depth run,
- one radar point-cloud run,
- one radar raw run.

### 6.3 Run sample

Represents one sample within a run, usually one frame.

### 6.4 Run asset

Represents run-level assets such as:

- video files,
- radar capture folders,
- point-cloud files,
- metadata bundles,
- calibration or session files.

### 6.5 Sample artifact

Represents sample-level assets such as:

- preview images,
- per-frame point-cloud file references,
- per-frame raw-frame files,
- derived pose artifacts.

### 6.6 Timeline model

Represents a model that assigns or refines time estimates for samples in a run.

### 6.7 Sample time estimate

Represents the time value assigned to a specific sample under a specific timeline model, optionally with residuals or diagnostics.

### 6.8 Anchor

Represents a manually identified cross-run or cross-modal correspondence event.

### 6.9 Anchor member

Represents the participation of a specific sample in an anchor.

### 6.10 Sync model

Represents a model that maps one run’s time axis to another run’s time axis.

### 6.11 Mapping version

Represents one produced set of source-target sample matches, generated by some method and often based on a sync model.

### 6.12 Sample mapping

Represents one candidate or final mapping pair between a source sample and a target sample.

---

## 7. Current synchronisation framing

The current framing is that synchronisation should be understood as a mapping from one modality’s notion of time to another’s.

Let:

- $t_k$ be a Kinect-frame timestamp or Kinect-side timeline value,
- $t_r$ be a radar-frame timestamp or radar-side timeline value,
- $f(t_k)$ be a model mapping Kinect time to radar time.

Once $f$ is estimated, source-to-target pairing can be done by choosing the target sample whose target-side time is closest to the predicted target time:

$$
j^* = \arg\min_j \left| t_r^{(j)} - f(t_k) \right|
$$

This is the general idea, but the design should not assume that the raw observed timestamps themselves are the correct time coordinates to use.

---

## 8. Naive piecewise affine model

One simple starting model is a piecewise affine mapping defined between consecutive anchors.

### 8.1 Anchor definition

A manual anchor pair can be written as:

$$
(t_{k,i}, t_{r,i})
$$

where the two timestamps correspond to samples believed to represent the same physical moment.

If anchors are ordered by Kinect time:

$$
t_{k,1} < t_{k,2} < \dots < t_{k,n}
$$

then each interval between consecutive anchors can be modelled separately.

### 8.2 Piecewise affine interpolation

For consecutive anchors

$$
(t_{k,i}, t_{r,i}), \quad (t_{k,i+1}, t_{r,i+1})
$$

fit:

$$
t_r = a_i t_k + b_i
$$

with

$$
a_i = \frac{t_{r,i+1} - t_{r,i}}{t_{k,i+1} - t_{k,i}}
$$

and

$$
b_i = t_{r,i} - a_i t_{k,i}
$$

This gives a piecewise affine function that interpolates exactly through the anchors.

### 8.3 Why this is attractive

The naive model is attractive because it is:

- easy to explain,
- easy to fit,
- exact at anchor points,
- locally adaptable if drift varies across the run.

### 8.4 Why this is not enough on its own

The problem is that the naive model implicitly treats the chosen timestamps as if they were true acquisition times. In this project that assumption is often violated, especially for radar.

---

## 9. Why raw observed timestamps are not trustworthy enough

### 9.1 Point-cloud radar timestamps reflect parse completion, not pure acquisition

For point-cloud radar, the observed time can be thought of as:

$$
y_i^{pc} = \tau_i^{pc} + \epsilon_i
$$

where:

- $\tau_i^{pc}$ is the true acquisition time,
- $\epsilon_i$ is variable delay from parsing, buffering, and OS scheduling.

Then frame-to-frame observed deltas become:

$$
y_i^{pc} - y_{i-1}^{pc}
=
(\tau_i^{pc} - \tau_{i-1}^{pc}) + (\epsilon_i - \epsilon_{i-1})
$$

Even if the radar acquires at a stable cadence, the observed timestamps can show considerable jitter because the delay term varies.

So short-term slopes inferred directly from observed times may reflect parser jitter rather than true clock behaviour.

### 9.2 Raw radar timestamps are smooth but inferred

For raw radar, individual frame times are inferred from start/end timestamps and frame count. If there are $N$ frames and logged start/end times $t_{start}$ and $t_{end}$, then a common estimate is:

$$
\tau_n^{raw}
=
t_{start}
+
\frac{n}{N-1}(t_{end} - t_{start})
$$

This gives a smooth cadence, but it inherits uncertainty from:

- errors in the logged start or end times,
- slight mismatch between nominal and true frame rate,
- accumulated drift across the run.

the absolute difference between the logged end time and the extrapolated end time across runs averaged approximately **0.3486 ± 0.2473 s**. That is a warning that the raw-radar timeline may be smooth but still meaningfully uncertain.

### 9.3 Kinect is also not a perfect absolute clock

Kinect-side times may also come from mixed notions of time:

- video PTS,
- internal elapsed time,
- host OS time.

These are useful, but they should not be collapsed into one “ground truth” timestamp without care. 

---

## 10. Revised synchronisation approach

The improved conceptual approach separates two problems:

1. estimating a useful timeline for each run,
2. synchronising one run’s timeline to another run’s timeline.

This is one of the most important design choices in the project.

### 10.1 Point-cloud radar timeline estimation

For point-cloud radar, a latent acquisition timeline can be estimated from frame index.

Let the latent time for frame $i$ be:

$$
\tau_i^{pc} = \alpha + \beta i
$$

where $\alpha$ and $\beta$ are estimated by regressing observed timestamps on frame index.

Residuals are:

$$
r_i = y_i^{pc} - (\alpha + \beta i)
$$

This yields:

- a smoothed latent timeline,
- per-frame residuals for diagnostics,
- a cleaner basis for cross-modal synchronisation than the raw observed timestamp alone.

### 10.2 Raw radar timeline estimation

For raw radar, the current practical baseline is the start/end constrained interpolation:

$$
\tau_n^{raw}
=
t_{start}
+
\frac{n}{N-1}(t_{end} - t_{start})
$$

This gives a uniform per-frame time axis for the run.

It is still uncertain, but it avoids the opposite problem of parser jitter.

### 10.3 Kinect-side timeline choice

A Kinect-side timeline may be based on one of several time sources depending on the downstream task:

- video PTS,
- Kinect internal elapsed time,
- best-effort wallclock estimate,
- a future latent timeline model if that becomes necessary.

One unresolved idea noted earlier is that a latent timeline model may also eventually be useful for RGB/Kinect rather than treating the current available timestamp as fixed and final.

### 10.4 Cross-modal synchronisation then uses the chosen timelines

Once each run has a chosen timeline model, the cross-modal sync model should operate between those timeline axes rather than directly between raw observed timestamps.

This is cleaner conceptually and makes the modelling assumptions easier to inspect.

---

## 11. Interpolation, support boundaries, and extrapolation

This project should treat support regions carefully.

### 11.1 No interpolation across inactive intervals

Interpolation should **not** be allowed across time gaps where a sensor was inactive or restarted. A run boundary matters. If the device stopped and restarted, that should produce separate runs rather than one continuous interval.

### 11.2 Support status should be explicit

Mappings should carry a support label or region type such as:

- within supported interpolation region,
- boundary region,
- extrapolated before first anchor,
- extrapolated after last anchor,
- outside valid run overlap,
- unsupported because of restart or missing coverage.

The exact taxonomy can be finalised separately, but the main point is that support status should be explicit, not implicit.

### 11.3 Extrapolation should be policy-controlled

The project should not silently extrapolate forever outside anchor-supported regions. Extrapolation behaviour should be tied to an explicit model policy.

Possible policies include:

- no extrapolation,
- local affine extrapolation from nearest anchor segment,
- bounded extrapolation up to a maximum tolerance,
- extrapolation allowed but marked low confidence.

Whatever policy is used should be represented in the sync model metadata.

---

## 12. Mapping outputs and ranking

The workbench should support generation of candidate mappings from a source run to a target run.

Typical mapping metadata may include:

- source run and source sample index,
- target run and target sample index,
- rank,
- whether the pair is the primary mapping,
- mapping method,
- predicted-minus-estimated time error,
- support status,
- confidence score,
- notes.

This makes it possible to keep multiple candidate matches rather than pretending there is always one absolutely certain pair.

---

## 13. Timestamp and time-representation philosophy

A recurring theme in this project is that “timestamp” is not a single simple thing. A sample may have several time-related fields with different meaning.

Examples include:

- native PTS,
- internal elapsed time,
- host OS time,
- inferred wallclock estimate,
- timeline-model-assigned time.

The workbench should preserve these distinctions rather than flattening them too early.

A practical consequence is:

> The system should store what was observed, what was inferred, and what was model-estimated as separate layers.

---

## 14. Practical data-management strategy for assets

The asset-management strategy is important because the data may be split across many locations:

- personal computer,
- external hard drive,
- OneDrive or shared storage,
- HPC or Linux machine,
- local subsets prepared for assistants.

### 14.1 Canonical metadata should stay portable

Canonical metadata files should avoid embedding machine-specific absolute paths where possible.

Instead, they should store:

- logical asset refs,
- storage keys,
- relative paths,
- run-level or sample-level references that are portable.

### 14.2 Path resolution should be configurable

Actual filesystem resolution should happen through configuration, not by rewriting the canonical tables.

A useful pattern is:

- one config file that defines available roots,
- roots grouped by modality or asset role,
- fallback search across multiple candidate roots,
- optional user-specific overrides.

### 14.3 Assistants should not need the full dataset

The design should make it possible for a research assistant to register only the local root(s) for the subset they actually have, for example only one subject and one modality.

This is another reason not to hard-code one canonical absolute path in the project metadata.

---

## 15. Relationship to the detailed schema and ER diagram

This file deliberately stays at the conceptual level. The detailed table definitions, column choices, and relationships should be maintained separately in:

- the detailed data-model markdown file,
- the Mermaid ER diagram file.

That separation is intentional:

- this file explains the reasoning and canonical concepts,
- the schema file defines the concrete tables/columns,
- the ER diagram provides the compact relational view.

---

## 16. Recommended reading order for an agent

A new agent session should ideally ingest the project in this order:

1. **This main context file**
2. **Detailed data model and tables**
3. **ER diagram**
4. **Workflows and usage**
5. **Open questions and future changes**

This order matters because the schema and workflows depend on the conceptual distinctions established here.

---

## 17. Current status of certainty

Not everything in this project is fully settled. The most stable ideas at present are:

- anchors should be first-class and model-independent,
- runs should represent uninterrupted capture segments,
- observed timestamps should be separated from timeline-model estimates,
- timeline models should be separated from cross-modal sync models,
- mapping outputs should be versioned separately from the models that produced them,
- canonical asset references should be portable and path resolution should be configurable.

Areas that are still less final include:

- the exact Kinect-side timeline strategy,
- the final extrapolation policy taxonomy,
- the exact confidence/scoring scheme for mappings,
- whether additional latent timeline models are needed for video/Kinect,
- some schema details and workflow ergonomics.

These should be treated as current design direction rather than immutable final truth.

---

## 18. Compact project summary

In one paragraph:

The Multi-Modal Synchronisation Workbench is a post hoc synchronisation framework for multimodal recordings captured on different devices and host systems with imperfect and heterogeneous timestamps. Its core design is to separate manual anchor correspondences, run-specific timeline estimation, cross-modal synchronisation models, and generated mapping outputs, while keeping provenance and uncertainty explicit. The project also treats asset storage as flexible and relocatable by using portable references plus modality-aware configurable roots rather than brittle absolute paths. The workbench is therefore not just a formula for matching frames; it is a structured system for representing uncertain time, human correspondence decisions, modelled timelines, and reproducible cross-modal mappings.

---
