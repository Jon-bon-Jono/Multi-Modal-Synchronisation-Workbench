# 06 — Open Questions and Future Changes

**Purpose of this file:** capture what is still undecided, what may change later, and what should not be treated as fully settled.

---

## 1. What is already treated as settled

The following are assumed stable unless there is a deliberate redesign:

- anchors are model-independent,
- runs are uninterrupted acquisition segments,
- observed timestamps are distinct from estimated timeline values,
- run timeline models are distinct from cross-run sync models,
- sync models are distinct from mapping outputs,
- canonical metadata should stay portable across machines,
- large payloads should stay out of the core tables.

This file is only about what is **not** settled.

---

## 2. Top unresolved questions

## 2.1 What should the canonical time coordinate actually be?

Several time notions exist at once:

- native device time,
- OS wallclock time,
- inferred wallclock time,
- latent timeline coordinates that may only be approximately wallclock-aligned.

The schema currently allows this ambiguity to exist, which is good. The unresolved issue is whether the long-term system should prefer one canonical coordinate for most downstream work.

**Main options:**
- use wallclock-like datetime values whenever possible,
- use numeric offsets relative to run start,
- allow both and make `time_kind` mandatory.

**Current leaning:** keep both possible; do not force one representation too early.

---

## 2.2 How many timeline models should be kept per run?

The current design allows many timeline models per run. That is flexible, but it can create clutter if every experiment produces many tiny variants.

**Open question:** what counts as a new timeline model version rather than just a diagnostic rerun of the same idea?

**Likely rule:** create a new version only when one of these changes:
- the source timing basis,
- the fitting method,
- segment boundaries,
- excluded samples,
- assumptions about cadence or drift.

---

## 2.3 Should point-cloud radar always get a latent timeline model?

The current design strongly suggests smoothing point-cloud timestamps by regressing observed time against frame index. That is sensible, but it is still a modelling choice.

**Unresolved point:** whether the smoothed timeline should become the default time axis for all downstream work, or whether raw observed timestamps should remain equally first-class in mapping workflows.

**Current leaning:** keep both, but make the smoothed timeline the normal starting point.

---

## 2.4 How should raw radar timing uncertainty be represented?

Raw radar time is inferred from start/end constraints and assumed cadence. The system can store a timeline, but that alone does not express uncertainty well.

**Open question:** should uncertainty be recorded explicitly?

Possible approaches:
- a single run-level uncertainty summary,
- per-sample uncertainty estimates,
- only notes/diagnostics outside the core tables.

**Current leaning:** start with run-level uncertainty summaries. Per-sample uncertainty can be added later if it proves useful.

---

## 2.5 How strict should interpolation and extrapolation rules be?

It is already agreed that interpolation must not cross inactive intervals or restarts. The open issue is how the UI and outputs should label behaviour near the boundaries.

**Still to settle:**
- whether extrapolation before the first anchor and after the last anchor should be allowed by default,
- whether different model types can expose different extrapolation policies,
- how aggressively mappings outside anchor support should be down-ranked or hidden.

**Current leaning:** allow extrapolation only when explicitly requested and label it clearly.

---

## 2.6 How should support regions be defined?

A mapping can be:
- between anchors,
- before the first anchor,
- after the last anchor,
- outside model support entirely,
- near a run boundary or discontinuity.

The unresolved issue is whether support should be a small fixed enum or a richer diagnostic object.

**Current leaning:** keep a compact enum in the canonical tables and push richer diagnostics to derived artifacts.

---

## 2.7 What anchor granularity is actually needed?

Anchors currently work at the sample level. That is the cleanest representation, but the human judgement behind an anchor may be fuzzy.

Examples:
- the best corresponding frame may be ambiguous within a small window,
- different modalities may expose the event differently,
- multiple nearby samples may be equally plausible.

**Open question:** should anchors eventually support interval or window semantics rather than only exact sample picks?

**Current leaning:** keep sample-level anchors for now and use confidence plus notes to express ambiguity.

---

## 2.8 What should anchor confidence mean?

Confidence exists in the design, but its semantics are not yet fixed.

Possible meanings:
- subjective annotator certainty,
- estimated temporal precision,
- agreement across annotators,
- confidence score derived from downstream residuals.

These are not the same thing.

**Current leaning:** treat anchor confidence as human certainty only, unless a separate machine-derived field is introduced later.

---

## 2.9 Do we need multiple anchor types?

The schema allows `anchor_type`, but the controlled vocabulary is not settled.

Potential anchor types:
- exact event,
- approximate event,
- boundary marker,
- restart marker,
- calibration marker.

**Current leaning:** keep the field, but do not over-engineer the enum until real usage demands it.

---

## 3. Mapping-policy questions

## 3.1 One-to-one mapping or many-to-one mapping?

For practical frame pairing, a source sample is often mapped to the nearest target sample. That does not guarantee global one-to-one behaviour.

**Still unresolved:**
- whether the primary mapping should enforce one-to-one pairings,
- whether multiple source samples can share the same target sample,
- whether one-to-one should be optional and method-dependent.

**Current leaning:** do not force one-to-one globally.

---

## 3.2 How many candidates should be stored per source sample?

The current design allows ranked candidate mappings, not just a single match.

**Open question:** how many ranks are actually worth storing?

Too many candidates create noise. Too few make debugging harder.

**Current leaning:** keep the top few only, with one primary match.

---

## 3.3 What diagnostics belong in canonical mapping rows?

Possible diagnostics include:
- predicted minus estimated time delta,
- absolute time delta,
- support label,
- confidence score,
- local anchor density,
- distance to nearest anchor,
- interpolation vs extrapolation flag.

Not all of these belong in the core table.

**Current leaning:** keep the canonical mapping rows compact and move richer diagnostics to derived artifacts.

---

## 3.4 Should mappings ever be edited by hand?

The design supports versioned mappings, but it is not fully settled whether manual correction of individual pairings should be a first-class workflow.

**Tension:**
- manual edits are sometimes useful,
- manual edits can also hide model problems.

**Current leaning:** prefer new mapping versions or new anchors over ad hoc manual patching.

---

## 4. Data-model and storage questions

## 4.1 How much should live in canonical tables versus external artifacts?

The current direction is to keep canonical tables small and stable, with large or messy data stored externally.

The unresolved part is the boundary.

Examples still open to judgement:
- small per-sample dictionaries,
- activity labels,
- compact pose summaries,
- quality flags,
- annotator notes.

**Current leaning:** if a field is needed for filtering, joins, or UI summaries, it may belong in a table; otherwise keep it in artifacts.

---

## 4.2 Should modality-specific sample tables survive long term?

The temporary ingestion package uses files such as `rgb_samples.zst`, `radar_pc_samples.zst`, and `radar_raw_samples.zst`. That is useful for ingestion, but it is not clear how much of that shape should survive in the canonical layer.

**Current leaning:** keep modality-specific wide tables as ingestion or export formats, not as the canonical internal model.

---

## 4.3 How should paths and storage roots be configured?

The project already leans toward portable asset references plus configurable roots. The unresolved details are operational:

- config file format,
- precedence rules across default roots and user overrides,
- per-modality root groups,
- whether failed resolution attempts should be cached.

**Current leaning:** per-modality root groups with user-local overrides.

---

## 4.4 Do we need stronger constraints on controlled vocabularies?

Fields such as `device_type`, `sample_kind`, `asset_role`, `time_kind`, and `support_status` will become messy if left completely free-form.

**Open question:** how strict should the controlled vocabularies be at this stage?

**Current leaning:** define compact recommended enums now, but do not lock the design so hard that small experiments become annoying.

---

## 4.5 Should canonical IDs be human-readable or opaque?

Readable IDs are easier to inspect. Opaque IDs are safer when names change.

Examples:
- `subject07_rgb_run1`,
- UUID-like identifiers.

**Current leaning:** human-readable IDs are fine at this stage.

---

## 5. Workflow and tooling questions

## 5.1 What should the first real UI support?

The long-term concept is a timeline-based workbench, but the first usable tool does not need every feature.

**Likely priorities:**
1. inspect runs and assets,
2. browse samples,
3. place anchors,
4. fit or select a sync model,
5. preview resulting mappings.

Still open:
- whether to start with a GUI immediately,
- whether a CLI plus plots is enough for the first pass,
- how much annotation support research assistants need on day one.

---

## 5.2 How should anchor placement be assisted?

Possible aids include:
- rough initial alignment from naive timestamps,
- jump-to-predicted-frame behaviour,
- local search windows,
- side-by-side previews,
- keyboard shortcuts for fast anchoring.

**Current leaning:** provide rough prealignment first, then manual refinement.

---

## 5.3 Do we need multi-user annotation support early?

The project may be shared with research assistants. That raises questions about:
- local path overrides,
- permission to edit canonical metadata,
- conflict handling when two people anchor the same run pair,
- whether annotator identity should be recorded.

**Current leaning:** support separate local asset resolution early; defer full collaborative editing until it is actually needed.

---

## 5.4 What is the minimal reproducibility package?

For migration, sharing, or publication, it is not yet fixed what the smallest useful bundle should contain.

Possible bundle contents:
- core canonical tables,
- ER diagram,
- asset-root config template,
- selected derived artifacts,
- anchor exports,
- mapping exports.

**Current leaning:** define a small portable package once the first end-to-end pipeline works.

---

## 6. Analytical questions to revisit once data are loaded

These are not pure schema questions. They need real data.

## 6.1 How bad is point-cloud timestamp jitter after smoothing?

Need to check whether a simple linear timeline model is good enough or whether some runs need piecewise treatment.

## 6.2 How stable is the raw radar cadence assumption?

Need to check whether start/end interpolation is acceptable across all runs or whether some sessions show evidence of dropped or irregular frames.

## 6.3 How dense do anchors need to be?

Need to measure how mapping quality degrades as anchor spacing increases.

## 6.4 Which events are easiest to anchor reliably?

Need to identify event types that produce the best cross-modal agreement.

## 6.5 Are some modality pairs not worth modelling directly?

For example, some mappings may be better composed through an intermediate timeline rather than fitted directly.

---

## 7. Likely future changes

These are plausible future extensions, not current requirements.

- add explicit run-boundary or discontinuity tables if restarts become more complex,
- add run-level uncertainty summaries for timeline models,
- add annotator identity and audit fields for anchors,
- add richer derived diagnostics outside the core relational schema,
- support interval-style anchors if exact sample anchors prove too brittle,
- add import/export helpers for assistant-specific subsets,
- formalise a reproducibility bundle for migration and publication.

---

## 8. Immediate next decisions worth making

The most useful near-term decisions are:

1. settle the first controlled vocabularies for `time_kind`, `support_status`, and `asset_role`,
2. define the first extrapolation policy options,
3. decide what minimum diagnostics every mapping row should store,
4. define the first local asset-root configuration format,
5. decide whether anchor confidence is human-only.

These are the questions most likely to unblock implementation without forcing a major redesign.
