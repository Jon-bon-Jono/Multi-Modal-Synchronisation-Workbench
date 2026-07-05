# v0.2.1 architecture overview

```mermaid
flowchart LR
    CLI[CLI / future GUI] --> Services[Application services]
    Services --> Core[Core schemas, enums, IDs, validation]
    Services --> Storage[CoreStore interface]
    Services --> Assets[Asset resolver]
    Services --> Artifacts[Artifact store]
    Storage --> SQLite[(SQLite canonical store)]
    Storage --> Export[Parquet/CSV export]
    Artifacts --> Bundles[NPZ / JSONL payload bundles]
    Assets --> Roots[User-local roots config]
```

The backend has no GUI dependency. A future Qt GUI should call the service layer
rather than directly reading pandas dataframes or SQLite tables.

## Main services

- `IngestionService`: reads the temporary package and writes the canonical store.
- `MappingService`: generates initial nearest-time mappings using selected source and target timelines.
- `ArtifactBuildService`: builds run-level payload bundles and writes artifact metadata.
- `PayloadService`: retrieves sample payloads by canonical sample identity.
- `PairInspectionService`: retrieves mapped-pair metadata, summaries, payload roles, and payload shapes.
- `ArtifactAuditService`: checks artifact file/metadata consistency.

## Mapping provenance

```mermaid
flowchart TD
    A[Selected source timeline] --> S[SYNC_MODEL: identity_time]
    B[Selected target timeline] --> S
    S --> M[MAPPING_VERSION]
    M --> R[SAMPLE_MAPPING rows]
```

Even the crude nearest-frame mapping is represented as a derived mapping version
from an explicit sync model.

## Initial Nearest Mapping

The v0.1 nearest mapping is an anchor-placement aid. It is intended to give the
future GUI or notebook workflow a default target frame to jump to when browsing
from RGB to radar.

For this mapping method, `is_primary=True` means “selected default navigation candidate under the configured primary policy”, not “trusted final synchronised correspondence”.

The default v0.1 `primary_policy` is `supported-only`, so weakly supported rows are kept as candidates but are not marked primary. Less conservative policies such as `within-max-delta` or `nearest-any` can be selected explicitly.

Final or anchor-derived mappings must be generated as separate `MAPPING_VERSION`
rows from an anchor-based `SYNC_MODEL`.

## Mapping overwrite policy

`map-nearest` and `map-nearest-all` refuse to reuse an existing `mapping_version_id` by default. If `--overwrite` is passed, existing `SAMPLE_MAPPING` rows for that mapping version are deleted before regenerated rows are inserted.