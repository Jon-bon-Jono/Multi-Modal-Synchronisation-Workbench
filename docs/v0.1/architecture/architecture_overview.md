# v0.1 architecture overview

```mermaid
flowchart LR
    CLI[CLI / future GUI] --> Services[Application services]
    Services --> Core[Core schemas, enums, IDs, validation]
    Services --> Storage[CoreStore interface]
    Services --> Assets[Asset resolver]
    Storage --> SQLite[(SQLite canonical store)]
    Storage --> Export[Parquet/CSV export]
    Assets --> Roots[User-local roots config]
```

The backend has no GUI dependency. A future Qt GUI should call the service layer
rather than directly reading pandas dataframes or SQLite tables.

## Main v0.1 services

- `IngestionService`: reads the temporary package and writes the canonical store.
- `MappingService`: generates initial nearest-time mappings using selected
  source and target timelines.

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

For this mapping method, `is_primary=True` means “default navigation candidate”,
not “trusted synchronised correspondence”. Rows with `support_status=weak_support`
may still be primary if they are the nearest available candidate.

Final or anchor-derived mappings must be generated as separate `MAPPING_VERSION`
rows from an anchor-based `SYNC_MODEL`.