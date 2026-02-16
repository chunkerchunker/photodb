# Full HDBSCAN Hierarchy Implementation

**Date:** 2026-02-15
**Status:** Approved

## Problem

The current HDBSCAN implementation uses HDBSCAN as a flat clustering tool. After `clusterer.fit()`, only `labels_` and `probabilities_` are extracted. The condensed tree, lambda values, cluster persistence, outlier scores, and prediction data are all discarded. This means:

- **Epsilon is a heuristic**: Calculated as the 90th percentile of pairwise distances, disconnected from HDBSCAN's density model
- **Core points are approximated**: Defined as `probability >= 0.8` rather than using HDBSCAN's actual core distance concept
- **No incremental prediction**: `approximate_predict()` is unavailable because `prediction_data` is never enabled
- **No hierarchy access**: Cannot re-extract clusters at different density levels without re-running HDBSCAN
- **No outlier detection**: GLOSH outlier scores are discarded

## Approach: Live Clusterer + Stored Tree

Store the condensed tree AND enable `prediction_data=True` so we can use `approximate_predict()` for incremental assignment. Serialize the fitted clusterer state alongside the tree.

### Two-Tier Incremental Assignment

1. **Primary**: `approximate_predict` using the cached clusterer from the most recent bootstrap
2. **Fallback**: Epsilon-ball assignment (current logic) when the clusterer is unavailable, stale, or returns noise for a detection that might fit a post-bootstrap cluster

## Database Schema Changes

### New Table: `hdbscan_run`

Stores serialized clusterer and condensed tree per bootstrap run.

```sql
CREATE TABLE hdbscan_run (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collection(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding_count INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    noise_count INTEGER NOT NULL,
    min_cluster_size INTEGER NOT NULL,
    min_samples INTEGER NOT NULL,
    condensed_tree JSONB NOT NULL,
    label_to_cluster_id JSONB NOT NULL,       -- maps HDBSCAN label -> database cluster ID
    clusterer_state BYTEA,                     -- pickled clusterer for approximate_predict
    is_active BOOLEAN NOT NULL DEFAULT TRUE    -- only one active run per collection
);

CREATE UNIQUE INDEX idx_hdbscan_run_active
    ON hdbscan_run(collection_id) WHERE is_active = TRUE;
```

### New Columns on `cluster`

```sql
ALTER TABLE cluster ADD COLUMN lambda_birth REAL;
ALTER TABLE cluster ADD COLUMN persistence REAL;
ALTER TABLE cluster ADD COLUMN hdbscan_run_id INTEGER REFERENCES hdbscan_run(id);
```

- `lambda_birth`: Density level where the cluster emerged in the condensed tree. Epsilon derived as `1/lambda_birth`.
- `persistence`: HDBSCAN cluster stability score. Higher = more stable cluster.
- `hdbscan_run_id`: Which bootstrap run produced this cluster.

### New Columns on `person_detection`

```sql
ALTER TABLE person_detection ADD COLUMN lambda_val REAL;
ALTER TABLE person_detection ADD COLUMN outlier_score REAL;
```

- `lambda_val`: Point's lambda value from the condensed tree (density at which it joins/leaves its cluster).
- `outlier_score`: GLOSH outlier score. 0 = strong inlier, 1 = strong outlier.

### Epsilon Derivation Change

`cluster.epsilon` is now set as `1/lambda_birth`, clamped to `[0.1, clustering_threshold * 1.5]`. The old percentile-of-pairwise-distances calculation (`calculate_cluster_epsilon` in `hdbscan_config.py`) is removed and replaced by a simpler `lambda_to_epsilon()` function.

Existing clusters retain their current percentile-based epsilon until re-bootstrap.

## Bootstrap Flow Changes

### `create_hdbscan_clusterer()` in `hdbscan_config.py`

- Add `prediction_data=True`
- Add `gen_min_span_tree=True`

### `_run_hdbscan_cpu()`

Returns the full `clusterer` object instead of just `(labels_, probabilities_)`.

### `_run_hdbscan_metal()`

Returns the full `clusterer` object. Since `prediction_data=True` may not work with precomputed sparse matrices, the Metal path uses GPU for the initial k-NN and clustering, then fits a second CPU clusterer with `prediction_data=True` on the raw embeddings using the labels from Metal as a seed. The CPU clusterer is the one serialized for `approximate_predict`.

**Alternative (simpler):** If the Metal clusterer supports `prediction_data` with precomputed input, use it directly. Test this first.

### `_assign_bootstrap_clusters()`

Additional extraction steps:

1. Extract `lambda_birth` per cluster from `clusterer.condensed_tree_`
2. Set `cluster.epsilon = 1/lambda_birth` (clamped)
3. Set `cluster.persistence` from `clusterer.cluster_persistence_`
4. Set `person_detection.lambda_val` per point from the condensed tree
5. Set `person_detection.outlier_score` from `clusterer.outlier_scores_`
6. Serialize condensed tree to JSONB (via `condensed_tree_.to_pandas().to_dict()`)
7. Serialize clusterer to BYTEA (via `joblib.dump` to bytes buffer)
8. Store both in `hdbscan_run`, along with `label_to_cluster_id` mapping
9. Mark previous `hdbscan_run` for this collection as `is_active = FALSE`

## Incremental Assignment Changes

### Primary Path: `approximate_predict`

1. Load the active `hdbscan_run` clusterer for this collection (deserialize from BYTEA, cached in `self._cached_clusterer` after first load)
2. Call `hdbscan.approximate_predict(clusterer, [embedding])`
3. If result is a valid cluster label (not -1):
   - Map HDBSCAN label to database cluster ID via `label_to_cluster_id`
   - Apply cannot-link constraint filtering
   - Apply verified cluster stricter threshold
   - Assign with returned strength as confidence
4. If result is -1 (noise): fall through to fallback path

### Fallback Path: Epsilon-Ball

Used when:
- No active `hdbscan_run` exists (pre-bootstrap collections)
- Clusterer is stale (embedding count grown >25% since bootstrap)
- `approximate_predict` returns -1 but epsilon-ball finds a match (post-bootstrap clusters)
- Clusters created after the bootstrap run (not in condensed tree)

This is the current `_cluster_single_detection` logic, unchanged.

### Staleness Detection

- Compare `hdbscan_run.embedding_count` vs current total embeddings
- When ratio exceeds configurable threshold (default 1.25x), log warning
- No auto-re-bootstrap — user triggers via `migrate_to_hdbscan.py` or maintenance command

### Clusterer Caching

- Deserialize once per `ClusteringStage` instance lifetime
- Cache as `self._cached_clusterer` and `self._cached_label_map`
- Invalidate when `hdbscan_run.id` changes

## Manual Assignment Impact

**No epsilon change on manual add.** Epsilon (derived from `lambda_birth`) represents the cluster's density-based boundary. Manual additions are identity assertions, not density evidence.

What changes on manual add:
- Centroid recomputed (already implemented)
- `face_count` updated (already implemented)
- Detection gets `cluster_status = 'manual'`, `outlier_score = NULL`, `lambda_val = NULL`

Epsilon updates only on re-bootstrap, where manual detections participate as regular data points and HDBSCAN determines new density structure naturally.

## Maintenance Changes

- Pool clustering in `utils/maintenance.py` extracts lambda/outlier values for newly created clusters and detections, but does NOT create `hdbscan_run` or serialize a clusterer (smaller-scope operation)
- New maintenance task: `check-staleness` — compares active `hdbscan_run.embedding_count` against current total, reports whether re-bootstrap is recommended

## Migration Strategy

Migration `011_hdbscan_hierarchy.sql`:
- Creates `hdbscan_run` table
- Adds `lambda_birth`, `persistence`, `hdbscan_run_id` to `cluster`
- Adds `lambda_val`, `outlier_score` to `person_detection`
- Adds `label_to_cluster_id` JSONB to `hdbscan_run`
- Existing clusters keep their current epsilon values as valid fallback

After migration, users run `migrate_to_hdbscan.py` (or a new `--re-bootstrap` flag) to populate the new columns and create the first `hdbscan_run`.

## Files Modified

| File | Changes |
|------|---------|
| `migrations/011_hdbscan_hierarchy.sql` | New migration |
| `src/photodb/utils/hdbscan_config.py` | Add `prediction_data`, `gen_min_span_tree`, replace `calculate_cluster_epsilon` with `lambda_to_epsilon` |
| `src/photodb/stages/clustering.py` | Return full clusterer from bootstrap, extract hierarchy data, two-tier incremental assignment, clusterer caching |
| `src/photodb/database/repository.py` | New methods for `hdbscan_run` CRUD, updated cluster/detection methods for new columns |
| `src/photodb/utils/maintenance.py` | Extract lambda/outlier from pool clustering, add staleness check |
| `scripts/migrate_to_hdbscan.py` | Update to use new bootstrap flow |
| `docs/DESIGN.md` | Update clustering section |
| `CLAUDE.md` | Update clustering configuration docs |