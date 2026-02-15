# Full HDBSCAN Hierarchy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace flat HDBSCAN usage with full hierarchy extraction, lambda-derived epsilon, persisted clusterer for `approximate_predict`, and stored condensed tree.

**Architecture:** Two-tier incremental assignment: primary path uses `approximate_predict` from a cached HDBSCAN clusterer; fallback uses epsilon-ball for stale/missing clusterers and post-bootstrap clusters. Bootstrap extracts full hierarchy (condensed tree, lambda values, persistence, outlier scores) and persists both derived values and the raw tree.

**Tech Stack:** Python, hdbscan library, PostgreSQL, pgvector, pickle (stdlib), numpy

**Design doc:** `docs/plans/2026-02-15-hdbscan-full-hierarchy-design.md`

---

### Task 1: Database Migration

**Files:**
- Create: `migrations/011_hdbscan_hierarchy.sql`

**Step 1: Write the migration**

```sql
-- Migration: Full HDBSCAN Hierarchy Support
-- Adds hdbscan_run table for persisting clusterer state and condensed tree.
-- Adds lambda_birth, persistence, hdbscan_run_id to cluster table.
-- Adds lambda_val, outlier_score to person_detection table.

-- =============================================================================
-- Part 1: Create hdbscan_run table
-- =============================================================================

CREATE TABLE IF NOT EXISTS hdbscan_run (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collection(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding_count INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    noise_count INTEGER NOT NULL,
    min_cluster_size INTEGER NOT NULL,
    min_samples INTEGER NOT NULL,
    condensed_tree JSONB NOT NULL,
    label_to_cluster_id JSONB NOT NULL,
    clusterer_state BYTEA,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Only one active run per collection
CREATE UNIQUE INDEX IF NOT EXISTS idx_hdbscan_run_active
    ON hdbscan_run(collection_id) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_hdbscan_run_collection
    ON hdbscan_run(collection_id);

COMMENT ON TABLE hdbscan_run IS 'Persisted HDBSCAN bootstrap runs with condensed tree and serialized clusterer';
COMMENT ON COLUMN hdbscan_run.condensed_tree IS 'HDBSCAN condensed tree as JSON (from condensed_tree_.to_pandas().to_dict())';
COMMENT ON COLUMN hdbscan_run.label_to_cluster_id IS 'Mapping from HDBSCAN label (int) to database cluster ID (int)';
COMMENT ON COLUMN hdbscan_run.clusterer_state IS 'Pickled HDBSCAN clusterer for approximate_predict';
COMMENT ON COLUMN hdbscan_run.is_active IS 'Only one active run per collection; previous runs kept for audit';

-- =============================================================================
-- Part 2: Add hierarchy columns to cluster
-- =============================================================================

ALTER TABLE cluster ADD COLUMN IF NOT EXISTS lambda_birth REAL;
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS persistence REAL;
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS hdbscan_run_id INTEGER REFERENCES hdbscan_run(id);

CREATE INDEX IF NOT EXISTS idx_cluster_hdbscan_run ON cluster(hdbscan_run_id)
    WHERE hdbscan_run_id IS NOT NULL;

COMMENT ON COLUMN cluster.lambda_birth IS 'Density level where cluster emerged in condensed tree (epsilon = 1/lambda_birth)';
COMMENT ON COLUMN cluster.persistence IS 'HDBSCAN cluster stability score (higher = more stable)';
COMMENT ON COLUMN cluster.hdbscan_run_id IS 'Which bootstrap run produced this cluster';

-- =============================================================================
-- Part 3: Add hierarchy columns to person_detection
-- =============================================================================

ALTER TABLE person_detection ADD COLUMN IF NOT EXISTS lambda_val REAL;
ALTER TABLE person_detection ADD COLUMN IF NOT EXISTS outlier_score REAL;

COMMENT ON COLUMN person_detection.lambda_val IS 'Point lambda value from condensed tree (density at join/leave)';
COMMENT ON COLUMN person_detection.outlier_score IS 'GLOSH outlier score (0=inlier, 1=outlier)';
```

**Step 2: Verify migration syntax**

Run: `psql $DATABASE_URL -f migrations/011_hdbscan_hierarchy.sql`
Expected: All statements succeed without errors.

**Step 3: Commit**

```bash
git add migrations/011_hdbscan_hierarchy.sql
git commit -m "feat(db): add migration for full HDBSCAN hierarchy support"
```

---

### Task 2: Update `hdbscan_config.py` — Clusterer Factory & Lambda-to-Epsilon

**Files:**
- Modify: `src/photodb/utils/hdbscan_config.py`
- Create: `tests/test_hdbscan_config.py`

**Step 1: Write failing tests**

```python
"""Tests for HDBSCAN configuration and utility functions."""

import numpy as np
import pytest
from photodb.utils.hdbscan_config import (
    create_hdbscan_clusterer,
    lambda_to_epsilon,
    extract_cluster_lambda_births,
    serialize_condensed_tree,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
)


class TestLambdaToEpsilon:
    def test_basic_conversion(self):
        # lambda_birth=2.0 -> epsilon=0.5
        assert lambda_to_epsilon(2.0) == pytest.approx(0.5)

    def test_clamped_to_min(self):
        # Very high lambda -> very small epsilon, clamped to 0.1
        assert lambda_to_epsilon(100.0) == 0.1

    def test_clamped_to_max(self):
        # Very low lambda -> very large epsilon, clamped to max
        result = lambda_to_epsilon(0.5, max_epsilon=0.675)
        assert result == 0.675

    def test_zero_lambda_returns_fallback(self):
        result = lambda_to_epsilon(0.0, fallback=0.45)
        assert result == 0.45

    def test_negative_lambda_returns_fallback(self):
        result = lambda_to_epsilon(-1.0, fallback=0.45)
        assert result == 0.45


class TestCreateHdbscanClusterer:
    def test_default_has_prediction_data(self):
        clusterer = create_hdbscan_clusterer()
        # prediction_data should be enabled
        assert clusterer.prediction_data is True

    def test_default_has_gen_min_span_tree(self):
        clusterer = create_hdbscan_clusterer()
        assert clusterer.gen_min_span_tree is True

    def test_precomputed_mode(self):
        clusterer = create_hdbscan_clusterer(precomputed=True)
        assert clusterer.metric == "precomputed"

    def test_eom_selection(self):
        clusterer = create_hdbscan_clusterer()
        assert clusterer.cluster_selection_method == "eom"


class TestSerializeCondensedTree:
    def test_roundtrip(self):
        """Fit a small clusterer and verify condensed tree serialization."""
        rng = np.random.RandomState(42)
        # 3 clusters of 10 points each
        cluster1 = rng.randn(10, 512) * 0.1 + np.array([1.0] + [0.0] * 511)
        cluster2 = rng.randn(10, 512) * 0.1 + np.array([0.0, 1.0] + [0.0] * 510)
        cluster3 = rng.randn(10, 512) * 0.1 + np.array([0.0, 0.0, 1.0] + [0.0] * 509)
        data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)

        clusterer = create_hdbscan_clusterer(min_cluster_size=3, min_samples=2)
        clusterer.fit(data)

        tree_json = serialize_condensed_tree(clusterer)
        assert isinstance(tree_json, dict)
        # Condensed tree has these columns
        assert "parent" in tree_json
        assert "child" in tree_json
        assert "lambda_val" in tree_json
        assert "child_size" in tree_json
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hdbscan_config.py -v`
Expected: FAIL — `lambda_to_epsilon`, `extract_cluster_lambda_births`, `serialize_condensed_tree` don't exist yet.

**Step 3: Implement changes to `hdbscan_config.py`**

Replace `calculate_cluster_epsilon` with new functions. Update `create_hdbscan_clusterer` to enable `prediction_data` and `gen_min_span_tree`.

The updated `hdbscan_config.py` should:

1. **`create_hdbscan_clusterer()`**: Add `prediction_data=True` and `gen_min_span_tree=True` to both precomputed and standard paths
2. **`lambda_to_epsilon(lambda_birth, min_epsilon=0.1, max_epsilon=None, fallback=0.45)`**: Convert lambda_birth to epsilon via `1/lambda_birth`, clamped to bounds. Return fallback for zero/negative lambda.
3. **`extract_cluster_lambda_births(clusterer)`**: Extract per-cluster lambda_birth values from the condensed tree. The condensed tree has rows where `child_size > 1` representing cluster nodes. The lambda_birth for a cluster is the lambda at which it first appears (minimum lambda_val for that cluster node as parent).
4. **`serialize_condensed_tree(clusterer)`**: Convert `clusterer.condensed_tree_.to_pandas().to_dict()` to a JSON-serializable dict.
5. Remove `calculate_cluster_epsilon()` (no longer used after all callers are updated in later tasks).

Keep `DEFAULT_*` constants. Add `DEFAULT_MIN_EPSILON = 0.1`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hdbscan_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/utils/hdbscan_config.py tests/test_hdbscan_config.py
git commit -m "feat(clustering): add lambda-to-epsilon conversion and condensed tree serialization"
```

---

### Task 3: Repository Methods for `hdbscan_run`

**Files:**
- Modify: `src/photodb/database/repository.py`
- Create: `tests/test_hdbscan_repository.py`

**Step 1: Write failing tests**

Tests should mock the database connection pool and verify SQL generation. Follow existing test patterns using `MagicMock`.

Test the following new repository methods:

- `create_hdbscan_run(collection_id, embedding_count, cluster_count, noise_count, min_cluster_size, min_samples, condensed_tree, label_to_cluster_id, clusterer_state) -> int`
  - Deactivates previous active run for collection
  - Inserts new run with `is_active=TRUE`
  - Returns new run ID

- `get_active_hdbscan_run(collection_id) -> Optional[Dict]`
  - Returns the active run for a collection or None
  - Dict has: id, embedding_count, label_to_cluster_id, clusterer_state, created_at

- `update_cluster_hierarchy(cluster_id, lambda_birth, persistence, hdbscan_run_id) -> None`
  - Updates lambda_birth, persistence, and hdbscan_run_id on a cluster

- `update_detection_hierarchy(detection_id, lambda_val, outlier_score) -> None`
  - Updates lambda_val and outlier_score on a person_detection

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hdbscan_repository.py -v`
Expected: FAIL — methods don't exist.

**Step 3: Implement repository methods**

Add the four methods to `PhotoRepository`. Follow existing patterns:
- Use `self.pool.transaction()` for writes
- Use `self.pool.get_connection()` for reads
- Use `dict_row` cursor factory for reads
- Use `_resolve_collection_id()` for collection_id parameter

For `create_hdbscan_run`, the deactivation + insert should be in a single transaction:
```python
with self.pool.transaction() as conn:
    with conn.cursor() as cursor:
        # Deactivate previous active run
        cursor.execute(
            "UPDATE hdbscan_run SET is_active = FALSE WHERE collection_id = %s AND is_active = TRUE",
            (collection_id,),
        )
        # Insert new run
        cursor.execute(
            """INSERT INTO hdbscan_run
               (collection_id, embedding_count, cluster_count, noise_count,
                min_cluster_size, min_samples, condensed_tree, label_to_cluster_id,
                clusterer_state, is_active)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
               RETURNING id""",
            (collection_id, embedding_count, cluster_count, noise_count,
             min_cluster_size, min_samples,
             Json(condensed_tree), Json(label_to_cluster_id),
             clusterer_state),
        )
        return cursor.fetchone()[0]
```

Note: Import `psycopg.types.json.Json` for JSONB parameters.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hdbscan_repository.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/database/repository.py tests/test_hdbscan_repository.py
git commit -m "feat(db): add repository methods for hdbscan_run CRUD and hierarchy columns"
```

---

### Task 4: Update Bootstrap — Return Full Clusterer

**Files:**
- Modify: `src/photodb/stages/clustering.py` (methods: `_run_hdbscan_cpu`, `_run_hdbscan_metal`, `_run_hdbscan_bootstrap`)

**Step 1: Write failing test**

Create `tests/test_clustering_stage.py`:

```python
"""Tests for ClusteringStage HDBSCAN bootstrap."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from photodb.stages.clustering import ClusteringStage


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.pool = MagicMock()
    repo.collection_id = 1
    return repo


@pytest.fixture
def stage(mock_repository):
    config = {
        "HDBSCAN_MIN_CLUSTER_SIZE": "3",
        "HDBSCAN_MIN_SAMPLES": "2",
        "CLUSTERING_THRESHOLD": "0.45",
    }
    s = ClusteringStage(mock_repository, config)
    s.collection_id = 1
    return s


class TestRunHdbscanCpu:
    def test_returns_clusterer_object(self, stage):
        """_run_hdbscan_cpu should return the full clusterer, not just labels."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.randn(10, 512) * 0.1 + np.array([1.0] + [0.0] * 511),
            rng.randn(10, 512) * 0.1 + np.array([0.0, 1.0] + [0.0] * 510),
        ]).astype(np.float32)

        clusterer = stage._run_hdbscan_cpu(data)

        # Should return the clusterer object, not a tuple
        import hdbscan
        assert isinstance(clusterer, hdbscan.HDBSCAN)
        assert hasattr(clusterer, 'labels_')
        assert hasattr(clusterer, 'probabilities_')
        assert hasattr(clusterer, 'condensed_tree_')
        assert hasattr(clusterer, 'outlier_scores_')
        assert hasattr(clusterer, 'cluster_persistence_')

    def test_prediction_data_available(self, stage):
        """Clusterer should have prediction data for approximate_predict."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.randn(10, 512) * 0.1 + np.array([1.0] + [0.0] * 511),
            rng.randn(10, 512) * 0.1 + np.array([0.0, 1.0] + [0.0] * 510),
        ]).astype(np.float32)

        clusterer = stage._run_hdbscan_cpu(data)

        # Should be able to call approximate_predict
        import hdbscan
        new_point = rng.randn(1, 512).astype(np.float32) * 0.1 + np.array([1.0] + [0.0] * 511)
        labels, strengths = hdbscan.approximate_predict(clusterer, new_point)
        assert len(labels) == 1
        assert len(strengths) == 1


class TestRunHdbscanBootstrap:
    def test_returns_clusterer_and_results(self, stage, mock_repository):
        """_run_hdbscan_bootstrap should return both results dict and clusterer."""
        rng = np.random.RandomState(42)
        embeddings_data = []
        for i in range(20):
            cluster_idx = i // 10
            offset = np.zeros(512)
            offset[cluster_idx] = 1.0
            emb = (rng.randn(512) * 0.1 + offset).astype(np.float32)
            embeddings_data.append({
                "detection_id": i + 1,
                "cluster_id": None,
                "cluster_status": None,
                "embedding": emb,
            })

        mock_repository.get_all_embeddings_for_collection.return_value = embeddings_data

        results, clusterer = stage._run_hdbscan_bootstrap()

        assert isinstance(results, dict)
        assert len(results) == 20
        assert clusterer is not None
        assert hasattr(clusterer, 'condensed_tree_')
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_clustering_stage.py -v`
Expected: FAIL — `_run_hdbscan_cpu` returns tuple, `_run_hdbscan_bootstrap` returns dict only.

**Step 3: Update the three methods**

**`_run_hdbscan_cpu`** (line 765-772): Return `clusterer` instead of `(clusterer.labels_, clusterer.probabilities_)`.

**`_run_hdbscan_metal`** (line 774-833): After getting labels from the sparse graph clusterer, fit a second CPU clusterer with `prediction_data=True` on the raw embeddings. Return the CPU clusterer. The Metal path is only for speed on the initial k-NN; the serialized clusterer needs to support `approximate_predict`.

Implementation approach for Metal:
1. Run existing Metal k-NN + sparse HDBSCAN to get labels quickly
2. Fit a CPU clusterer with `prediction_data=True` on raw embeddings
3. Return the CPU clusterer (which has consistent labels due to same data)

Note: If you find that `prediction_data=True` works with precomputed matrices in testing, use that instead. Test this empirically.

**`_run_hdbscan_bootstrap`** (line 694-763):
- Change return type to `Tuple[Dict[int, Dict], Any]` (results dict + clusterer)
- After getting clusterer from `_run_hdbscan_cpu`/`_run_hdbscan_metal`, extract labels/probabilities/outlier_scores from the clusterer object
- Add `outlier_score` to the per-detection results dict
- Return `(results, clusterer)`

Update callers: `run_bootstrap()` (line 678) passes the clusterer to `_assign_bootstrap_clusters`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_clustering_stage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/stages/clustering.py tests/test_clustering_stage.py
git commit -m "feat(clustering): return full HDBSCAN clusterer from bootstrap methods"
```

---

### Task 5: Update Bootstrap — Extract & Persist Hierarchy

**Files:**
- Modify: `src/photodb/stages/clustering.py` (methods: `run_bootstrap`, `_assign_bootstrap_clusters`)

**Step 1: Write failing test**

Add to `tests/test_clustering_stage.py`:

```python
class TestAssignBootstrapClusters:
    def test_stores_hdbscan_run(self, stage, mock_repository):
        """Bootstrap should create an hdbscan_run record."""
        # ... setup mock embeddings, mock clusterer, call run_bootstrap
        # Assert: mock_repository.create_hdbscan_run was called
        # Assert: call args include condensed_tree (dict), label_to_cluster_id (dict),
        #         clusterer_state (bytes)

    def test_sets_lambda_birth_on_clusters(self, stage, mock_repository):
        """Bootstrap should set lambda_birth from condensed tree, not percentile."""
        # Assert: mock_repository.update_cluster_hierarchy was called for each cluster
        # Assert: lambda_birth is a positive float

    def test_sets_outlier_score_on_detections(self, stage, mock_repository):
        """Bootstrap should set outlier_score from GLOSH."""
        # Assert: mock_repository.update_detection_hierarchy was called
        # Assert: outlier_score is between 0 and 1

    def test_epsilon_derived_from_lambda(self, stage, mock_repository):
        """Cluster epsilon should be 1/lambda_birth, not percentile."""
        # Assert: create_cluster_with_epsilon called with epsilon close to 1/lambda_birth
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_clustering_stage.py::TestAssignBootstrapClusters -v`
Expected: FAIL

**Step 3: Implement hierarchy extraction**

Update `run_bootstrap()`:
1. Receive `(results, clusterer)` from `_run_hdbscan_bootstrap()`
2. Pass `clusterer` to `_assign_bootstrap_clusters(results, clusterer)`

Update `_assign_bootstrap_clusters(self, bootstrap_results, clusterer)`:

After creating all clusters, add these steps:

1. **Extract lambda_birth per cluster** using `extract_cluster_lambda_births(clusterer)` from hdbscan_config
2. **Set epsilon = lambda_to_epsilon(lambda_birth)** instead of `_calculate_cluster_epsilon()`
3. **Extract persistence** from `clusterer.cluster_persistence_` (array indexed by cluster label)
4. **Extract outlier_scores** from `clusterer.outlier_scores_` (array indexed by point position)
5. **Call `repository.update_cluster_hierarchy()`** for each cluster with lambda_birth, persistence, hdbscan_run_id
6. **Call `repository.update_detection_hierarchy()`** for each detection with lambda_val, outlier_score
7. **Serialize and persist**:
   - `condensed_tree = serialize_condensed_tree(clusterer)`
   - `clusterer_state = pickle.dumps(clusterer)`
   - `label_to_cluster_id = {str(label): cluster_id for label, cluster_id in mapping.items()}`
   - Call `repository.create_hdbscan_run(...)`

8. **Remove the call to `_calculate_cluster_epsilon()`** — epsilon now comes from lambda_birth

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_clustering_stage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/stages/clustering.py tests/test_clustering_stage.py
git commit -m "feat(clustering): extract and persist full HDBSCAN hierarchy during bootstrap"
```

---

### Task 6: Two-Tier Incremental Assignment

**Files:**
- Modify: `src/photodb/stages/clustering.py` (methods: `__init__`, `_cluster_single_detection`, new `_try_approximate_predict`)

**Step 1: Write failing test**

Add to `tests/test_clustering_stage.py`:

```python
class TestIncrementalAssignment:
    def test_uses_approximate_predict_when_available(self, stage, mock_repository):
        """Should try approximate_predict before epsilon-ball."""
        # Setup: mock get_active_hdbscan_run to return a run with clusterer_state
        # Setup: mock a pickled clusterer that returns a valid label
        # Call: stage._cluster_single_detection(detection)
        # Assert: approximate_predict was used (detection assigned to predicted cluster)

    def test_falls_back_to_epsilon_ball_on_noise(self, stage, mock_repository):
        """Should fall back to epsilon-ball when approximate_predict returns -1."""
        # Setup: mock approximate_predict returning label=-1
        # Setup: mock get_clusters_with_epsilon returning clusters within epsilon
        # Call: stage._cluster_single_detection(detection)
        # Assert: epsilon-ball path was used

    def test_falls_back_when_no_hdbscan_run(self, stage, mock_repository):
        """Should use epsilon-ball when no active hdbscan_run exists."""
        mock_repository.get_active_hdbscan_run.return_value = None
        # Call: stage._cluster_single_detection(detection)
        # Assert: epsilon-ball path used (get_clusters_with_epsilon called)

    def test_respects_cannot_link_in_approximate_predict(self, stage, mock_repository):
        """Cannot-link constraints should filter approximate_predict results."""
        # Setup: approximate_predict assigns to cluster X
        # Setup: cannot-link constraint forbids cluster X
        # Assert: detection not assigned to cluster X
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_clustering_stage.py::TestIncrementalAssignment -v`
Expected: FAIL

**Step 3: Implement two-tier assignment**

Add to `__init__`:
```python
self._cached_clusterer = None
self._cached_label_map = None
self._cached_hdbscan_run_id = None
```

Add new method `_try_approximate_predict(self, embedding) -> Optional[Tuple[int, float]]`:
1. Load active hdbscan_run if not cached (or if run_id changed)
2. Deserialize clusterer from `clusterer_state` bytes via `pickle.loads()`
3. Cache clusterer and label_to_cluster_id map
4. Check staleness: compare `hdbscan_run.embedding_count` against current count. If stale (>25% growth), log warning but still try
5. Call `hdbscan.approximate_predict(self._cached_clusterer, embedding.reshape(1, -1))`
6. If label >= 0: map to database cluster_id via label_map, return `(cluster_id, strength)`
7. If label == -1: return None

Update `_cluster_single_detection`:
1. **First**, try `_try_approximate_predict(embedding)`
2. If it returns a cluster_id:
   - Filter by cannot-link constraints
   - Apply verified cluster stricter threshold
   - If passes: `_assign_to_cluster(detection_id, cluster_id, confidence, embedding)`
   - If filtered out: fall through to epsilon-ball
3. If it returns None (or no active run): run existing epsilon-ball logic unchanged

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_clustering_stage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/stages/clustering.py tests/test_clustering_stage.py
git commit -m "feat(clustering): two-tier incremental assignment with approximate_predict primary"
```

---

### Task 7: Update Maintenance Pool Clustering

**Files:**
- Modify: `src/photodb/utils/maintenance.py` (method: `cluster_unassigned_pool`, ~line 434-602)

**Step 1: Write failing test**

Create `tests/test_maintenance_clustering.py`:

```python
"""Tests for maintenance pool clustering with hierarchy extraction."""

from unittest.mock import MagicMock
from photodb.utils.maintenance import MaintenanceUtilities


class TestPoolClusteringHierarchy:
    def test_sets_lambda_and_outlier_on_new_clusters(self):
        """Pool clustering should extract lambda/outlier values."""
        # Setup mock repo with unassigned detections that have embeddings
        # Call cluster_unassigned_pool
        # Assert: update_cluster_hierarchy called for new clusters
        # Assert: update_detection_hierarchy called for assigned detections

    def test_uses_lambda_for_epsilon(self):
        """Pool clustering should use lambda-derived epsilon, not percentile."""
        # Assert: create_cluster_with_epsilon called with epsilon from lambda_to_epsilon
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_maintenance_clustering.py -v`
Expected: FAIL

**Step 3: Implement changes**

In `cluster_unassigned_pool` (line 498-601):

1. Replace `clusterer = create_hdbscan_clusterer(...)` + `clusterer.fit(embeddings)` with same pattern, but keep full clusterer object
2. Replace `calculate_cluster_epsilon(...)` (line 550-552) with `lambda_to_epsilon(lambda_birth)` using `extract_cluster_lambda_births(clusterer)`
3. After assigning detections to clusters, call `repo.update_cluster_hierarchy()` and `repo.update_detection_hierarchy()` for lambda_val and outlier_score
4. Do NOT create hdbscan_run or serialize clusterer (pool clustering is smaller scope)

Remove the now-unused import of `calculate_cluster_epsilon` from this file.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_maintenance_clustering.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/utils/maintenance.py tests/test_maintenance_clustering.py
git commit -m "feat(maintenance): extract lambda/outlier values in pool clustering"
```

---

### Task 8: Add Staleness Check Maintenance Command

**Files:**
- Modify: `src/photodb/utils/maintenance.py`
- Modify: `src/photodb/cli_maintenance.py`

**Step 1: Implement staleness check**

Add method to `MaintenanceUtilities`:

```python
def check_hdbscan_staleness(self, threshold: float = 1.25) -> Dict[str, Any]:
    """Check if the active HDBSCAN run is stale.

    Returns dict with:
        is_stale: bool
        active_run_id: Optional[int]
        bootstrap_embedding_count: int
        current_embedding_count: int
        growth_ratio: float
        recommendation: str
    """
```

Logic:
1. Get active hdbscan_run via `repo.get_active_hdbscan_run()`
2. If none: return `is_stale=True, recommendation="No HDBSCAN run found. Run bootstrap."`
3. Get current embedding count via `repo.get_all_embeddings_for_collection()` (just count)
4. Calculate `growth_ratio = current / run.embedding_count`
5. Return staleness assessment

Add CLI command in `cli_maintenance.py`:

```python
@cli.command()
def check_staleness():
    """Check if HDBSCAN bootstrap needs re-running."""
```

**Step 2: Run lint/type checks**

Run: `uv run ruff check src/photodb/utils/maintenance.py src/photodb/cli_maintenance.py`
Expected: No errors

**Step 3: Commit**

```bash
git add src/photodb/utils/maintenance.py src/photodb/cli_maintenance.py
git commit -m "feat(maintenance): add HDBSCAN staleness check command"
```

---

### Task 9: Update Migration Script

**Files:**
- Modify: `scripts/migrate_to_hdbscan.py`

**Step 1: Update the script**

The script should now show additional information in its summary:
- Number of clusters with lambda_birth set
- Active hdbscan_run ID and embedding count
- Whether condensed tree and clusterer state were persisted

The actual bootstrap logic is already updated (Tasks 4-5), so this is just the reporting/summary changes.

Add to the summary section after successful bootstrap:

```python
# Check hdbscan_run was created
active_run = repository.get_active_hdbscan_run()
if active_run:
    logger.info(f"  Active HDBSCAN run: #{active_run['id']}")
    logger.info(f"  Embeddings in run: {active_run['embedding_count']}")
    logger.info(f"  Clusterer state: {'persisted' if active_run.get('clusterer_state') else 'not persisted'}")
```

**Step 2: Test manually**

Run: `uv run python scripts/migrate_to_hdbscan.py --dry-run --collection-id 1`
Expected: Dry run output shows current state without errors.

**Step 3: Commit**

```bash
git add scripts/migrate_to_hdbscan.py
git commit -m "feat(scripts): update migration script for full hierarchy reporting"
```

---

### Task 10: Clean Up Old Epsilon Calculation

**Files:**
- Modify: `src/photodb/utils/hdbscan_config.py`
- Modify: `src/photodb/stages/clustering.py`

**Step 1: Remove dead code**

1. Delete `calculate_cluster_epsilon()` from `hdbscan_config.py` (lines 56-104)
2. Delete `_calculate_cluster_epsilon()` from `clustering.py` (lines 835-858)
3. Remove any remaining imports of `calculate_cluster_epsilon` across the codebase
4. Remove `DEFAULT_EPSILON_PERCENTILE` constant if no longer used

**Step 2: Verify no remaining references**

Run: `uv run ruff check && uv run pytest -x`
Expected: No errors, all tests pass.

**Step 3: Commit**

```bash
git add src/photodb/utils/hdbscan_config.py src/photodb/stages/clustering.py
git commit -m "refactor(clustering): remove percentile-based epsilon calculation"
```

---

### Task 11: Update Documentation

**Files:**
- Modify: `docs/DESIGN.md`
- Modify: `CLAUDE.md`

**Step 1: Update DESIGN.md**

Update the Stage 5: Clustering section to reflect:
- Two-tier incremental assignment (approximate_predict + epsilon-ball fallback)
- Lambda-derived epsilon instead of percentile
- Condensed tree and clusterer persistence
- `hdbscan_run` table in schema docs
- New columns on cluster and person_detection
- FISHDBC future work note

**Step 2: Update CLAUDE.md**

Update the Clustering Stage Configuration section:
- Remove `EPSILON_PERCENTILE` (no longer used)
- Document new behavior: epsilon derived from lambda_birth
- Add `hdbscan_run` persistence info
- Add staleness check command

**Step 3: Commit**

```bash
git add docs/DESIGN.md CLAUDE.md
git commit -m "docs: update clustering documentation for full HDBSCAN hierarchy"
```

---

### Task 12: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass.

**Step 2: Run linting and type checks**

Run: `uv run ruff check && uv run ruff format --check`
Expected: No errors.

**Step 3: Final commit if any formatting fixes needed**

```bash
git add -A
git commit -m "chore: formatting fixes"
```
