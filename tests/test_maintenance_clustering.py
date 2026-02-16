"""Tests for MaintenanceUtilities.cluster_unassigned_pool hierarchy persistence."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from photodb.utils.maintenance import MaintenanceUtilities
from photodb.utils.hdbscan_config import (
    create_hdbscan_clusterer,
    extract_cluster_lambda_births,
    lambda_to_epsilon,
)


def _make_pool_embeddings(n_per_cluster=10, n_clusters=2, seed=42):
    """Create normalized embeddings that form well-separated clusters.

    Returns:
        Tuple of (detection_ids, embeddings) where:
        - detection_ids: list of detection IDs (ints starting at 1)
        - embeddings: numpy array of shape (n_per_cluster * n_clusters, 512)
    """
    rng = np.random.RandomState(seed)
    total = n_per_cluster * n_clusters
    detection_ids = list(range(1, total + 1))
    raw_embeddings = []

    for i in range(total):
        cluster_idx = i // n_per_cluster
        offset = np.zeros(512)
        offset[cluster_idx] = 1.0
        emb = (rng.randn(512) * 0.1 + offset).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        raw_embeddings.append(emb)

    embeddings = np.array(raw_embeddings, dtype=np.float32)
    return detection_ids, embeddings


def _make_db_rows(detection_ids, embeddings):
    """Create DB-style rows as returned by cursor.fetchall().

    Returns list of (detection_id, embedding) tuples.
    """
    return [(did, emb.tolist()) for did, emb in zip(detection_ids, embeddings)]


@pytest.fixture
def mock_pool():
    """Create a mock ConnectionPool."""
    pool = MagicMock()
    return pool


@pytest.fixture
def maintenance(mock_pool):
    """Create a MaintenanceUtilities instance with mocked pool and repo."""
    m = MaintenanceUtilities(mock_pool, collection_id=1)
    m.repo = MagicMock()
    return m


class TestPoolClusteringHierarchy:
    def test_sets_lambda_and_outlier_on_new_clusters(self, maintenance, mock_pool):
        """Pool clustering should extract lambda/outlier values."""
        detection_ids, embeddings = _make_pool_embeddings()
        rows = _make_db_rows(detection_ids, embeddings)

        # Mock the DB query to return our test data
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        # Make create_cluster_with_epsilon return incrementing IDs
        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        maintenance.repo.create_cluster_with_epsilon.side_effect = _make_cluster_id

        result = maintenance.cluster_unassigned_pool(min_cluster_size=3)

        # Should have created clusters
        assert result > 0

        # update_cluster_hierarchy should have been called for each cluster
        assert maintenance.repo.update_cluster_hierarchy.call_count == result

        # Each cluster should have lambda_birth and persistence
        for c in maintenance.repo.update_cluster_hierarchy.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            if not kwargs:
                # Positional: cluster_id, lambda_birth, persistence, hdbscan_run_id
                positional = c[0] if c[0] else ()
                param_names = ["cluster_id", "lambda_birth", "persistence", "hdbscan_run_id"]
                kwargs = dict(zip(param_names, positional))

            # lambda_birth should be a positive float (or None if not in condensed tree)
            lb = kwargs.get("lambda_birth")
            if lb is not None:
                assert isinstance(lb, float), f"lambda_birth should be float, got {type(lb)}"
                assert lb > 0, f"lambda_birth should be positive, got {lb}"

            # hdbscan_run_id should be None for pool clustering
            assert kwargs.get("hdbscan_run_id") is None

        # update_detection_hierarchy should have been called for assigned detections
        assert maintenance.repo.update_detection_hierarchy.call_count > 0

        # Each detection hierarchy call should have outlier_score
        for c in maintenance.repo.update_detection_hierarchy.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            if not kwargs:
                positional = c[0] if c[0] else ()
                param_names = ["detection_id", "lambda_val", "outlier_score"]
                kwargs = dict(zip(param_names, positional))

            os_val = kwargs.get("outlier_score")
            assert os_val is not None, "outlier_score should not be None"
            assert isinstance(os_val, float), f"outlier_score should be float, got {type(os_val)}"
            assert 0.0 <= os_val <= 1.0, f"outlier_score should be in [0,1], got {os_val}"

            # lambda_val should come from condensed tree (positive float when available)
            lv = kwargs.get("lambda_val")
            if lv is not None:
                assert isinstance(lv, float), f"lambda_val should be float, got {type(lv)}"
                assert lv > 0, f"lambda_val should be positive, got {lv}"

    def test_uses_lambda_for_epsilon(self, maintenance, mock_pool):
        """Pool clustering should use lambda-derived epsilon, not percentile."""
        detection_ids, embeddings = _make_pool_embeddings()
        rows = _make_db_rows(detection_ids, embeddings)

        # Mock the DB query
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        maintenance.repo.create_cluster_with_epsilon.side_effect = _make_cluster_id

        result = maintenance.cluster_unassigned_pool(min_cluster_size=3)
        assert result > 0

        # Reproduce the HDBSCAN clustering to get expected lambda values
        clusterer = create_hdbscan_clusterer(min_cluster_size=3)
        # Normalize embeddings the same way the method does
        normalized = []
        for emb in embeddings:
            arr = np.array(emb)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            normalized.append(arr)
        normalized_arr = np.array(normalized)
        clusterer.fit(normalized_arr)

        lambda_births = extract_cluster_lambda_births(clusterer)

        # Verify that create_cluster_with_epsilon was called with lambda-derived epsilon
        for c in maintenance.repo.create_cluster_with_epsilon.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            if not kwargs:
                continue
            epsilon_used = kwargs.get("epsilon")
            if epsilon_used is None:
                continue

            # The epsilon should match a lambda_to_epsilon value from at least one cluster
            # (we can't easily map which call goes to which label, but we can verify
            # the epsilon is in the set of expected values)
            expected_epsilons = {lambda_to_epsilon(lb) for lb in lambda_births.values()}
            assert any(abs(epsilon_used - expected) < 1e-6 for expected in expected_epsilons), (
                f"epsilon {epsilon_used} should match one of the lambda-derived values: "
                f"{expected_epsilons}"
            )

    def test_no_hdbscan_run_created(self, maintenance, mock_pool):
        """Pool clustering should NOT create an hdbscan_run record."""
        detection_ids, embeddings = _make_pool_embeddings()
        rows = _make_db_rows(detection_ids, embeddings)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        maintenance.repo.create_cluster_with_epsilon.side_effect = _make_cluster_id

        result = maintenance.cluster_unassigned_pool(min_cluster_size=3)
        assert result > 0

        # create_hdbscan_run should NOT have been called (pool clustering is smaller scope)
        maintenance.repo.create_hdbscan_run.assert_not_called()

    def test_persistence_set_on_clusters(self, maintenance, mock_pool):
        """Pool clustering should set persistence from clusterer.cluster_persistence_."""
        detection_ids, embeddings = _make_pool_embeddings()
        rows = _make_db_rows(detection_ids, embeddings)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        maintenance.repo.create_cluster_with_epsilon.side_effect = _make_cluster_id

        result = maintenance.cluster_unassigned_pool(min_cluster_size=3)
        assert result > 0

        # Check that persistence is set (float or None)
        for c in maintenance.repo.update_cluster_hierarchy.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            if not kwargs:
                positional = c[0] if c[0] else ()
                param_names = ["cluster_id", "lambda_birth", "persistence", "hdbscan_run_id"]
                kwargs = dict(zip(param_names, positional))

            persistence = kwargs.get("persistence")
            if persistence is not None:
                assert isinstance(persistence, float)
                assert persistence >= 0

    def test_not_enough_detections_returns_zero(self, maintenance, mock_pool):
        """When there are fewer detections than min_cluster_size, should return 0."""
        # Only 2 rows, below min_cluster_size=3
        rows = [(1, np.random.randn(512).tolist()), (2, np.random.randn(512).tolist())]

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        result = maintenance.cluster_unassigned_pool(min_cluster_size=3)
        assert result == 0
        maintenance.repo.update_cluster_hierarchy.assert_not_called()
        maintenance.repo.update_detection_hierarchy.assert_not_called()

    def test_hierarchy_call_count_matches_assigned_detections(self, maintenance, mock_pool):
        """update_detection_hierarchy should be called once per assigned (non-noise) detection."""
        detection_ids, embeddings = _make_pool_embeddings()
        rows = _make_db_rows(detection_ids, embeddings)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        maintenance.repo.create_cluster_with_epsilon.side_effect = _make_cluster_id

        maintenance.cluster_unassigned_pool(min_cluster_size=3)

        # The number of force_update_detection_cluster calls should equal
        # the number of update_detection_hierarchy calls (one per assigned detection)
        assert (
            maintenance.repo.update_detection_hierarchy.call_count
            == maintenance.repo.force_update_detection_cluster.call_count
        )
