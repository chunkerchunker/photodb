"""Tests for HDBSCAN repository methods."""

import pytest
from unittest.mock import MagicMock
from photodb.database.repository import PhotoRepository


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    return pool


@pytest.fixture
def repo(mock_pool):
    repo = PhotoRepository(mock_pool, collection_id=1)
    return repo


class TestCreateHdbscanRun:
    def test_deactivates_previous_run(self, repo, mock_pool):
        """Should deactivate previous active run before inserting new one."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        result = repo.create_hdbscan_run(
            embedding_count=100,
            cluster_count=5,
            noise_count=10,
            min_cluster_size=3,
            min_samples=2,
            condensed_tree={"parent": [], "child": []},
            label_to_cluster_id={"0": 1, "1": 2},
            clusterer_state=b"pickled_data",
        )

        # Should have made 2 execute calls (deactivate + insert)
        assert mock_cursor.execute.call_count == 2
        # First call should be the deactivation UPDATE
        first_sql = mock_cursor.execute.call_args_list[0][0][0]
        assert "is_active = FALSE" in first_sql
        # Return value should be the new ID
        assert result == 42

    def test_returns_new_run_id(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (99,)
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        result = repo.create_hdbscan_run(
            embedding_count=50,
            cluster_count=3,
            noise_count=5,
            min_cluster_size=3,
            min_samples=2,
            condensed_tree={},
            label_to_cluster_id={},
        )
        assert result == 99

    def test_insert_sql_contains_expected_columns(self, repo, mock_pool):
        """Should insert with all expected columns."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        repo.create_hdbscan_run(
            embedding_count=100,
            cluster_count=5,
            noise_count=10,
            min_cluster_size=3,
            min_samples=2,
            condensed_tree={"parent": [1]},
            label_to_cluster_id={"0": 1},
        )

        insert_sql = mock_cursor.execute.call_args_list[1][0][0]
        assert "embedding_count" in insert_sql
        assert "cluster_count" in insert_sql
        assert "noise_count" in insert_sql
        assert "condensed_tree" in insert_sql
        assert "label_to_cluster_id" in insert_sql
        assert "clusterer_state" in insert_sql
        assert "RETURNING id" in insert_sql

    def test_uses_resolved_collection_id(self, repo, mock_pool):
        """Should use the provided collection_id or default."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        repo.create_hdbscan_run(
            embedding_count=10,
            cluster_count=1,
            noise_count=0,
            min_cluster_size=3,
            min_samples=2,
            condensed_tree={},
            label_to_cluster_id={},
            collection_id=5,
        )

        # The deactivation query should use collection_id=5
        deactivate_params = mock_cursor.execute.call_args_list[0][0][1]
        assert deactivate_params == (5,)


class TestGetActiveHdbscanRun:
    def test_returns_dict_when_exists(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "embedding_count": 100,
            "cluster_count": 5,
            "label_to_cluster_id": {"0": 1},
            "clusterer_state": b"data",
            "created_at": "2026-01-01",
        }
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = lambda s: mock_conn
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        result = repo.get_active_hdbscan_run()
        assert result is not None
        assert result["id"] == 1
        assert result["embedding_count"] == 100
        assert result["cluster_count"] == 5

    def test_returns_none_when_no_active_run(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = lambda s: mock_conn
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        result = repo.get_active_hdbscan_run()
        assert result is None

    def test_queries_with_collection_id(self, repo, mock_pool):
        """Should query with correct collection_id."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.get_connection.return_value.__enter__ = lambda s: mock_conn
        mock_pool.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        repo.get_active_hdbscan_run(collection_id=7)

        sql = mock_cursor.execute.call_args[0][0]
        params = mock_cursor.execute.call_args[0][1]
        assert "is_active = TRUE" in sql
        assert params == (7,)


class TestUpdateClusterHierarchy:
    def test_updates_columns(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        repo.update_cluster_hierarchy(
            cluster_id=5, lambda_birth=2.5, persistence=0.9, hdbscan_run_id=1
        )

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "lambda_birth" in sql
        assert "persistence" in sql
        assert "hdbscan_run_id" in sql
        assert "updated_at = NOW()" in sql

    def test_passes_correct_params(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        repo.update_cluster_hierarchy(
            cluster_id=5, lambda_birth=2.5, persistence=0.9, hdbscan_run_id=1
        )

        params = mock_cursor.execute.call_args[0][1]
        assert params == (2.5, 0.9, 1, 5)

    def test_allows_none_values(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        repo.update_cluster_hierarchy(cluster_id=5)

        params = mock_cursor.execute.call_args[0][1]
        assert params == (None, None, None, 5)


class TestUpdateDetectionHierarchy:
    def test_updates_columns(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        repo.update_detection_hierarchy(detection_id=10, lambda_val=1.5, outlier_score=0.2)

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "lambda_val" in sql
        assert "outlier_score" in sql

    def test_passes_correct_params(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        repo.update_detection_hierarchy(detection_id=10, lambda_val=1.5, outlier_score=0.2)

        params = mock_cursor.execute.call_args[0][1]
        assert params == (1.5, 0.2, 10)

    def test_allows_none_values(self, repo, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.transaction.return_value.__enter__ = lambda s: mock_conn
        mock_pool.transaction.return_value.__exit__ = MagicMock(return_value=False)

        repo.update_detection_hierarchy(detection_id=10)

        params = mock_cursor.execute.call_args[0][1]
        assert params == (None, None, 10)
