"""Tests for ClusteringStage HDBSCAN bootstrap."""

import numpy as np
import pytest
from unittest.mock import MagicMock
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
        data = np.vstack(
            [
                rng.randn(10, 512) * 0.1 + np.array([1.0] + [0.0] * 511),
                rng.randn(10, 512) * 0.1 + np.array([0.0, 1.0] + [0.0] * 510),
            ]
        ).astype(np.float32)

        clusterer = stage._run_hdbscan_cpu(data)

        # Should return the clusterer object, not a tuple
        import hdbscan

        assert isinstance(clusterer, hdbscan.HDBSCAN)
        assert hasattr(clusterer, "labels_")
        assert hasattr(clusterer, "probabilities_")
        assert hasattr(clusterer, "condensed_tree_")
        assert hasattr(clusterer, "outlier_scores_")
        assert hasattr(clusterer, "cluster_persistence_")

    def test_prediction_data_available(self, stage):
        """Clusterer should have prediction data for approximate_predict."""
        rng = np.random.RandomState(42)
        data = np.vstack(
            [
                rng.randn(10, 512) * 0.1 + np.array([1.0] + [0.0] * 511),
                rng.randn(10, 512) * 0.1 + np.array([0.0, 1.0] + [0.0] * 510),
            ]
        ).astype(np.float32)

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
            embeddings_data.append(
                {
                    "detection_id": i + 1,
                    "cluster_id": None,
                    "cluster_status": None,
                    "embedding": emb,
                }
            )

        mock_repository.get_all_embeddings_for_collection.return_value = embeddings_data

        results, clusterer = stage._run_hdbscan_bootstrap()

        assert isinstance(results, dict)
        assert len(results) == 20
        assert clusterer is not None
        assert hasattr(clusterer, "condensed_tree_")

    def test_results_contain_outlier_score(self, stage, mock_repository):
        """Each result entry should contain an outlier_score field."""
        rng = np.random.RandomState(42)
        embeddings_data = []
        for i in range(20):
            cluster_idx = i // 10
            offset = np.zeros(512)
            offset[cluster_idx] = 1.0
            emb = (rng.randn(512) * 0.1 + offset).astype(np.float32)
            embeddings_data.append(
                {
                    "detection_id": i + 1,
                    "cluster_id": None,
                    "cluster_status": None,
                    "embedding": emb,
                }
            )

        mock_repository.get_all_embeddings_for_collection.return_value = embeddings_data

        results, clusterer = stage._run_hdbscan_bootstrap()

        for detection_id, result in results.items():
            assert "outlier_score" in result, f"Detection {detection_id} missing outlier_score"
            assert isinstance(result["outlier_score"], float)

    def test_empty_embeddings_returns_empty_and_none(self, stage, mock_repository):
        """When no embeddings exist, should return empty dict and None clusterer."""
        mock_repository.get_all_embeddings_for_collection.return_value = []

        results, clusterer = stage._run_hdbscan_bootstrap()

        assert results == {}
        assert clusterer is None
