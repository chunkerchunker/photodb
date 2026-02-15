"""Tests for ClusteringStage HDBSCAN bootstrap."""

import pickle

import numpy as np
import pytest
from unittest.mock import MagicMock
from photodb.stages.clustering import ClusteringStage
from photodb.utils.hdbscan_config import lambda_to_epsilon


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


def _make_bootstrap_data(n_per_cluster=10, n_clusters=2, seed=42):
    """Create embeddings and a fitted HDBSCAN clusterer for bootstrap tests.

    Returns:
        Tuple of (embeddings_data, bootstrap_results, clusterer) where:
        - embeddings_data: list of dicts as returned by get_all_embeddings_for_collection
        - bootstrap_results: dict mapping detection_id to {label, probability, is_core,
            outlier_score}
        - clusterer: fitted HDBSCAN clusterer
    """
    from photodb.utils.hdbscan_config import create_hdbscan_clusterer

    rng = np.random.RandomState(seed)
    total = n_per_cluster * n_clusters
    embeddings_data = []
    raw_embeddings = []

    for i in range(total):
        cluster_idx = i // n_per_cluster
        offset = np.zeros(512)
        offset[cluster_idx] = 1.0
        emb = (rng.randn(512) * 0.1 + offset).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        raw_embeddings.append(emb)
        embeddings_data.append(
            {
                "detection_id": i + 1,
                "cluster_id": None,
                "cluster_status": None,
                "embedding": emb,
            }
        )

    embeddings_arr = np.array(raw_embeddings, dtype=np.float32)

    # Fit HDBSCAN
    clusterer = create_hdbscan_clusterer(min_cluster_size=3, min_samples=2)
    clusterer.fit(embeddings_arr)

    # Build bootstrap_results
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    outlier_scores = clusterer.outlier_scores_
    bootstrap_results = {}
    for i, det_data in enumerate(embeddings_data):
        detection_id = det_data["detection_id"]
        is_core = bool(probabilities[i] >= 0.8)
        bootstrap_results[detection_id] = {
            "label": int(labels[i]),
            "probability": float(probabilities[i]),
            "is_core": is_core,
            "outlier_score": float(outlier_scores[i]),
        }

    return embeddings_data, bootstrap_results, clusterer


class TestAssignBootstrapClusters:
    """Tests for _assign_bootstrap_clusters hierarchy persistence."""

    def test_stores_hdbscan_run(self, stage, mock_repository):
        """Bootstrap should create an hdbscan_run record."""
        embeddings_data, bootstrap_results, clusterer = _make_bootstrap_data()
        mock_repository.get_all_embeddings_for_collection.return_value = embeddings_data

        # Make create_cluster_with_epsilon return incrementing IDs
        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        mock_repository.create_cluster_with_epsilon.side_effect = _make_cluster_id
        mock_repository.create_hdbscan_run.return_value = 100

        stage._assign_bootstrap_clusters(bootstrap_results, clusterer)

        # Verify create_hdbscan_run was called exactly once
        mock_repository.create_hdbscan_run.assert_called_once()
        call_kwargs = mock_repository.create_hdbscan_run.call_args
        args = call_kwargs.kwargs if call_kwargs.kwargs else {}
        # If called with positional args, get them from the call
        if not args:
            args = call_kwargs[1] if len(call_kwargs) > 1 and call_kwargs[1] else {}
        if not args:
            # Positional call - map to parameter names
            positional = call_kwargs[0] if call_kwargs[0] else ()
            param_names = [
                "embedding_count",
                "cluster_count",
                "noise_count",
                "min_cluster_size",
                "min_samples",
                "condensed_tree",
                "label_to_cluster_id",
                "clusterer_state",
            ]
            args = dict(zip(param_names, positional))

        # condensed_tree should be a dict with the expected columns
        assert isinstance(args["condensed_tree"], dict)
        assert "parent" in args["condensed_tree"]
        assert "child" in args["condensed_tree"]
        assert "lambda_val" in args["condensed_tree"]

        # label_to_cluster_id should be a dict with string keys
        assert isinstance(args["label_to_cluster_id"], dict)
        for key in args["label_to_cluster_id"]:
            assert isinstance(key, str), f"label_to_cluster_id key should be str, got {type(key)}"

        # clusterer_state should be bytes (pickled)
        assert isinstance(args["clusterer_state"], bytes)
        # Verify it can be unpickled
        unpickled = pickle.loads(args["clusterer_state"])
        assert hasattr(unpickled, "labels_")

    def test_sets_lambda_birth_on_clusters(self, stage, mock_repository):
        """Bootstrap should set lambda_birth from condensed tree."""
        embeddings_data, bootstrap_results, clusterer = _make_bootstrap_data()
        mock_repository.get_all_embeddings_for_collection.return_value = embeddings_data

        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        mock_repository.create_cluster_with_epsilon.side_effect = _make_cluster_id
        mock_repository.create_hdbscan_run.return_value = 100

        stage._assign_bootstrap_clusters(bootstrap_results, clusterer)

        # update_cluster_hierarchy should have been called for each cluster
        n_clusters = len(set(clusterer.labels_[clusterer.labels_ >= 0]))
        assert mock_repository.update_cluster_hierarchy.call_count == n_clusters

        # Each call should have a positive lambda_birth
        for c in mock_repository.update_cluster_hierarchy.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            if not kwargs:
                # Positional: cluster_id, lambda_birth, persistence, hdbscan_run_id
                kwargs = {
                    "cluster_id": c[0][0] if c[0] else None,
                    "lambda_birth": c[0][1] if len(c[0]) > 1 else None,
                }
            lb = kwargs.get("lambda_birth")
            assert lb is not None, "lambda_birth should not be None"
            assert isinstance(lb, float), f"lambda_birth should be float, got {type(lb)}"
            assert lb > 0, f"lambda_birth should be positive, got {lb}"

    def test_sets_outlier_score_on_detections(self, stage, mock_repository):
        """Bootstrap should set outlier_score from GLOSH."""
        embeddings_data, bootstrap_results, clusterer = _make_bootstrap_data()
        mock_repository.get_all_embeddings_for_collection.return_value = embeddings_data

        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        mock_repository.create_cluster_with_epsilon.side_effect = _make_cluster_id
        mock_repository.create_hdbscan_run.return_value = 100

        stage._assign_bootstrap_clusters(bootstrap_results, clusterer)

        # update_detection_hierarchy should have been called for every detection
        assert mock_repository.update_detection_hierarchy.call_count == len(bootstrap_results)

        # Each call should have an outlier_score between 0 and 1
        for c in mock_repository.update_detection_hierarchy.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            if not kwargs:
                kwargs = {
                    "detection_id": c[0][0] if c[0] else None,
                    "lambda_val": c[0][1] if len(c[0]) > 1 else None,
                    "outlier_score": c[0][2] if len(c[0]) > 2 else None,
                }
            os_val = kwargs.get("outlier_score")
            assert os_val is not None, "outlier_score should not be None"
            assert isinstance(os_val, float), f"outlier_score should be float, got {type(os_val)}"
            assert 0.0 <= os_val <= 1.0, f"outlier_score should be in [0,1], got {os_val}"

            # lambda_val should come from condensed tree (positive float)
            lv = kwargs.get("lambda_val")
            if lv is not None:
                assert isinstance(lv, float), f"lambda_val should be float, got {type(lv)}"
                assert lv > 0, f"lambda_val should be positive, got {lv}"

    def test_epsilon_derived_from_lambda(self, stage, mock_repository):
        """Cluster epsilon should be 1/lambda_birth, not percentile."""
        embeddings_data, bootstrap_results, clusterer = _make_bootstrap_data()
        mock_repository.get_all_embeddings_for_collection.return_value = embeddings_data

        cluster_id_counter = [0]

        def _make_cluster_id(**kwargs):
            cluster_id_counter[0] += 1
            return cluster_id_counter[0]

        mock_repository.create_cluster_with_epsilon.side_effect = _make_cluster_id
        mock_repository.create_hdbscan_run.return_value = 100

        stage._assign_bootstrap_clusters(bootstrap_results, clusterer)

        # Collect the epsilon values passed to create_cluster_with_epsilon
        from photodb.utils.hdbscan_config import extract_cluster_lambda_births

        lambda_births = extract_cluster_lambda_births(clusterer)

        # We need to map HDBSCAN labels to call order. The calls happen in
        # labels_to_detections iteration order (label >= 0).
        call_idx = 0
        labels_in_order = sorted(label for label in set(clusterer.labels_) if label >= 0)
        for c in mock_repository.create_cluster_with_epsilon.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            if not kwargs:
                continue
            epsilon_used = kwargs.get("epsilon")
            if epsilon_used is None:
                continue

            # For labels with lambda_birth, epsilon should match lambda_to_epsilon
            if call_idx < len(labels_in_order):
                label = labels_in_order[call_idx]
                lb = lambda_births.get(label)
                if lb is not None:
                    expected_epsilon = lambda_to_epsilon(lb)
                    assert abs(epsilon_used - expected_epsilon) < 1e-6, (
                        f"epsilon {epsilon_used} should be close to "
                        f"lambda_to_epsilon({lb}) = {expected_epsilon}"
                    )
            call_idx += 1
