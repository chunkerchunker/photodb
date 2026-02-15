"""Tests for ClusteringStage HDBSCAN bootstrap and incremental assignment."""

import pickle
from datetime import datetime, timezone

import numpy as np
import pytest
from unittest.mock import MagicMock
from photodb.stages.clustering import ClusteringStage
from photodb.utils.hdbscan_config import create_hdbscan_clusterer, lambda_to_epsilon


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


def _make_clusterer_and_label_map(seed=42):
    """Create a fitted HDBSCAN clusterer and a label_to_cluster_id mapping.

    Returns:
        Tuple of (clusterer, label_to_cluster_id, embeddings) where:
        - clusterer: fitted HDBSCAN clusterer with prediction_data
        - label_to_cluster_id: dict mapping string HDBSCAN labels to DB cluster IDs
        - embeddings: numpy array of the training embeddings (normalized)
    """
    rng = np.random.RandomState(seed)
    # Two well-separated clusters in 512-d space
    raw_embeddings = []
    for i in range(20):
        cluster_idx = i // 10
        offset = np.zeros(512)
        offset[cluster_idx] = 1.0
        emb = (rng.randn(512) * 0.1 + offset).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        raw_embeddings.append(emb)

    embeddings = np.array(raw_embeddings, dtype=np.float32)

    clusterer = create_hdbscan_clusterer(min_cluster_size=3, min_samples=2)
    clusterer.fit(embeddings)

    # Build label -> cluster_id mapping with string keys (like DB stores)
    unique_labels = sorted(set(clusterer.labels_[clusterer.labels_ >= 0]))
    # Map HDBSCAN labels to database cluster IDs (starting at 100)
    label_to_cluster_id = {str(label): 100 + label for label in unique_labels}

    return clusterer, label_to_cluster_id, embeddings


def _make_hdbscan_run(clusterer, label_to_cluster_id, embedding_count=20):
    """Create a mock hdbscan_run dict as returned by get_active_hdbscan_run."""
    return {
        "id": 1,
        "embedding_count": embedding_count,
        "cluster_count": len(label_to_cluster_id),
        "label_to_cluster_id": label_to_cluster_id,
        "clusterer_state": pickle.dumps(clusterer),
        "created_at": datetime.now(timezone.utc),
    }


class TestIncrementalAssignment:
    """Tests for two-tier incremental assignment (approximate_predict + epsilon-ball)."""

    def test_uses_approximate_predict_when_available(self, stage, mock_repository):
        """Should try approximate_predict before epsilon-ball."""
        clusterer, label_to_cluster_id, embeddings = _make_clusterer_and_label_map()
        run = _make_hdbscan_run(clusterer, label_to_cluster_id)

        mock_repository.get_active_hdbscan_run.return_value = run
        # Staleness check: return same count as the run
        mock_repository.get_all_embeddings_for_collection.return_value = [{}] * 20

        # Create a new embedding close to cluster 0 (offset at dim 0)
        rng = np.random.RandomState(99)
        new_emb = (rng.randn(512) * 0.05 + np.array([1.0] + [0.0] * 511)).astype(np.float32)
        norm = np.linalg.norm(new_emb)
        new_emb = new_emb / norm

        detection = {"id": 999, "embedding": new_emb}

        # HDBSCAN label 0 -> cluster_id 100; make it non-verified
        mock_cluster = MagicMock()
        mock_cluster.verified = False
        mock_repository.get_cluster_by_id.return_value = mock_cluster

        # No cannot-link constraints
        mock_repository.get_cannot_linked_detections.return_value = []
        mock_repository.get_detections_in_cluster.return_value = []

        # Mock _assign_to_cluster to track calls (it uses raw SQL)
        stage._assign_to_cluster = MagicMock()

        stage._cluster_single_detection(detection)

        # Should have called _assign_to_cluster with the predicted cluster_id (100)
        stage._assign_to_cluster.assert_called_once()
        call_args = stage._assign_to_cluster.call_args
        assert call_args[0][0] == 999  # detection_id
        assert call_args[0][1] == 100  # cluster_id from label 0

        # Should NOT have called get_clusters_with_epsilon (epsilon-ball path)
        mock_repository.get_clusters_with_epsilon.assert_not_called()

    def test_falls_back_to_epsilon_ball_on_noise(self, stage, mock_repository):
        """Should fall back to epsilon-ball when approximate_predict returns -1 (noise)."""
        clusterer, label_to_cluster_id, embeddings = _make_clusterer_and_label_map()
        run = _make_hdbscan_run(clusterer, label_to_cluster_id)

        mock_repository.get_active_hdbscan_run.return_value = run
        mock_repository.get_all_embeddings_for_collection.return_value = [{}] * 20

        # Create an embedding far from both clusters (should be noise)
        rng = np.random.RandomState(77)
        noise_emb = rng.randn(512).astype(np.float32) * 0.5
        norm = np.linalg.norm(noise_emb)
        noise_emb = noise_emb / norm

        detection = {"id": 888, "embedding": noise_emb}

        # Set up epsilon-ball path
        mock_repository.get_clusters_with_epsilon.return_value = []
        mock_repository.get_cannot_linked_detections.return_value = []
        mock_repository.get_detections_in_cluster.return_value = []

        # Mock unassigned pool
        mock_repository.find_similar_unassigned_detections.return_value = []

        stage._cluster_single_detection(detection)

        # Should have fallen through to epsilon-ball path
        mock_repository.get_clusters_with_epsilon.assert_called_once()
        # Should have been added to unassigned pool (no clusters within epsilon)
        mock_repository.update_detection_unassigned.assert_called_once_with(888)

    def test_falls_back_when_no_hdbscan_run(self, stage, mock_repository):
        """Should use epsilon-ball when no active hdbscan_run exists."""
        mock_repository.get_active_hdbscan_run.return_value = None

        rng = np.random.RandomState(55)
        emb = rng.randn(512).astype(np.float32)
        norm = np.linalg.norm(emb)
        emb = emb / norm

        detection = {"id": 777, "embedding": emb}

        # Set up epsilon-ball path
        mock_repository.get_clusters_with_epsilon.return_value = []
        mock_repository.get_cannot_linked_detections.return_value = []
        mock_repository.get_detections_in_cluster.return_value = []
        mock_repository.find_similar_unassigned_detections.return_value = []

        stage._cluster_single_detection(detection)

        # Should have used epsilon-ball path since no hdbscan_run
        mock_repository.get_clusters_with_epsilon.assert_called_once()
        mock_repository.update_detection_unassigned.assert_called_once_with(777)

    def test_respects_cannot_link_in_approximate_predict(self, stage, mock_repository):
        """Cannot-link constraints should filter approximate_predict results."""
        clusterer, label_to_cluster_id, embeddings = _make_clusterer_and_label_map()
        run = _make_hdbscan_run(clusterer, label_to_cluster_id)

        mock_repository.get_active_hdbscan_run.return_value = run
        mock_repository.get_all_embeddings_for_collection.return_value = [{}] * 20

        # Create a new embedding close to cluster 0 (should predict label 0 -> cluster_id 100)
        rng = np.random.RandomState(99)
        new_emb = (rng.randn(512) * 0.05 + np.array([1.0] + [0.0] * 511)).astype(np.float32)
        norm = np.linalg.norm(new_emb)
        new_emb = new_emb / norm

        detection = {"id": 666, "embedding": new_emb}

        # Set up cannot-link: detection 666 cannot link to a detection in cluster 100
        # The _filter_cannot_link_clusters checks:
        # 1. Direct cannot-link detections for this detection
        mock_repository.get_cannot_linked_detections.return_value = [
            {"id": 50, "cluster_id": 100}  # Forbidden cluster
        ]
        # 2. Cluster-level check (detections in cluster check)
        mock_repository.get_detections_in_cluster.return_value = []

        # Set up epsilon-ball fallback path
        mock_repository.get_clusters_with_epsilon.return_value = []
        mock_repository.find_similar_unassigned_detections.return_value = []

        # Mock _assign_to_cluster to verify it is NOT called with cluster 100
        stage._assign_to_cluster = MagicMock()

        stage._cluster_single_detection(detection)

        # Should NOT have assigned to cluster 100 (blocked by cannot-link)
        if stage._assign_to_cluster.called:
            # If assigned, it should NOT be to cluster 100
            assigned_cluster = stage._assign_to_cluster.call_args[0][1]
            assert assigned_cluster != 100, "Should not assign to cannot-linked cluster 100"

        # Should have fallen through to epsilon-ball
        mock_repository.get_clusters_with_epsilon.assert_called_once()

    def test_skips_approximate_predict_in_bootstrap_mode(self, stage, mock_repository):
        """In bootstrap mode, approximate_predict should be skipped entirely."""
        clusterer, label_to_cluster_id, embeddings = _make_clusterer_and_label_map()
        run = _make_hdbscan_run(clusterer, label_to_cluster_id)
        mock_repository.get_active_hdbscan_run.return_value = run

        rng = np.random.RandomState(99)
        new_emb = (rng.randn(512) * 0.05 + np.array([1.0] + [0.0] * 511)).astype(np.float32)
        norm = np.linalg.norm(new_emb)
        new_emb = new_emb / norm

        detection = {"id": 555, "embedding": new_emb}

        # Enable bootstrap mode
        stage.bootstrap_mode = True

        # Set up epsilon-ball path
        mock_repository.get_clusters_with_epsilon.return_value = []
        mock_repository.get_cannot_linked_detections.return_value = []
        mock_repository.get_detections_in_cluster.return_value = []
        mock_repository.find_similar_unassigned_detections.return_value = []

        stage._cluster_single_detection(detection)

        # Should NOT have called get_active_hdbscan_run (approximate_predict skipped)
        mock_repository.get_active_hdbscan_run.assert_not_called()
        # Should have used epsilon-ball path directly
        mock_repository.get_clusters_with_epsilon.assert_called_once()

    def test_verified_cluster_applies_strict_threshold(self, stage, mock_repository):
        """Verified clusters should use stricter distance threshold."""
        clusterer, label_to_cluster_id, embeddings = _make_clusterer_and_label_map()
        run = _make_hdbscan_run(clusterer, label_to_cluster_id)

        mock_repository.get_active_hdbscan_run.return_value = run
        mock_repository.get_all_embeddings_for_collection.return_value = [{}] * 20

        # Create embedding close-ish to cluster 0 but beyond strict threshold
        # We'll set epsilon very small so strict_threshold = epsilon * 0.8 is tiny
        rng = np.random.RandomState(99)
        new_emb = (rng.randn(512) * 0.05 + np.array([1.0] + [0.0] * 511)).astype(np.float32)
        norm = np.linalg.norm(new_emb)
        new_emb = new_emb / norm

        detection = {"id": 444, "embedding": new_emb}

        # Make cluster verified with very small epsilon (so strict_threshold is exceeded)
        mock_cluster = MagicMock()
        mock_cluster.verified = True
        mock_cluster.epsilon = 0.001  # Very tight epsilon
        # Set centroid to cluster 0 center (not exactly matching the embedding)
        cluster_0_center = np.zeros(512, dtype=np.float32)
        cluster_0_center[0] = 1.0
        mock_cluster.centroid = cluster_0_center.tolist()
        mock_repository.get_cluster_by_id.return_value = mock_cluster

        # No cannot-link constraints
        mock_repository.get_cannot_linked_detections.return_value = []
        mock_repository.get_detections_in_cluster.return_value = []

        # Set up epsilon-ball fallback path
        mock_repository.get_clusters_with_epsilon.return_value = []
        mock_repository.find_similar_unassigned_detections.return_value = []

        stage._assign_to_cluster = MagicMock()

        stage._cluster_single_detection(detection)

        # Distance to centroid will be > 0.001 * 0.8 = 0.0008
        # So approximate_predict should have been rejected for this verified cluster
        # and epsilon-ball should have been used
        mock_repository.get_clusters_with_epsilon.assert_called_once()

    def test_caches_clusterer_across_calls(self, stage, mock_repository):
        """Clusterer should be cached and reused for the same hdbscan_run."""
        clusterer, label_to_cluster_id, embeddings = _make_clusterer_and_label_map()
        run = _make_hdbscan_run(clusterer, label_to_cluster_id)

        mock_repository.get_active_hdbscan_run.return_value = run
        mock_repository.get_all_embeddings_for_collection.return_value = [{}] * 20

        rng = np.random.RandomState(99)
        emb = (rng.randn(512) * 0.05 + np.array([1.0] + [0.0] * 511)).astype(np.float32)
        norm = np.linalg.norm(emb)
        emb = emb / norm

        # First call
        stage._try_approximate_predict(emb)

        # After first call, cache should be populated
        assert stage._cached_hdbscan_run_id == 1
        assert stage._cached_clusterer is not None
        assert stage._cached_label_map is not None

        cached_clusterer_ref = stage._cached_clusterer

        # Second call with same run_id should reuse cache
        stage._try_approximate_predict(emb)

        # Clusterer object should be the same reference (not re-deserialized)
        assert stage._cached_clusterer is cached_clusterer_ref

    def test_refreshes_cache_on_new_run(self, stage, mock_repository):
        """Cache should be refreshed when hdbscan_run_id changes."""
        clusterer, label_to_cluster_id, embeddings = _make_clusterer_and_label_map()
        run1 = _make_hdbscan_run(clusterer, label_to_cluster_id)
        run1["id"] = 1

        mock_repository.get_active_hdbscan_run.return_value = run1
        mock_repository.get_all_embeddings_for_collection.return_value = [{}] * 20

        rng = np.random.RandomState(99)
        emb = (rng.randn(512) * 0.05 + np.array([1.0] + [0.0] * 511)).astype(np.float32)
        norm = np.linalg.norm(emb)
        emb = emb / norm

        # First call with run_id=1
        stage._try_approximate_predict(emb)
        assert stage._cached_hdbscan_run_id == 1
        old_clusterer = stage._cached_clusterer

        # Change to run_id=2
        run2 = _make_hdbscan_run(clusterer, label_to_cluster_id)
        run2["id"] = 2
        mock_repository.get_active_hdbscan_run.return_value = run2

        stage._try_approximate_predict(emb)
        assert stage._cached_hdbscan_run_id == 2
        # Should have re-deserialized (different object)
        assert stage._cached_clusterer is not old_clusterer
