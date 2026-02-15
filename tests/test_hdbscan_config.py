"""Tests for HDBSCAN configuration and utility functions."""

import numpy as np
import pytest
from photodb.utils.hdbscan_config import (
    create_hdbscan_clusterer,
    lambda_to_epsilon,
    extract_cluster_lambda_births,
    serialize_condensed_tree,
)


class TestLambdaToEpsilon:
    def test_basic_conversion(self):
        assert lambda_to_epsilon(2.0) == pytest.approx(0.5)

    def test_clamped_to_min(self):
        assert lambda_to_epsilon(100.0) == 0.1

    def test_clamped_to_max(self):
        result = lambda_to_epsilon(0.5, max_epsilon=0.675)
        assert result == 0.675

    def test_zero_lambda_returns_fallback(self):
        assert lambda_to_epsilon(0.0, fallback=0.45) == 0.45

    def test_negative_lambda_returns_fallback(self):
        assert lambda_to_epsilon(-1.0, fallback=0.45) == 0.45


class TestCreateHdbscanClusterer:
    def test_default_has_prediction_data(self):
        clusterer = create_hdbscan_clusterer()
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


class TestExtractClusterLambdaBirths:
    def test_extracts_lambda_for_each_cluster(self):
        """Fit clusterer on well-separated data and verify lambda extraction."""
        rng = np.random.RandomState(42)
        cluster1 = rng.randn(10, 512) * 0.05 + np.array([1.0] + [0.0] * 511)
        cluster2 = rng.randn(10, 512) * 0.05 + np.array([0.0, 1.0] + [0.0] * 510)
        data = np.vstack([cluster1, cluster2]).astype(np.float32)

        clusterer = create_hdbscan_clusterer(min_cluster_size=3, min_samples=2)
        clusterer.fit(data)

        lambda_births = extract_cluster_lambda_births(clusterer)
        # Should have entries for each cluster found
        n_clusters = len(set(clusterer.labels_) - {-1})
        assert len(lambda_births) == n_clusters
        # All lambda values should be positive
        for label, lb in lambda_births.items():
            assert lb > 0, f"Cluster {label} has non-positive lambda_birth: {lb}"


class TestSerializeCondensedTree:
    def test_roundtrip(self):
        rng = np.random.RandomState(42)
        cluster1 = rng.randn(10, 512) * 0.05 + np.array([1.0] + [0.0] * 511)
        cluster2 = rng.randn(10, 512) * 0.05 + np.array([0.0, 1.0] + [0.0] * 510)
        data = np.vstack([cluster1, cluster2]).astype(np.float32)

        clusterer = create_hdbscan_clusterer(min_cluster_size=3, min_samples=2)
        clusterer.fit(data)

        tree_json = serialize_condensed_tree(clusterer)
        assert isinstance(tree_json, dict)
        assert "parent" in tree_json
        assert "child" in tree_json
        assert "lambda_val" in tree_json
        assert "child_size" in tree_json
        # All values should be JSON-serializable (int or float)
        import json

        json.dumps(tree_json)  # Should not raise
