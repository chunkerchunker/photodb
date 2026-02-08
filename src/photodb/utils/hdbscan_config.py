"""
HDBSCAN configuration and factory functions.

This module provides shared HDBSCAN configuration used by both the
ClusteringStage (for bootstrap) and MaintenanceUtilities (for pool clustering).
"""

from typing import Any
import numpy as np

# Default HDBSCAN parameters
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MIN_SAMPLES = 2
DEFAULT_EPSILON_PERCENTILE = 90.0
DEFAULT_CORE_PROBABILITY_THRESHOLD = 0.8
DEFAULT_CLUSTERING_THRESHOLD = 0.45


def create_hdbscan_clusterer(
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    metric: str = "euclidean",
    precomputed: bool = False,
) -> Any:
    """
    Create an HDBSCAN clusterer with consistent configuration.

    Args:
        min_cluster_size: Minimum number of points to form a cluster
        min_samples: Core point requirement (number of points in neighborhood)
        metric: Distance metric to use (ignored if precomputed=True)
        precomputed: If True, expects a precomputed distance matrix

    Returns:
        Configured HDBSCAN clusterer instance
    """
    import hdbscan

    if precomputed:
        return hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
            cluster_selection_method="eom",
        )

    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )


def calculate_cluster_epsilon(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label: int,
    percentile: float = DEFAULT_EPSILON_PERCENTILE,
    fallback_threshold: float = DEFAULT_CLUSTERING_THRESHOLD,
) -> float:
    """
    Calculate the epsilon threshold for a specific cluster.

    Uses the percentile of pairwise distances between points to determine
    the cluster's natural spread.

    Args:
        embeddings: All embeddings array
        labels: HDBSCAN labels for all embeddings
        label: The specific cluster label to calculate epsilon for
        percentile: Percentile of pairwise distances to use (must be between 0 and 100)
        fallback_threshold: Default epsilon for single-point clusters

    Returns:
        Epsilon value (distance threshold) for the cluster

    Raises:
        ValueError: If percentile is not between 0 and 100
    """
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

    from scipy.spatial.distance import pdist

    cluster_mask = labels == label
    cluster_embeddings = embeddings[cluster_mask]

    if len(cluster_embeddings) < 2:
        return fallback_threshold

    pairwise_distances = pdist(cluster_embeddings, metric="euclidean")

    if len(pairwise_distances) == 0:
        return fallback_threshold

    epsilon = float(np.percentile(pairwise_distances, percentile))

    # Clamp to reasonable bounds
    min_epsilon = 0.1
    max_epsilon = fallback_threshold * 1.5

    return max(min_epsilon, min(epsilon, max_epsilon))
