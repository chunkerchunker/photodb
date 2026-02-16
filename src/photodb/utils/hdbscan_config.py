"""
HDBSCAN configuration and factory functions.

This module provides shared HDBSCAN configuration used by both the
ClusteringStage (for bootstrap) and MaintenanceUtilities (for pool clustering).
"""

from typing import Any
import numpy as np

from .. import config as _config

# Default HDBSCAN parameters — re-exported from config for backward compatibility
DEFAULT_MIN_CLUSTER_SIZE = _config.HDBSCAN_MIN_CLUSTER_SIZE
DEFAULT_MIN_SAMPLES = _config.HDBSCAN_MIN_SAMPLES
DEFAULT_CORE_PROBABILITY_THRESHOLD = _config.CORE_PROBABILITY_THRESHOLD
DEFAULT_CLUSTERING_THRESHOLD = _config.CLUSTERING_THRESHOLD
DEFAULT_MIN_EPSILON = _config.MIN_EPSILON


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
            gen_min_span_tree=True,
        )

    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
        prediction_data=True,
        gen_min_span_tree=True,
    )


def lambda_to_epsilon(
    lambda_birth: float,
    min_epsilon: float = DEFAULT_MIN_EPSILON,
    max_epsilon: float | None = None,
    fallback: float = DEFAULT_CLUSTERING_THRESHOLD,
) -> float:
    """
    Convert a lambda_birth value from the HDBSCAN condensed tree to an epsilon threshold.

    Lambda values are the inverse of distance. This function converts them back to
    distance-based epsilon thresholds, clamped to reasonable bounds.

    Args:
        lambda_birth: The lambda_birth value from the condensed tree.
            If <= 0, returns the fallback value.
        min_epsilon: Minimum allowed epsilon value (floor clamp).
        max_epsilon: Maximum allowed epsilon value (ceiling clamp).
            If None, defaults to fallback * 1.5.
        fallback: Value to return when lambda_birth is invalid (<= 0).

    Returns:
        Epsilon threshold clamped to [min_epsilon, max_epsilon].
    """
    if lambda_birth <= 0:
        return fallback

    epsilon = 1.0 / lambda_birth

    if max_epsilon is None:
        max_epsilon = fallback * 1.5

    return max(min_epsilon, min(epsilon, max_epsilon))


def extract_cluster_lambda_births(clusterer: Any) -> dict[int, float]:
    """
    Extract per-cluster lambda_birth from the HDBSCAN condensed tree.

    The lambda_birth of a cluster is the minimum lambda_val at which that cluster
    node first appears as a parent in the condensed tree. This represents the
    density at which the cluster "is born" (splits from a parent cluster).

    Args:
        clusterer: A fitted HDBSCAN clusterer with condensed_tree_ attribute.

    Returns:
        Dictionary mapping cluster label (int) to lambda_birth (float).
        Only includes labels >= 0 (excludes noise label -1).
    """
    tree_df = clusterer.condensed_tree_.to_pandas()
    n_samples = len(clusterer.labels_)

    # Cluster nodes in the condensed tree have parent >= n_samples
    # Find the lambda at which each cluster node first appears as a parent
    cluster_nodes = tree_df[tree_df["parent"] >= n_samples]

    # Group by parent node and get the minimum lambda (birth)
    node_births = cluster_nodes.groupby("parent")["lambda_val"].min()

    # Map condensed tree node IDs to HDBSCAN label assignments
    # For each unique label >= 0, find which condensed tree node contains its points
    labels = clusterer.labels_
    unique_labels = set(labels[labels >= 0])

    label_to_lambda: dict[int, float] = {}
    for label in unique_labels:
        # Get indices of points with this label
        point_indices = np.where(labels == label)[0]
        if len(point_indices) == 0:
            continue

        # Find which condensed tree leaf entries contain these points
        # Leaf entries have child < n_samples (they are individual points)
        leaf_entries = tree_df[
            (tree_df["child"].isin(point_indices)) & (tree_df["child_size"] == 1)
        ]

        if leaf_entries.empty:
            continue

        # The most common parent of these leaf entries is the cluster node for this label
        most_common_parent = leaf_entries["parent"].mode()
        if len(most_common_parent) == 0:
            continue

        parent_node = most_common_parent.iloc[0]

        if parent_node in node_births.index:
            label_to_lambda[int(label)] = float(node_births[parent_node])

    return label_to_lambda


def serialize_condensed_tree(clusterer: Any) -> dict[str, list[int | float]]:
    """
    Serialize HDBSCAN condensed tree to a JSON-compatible dict.

    Converts the condensed tree DataFrame into a dictionary of lists with
    native Python types suitable for JSON serialization.

    Args:
        clusterer: A fitted HDBSCAN clusterer with condensed_tree_ attribute.

    Returns:
        Dictionary with keys 'parent', 'child', 'lambda_val', 'child_size',
        each mapping to a list of int or float values.
    """
    df = clusterer.condensed_tree_.to_pandas()
    # Convert to dict with native Python types for JSON serialization
    return {
        col: [float(v) if isinstance(v, (np.floating, float)) else int(v) for v in df[col]]
        for col in df.columns
    }
