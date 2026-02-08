#!/usr/bin/env python
"""
Benchmark: Metal/MPS HDBSCAN vs Standard HDBSCAN

Compares clustering quality and speed between:
1. Standard HDBSCAN (CPU, single-threaded MST)
2. Metal Hybrid (MPS GPU for KNN + sparse graph HDBSCAN)

Usage:
    uv run python scripts/benchmark_metal_hdbscan.py --sample-size 5000
    uv run python scripts/benchmark_metal_hdbscan.py --sample-size 5000 --skip-standard  # Skip slow standard run
"""

import argparse
import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Check for required packages
try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False

import hdbscan
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

from photodb.database.connection import ConnectionPool


def load_embeddings(pool, sample_size: int, collection_id: int = 1) -> np.ndarray:
    """Load face embeddings from database."""
    print(f"Loading up to {sample_size} embeddings from database...")

    with pool.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT fe.embedding
                   FROM face_embedding fe
                   JOIN person_detection pd ON fe.person_detection_id = pd.id
                   WHERE pd.collection_id = %s
                   LIMIT %s""",
                (collection_id, sample_size),
            )
            rows = cursor.fetchall()

    embeddings = np.array([np.array(row[0]) for row in rows], dtype=np.float32)
    print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return embeddings / norms


def run_standard_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 3) -> tuple:
    """Run standard HDBSCAN (CPU)."""
    print("\n" + "=" * 60)
    print("STANDARD HDBSCAN (CPU)")
    print("=" * 60)

    start = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    clusterer.fit(embeddings)
    elapsed = time.time() - start

    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f"Time: {elapsed:.2f}s")
    print(f"Clusters: {n_clusters}")
    print(f"Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")

    return labels, elapsed


class MetalKNN:
    """KNN using PyTorch MPS (Metal) for GPU acceleration."""

    def __init__(self, k: int = 60, batch_size: int = 2000):
        self.k = k
        self.batch_size = batch_size

        if HAS_MPS:
            self.device = torch.device("mps")
            print(f"Using device: MPS (Apple Silicon GPU)")
        elif HAS_TORCH and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: CUDA")
        elif HAS_TORCH:
            self.device = torch.device("cpu")
            print(f"Using device: CPU (no GPU acceleration)")
        else:
            raise RuntimeError("PyTorch not available")

    def compute_knn(self, X: np.ndarray) -> tuple:
        """Compute k-nearest neighbors using GPU."""
        X_torch = torch.from_numpy(X).to(self.device)
        n_samples = X.shape[0]

        knn_indices = np.zeros((n_samples, self.k), dtype=np.int64)
        knn_dists = np.zeros((n_samples, self.k), dtype=np.float32)

        start = time.time()

        for i in range(0, n_samples, self.batch_size):
            end = min(i + self.batch_size, n_samples)
            batch = X_torch[i:end]

            # Compute distances between batch and all points
            dists = torch.cdist(batch, X_torch)

            # Get k+1 nearest (includes self)
            values, indices = dists.topk(self.k + 1, dim=1, largest=False, sorted=True)

            # Exclude self (first column)
            knn_indices[i:end] = indices[:, 1:].cpu().numpy()
            knn_dists[i:end] = values[:, 1:].cpu().numpy()

        self.knn_time = time.time() - start
        print(f"KNN search time: {self.knn_time:.2f}s")

        return knn_dists, knn_indices


def build_symmetric_sparse_graph(knn_dists: np.ndarray, knn_indices: np.ndarray) -> csr_matrix:
    """Build symmetric sparse distance matrix from KNN results."""
    n_samples = knn_indices.shape[0]
    k = knn_indices.shape[1]

    # Create asymmetric graph first
    rows = np.repeat(np.arange(n_samples), k)
    cols = knn_indices.flatten()
    vals = knn_dists.flatten()

    # Build sparse matrix
    sparse = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))

    # Make symmetric: take minimum of (i,j) and (j,i) where both exist
    # For entries that only exist one way, keep that value
    sparse_t = sparse.T

    # Element-wise minimum where both exist, otherwise take the existing value
    symmetric = sparse.minimum(sparse_t)

    # Add entries that only exist in one direction
    mask = (sparse > 0) != (sparse_t > 0)
    symmetric = symmetric + sparse.multiply(mask) + sparse_t.multiply(mask.T)

    return symmetric


def run_metal_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 3, k: int = 60) -> tuple:
    """Run Metal/MPS hybrid HDBSCAN."""
    print("\n" + "=" * 60)
    print(f"METAL HYBRID HDBSCAN (MPS KNN + Sparse Graph)")
    print("=" * 60)

    if not HAS_TORCH:
        print("ERROR: PyTorch not installed. Install with: uv add torch")
        return None, None

    total_start = time.time()

    # Step 1: GPU-accelerated KNN
    print(f"\nStep 1: Computing {k}-NN on GPU...")
    knn = MetalKNN(k=k)
    knn_dists, knn_indices = knn.compute_knn(embeddings)

    # Step 2: Build symmetric sparse graph
    print("\nStep 2: Building symmetric sparse graph...")
    sparse_start = time.time()
    sparse_graph = build_symmetric_sparse_graph(knn_dists, knn_indices)
    sparse_time = time.time() - sparse_start
    print(f"Sparse graph time: {sparse_time:.2f}s")
    print(f"Graph density: {sparse_graph.nnz / (embeddings.shape[0]**2) * 100:.4f}%")

    # Step 3: HDBSCAN on sparse graph
    print("\nStep 3: Running HDBSCAN on sparse graph...")
    cluster_start = time.time()

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric="precomputed",
            cluster_selection_method="eom",
        )
        clusterer.fit(sparse_graph)
        cluster_time = time.time() - cluster_start

        labels = clusterer.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        total_time = time.time() - total_start

        print(f"\nClustering time: {cluster_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print(f"  - KNN: {knn.knn_time:.2f}s")
        print(f"  - Sparse graph: {sparse_time:.2f}s")
        print(f"  - HDBSCAN: {cluster_time:.2f}s")
        print(f"Clusters: {n_clusters}")
        print(f"Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")

        return labels, total_time

    except Exception as e:
        print(f"ERROR: HDBSCAN failed: {e}")
        print("This often means disconnected graph - try increasing k")
        return None, None


def compare_results(labels_std, labels_metal):
    """Compare clustering results."""
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if labels_std is None or labels_metal is None:
        print("Cannot compare - one or both runs failed")
        return

    # Adjusted Rand Index
    ari = adjusted_rand_score(labels_std, labels_metal)
    print(f"Adjusted Rand Index: {ari:.4f}")

    if ari > 0.95:
        print("  -> Excellent agreement (>0.95)")
    elif ari > 0.80:
        print("  -> Good agreement (>0.80)")
    elif ari > 0.50:
        print("  -> Moderate agreement (>0.50)")
    else:
        print("  -> Poor agreement - results differ significantly")

    # Cluster count comparison
    n_std = len(set(labels_std)) - (1 if -1 in labels_std else 0)
    n_metal = len(set(labels_metal)) - (1 if -1 in labels_metal else 0)
    print(f"\nCluster count: Standard={n_std}, Metal={n_metal}")

    # Noise comparison
    noise_std = np.sum(labels_std == -1)
    noise_metal = np.sum(labels_metal == -1)
    print(f"Noise points: Standard={noise_std}, Metal={noise_metal}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Metal HDBSCAN vs Standard")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Number of embeddings to sample (default: 5000)")
    parser.add_argument("--collection-id", type=int, default=1,
                        help="Collection ID to load from (default: 1)")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="HDBSCAN min_cluster_size (default: 3)")
    parser.add_argument("--k", type=int, default=60,
                        help="Number of neighbors for Metal KNN (default: 60)")
    parser.add_argument("--skip-standard", action="store_true",
                        help="Skip standard HDBSCAN (useful if you know it's slow)")
    parser.add_argument("--metal-only", action="store_true",
                        help="Only run Metal version (alias for --skip-standard)")
    args = parser.parse_args()

    # Connect to database
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)

    try:
        # Load embeddings
        embeddings = load_embeddings(pool, args.sample_size, args.collection_id)

        if len(embeddings) < args.min_cluster_size:
            print(f"ERROR: Not enough embeddings ({len(embeddings)}) for clustering")
            return 1

        # Normalize
        embeddings = normalize_embeddings(embeddings)

        # Run benchmarks
        labels_std = None
        time_std = None

        if not args.skip_standard and not args.metal_only:
            labels_std, time_std = run_standard_hdbscan(
                embeddings, args.min_cluster_size
            )
        else:
            print("\nSkipping standard HDBSCAN (--skip-standard or --metal-only)")

        labels_metal, time_metal = run_metal_hdbscan(
            embeddings, args.min_cluster_size, args.k
        )

        # Compare
        if labels_std is not None and labels_metal is not None:
            compare_results(labels_std, labels_metal)

            if time_std and time_metal:
                speedup = time_std / time_metal
                print(f"\nSpeedup: {speedup:.1f}x faster with Metal")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if time_std:
            print(f"Standard HDBSCAN: {time_std:.2f}s")
        if time_metal:
            print(f"Metal Hybrid:     {time_metal:.2f}s")
        if time_std and time_metal:
            print(f"Speedup:          {time_std/time_metal:.1f}x")

        return 0

    finally:
        pool.close_all()


if __name__ == "__main__":
    exit(main())
