import time
import numpy as np
import hdbscan
import torch
import warnings
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Try to import fast_hdbscan
try:
    import fast_hdbscan
    HAS_FAST_HDBSCAN = True
except ImportError:
    HAS_FAST_HDBSCAN = False
    print("Warning: fast_hdbscan not found. Skipping 'Fast CPU' benchmark.")

def generate_data(n_samples=30000, n_features=128, n_clusters=20):
    print(f"Generating {n_samples} samples with {n_features} dimensions...")
    data, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    return data.astype(np.float32), labels

class MetalKNN:
    def __init__(self, k=15, batch_size=2000):
        """
        Custom KNN implementation using PyTorch MPS.
        batch_size: split the query to avoid OOM on GPU for large N.
        """
        self.k = k
        self.batch_size = batch_size
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Metal utilizing device: MPS")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Metal utilizing device: CUDA (Example Only)")
        else:
            self.device = torch.device("cpu")
            print("Metal utilizing device: CPU (Fallback - No Speedup)")

    def fit_query(self, X):
        X_torch = torch.from_numpy(X).to(self.device)
        n_samples = X.shape[0]
        
        # Pre-allocate output buffers
        # KNN indices and Distances
        knn_indices = np.zeros((n_samples, self.k), dtype=np.int64)
        knn_dists = np.zeros((n_samples, self.k), dtype=np.float32)
        
        # Process in chunks to respect memory
        start_time = time.time()
        
        # We need to compute pairwise distance. For full N*N, it's heavy.
        # But wait, HDBSCAN usually builds a Minimum Spanning Tree from the KNN graph.
        # We just need the 'k' nearest neighbors for every point.
        
        for i in range(0, n_samples, self.batch_size):
            end = min(i + self.batch_size, n_samples)
            batch = X_torch[i:end]
            
            # Compute distances between this batch and ALL points
            # cdist is efficient, but N*N can be huge. 
            # If N=50k, 50k*50k floats = 2.5B * 4 bytes = 10GB. M1/M2/M3 implies unified memory, so might fit.
            # For larger datasets, we might need a more blocked approach or iterative.
            # Here we do batch x ALL.
            dists = torch.cdist(batch, X_torch)
            
            # Get topk
            # largest=False because we want smallest distances
            # We strictly want k+1 because the point itself is included (dist=0)
            values, indices = dists.topk(self.k + 1, dim=1, largest=False, sorted=True)
            
            # Transfer back to CPU
            # Exclude the first point (itself)
            knn_indices[i:end] = indices[:, 1:].cpu().numpy()
            knn_dists[i:end] = values[:, 1:].cpu().numpy()
            
        self.last_duration = time.time() - start_time
        print(f"KNN Search on {self.device} took: {self.last_duration:.4f}s")
        return knn_dists, knn_indices

def run_standard_hdbscan(data):
    print("\n--- Standard HDBSCAN (CPU) ---")
    start = time.time()
    clusterer = hdbscan.HDBSCAN(algorithm='best', core_dist_n_jobs=1)
    clusterer.fit(data)
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    return clusterer.labels_

def run_fast_hdbscan(data):
    if not HAS_FAST_HDBSCAN:
        return None
    print("\n--- Fast HDBSCAN (CPU - Numba) ---")
    start = time.time()
    clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=5)
    clusterer.fit(data)
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    return clusterer.labels_

def run_hybrid_hdbscan(data):
    print("\n--- Hybrid HDBSCAN (Metal KNN + CPU Clustering) ---")
    total_start = time.time()
    
    # 1. Compute KNN on GPU
    # Increase K to ensure connectivity in high dimensions (e.g. 128 dims)
    # If clusters are very far apart, a low K creates disconnected components.
    k_val = 60
    knn_solver = MetalKNN(k=k_val) 
    knn_dists, knn_indices = knn_solver.fit_query(data)
    
    # 2. Pass to HDBSCAN
    try:
        from scipy.sparse import csr_matrix
        N = data.shape[0]
        # Create the sparse matrix
        rows = np.repeat(np.arange(N), knn_indices.shape[1])
        cols = knn_indices.flatten()
        vals = knn_dists.flatten()
        
        # Note: We duplicate edges for symmetry effectively? data is asymmetric from KNN?
        # HDBSCAN usually expects undirected.
        # Let's simple create the matrix.
        sparse_graph = csr_matrix((vals, (rows, cols)), shape=(N, N))
        
        clustering_start = time.time()
        # allow_single_cluster=True might help if it's super merging, but connectivity is the key.
        clusterer = hdbscan.HDBSCAN(metric='precomputed', allow_single_cluster=False)
        clusterer.fit(sparse_graph)
        clustering_end = time.time()
        
        total_end = time.time()
        print(f"Total Hybrid Time: {total_end - total_start:.4f}s")
        print(f"  - KNN GPU Time: {knn_solver.last_duration:.4f}s")
        print(f"  - CPU Clustering Time: {clustering_end - clustering_start:.4f}s")
        return clusterer.labels_
        
    except Exception as e:
        print(f"Hybrid method failed in clustering phase: {e}")
        # Even if clustering fails, we proved the speed of KNN
        print(f"  But KNN GPU Time was: {knn_solver.last_duration if hasattr(knn_solver, 'last_duration') else 'N/A'}s")
        return None


def main():
    # 1. Setup
    n_samples = 20000 # Start safe. User can bump this up.
    data, truth = generate_data(n_samples=n_samples)
    
    # 2. Benchmarks
    labels_std = run_standard_hdbscan(data)
    if HAS_FAST_HDBSCAN:
        labels_fast = run_fast_hdbscan(data)
    
    # 3. Hybrid
    # Using Torch MPS
    try:
        labels_hybrid = run_hybrid_hdbscan(data)
    except Exception as e:
        print(f"Hybrid run failed: {e}")
        labels_hybrid = None

    # 4. Compare Accuracy (ARI)
    print("\n--- Accuracy Check (Adjusted Rand Index) ---")
    print("Standard vs Truth:", adjusted_rand_score(truth, labels_std))
    
    if HAS_FAST_HDBSCAN:
        print("Fast vs Truth:    ", adjusted_rand_score(truth, labels_fast))
        
    if labels_hybrid is not None:
        print("Hybrid vs Truth:  ", adjusted_rand_score(truth, labels_hybrid))
        print("Hybrid vs Std:    ", adjusted_rand_score(labels_std, labels_hybrid))

if __name__ == "__main__":
    main()
