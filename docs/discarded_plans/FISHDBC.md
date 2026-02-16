  FISHDBC Investigation: Full Analysis

  What is FISHDBC?

  FISHDBC (Flexible, Incremental, Scalable, Hierarchical Density-Based Clustering) is a 2019 research algorithm (<https://arxiv.org/abs/1910.07283>) that combines HNSW (approximate nearest-neighbor graph) with
  incremental MST maintenance to produce HDBSCAN-equivalent clustering without requiring all data upfront.

  Core idea: As you insert points into an HNSW graph, the distances computed during insertion are captured and fed into an incremental MST. When you call cluster(), the MST is passed through HDBSCAN's internal
  pipeline (condense_tree -> compute_stability -> get_clusters) to extract the hierarchy.

  Benchmark Results on Apple Silicon (M-series)

  Batch Performance (2,000 points, 512-d normalized embeddings)
  ┌───────────────────────┬──────┬──────────┬───────┬─────────┐
  │        Method         │ Time │ Clusters │ Noise │ Memory  │
  ├───────────────────────┼──────┼──────────┼───────┼─────────┤
  │ HDBSCAN batch         │ 4.4s │ 24       │ 35.2% │ +17 MB  │
  ├───────────────────────┼──────┼──────────┼───────┼─────────┤
  │ FISHDBC (m=5, ef=32)  │ 2.0s │ 11       │ 37.0% │ +112 MB │
  ├───────────────────────┼──────┼──────────┼───────┼─────────┤
  │ FISHDBC (m=16, ef=64) │ 7.2s │ 16       │ 32.8% │ +192 MB │
  └───────────────────────┴──────┴──────────┴───────┴─────────┘
  Batch Performance (5,000 points)
  ┌───────────────────────┬───────┬──────────┬───────────┐
  │        Method         │ Time  │ Clusters │  Memory   │
  ├───────────────────────┼───────┼──────────┼───────────┤
  │ HDBSCAN batch         │ 28.8s │ 138      │ +47 MB    │
  ├───────────────────────┼───────┼──────────┼───────────┤
  │ FISHDBC (m=5, ef=32)  │ 6.2s  │ 110      │ +401 MB   │
  ├───────────────────────┼───────┼──────────┼───────────┤
  │ FISHDBC (m=16, ef=64) │ 29.1s │ 143      │ +1,138 MB │
  └───────────────────────┴───────┴──────────┴───────────┘
  Quality (ARI agreement with HDBSCAN)
  ┌───────────────────────┬─────────────────────┬─────────────────────┐
  │    FISHDBC Config     │ ARI vs HDBSCAN (2k) │ ARI vs HDBSCAN (5k) │
  ├───────────────────────┼─────────────────────┼─────────────────────┤
  │ m=5, ef=32 (fast)     │ 0.858               │ 0.455               │
  ├───────────────────────┼─────────────────────┼─────────────────────┤
  │ m=16, ef=64 (quality) │ 0.946               │ 0.810               │
  └───────────────────────┴─────────────────────┴─────────────────────┘
  Key finding: At low connectivity (m=5), FISHDBC finds significantly fewer clusters and diverges from HDBSCAN. At high connectivity (m=16), quality approaches HDBSCAN but speed and memory advantages vanish.

  Incremental Scenario (1,600 initial + 400 added in batches of 10)
  ┌───────────────────────────────┬──────────────────────┬────────────────────┬─────────────────────┐
  │            Method             │       Initial        │   Per-new-point    │   Label Stability   │
  ├───────────────────────────────┼──────────────────────┼────────────────────┼─────────────────────┤
  │ HDBSCAN + approximate_predict │ 2.85s bootstrap      │ 0.85ms predict     │ 100% stable         │
  ├───────────────────────────────┼──────────────────────┼────────────────────┼─────────────────────┤
  │ FISHDBC incremental           │ 1.46s insert+cluster │ 57ms add+recluster │ ~4% churn per batch │
  └───────────────────────────────┴──────────────────────┴────────────────────┴─────────────────────┘
  Insertion Rate Scaling (pure-Python HNSW)
  ┌────────┬─────────────┬──────────────┬──────────┐
  │   N    │ Insert Rate │ Cluster Time │ Peak RSS │
  ├────────┼─────────────┼──────────────┼──────────┤
  │ 500    │ 2,117 pts/s │ 0.03s        │ 169 MB   │
  ├────────┼─────────────┼──────────────┼──────────┤
  │ 1,000  │ 1,833 pts/s │ 0.09s        │ 217 MB   │
  ├────────┼─────────────┼──────────────┼──────────┤
  │ 2,000  │ 1,441 pts/s │ 0.27s        │ 298 MB   │
  ├────────┼─────────────┼──────────────┼──────────┤
  │ 5,000  │ 1,046 pts/s │ 0.98s        │ 634 MB   │
  ├────────┼─────────────┼──────────────┼──────────┤
  │ 10,000 │ 851 pts/s   │ 2.81s        │ 1,153 MB │
  └────────┴─────────────┴──────────────┴──────────┘
  Pros vs Current Architecture
  ┌─────────────────────────────┬───────────────────────────────────────────────────────────────────────────────┐
  │          Advantage          │                                    Details                                    │
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ True incremental insertion  │ No separate bootstrap/incremental modes. Add points as they arrive.           │
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ Arbitrary distance function │ Natively supports cosine distance (no normalized-Euclidean workaround needed) │
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ Lower batch memory          │ O(n×k) vs O(n²) distance matrix — matters for very large datasets             │
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ Same output format          │ Produces identical condensed tree, persistence, lambda values as HDBSCAN      │
  ├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ Faster batch at low quality │ 2s vs 4.4s at m=5 (but with worse cluster quality)                            │
  └─────────────────────────────┴───────────────────────────────────────────────────────────────────────────────┘
  Cons vs Current Architecture
  Disadvantage: No approximate_predict
  Severity: Critical
  Details: Every cluster() call re-labels ALL points. No stable incremental assignment. Issue #7 on GitHub confirms this is unresolved.
  ────────────────────────────────────────
  Disadvantage: Label instability
  Severity: Critical
  Details: Mean 4% of points change labels per re-cluster. Max observed: 1,121 out of 2,000 points changed in a single re-cluster. This would cause constant database churn in your system.
  ────────────────────────────────────────
  Disadvantage: No GPU acceleration
  Severity: High
  Details: Pure Python HNSW — cannot use Metal/MPS. Your current system gets ~8x speedup from GPU k-NN.
  ────────────────────────────────────────
  Disadvantage: ~6x memory overhead
  Severity: High
  Details: 112-1,138 MB vs 17-47 MB for HDBSCAN. Python dicts for HNSW layers are extremely memory-inefficient.
  ────────────────────────────────────────
  Disadvantage: Quality degrades at scale
  Severity: High
  Details: ARI vs HDBSCAN drops from 0.946 to 0.810 at 5k points (m=16), and to 0.455 (m=5).
  ────────────────────────────────────────
  Disadvantage: Not serializable
  Severity: High
  Details: Cannot pickle the FISHDBC object (GitHub Issue #2). Your system relies on persisting the clusterer to the database.
  ────────────────────────────────────────
  Disadvantage: Not production-ready
  Severity: High
  Details: Alpha quality, single contributor, last code update Nov 2019, no tests, no CI, no PyPI, incompatible with NumPy 2.0 out of the box.
  ────────────────────────────────────────
  Disadvantage: Not thread-safe
  Severity: Medium
  Details: No synchronization on any data structures.
  ────────────────────────────────────────
  Disadvantage: Insertion rate degrades
  Severity: Medium
  Details: ~850 pts/s at 10k, trending down. Pure Python HNSW is orders of magnitude slower than hnswlib/FAISS.
  ────────────────────────────────────────
  Disadvantage: Breaks two-tier model
  Severity: High
  Details: Your system's strength is: fast batch bootstrap + O(1ms) approximate_predict per new face + epsilon-ball fallback. FISHDBC replaces this with "add-then-recluster-everything" (57ms+ per face, with churn).
  The Fundamental Problem

  FISHDBC solves a different problem than what PhotoDB needs. PhotoDB's incremental workflow requires:

  1. Stable cluster identities — a person's cluster ID must not change when unrelated faces are added
  2. Fast per-point assignment — ~1ms to assign a new face, not 50ms+ with global re-clustering
  3. No cascading label changes — adding one face must not reassign 4% of all faces

  FISHDBC provides incremental insertion into the clustering data structure, but not incremental assignment. Every call to cluster() is a global re-computation that can and does change existing labels. This is
  architecturally incompatible with PhotoDB's design where cluster IDs are foreign keys in the database, linked to persons, verified by users, and constrained by cannot-link rules.

  Recommendation

  Do not adopt FISHDBC. The current HDBSCAN + approximate_predict + epsilon-ball architecture is better suited to PhotoDB's requirements in every measurable dimension:

- 67x faster incremental assignment (0.85ms vs 57ms)
- 100% label stability vs ~4% churn
- 6-24x less memory
- GPU-accelerated batch clustering
- Production-quality dependencies (scikit-learn-contrib/hdbscan)
- Serializable clusterer for database persistence

  The FISHDBC paper's value is theoretical — it proves that incremental HDBSCAN is possible — but the implementation is a research prototype that's incompatible with real-world incremental face clustering workflows.
   If a production-quality, GPU-accelerated implementation with stable label tracking emerged in the future, it would be worth re-evaluating.
