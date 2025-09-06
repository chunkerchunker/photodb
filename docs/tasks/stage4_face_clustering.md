# Stage 4: Face Clustering

This stage clusters detected faces using embeddings and cosine similarity. The clustering process is designed to be incremental, concurrent-safe, and handle ambiguous cases for later human review.

## Overview

- **Purpose**: Group faces belonging to the same person into clusters
- **Method**: Cosine similarity on 512-dimensional face embeddings
- **Threshold**: 0.45 cosine distance (configurable via CLUSTERING_THRESHOLD env var)
- **Integration**: Runs after face detection stage, tracks status via processing_status table

## Algorithm: Per-Face Incremental Clustering

For each unprocessed face (faces without cluster assignment):

### 1. Query unprocessed faces

```sql
-- Get faces that haven't been clustered yet
SELECT f.id, f.photo_id, fe.embedding
FROM face f
JOIN face_embedding fe ON f.id = fe.face_id
WHERE f.cluster_id IS NULL 
  AND f.cluster_status IS NULL
ORDER BY f.id;
```

### 2. Find nearest cluster candidates (KNN search)

```sql
-- Find top 10 nearest clusters using cosine similarity
-- Note: <=> operator computes cosine distance for normalized vectors
SELECT id, centroid <=> $embedding AS distance
FROM cluster
WHERE centroid IS NOT NULL
ORDER BY centroid <=> $embedding
LIMIT 10;
```

### 3. Apply clustering decision rules

Based on the distance to nearest clusters, apply one of three rules:

#### Rule A: Create new cluster (no matches below threshold)
If no cluster has distance < 0.45:
- Create new cluster with this face as first member
- Set cluster centroid to face embedding
- Assign `face.cluster_id` with `cluster_confidence = 1.0`
- Set `face.cluster_status = 'auto'`

#### Rule B: Assign to single cluster (one clear match)
If exactly one cluster has distance < 0.45:
- Assign face to that cluster
- Set `face.cluster_confidence = 1.0 - distance` (higher is better)
- Set `face.cluster_status = 'auto'`
- Update cluster centroid incrementally:

```sql
-- Incremental centroid update with proper face counting
-- Note: This requires transaction to ensure consistency
BEGIN;

-- Lock the cluster row to prevent concurrent updates
SELECT * FROM cluster WHERE id = $cluster_id FOR UPDATE;

-- Update centroid and face count atomically
UPDATE cluster
SET centroid = (centroid * face_count + $embedding::vector) / (face_count + 1),
    face_count = face_count + 1,
    updated_at = NOW()
WHERE id = $cluster_id;

-- Update face assignment
UPDATE face 
SET cluster_id = $cluster_id,
    cluster_confidence = $confidence,
    cluster_status = 'auto'
WHERE id = $face_id;

COMMIT;
```

#### Rule C: Mark for manual review (multiple potential matches)
If multiple clusters have distance < 0.45:
- Create `face_match_candidate` records for each potential cluster
- Set `face.cluster_status = 'pending'` (no cluster_id assigned yet)
- Human review will later resolve the ambiguity

```sql
-- Insert all potential matches for later review
INSERT INTO face_match_candidate (face_id, cluster_id, similarity, status)
SELECT $face_id, id, (1.0 - distance), 'pending'
FROM (
  SELECT id, centroid <=> $embedding AS distance
  FROM cluster
  WHERE centroid IS NOT NULL
    AND centroid <=> $embedding < 0.45
) matches;

-- Mark face as pending review
UPDATE face 
SET cluster_status = 'pending'
WHERE id = $face_id;

```

## Periodic Maintenance Tasks

These tasks run periodically to maintain cluster quality and consistency:

### 1. Recompute Centroids (Daily)

Recalculate cluster centroids from all member faces to correct for drift:

```sql
-- Recompute all cluster centroids from their member faces
UPDATE cluster c
SET centroid = sub.avg_embedding,
    face_count = sub.count,
    updated_at = NOW()
FROM (
    SELECT f.cluster_id, 
           AVG(fe.embedding) AS avg_embedding,
           COUNT(*) as count
    FROM face f
    JOIN face_embedding fe ON f.id = fe.face_id
    WHERE f.cluster_id IS NOT NULL
    GROUP BY f.cluster_id
) sub
WHERE c.id = sub.cluster_id;
```

### 2. Update Medoids and Representative Faces (Weekly)

Find the face closest to each cluster centroid (medoid) for visualization:

```python
def update_cluster_medoid(conn, cluster_id):
    """Update medoid_face_id to be the face closest to the cluster centroid."""
    with conn.cursor() as cur:
        # Get all faces and embeddings in this cluster
        cur.execute("""
            SELECT f.id, fe.embedding 
            FROM face f
            JOIN face_embedding fe ON f.id = fe.face_id
            WHERE f.cluster_id = %s
        """, (cluster_id,))
        
        rows = cur.fetchall()
        if not rows:
            return
            
        face_ids = [row[0] for row in rows]
        embeddings = np.array([row[1] for row in rows])
        
        # Compute centroid
        centroid = embeddings.mean(axis=0)
        
        # Find face closest to centroid (medoid)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        medoid_idx = np.argmin(distances)
        medoid_face_id = face_ids[medoid_idx]
        
        # Update cluster with medoid
        cur.execute("""
            UPDATE cluster
            SET medoid_face_id = %s,
                representative_face_id = %s,
                updated_at = NOW()
            WHERE id = %s
        """, (medoid_face_id, medoid_face_id, cluster_id))
        
        conn.commit()
```

### 3. Merge Similar Clusters (Weekly)

Identify and merge clusters that have become too similar:

```sql
-- Find cluster pairs with high similarity
WITH cluster_pairs AS (
    SELECT c1.id as cluster1_id, 
           c2.id as cluster2_id,
           c1.centroid <=> c2.centroid as distance
    FROM cluster c1
    CROSS JOIN cluster c2
    WHERE c1.id < c2.id  -- Avoid duplicates
      AND c1.centroid IS NOT NULL
      AND c2.centroid IS NOT NULL
      AND c1.centroid <=> c2.centroid < 0.3  -- Very similar clusters
)
SELECT * FROM cluster_pairs
ORDER BY distance;

-- Merge logic would reassign all faces from smaller cluster to larger
```

### 4. Clean Up Empty Clusters (Daily)

Remove clusters with no assigned faces:

```sql
-- Delete empty clusters
DELETE FROM cluster
WHERE id NOT IN (
    SELECT DISTINCT cluster_id 
    FROM face 
    WHERE cluster_id IS NOT NULL
);
```

## Implementation Notes

### Concurrency and Safety

1. **Transaction Boundaries**: All cluster updates must use transactions with proper locking
2. **Row-Level Locking**: Use `SELECT FOR UPDATE` when modifying clusters to prevent race conditions
3. **Batch Processing**: Process faces in batches to reduce lock contention
4. **Idempotency**: Stage must be idempotent - reprocessing should not create duplicate clusters

### Performance Optimizations

1. **Required Indexes**:
   ```sql
   -- Vector similarity search on clusters
   CREATE INDEX idx_cluster_centroid ON cluster 
   USING ivfflat(centroid vector_cosine_ops) WITH (lists = 100);
   
   -- Face clustering queries
   CREATE INDEX idx_face_cluster_id ON face(cluster_id);
   CREATE INDEX idx_face_cluster_status ON face(cluster_status);
   ```

2. **Embedding Normalization**: Ensure all embeddings are L2-normalized before storage for accurate cosine similarity

3. **Batch KNN Search**: Process multiple faces together to amortize index lookup costs

### Integration with PhotoDB Pipeline

1. **Stage Configuration**:
   - Environment variable: `CLUSTERING_THRESHOLD` (default: 0.45)
   - Stage name in processing_status: `'clustering'`
   - Runs after `'faces'` stage completes

2. **Error Handling**:
   - Failed clustering attempts set processing_status to 'failed' with error message
   - Corrupted embeddings skip clustering with logged warning
   - Dimension mismatches (not 512-d) are rejected

3. **Future Enhancements**:
   - Person association UI for resolving `face_match_candidate` records
   - Cluster quality metrics and monitoring
   - Active learning to improve threshold selection
   - Multi-modal clustering using photo metadata (time, location)
