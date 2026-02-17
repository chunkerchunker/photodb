-- Partial index for find_similar_cluster_pairs: covers the CTE scan of
-- eligible (non-hidden, centroid-present) clusters per collection.
CREATE INDEX IF NOT EXISTS idx_cluster_eligible
ON cluster(collection_id, id)
WHERE centroid IS NOT NULL AND NOT hidden;
