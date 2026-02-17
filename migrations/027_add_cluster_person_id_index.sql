-- Migration: Add partial index on cluster(person_id) for people grid queries
-- Covers: getClustersGroupedByPerson, getPeople, getHiddenPeople, getPersonById, searchClusters

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cluster_person_visible
ON cluster(person_id, collection_id, face_count DESC)
WHERE face_count > 0 AND (hidden = false OR hidden IS NULL);
