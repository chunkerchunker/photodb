-- Migration 022: Optimize unassigned detection pool index for vector search
--
-- The find_similar_unassigned_detections query filters by collection_id first,
-- but the existing idx_person_detection_unassigned_pool index doesn't include
-- collection_id, forcing a full scan of all unassigned detections across all
-- collections before the expensive vector distance calculation.
--
-- This replaces it with a collection-aware index that also INCLUDEs the id
-- column for index-only access to the face_embedding join key.

-- Drop the old index that doesn't include collection_id
DROP INDEX IF EXISTS idx_person_detection_unassigned_pool;

-- New index: collection_id as leading column for equality lookup,
-- then range columns for >= filtering, INCLUDE id for join without heap access
CREATE INDEX idx_person_detection_unassigned_pool
ON person_detection(collection_id, face_confidence, face_bbox_width, face_bbox_height)
INCLUDE (id)
WHERE cluster_id IS NULL AND cluster_status = 'unassigned';
