-- Migration 008: Add indexes for database scalability
-- Optimizes queries that will become slow as data grows

-- 1. Cluster listing indexes
-- getClusters() and getClustersCount() filter by face_count > 0 and hidden = false/NULL
-- getHiddenClusters() filters by face_count > 0 and hidden = true
-- These partial indexes only include relevant rows, keeping them small and fast

CREATE INDEX IF NOT EXISTS idx_cluster_visible
ON cluster(face_count DESC, id)
WHERE face_count > 0 AND (hidden = false OR hidden IS NULL);

-- Note: idx_cluster_hidden already exists from migration 003 with different columns
-- This index is specifically for the hidden cluster listing query with ORDER BY face_count
CREATE INDEX IF NOT EXISTS idx_cluster_hidden_listing
ON cluster(face_count DESC, id)
WHERE face_count > 0 AND hidden = true;

-- 2. Unassigned detection pool index
-- find_similar_unassigned_detections() and seed selection queries filter by:
--   cluster_id IS NULL AND cluster_status = 'unassigned'
--   AND face_confidence >= X AND face_bbox_width >= Y AND face_bbox_height >= Y
-- This composite partial index covers the common filtering pattern

CREATE INDEX IF NOT EXISTS idx_person_detection_unassigned_pool
ON person_detection(face_confidence DESC, face_bbox_width, face_bbox_height)
WHERE cluster_id IS NULL AND cluster_status = 'unassigned';

-- 3. Processing status - completed photos per stage
-- get_unprocessed_photos() uses LEFT JOIN anti-pattern to find photos without completed status
-- This index supports a more efficient EXISTS subquery pattern
-- Also useful for progress tracking queries

CREATE INDEX IF NOT EXISTS idx_processing_status_completed
ON processing_status(photo_id)
WHERE status = 'completed';
