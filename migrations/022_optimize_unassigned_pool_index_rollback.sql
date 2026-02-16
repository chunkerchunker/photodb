-- Rollback migration 022: Restore original unassigned pool index

DROP INDEX IF EXISTS idx_person_detection_unassigned_pool;

CREATE INDEX idx_person_detection_unassigned_pool
ON person_detection(face_confidence DESC, face_bbox_width, face_bbox_height)
WHERE cluster_id IS NULL AND cluster_status = 'unassigned';
