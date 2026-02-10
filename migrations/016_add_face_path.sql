-- Migration 016: Add face_path column for cropped face images
-- Adds a column to store the path to cropped face images extracted during detection

ALTER TABLE person_detection ADD COLUMN IF NOT EXISTS face_path text;

COMMENT ON COLUMN person_detection.face_path IS 'Path to cropped face image (WebP in faces/ subdir)';

-- Index for queries that filter by face_path existence (e.g., backfill queries)
CREATE INDEX IF NOT EXISTS idx_person_detection_face_path_null
ON person_detection(id)
WHERE face_path IS NULL AND face_bbox_x IS NOT NULL;
