-- Rollback Migration 016: Remove face_path column

DROP INDEX IF EXISTS idx_person_detection_face_path_null;

ALTER TABLE person_detection DROP COLUMN IF EXISTS face_path;
