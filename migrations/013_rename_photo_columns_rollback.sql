-- Rollback Migration 013: Revert photo column renames

-- Rename columns back
ALTER TABLE photo RENAME COLUMN orig_path TO filename;
ALTER TABLE photo RENAME COLUMN med_path TO normalized_path;
ALTER TABLE photo RENAME COLUMN med_width TO normalized_width;
ALTER TABLE photo RENAME COLUMN med_height TO normalized_height;

-- Restore original comments
COMMENT ON COLUMN photo.filename IS 'Relative path from INGEST_PATH';
COMMENT ON COLUMN photo.normalized_path IS 'Path to normalized image in IMG_PATH';
COMMENT ON COLUMN photo.normalized_width IS 'Normalized image width in pixels';
COMMENT ON COLUMN photo.normalized_height IS 'Normalized image height in pixels';
