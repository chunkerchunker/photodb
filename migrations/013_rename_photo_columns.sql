-- Migration 013: Rename photo columns for multi-size image support
-- Renames columns to distinguish original path from normalized sizes:
--   filename -> orig_path
--   normalized_path -> med_path
--   normalized_width -> med_width
--   normalized_height -> med_height

-- Rename columns
ALTER TABLE photo RENAME COLUMN filename TO orig_path;
ALTER TABLE photo RENAME COLUMN normalized_path TO med_path;
ALTER TABLE photo RENAME COLUMN normalized_width TO med_width;
ALTER TABLE photo RENAME COLUMN normalized_height TO med_height;

-- Update comments on the columns
COMMENT ON COLUMN photo.orig_path IS 'Original file path (relative from INGEST_PATH)';
COMMENT ON COLUMN photo.med_path IS 'Path to medium-sized normalized image (WebP in med/ subdir)';
COMMENT ON COLUMN photo.med_width IS 'Medium image width in pixels';
COMMENT ON COLUMN photo.med_height IS 'Medium image height in pixels';

-- Drop and recreate the unique constraints with new column names
-- Note: The constraints reference the old column names internally, but PostgreSQL
-- automatically updates them when columns are renamed. We just need to verify they work.

-- Verify the indexes still exist (they should, but with updated column references)
-- No action needed - PostgreSQL handles this automatically during column rename.
