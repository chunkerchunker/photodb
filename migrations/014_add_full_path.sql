-- Migration 014: Add full_path column for full-size WebP images
-- Adds a column to store the path to full-size normalized WebP images

ALTER TABLE photo ADD COLUMN IF NOT EXISTS full_path text;

COMMENT ON COLUMN photo.full_path IS 'Path to full-size normalized image (WebP in full/ subdir)';

-- Add index for queries that filter by full_path existence
CREATE INDEX IF NOT EXISTS idx_photo_full_path ON photo(full_path) WHERE full_path IS NOT NULL;
