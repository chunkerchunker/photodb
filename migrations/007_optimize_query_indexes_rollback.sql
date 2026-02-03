-- Rollback Migration 007: Revert query index optimizations

-- Remove the new indexes
DROP INDEX IF EXISTS idx_metadata_year_month;
DROP INDEX IF EXISTS idx_processing_status_stage_status;

-- Restore the redundant filename index (if desired)
CREATE INDEX IF NOT EXISTS idx_photo_filename ON photo(filename);
