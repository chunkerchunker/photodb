-- Migration 007: Optimize query indexes
-- Adds indexes to improve query performance based on EXPLAIN ANALYZE findings

-- 1. Expression index for date-based photo queries
-- Improves getPhotosByMonth which uses EXTRACT(YEAR/MONTH FROM captured_at)
-- Without this, the query can't use idx_metadata_captured_at efficiently
-- Note: Must cast to timestamp (without tz) for EXTRACT to be immutable
CREATE INDEX IF NOT EXISTS idx_metadata_year_month
ON metadata (EXTRACT(YEAR FROM captured_at AT TIME ZONE 'UTC'), EXTRACT(MONTH FROM captured_at AT TIME ZONE 'UTC'));

-- 2. Properly ordered index for processing status queries
-- The existing idx_processing_status is (status, stage) but queries filter by stage first
-- This index supports: WHERE stage = $1 AND status = $2
CREATE INDEX IF NOT EXISTS idx_processing_status_stage_status
ON processing_status(stage, status);

-- 3. Remove redundant filename index
-- photo.filename has UNIQUE constraint which creates an implicit unique index
-- idx_photo_filename is redundant and wastes space/slows writes
DROP INDEX IF EXISTS idx_photo_filename;
