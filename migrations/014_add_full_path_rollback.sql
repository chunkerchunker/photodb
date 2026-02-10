-- Rollback Migration 014: Remove full_path column

DROP INDEX IF EXISTS idx_photo_full_path;
ALTER TABLE photo DROP COLUMN IF EXISTS full_path;
