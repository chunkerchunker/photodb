-- Rollback: Remove hidden flag from person table

DROP INDEX IF EXISTS idx_person_collection_hidden;
DROP INDEX IF EXISTS idx_person_hidden;
ALTER TABLE person DROP COLUMN IF EXISTS hidden;
