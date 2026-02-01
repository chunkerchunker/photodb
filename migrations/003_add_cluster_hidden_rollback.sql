-- Rollback: Remove hidden column from cluster table

DROP INDEX IF EXISTS idx_cluster_hidden;
ALTER TABLE cluster DROP COLUMN IF EXISTS hidden;
