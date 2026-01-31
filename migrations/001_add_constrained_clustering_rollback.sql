-- Rollback Migration: Remove Constrained Clustering Support
-- Description: Reverses the changes from 001_add_constrained_clustering.sql
-- WARNING: This will DELETE all constraint data (must_link, cannot_link, cluster_cannot_link)
-- Created: 2025-01-30

BEGIN;

-- ============================================================================
-- 1. Drop helper functions
-- ============================================================================

DROP FUNCTION IF EXISTS add_must_link(bigint, bigint, text);
DROP FUNCTION IF EXISTS add_cannot_link(bigint, bigint, text);
DROP FUNCTION IF EXISTS check_constraint_violations();

-- ============================================================================
-- 2. Drop indexes
-- ============================================================================

DROP INDEX IF EXISTS idx_face_unassigned;
DROP INDEX IF EXISTS idx_cluster_verified;
DROP INDEX IF EXISTS idx_must_link_face1;
DROP INDEX IF EXISTS idx_must_link_face2;
DROP INDEX IF EXISTS idx_cannot_link_face1;
DROP INDEX IF EXISTS idx_cannot_link_face2;
DROP INDEX IF EXISTS idx_cluster_cannot_link_c1;
DROP INDEX IF EXISTS idx_cluster_cannot_link_c2;

-- ============================================================================
-- 3. Drop constraint tables (WARNING: deletes all constraint data)
-- ============================================================================

DROP TABLE IF EXISTS cluster_cannot_link;
DROP TABLE IF EXISTS cannot_link;
DROP TABLE IF EXISTS must_link;

-- ============================================================================
-- 4. Reset cluster_status values that use new types
-- ============================================================================

-- Set any 'unassigned' or 'constrained' faces back to NULL
UPDATE face
SET cluster_status = NULL, unassigned_since = NULL
WHERE cluster_status IN ('unassigned', 'constrained');

-- ============================================================================
-- 5. Revert cluster_status CHECK constraint
-- ============================================================================

ALTER TABLE face DROP CONSTRAINT IF EXISTS face_cluster_status_check;

ALTER TABLE face
ADD CONSTRAINT face_cluster_status_check
CHECK (cluster_status IN ('auto', 'pending', 'manual'));

-- ============================================================================
-- 6. Remove new columns
-- ============================================================================

ALTER TABLE face DROP COLUMN IF EXISTS unassigned_since;
ALTER TABLE cluster DROP COLUMN IF EXISTS verified;
ALTER TABLE cluster DROP COLUMN IF EXISTS verified_at;
ALTER TABLE cluster DROP COLUMN IF EXISTS verified_by;

-- ============================================================================
-- 7. Remove migration record
-- ============================================================================

DELETE FROM schema_migrations WHERE version = '001';

COMMIT;

-- ============================================================================
-- Post-rollback notes:
-- ============================================================================
--
-- After running this rollback:
--
-- 1. All constraint data (must_link, cannot_link, cluster_cannot_link) is DELETED
-- 2. Faces with 'unassigned' or 'constrained' status are reset to NULL
-- 3. Cluster verification status is removed
-- 4. The schema is back to pre-migration state
--
-- To verify the rollback:
--   SELECT version FROM schema_migrations;  -- Should not include '001'
--   \d face  -- Should not have unassigned_since column
--   \d cluster  -- Should not have verified columns
