-- Migration: Add Constrained Clustering Support
-- Description: Adds must-link/cannot-link constraints, cluster verification, and unassigned pool tracking
-- Created: 2025-01-30

BEGIN;

-- ============================================================================
-- 1. Add new columns to existing tables
-- ============================================================================

-- Add unassigned_since to face table for tracking outlier pool membership
ALTER TABLE face
ADD COLUMN IF NOT EXISTS unassigned_since timestamptz DEFAULT NULL;

-- Add verification columns to cluster table
ALTER TABLE cluster
ADD COLUMN IF NOT EXISTS verified boolean DEFAULT false;

ALTER TABLE cluster
ADD COLUMN IF NOT EXISTS verified_at timestamptz DEFAULT NULL;

ALTER TABLE cluster
ADD COLUMN IF NOT EXISTS verified_by text DEFAULT NULL;

-- ============================================================================
-- 2. Update cluster_status CHECK constraint to include new values
-- ============================================================================

-- Drop the old constraint and add new one with additional values
-- Note: This requires knowing the constraint name. If it fails, the constraint may have a different name.
DO $$
BEGIN
    -- Try to drop the constraint if it exists
    ALTER TABLE face DROP CONSTRAINT IF EXISTS face_cluster_status_check;
EXCEPTION
    WHEN undefined_object THEN
        -- Constraint doesn't exist with this name, try alternatives
        NULL;
END $$;

-- Add the new constraint with all valid values
ALTER TABLE face
ADD CONSTRAINT face_cluster_status_check
CHECK (cluster_status IN ('auto', 'pending', 'manual', 'unassigned', 'constrained'));

-- ============================================================================
-- 3. Create constraint tables
-- ============================================================================

-- Must-link constraint: forces faces to be in the same cluster
CREATE TABLE IF NOT EXISTS must_link (
    id bigserial PRIMARY KEY,
    face_id_1 bigint NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    face_id_2 bigint NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    created_by text DEFAULT 'human',  -- 'human' or 'system'
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(face_id_1, face_id_2),
    CHECK (face_id_1 < face_id_2)  -- Canonical ordering to prevent duplicates
);

-- Cannot-link constraint: prevents faces from being in the same cluster
CREATE TABLE IF NOT EXISTS cannot_link (
    id bigserial PRIMARY KEY,
    face_id_1 bigint NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    face_id_2 bigint NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    created_by text DEFAULT 'human',
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(face_id_1, face_id_2),
    CHECK (face_id_1 < face_id_2)
);

-- Cluster-level cannot-link for efficiency (prevents merging)
CREATE TABLE IF NOT EXISTS cluster_cannot_link (
    id bigserial PRIMARY KEY,
    cluster_id_1 bigint NOT NULL REFERENCES cluster(id) ON DELETE CASCADE,
    cluster_id_2 bigint NOT NULL REFERENCES cluster(id) ON DELETE CASCADE,
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(cluster_id_1, cluster_id_2),
    CHECK (cluster_id_1 < cluster_id_2)
);

-- ============================================================================
-- 4. Create indexes for new columns and tables
-- ============================================================================

-- Index for finding unassigned faces
CREATE INDEX IF NOT EXISTS idx_face_unassigned
ON face(unassigned_since)
WHERE cluster_id IS NULL;

-- Index for verified clusters
CREATE INDEX IF NOT EXISTS idx_cluster_verified
ON cluster(verified)
WHERE verified = true;

-- Indexes for must_link lookups (both directions)
CREATE INDEX IF NOT EXISTS idx_must_link_face1 ON must_link(face_id_1);
CREATE INDEX IF NOT EXISTS idx_must_link_face2 ON must_link(face_id_2);

-- Indexes for cannot_link lookups (both directions)
CREATE INDEX IF NOT EXISTS idx_cannot_link_face1 ON cannot_link(face_id_1);
CREATE INDEX IF NOT EXISTS idx_cannot_link_face2 ON cannot_link(face_id_2);

-- Indexes for cluster_cannot_link lookups
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c1 ON cluster_cannot_link(cluster_id_1);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c2 ON cluster_cannot_link(cluster_id_2);

-- ============================================================================
-- 5. Create helper functions (optional but useful)
-- ============================================================================

-- Function to add a must-link constraint with canonical ordering
CREATE OR REPLACE FUNCTION add_must_link(p_face_id_1 bigint, p_face_id_2 bigint, p_created_by text DEFAULT 'human')
RETURNS bigint AS $$
DECLARE
    v_id bigint;
    v_face_1 bigint;
    v_face_2 bigint;
BEGIN
    -- Ensure canonical ordering
    IF p_face_id_1 < p_face_id_2 THEN
        v_face_1 := p_face_id_1;
        v_face_2 := p_face_id_2;
    ELSE
        v_face_1 := p_face_id_2;
        v_face_2 := p_face_id_1;
    END IF;

    INSERT INTO must_link (face_id_1, face_id_2, created_by)
    VALUES (v_face_1, v_face_2, p_created_by)
    ON CONFLICT (face_id_1, face_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to add a cannot-link constraint with canonical ordering
CREATE OR REPLACE FUNCTION add_cannot_link(p_face_id_1 bigint, p_face_id_2 bigint, p_created_by text DEFAULT 'human')
RETURNS bigint AS $$
DECLARE
    v_id bigint;
    v_face_1 bigint;
    v_face_2 bigint;
BEGIN
    -- Ensure canonical ordering
    IF p_face_id_1 < p_face_id_2 THEN
        v_face_1 := p_face_id_1;
        v_face_2 := p_face_id_2;
    ELSE
        v_face_1 := p_face_id_2;
        v_face_2 := p_face_id_1;
    END IF;

    INSERT INTO cannot_link (face_id_1, face_id_2, created_by)
    VALUES (v_face_1, v_face_2, p_created_by)
    ON CONFLICT (face_id_1, face_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check if a constraint violation exists
CREATE OR REPLACE FUNCTION check_constraint_violations()
RETURNS TABLE(constraint_id bigint, cluster_id bigint, face_id_1 bigint, face_id_2 bigint) AS $$
BEGIN
    RETURN QUERY
    SELECT cl.id, f1.cluster_id, cl.face_id_1, cl.face_id_2
    FROM cannot_link cl
    JOIN face f1 ON cl.face_id_1 = f1.id
    JOIN face f2 ON cl.face_id_2 = f2.id
    WHERE f1.cluster_id = f2.cluster_id
      AND f1.cluster_id IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 6. Migration metadata (optional - for tracking migrations)
-- ============================================================================

-- Create migrations tracking table if it doesn't exist
CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz DEFAULT NOW(),
    description text
);

-- Record this migration
INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Add constrained clustering support')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ============================================================================
-- Post-migration notes:
-- ============================================================================
--
-- After running this migration:
--
-- 1. Existing faces with cluster_status values will continue to work
-- 2. New cluster_status values 'unassigned' and 'constrained' are now valid
-- 3. All clusters start with verified=false
-- 4. Constraint tables are empty and ready for use
--
-- To verify the migration:
--   SELECT version, applied_at, description FROM schema_migrations;
--   SELECT COUNT(*) FROM must_link;
--   SELECT COUNT(*) FROM cannot_link;
--   SELECT COUNT(*) FROM cluster WHERE verified = true;
--
-- To rollback (if needed), run: migrations/001_add_constrained_clustering_rollback.sql
