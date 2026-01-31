-- Reset Clustering Data
-- Description: Clears all clustering-related data while preserving face detections and embeddings
-- Use this to re-run clustering from scratch
--
-- PRESERVED:
--   - face records (id, photo_id, bbox_*, confidence)
--   - face_embedding records
--   - processing_status for 'faces' stage
--
-- CLEARED:
--   - face clustering fields (cluster_id, cluster_status, cluster_confidence, unassigned_since)
--   - cluster table (all clusters deleted)
--   - must_link, cannot_link, cluster_cannot_link constraints
--   - face_match_candidate records
--   - processing_status for 'clustering' stage

BEGIN;

-- ============================================================================
-- 1. Show current state (for logging)
-- ============================================================================

DO $$
DECLARE
    v_faces int;
    v_clustered_faces int;
    v_clusters int;
    v_must_links int;
    v_cannot_links int;
    v_candidates int;
BEGIN
    SELECT COUNT(*) INTO v_faces FROM face;
    SELECT COUNT(*) INTO v_clustered_faces FROM face WHERE cluster_id IS NOT NULL;
    SELECT COUNT(*) INTO v_clusters FROM cluster;
    SELECT COUNT(*) INTO v_must_links FROM must_link;
    SELECT COUNT(*) INTO v_cannot_links FROM cannot_link;
    SELECT COUNT(*) INTO v_candidates FROM face_match_candidate;

    RAISE NOTICE '=== Current State ===';
    RAISE NOTICE 'Total faces: %', v_faces;
    RAISE NOTICE 'Clustered faces: %', v_clustered_faces;
    RAISE NOTICE 'Clusters: %', v_clusters;
    RAISE NOTICE 'Must-link constraints: %', v_must_links;
    RAISE NOTICE 'Cannot-link constraints: %', v_cannot_links;
    RAISE NOTICE 'Match candidates: %', v_candidates;
    RAISE NOTICE '=====================';
END $$;

-- ============================================================================
-- 2. Clear face clustering assignments
-- ============================================================================

UPDATE face
SET cluster_id = NULL,
    cluster_status = NULL,
    cluster_confidence = 0,
    unassigned_since = NULL,
    person_id = NULL;

-- ============================================================================
-- 3. Clear face match candidates
-- ============================================================================

DELETE FROM face_match_candidate;

-- ============================================================================
-- 4. Clear constraint tables
-- ============================================================================

DELETE FROM cluster_cannot_link;
DELETE FROM cannot_link;
DELETE FROM must_link;

-- ============================================================================
-- 5. Clear clusters
-- ============================================================================

DELETE FROM cluster;

-- ============================================================================
-- 6. Reset clustering processing status
-- ============================================================================

DELETE FROM processing_status WHERE stage = 'clustering';

-- ============================================================================
-- 7. Reset cluster ID sequence
-- ============================================================================

-- Find the sequence name and reset it
DO $$
DECLARE
    seq_name text;
BEGIN
    SELECT pg_get_serial_sequence('cluster', 'id') INTO seq_name;
    IF seq_name IS NOT NULL THEN
        EXECUTE 'ALTER SEQUENCE ' || seq_name || ' RESTART WITH 1';
        RAISE NOTICE 'Reset cluster ID sequence to 1';
    END IF;
END $$;

-- ============================================================================
-- 8. Show final state
-- ============================================================================

DO $$
DECLARE
    v_faces int;
    v_embeddings int;
    v_clusters int;
BEGIN
    SELECT COUNT(*) INTO v_faces FROM face;
    SELECT COUNT(*) INTO v_embeddings FROM face_embedding;
    SELECT COUNT(*) INTO v_clusters FROM cluster;

    RAISE NOTICE '=== After Reset ===';
    RAISE NOTICE 'Faces preserved: %', v_faces;
    RAISE NOTICE 'Embeddings preserved: %', v_embeddings;
    RAISE NOTICE 'Clusters remaining: % (should be 0)', v_clusters;
    RAISE NOTICE '===================';
    RAISE NOTICE 'Ready for re-clustering with: uv run process-local /path --stage clustering';
END $$;

COMMIT;
