-- Reset Clustering Data
-- Description: Clears all clustering-related data while preserving face detections and embeddings
-- Use this to re-run clustering from scratch
--
-- PRESERVED:
--   - person_detection records (id, photo_id, bbox_*, confidence)
--   - face_embedding records
--   - processing_status for 'detection' stage
--
-- CLEARED:
--   - person_detection clustering fields (cluster_id, cluster_status, cluster_confidence, unassigned_since)
--   - cluster table (all clusters deleted)
--   - cannot_link constraints
--   - face_match_candidate records (cascaded from cluster delete)
--   - cluster_cannot_link constraints (cascaded from cluster delete)
--   - processing_status for 'clustering' stage

BEGIN;

-- ============================================================================
-- 1. Show current state (for logging)
-- ============================================================================

DO $$
DECLARE
    v_detections int;
    v_clustered_detections int;
    v_clusters int;
    v_cannot_links int;
    v_candidates int;
BEGIN
    SELECT COUNT(*) INTO v_detections FROM person_detection;
    SELECT COUNT(*) INTO v_clustered_detections FROM person_detection WHERE cluster_id IS NOT NULL;
    SELECT COUNT(*) INTO v_clusters FROM cluster;
    SELECT COUNT(*) INTO v_cannot_links FROM cannot_link;
    SELECT COUNT(*) INTO v_candidates FROM face_match_candidate;

    RAISE NOTICE '=== Current State ===';
    RAISE NOTICE 'Total detections: %', v_detections;
    RAISE NOTICE 'Clustered detections: %', v_clustered_detections;
    RAISE NOTICE 'Clusters: %', v_clusters;
    RAISE NOTICE 'Cannot-link constraints: %', v_cannot_links;
    RAISE NOTICE 'Match candidates: %', v_candidates;
    RAISE NOTICE '=====================';
END $$;

-- ============================================================================
-- 2. Clear person_detection clustering assignments
-- ============================================================================

UPDATE person_detection
SET cluster_id = NULL,
    cluster_status = NULL,
    cluster_confidence = 0,
    unassigned_since = NULL,
    is_core = false;

-- ============================================================================
-- 3. Clear cannot_link constraints (no FK to cluster, must delete explicitly)
-- ============================================================================

DELETE FROM cannot_link;

-- ============================================================================
-- 4. Clear clusters (cascades to face_match_candidate and cluster_cannot_link)
-- ============================================================================

DELETE FROM cluster;

-- ============================================================================
-- 5. Reset clustering processing status
-- ============================================================================

DELETE FROM processing_status WHERE stage = 'clustering';

-- ============================================================================
-- 6. Reset cluster ID sequence
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
-- 7. Show final state
-- ============================================================================

DO $$
DECLARE
    v_detections int;
    v_embeddings int;
    v_clusters int;
BEGIN
    SELECT COUNT(*) INTO v_detections FROM person_detection;
    SELECT COUNT(*) INTO v_embeddings FROM face_embedding;
    SELECT COUNT(*) INTO v_clusters FROM cluster;

    RAISE NOTICE '=== After Reset ===';
    RAISE NOTICE 'Detections preserved: %', v_detections;
    RAISE NOTICE 'Embeddings preserved: %', v_embeddings;
    RAISE NOTICE 'Clusters remaining: % (should be 0)', v_clusters;
    RAISE NOTICE '===================';
    RAISE NOTICE 'Ready for re-clustering with: uv run python scripts/migrate_to_hdbscan.py';
END $$;

COMMIT;
