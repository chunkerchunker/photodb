-- Migration: HDBSCAN Clustering Support
-- This migration adds support for the hybrid HDBSCAN → Incremental DBSCAN clustering approach.
--
-- Changes:
-- 1. Add columns for HDBSCAN core points and per-cluster epsilon
-- 2. Drop must_link table (replaced by direct person_id assignment)
-- 3. Add hdbscan/hdbscan_core to allowed cluster_status values

-- =============================================================================
-- Part 1: Add HDBSCAN columns
-- =============================================================================

-- Add is_core column to person_detection to track density-based core points
ALTER TABLE person_detection ADD COLUMN IF NOT EXISTS is_core BOOLEAN DEFAULT FALSE;

-- Add per-cluster epsilon (distance threshold derived from core point distances)
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS epsilon REAL;

-- Track core point count for each cluster
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS core_count INTEGER DEFAULT 0;

-- Index for efficient core point queries
CREATE INDEX IF NOT EXISTS idx_person_detection_is_core
ON person_detection(is_core) WHERE is_core = true;

-- Index for efficient epsilon-ball queries on clustered detections
CREATE INDEX IF NOT EXISTS idx_person_detection_cluster_core
ON person_detection(cluster_id, is_core) WHERE cluster_id IS NOT NULL;

COMMENT ON COLUMN person_detection.is_core IS 'True if this detection is a core point (high density) from HDBSCAN clustering';
COMMENT ON COLUMN cluster.epsilon IS 'Per-cluster distance threshold derived from 90th percentile of core point distances';
COMMENT ON COLUMN cluster.core_count IS 'Number of core points in this cluster';

-- =============================================================================
-- Part 2: Drop must_link table
-- =============================================================================
-- The must_link constraint system has been replaced by direct person_id assignment.
-- Clusters are now linked to the same person via the cluster.person_id foreign key.

DROP TABLE IF EXISTS must_link;

-- =============================================================================
-- Part 3: Add HDBSCAN cluster status values
-- =============================================================================
-- Adds 'hdbscan' and 'hdbscan_core' to the allowed cluster_status values.
-- These indicate detections assigned during HDBSCAN bootstrap clustering.

ALTER TABLE person_detection DROP CONSTRAINT IF EXISTS person_detection_cluster_status_check;

ALTER TABLE person_detection ADD CONSTRAINT person_detection_cluster_status_check
    CHECK (cluster_status = ANY (ARRAY[
        'auto'::text,
        'pending'::text,
        'manual'::text,
        'unassigned'::text,
        'constrained'::text,
        'hdbscan'::text,
        'hdbscan_core'::text
    ]));

-- =============================================================================
-- Part 4: Add person representative detection
-- =============================================================================
-- Each person can have a representative detection (face photo) for display.
-- This is set from the first cluster's representative when the person is created,
-- and can be updated by the user.

ALTER TABLE person ADD COLUMN IF NOT EXISTS representative_detection_id BIGINT
    REFERENCES person_detection(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_person_representative
ON person(representative_detection_id) WHERE representative_detection_id IS NOT NULL;

COMMENT ON COLUMN person.representative_detection_id IS 'Representative face detection for this person, used for display in person listings';
