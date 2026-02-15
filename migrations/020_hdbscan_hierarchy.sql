-- Migration: Full HDBSCAN Hierarchy Support
-- Adds hdbscan_run table for persisting clusterer state and condensed tree.
-- Adds lambda_birth, persistence, hdbscan_run_id to cluster table.
-- Adds lambda_val, outlier_score to person_detection table.

-- =============================================================================
-- Part 1: Create hdbscan_run table
-- =============================================================================

CREATE TABLE IF NOT EXISTS hdbscan_run (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collection(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding_count INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    noise_count INTEGER NOT NULL,
    min_cluster_size INTEGER NOT NULL,
    min_samples INTEGER NOT NULL,
    condensed_tree JSONB NOT NULL,
    label_to_cluster_id JSONB NOT NULL,
    clusterer_state BYTEA,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Only one active run per collection
CREATE UNIQUE INDEX IF NOT EXISTS idx_hdbscan_run_active
    ON hdbscan_run(collection_id) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_hdbscan_run_collection
    ON hdbscan_run(collection_id);

COMMENT ON TABLE hdbscan_run IS 'Persisted HDBSCAN bootstrap runs with condensed tree and serialized clusterer';
COMMENT ON COLUMN hdbscan_run.condensed_tree IS 'HDBSCAN condensed tree as JSON (from condensed_tree_.to_pandas().to_dict())';
COMMENT ON COLUMN hdbscan_run.label_to_cluster_id IS 'Mapping from HDBSCAN label (int) to database cluster ID (int)';
COMMENT ON COLUMN hdbscan_run.clusterer_state IS 'Pickled HDBSCAN clusterer for approximate_predict';
COMMENT ON COLUMN hdbscan_run.is_active IS 'Only one active run per collection; previous runs kept for audit';

-- =============================================================================
-- Part 2: Add hierarchy columns to cluster
-- =============================================================================

ALTER TABLE cluster ADD COLUMN IF NOT EXISTS lambda_birth REAL;
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS persistence REAL;
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS hdbscan_run_id INTEGER REFERENCES hdbscan_run(id);

CREATE INDEX IF NOT EXISTS idx_cluster_hdbscan_run ON cluster(hdbscan_run_id)
    WHERE hdbscan_run_id IS NOT NULL;

COMMENT ON COLUMN cluster.lambda_birth IS 'Density level where cluster emerged in condensed tree (epsilon = 1/lambda_birth)';
COMMENT ON COLUMN cluster.persistence IS 'HDBSCAN cluster stability score (higher = more stable)';
COMMENT ON COLUMN cluster.hdbscan_run_id IS 'Which bootstrap run produced this cluster';

-- =============================================================================
-- Part 3: Add hierarchy columns to person_detection
-- =============================================================================

ALTER TABLE person_detection ADD COLUMN IF NOT EXISTS lambda_val REAL;
ALTER TABLE person_detection ADD COLUMN IF NOT EXISTS outlier_score REAL;

COMMENT ON COLUMN person_detection.lambda_val IS 'Point lambda value from condensed tree (density at join/leave)';
COMMENT ON COLUMN person_detection.outlier_score IS 'GLOSH outlier score (0=inlier, 1=outlier)';
