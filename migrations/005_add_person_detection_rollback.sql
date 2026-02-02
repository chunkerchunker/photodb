-- Rollback Migration 005: Restore face table from person_detection
-- Description: Reverses the changes from 005_add_person_detection.sql
-- WARNING: This will lose any body bbox data and age/gender data not present in original face table
-- Created: 2025-02-01

BEGIN;

-- ============================================================================
-- 1. Recreate the face table
-- ============================================================================

CREATE TABLE IF NOT EXISTS face (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL,
    -- Bounding box coordinates (normalized 0.0-1.0 or pixel values)
    bbox_x real NOT NULL,
    bbox_y real NOT NULL,
    bbox_width real NOT NULL,
    bbox_height real NOT NULL,
    confidence DECIMAL(3, 2) NOT NULL DEFAULT 0,
    -- Detection metadata
    person_id bigint,
    -- Clustering fields
    cluster_status text CHECK (cluster_status IN ('auto', 'pending', 'manual', 'unassigned', 'constrained')) DEFAULT NULL,
    cluster_id bigint,
    cluster_confidence DECIMAL(3, 2) DEFAULT 0,
    -- Unassigned pool tracking
    unassigned_since timestamptz DEFAULT NULL,
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL,
    FOREIGN KEY (cluster_id) REFERENCES "cluster"(id) ON DELETE SET NULL
);

-- ============================================================================
-- 2. Migrate data back from person_detection to face
-- ============================================================================

INSERT INTO face (
    id,
    photo_id,
    bbox_x,
    bbox_y,
    bbox_width,
    bbox_height,
    confidence,
    person_id,
    cluster_status,
    cluster_id,
    cluster_confidence,
    unassigned_since
)
SELECT
    id,
    photo_id,
    COALESCE(face_bbox_x, 0),
    COALESCE(face_bbox_y, 0),
    COALESCE(face_bbox_width, 0),
    COALESCE(face_bbox_height, 0),
    COALESCE(face_confidence, 0),
    person_id,
    cluster_status,
    cluster_id,
    cluster_confidence,
    unassigned_since
FROM person_detection
WHERE face_bbox_x IS NOT NULL;  -- Only migrate records that had face data

-- Reset face sequence
SELECT setval('face_id_seq', COALESCE((SELECT MAX(id) FROM face), 0) + 1, false);

-- ============================================================================
-- 3. Restore face table indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_face_photo_id ON face(photo_id);
CREATE INDEX IF NOT EXISTS idx_face_person_id ON face(person_id);
CREATE INDEX IF NOT EXISTS idx_face_confidence ON face(confidence);
CREATE INDEX IF NOT EXISTS idx_face_cluster_status ON face(cluster_status);
CREATE INDEX IF NOT EXISTS idx_face_cluster_id ON face(cluster_id);
CREATE INDEX IF NOT EXISTS idx_face_unassigned ON face(unassigned_since) WHERE cluster_id IS NULL;

-- ============================================================================
-- 4. Restore foreign keys in dependent tables
-- ============================================================================

-- 4a. face_embedding: rename person_detection_id back to face_id
ALTER TABLE face_embedding DROP CONSTRAINT IF EXISTS face_embedding_person_detection_id_fkey;
ALTER TABLE face_embedding RENAME COLUMN person_detection_id TO face_id;
ALTER TABLE face_embedding ADD CONSTRAINT face_embedding_face_id_fkey
    FOREIGN KEY (face_id) REFERENCES face(id) ON DELETE CASCADE;

-- 4b. cluster: rename representative_detection_id back to representative_face_id
ALTER TABLE "cluster" DROP CONSTRAINT IF EXISTS cluster_representative_detection_id_fkey;
ALTER TABLE "cluster" RENAME COLUMN representative_detection_id TO representative_face_id;
ALTER TABLE "cluster" ADD CONSTRAINT cluster_representative_face_id_fkey
    FOREIGN KEY (representative_face_id) REFERENCES face(id) ON DELETE SET NULL;

-- 4c. cluster: rename medoid_detection_id back to medoid_face_id
ALTER TABLE "cluster" DROP CONSTRAINT IF EXISTS cluster_medoid_detection_id_fkey;
ALTER TABLE "cluster" RENAME COLUMN medoid_detection_id TO medoid_face_id;
ALTER TABLE "cluster" ADD CONSTRAINT cluster_medoid_face_id_fkey
    FOREIGN KEY (medoid_face_id) REFERENCES face(id) ON DELETE SET NULL;

-- 4d. must_link: rename detection_id_1/2 back to face_id_1/2
ALTER TABLE must_link DROP CONSTRAINT IF EXISTS must_link_detection_id_1_fkey;
ALTER TABLE must_link DROP CONSTRAINT IF EXISTS must_link_detection_id_2_fkey;
ALTER TABLE must_link DROP CONSTRAINT IF EXISTS must_link_detection_id_1_detection_id_2_key;
ALTER TABLE must_link DROP CONSTRAINT IF EXISTS must_link_check;
ALTER TABLE must_link RENAME COLUMN detection_id_1 TO face_id_1;
ALTER TABLE must_link RENAME COLUMN detection_id_2 TO face_id_2;
ALTER TABLE must_link ADD CONSTRAINT must_link_face_id_1_fkey
    FOREIGN KEY (face_id_1) REFERENCES face(id) ON DELETE CASCADE;
ALTER TABLE must_link ADD CONSTRAINT must_link_face_id_2_fkey
    FOREIGN KEY (face_id_2) REFERENCES face(id) ON DELETE CASCADE;
ALTER TABLE must_link ADD CONSTRAINT must_link_face_id_1_face_id_2_key
    UNIQUE (face_id_1, face_id_2);
ALTER TABLE must_link ADD CONSTRAINT must_link_check
    CHECK (face_id_1 < face_id_2);

-- 4e. cannot_link: rename detection_id_1/2 back to face_id_1/2
ALTER TABLE cannot_link DROP CONSTRAINT IF EXISTS cannot_link_detection_id_1_fkey;
ALTER TABLE cannot_link DROP CONSTRAINT IF EXISTS cannot_link_detection_id_2_fkey;
ALTER TABLE cannot_link DROP CONSTRAINT IF EXISTS cannot_link_detection_id_1_detection_id_2_key;
ALTER TABLE cannot_link DROP CONSTRAINT IF EXISTS cannot_link_check;
ALTER TABLE cannot_link RENAME COLUMN detection_id_1 TO face_id_1;
ALTER TABLE cannot_link RENAME COLUMN detection_id_2 TO face_id_2;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_face_id_1_fkey
    FOREIGN KEY (face_id_1) REFERENCES face(id) ON DELETE CASCADE;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_face_id_2_fkey
    FOREIGN KEY (face_id_2) REFERENCES face(id) ON DELETE CASCADE;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_face_id_1_face_id_2_key
    UNIQUE (face_id_1, face_id_2);
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_check
    CHECK (face_id_1 < face_id_2);

-- 4f. face_match_candidate: rename detection_id back to face_id
ALTER TABLE face_match_candidate DROP CONSTRAINT IF EXISTS face_match_candidate_detection_id_fkey;
ALTER TABLE face_match_candidate RENAME COLUMN detection_id TO face_id;
ALTER TABLE face_match_candidate ADD CONSTRAINT face_match_candidate_face_id_fkey
    FOREIGN KEY (face_id) REFERENCES face(id) ON DELETE CASCADE;

-- ============================================================================
-- 5. Restore indexes for renamed columns
-- ============================================================================

-- Drop new indexes
DROP INDEX IF EXISTS idx_must_link_detection1;
DROP INDEX IF EXISTS idx_must_link_detection2;
DROP INDEX IF EXISTS idx_cannot_link_detection1;
DROP INDEX IF EXISTS idx_cannot_link_detection2;
DROP INDEX IF EXISTS idx_face_match_candidate_detection;

-- Create old indexes
CREATE INDEX IF NOT EXISTS idx_must_link_face1 ON must_link(face_id_1);
CREATE INDEX IF NOT EXISTS idx_must_link_face2 ON must_link(face_id_2);
CREATE INDEX IF NOT EXISTS idx_cannot_link_face1 ON cannot_link(face_id_1);
CREATE INDEX IF NOT EXISTS idx_cannot_link_face2 ON cannot_link(face_id_2);
CREATE INDEX IF NOT EXISTS idx_face_match_candidate_face ON face_match_candidate(face_id);

-- ============================================================================
-- 6. Remove age/gender columns from person table
-- ============================================================================

ALTER TABLE person DROP COLUMN IF EXISTS estimated_birth_year;
ALTER TABLE person DROP COLUMN IF EXISTS birth_year_stddev;
ALTER TABLE person DROP COLUMN IF EXISTS gender;
ALTER TABLE person DROP COLUMN IF EXISTS gender_confidence;
ALTER TABLE person DROP COLUMN IF EXISTS age_gender_sample_count;
ALTER TABLE person DROP COLUMN IF EXISTS age_gender_updated_at;

-- ============================================================================
-- 7. Restore helper functions to use face_id
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
-- 8. Drop person_detection table and its indexes
-- ============================================================================

DROP INDEX IF EXISTS idx_person_detection_photo_id;
DROP INDEX IF EXISTS idx_person_detection_person_id;
DROP INDEX IF EXISTS idx_person_detection_face_confidence;
DROP INDEX IF EXISTS idx_person_detection_cluster_status;
DROP INDEX IF EXISTS idx_person_detection_cluster_id;
DROP INDEX IF EXISTS idx_person_detection_unassigned;
DROP INDEX IF EXISTS idx_person_detection_gender;
DROP INDEX IF EXISTS idx_person_detection_age;

DROP TABLE IF EXISTS person_detection CASCADE;

-- ============================================================================
-- 9. Remove migration record
-- ============================================================================

DELETE FROM schema_migrations WHERE version = '005';

COMMIT;

-- ============================================================================
-- Post-rollback notes:
-- ============================================================================
--
-- After running this rollback:
--
-- 1. The `person_detection` table has been replaced by `face` table
-- 2. Only records with face_bbox data have been migrated back
-- 3. Body-only detections and age/gender data have been LOST
-- 4. The person table no longer has age/gender columns
-- 5. All foreign key references have been restored to original names
--
-- WARNING: Any new person_detection records created after the original
-- migration that have body-only data (no face bbox) will be LOST.
--
-- To verify the rollback:
--   SELECT version FROM schema_migrations;  -- Should not include '005'
--   SELECT COUNT(*) FROM face;
--   \d face
--   \d person
