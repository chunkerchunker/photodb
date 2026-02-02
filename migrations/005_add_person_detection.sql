-- Migration 005: Add person_detection table replacing face table
-- Description: Creates unified person detection table with face/body bboxes and age/gender fields
-- Created: 2025-02-01

BEGIN;

-- ============================================================================
-- 1. Create person_detection table (before dropping face to allow data migration)
-- ============================================================================

CREATE TABLE IF NOT EXISTS person_detection (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL,

    -- Face bounding box (nullable - may have body-only detection)
    face_bbox_x real,
    face_bbox_y real,
    face_bbox_width real,
    face_bbox_height real,
    face_confidence real,

    -- Body bounding box (nullable - may have face-only detection)
    body_bbox_x real,
    body_bbox_y real,
    body_bbox_width real,
    body_bbox_height real,
    body_confidence real,

    -- Age/gender estimation
    age_estimate real,
    gender char(1) CHECK (gender IN ('M', 'F', 'U')),
    gender_confidence real,
    mivolo_output jsonb,

    -- Clustering fields (migrated from face table)
    person_id bigint,
    cluster_status text CHECK (cluster_status IN ('auto', 'pending', 'manual', 'unassigned', 'constrained')) DEFAULT NULL,
    cluster_id bigint,
    cluster_confidence real DEFAULT 0,

    -- Unassigned pool tracking (migrated from face table)
    unassigned_since timestamptz DEFAULT NULL,

    -- Detector metadata
    detector_model text,
    detector_version text,

    -- Timestamps
    created_at timestamptz DEFAULT now(),

    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL,
    FOREIGN KEY (cluster_id) REFERENCES "cluster"(id) ON DELETE SET NULL
);

-- ============================================================================
-- 2. Create indexes for person_detection
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_person_detection_photo_id ON person_detection(photo_id);
CREATE INDEX IF NOT EXISTS idx_person_detection_person_id ON person_detection(person_id);
CREATE INDEX IF NOT EXISTS idx_person_detection_face_confidence ON person_detection(face_confidence);
CREATE INDEX IF NOT EXISTS idx_person_detection_cluster_status ON person_detection(cluster_status);
CREATE INDEX IF NOT EXISTS idx_person_detection_cluster_id ON person_detection(cluster_id);
CREATE INDEX IF NOT EXISTS idx_person_detection_unassigned ON person_detection(unassigned_since) WHERE cluster_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_person_detection_gender ON person_detection(gender);
CREATE INDEX IF NOT EXISTS idx_person_detection_age ON person_detection(age_estimate);

-- ============================================================================
-- 3. Migrate existing face data to person_detection
-- ============================================================================

INSERT INTO person_detection (
    id,
    photo_id,
    face_bbox_x,
    face_bbox_y,
    face_bbox_width,
    face_bbox_height,
    face_confidence,
    person_id,
    cluster_status,
    cluster_id,
    cluster_confidence,
    unassigned_since,
    detector_model,
    created_at
)
SELECT
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
    unassigned_since,
    'legacy_face_detector',
    now()
FROM face;

-- ============================================================================
-- 4. Update foreign keys in dependent tables
-- ============================================================================

-- 4a. face_embedding: rename face_id to person_detection_id
ALTER TABLE face_embedding DROP CONSTRAINT IF EXISTS face_embedding_face_id_fkey;
ALTER TABLE face_embedding RENAME COLUMN face_id TO person_detection_id;
ALTER TABLE face_embedding ADD CONSTRAINT face_embedding_person_detection_id_fkey
    FOREIGN KEY (person_detection_id) REFERENCES person_detection(id) ON DELETE CASCADE;

-- 4b. cluster: rename representative_face_id to representative_detection_id
ALTER TABLE "cluster" DROP CONSTRAINT IF EXISTS cluster_representative_face_id_fkey;
ALTER TABLE "cluster" RENAME COLUMN representative_face_id TO representative_detection_id;
ALTER TABLE "cluster" ADD CONSTRAINT cluster_representative_detection_id_fkey
    FOREIGN KEY (representative_detection_id) REFERENCES person_detection(id) ON DELETE SET NULL;

-- 4c. cluster: rename medoid_face_id to medoid_detection_id
ALTER TABLE "cluster" DROP CONSTRAINT IF EXISTS cluster_medoid_face_id_fkey;
ALTER TABLE "cluster" RENAME COLUMN medoid_face_id TO medoid_detection_id;
ALTER TABLE "cluster" ADD CONSTRAINT cluster_medoid_detection_id_fkey
    FOREIGN KEY (medoid_detection_id) REFERENCES person_detection(id) ON DELETE SET NULL;

-- 4d. must_link: rename face_id_1/2 to detection_id_1/2
ALTER TABLE must_link DROP CONSTRAINT IF EXISTS must_link_face_id_1_fkey;
ALTER TABLE must_link DROP CONSTRAINT IF EXISTS must_link_face_id_2_fkey;
ALTER TABLE must_link DROP CONSTRAINT IF EXISTS must_link_face_id_1_face_id_2_key;
ALTER TABLE must_link DROP CONSTRAINT IF EXISTS must_link_check;
ALTER TABLE must_link RENAME COLUMN face_id_1 TO detection_id_1;
ALTER TABLE must_link RENAME COLUMN face_id_2 TO detection_id_2;
ALTER TABLE must_link ADD CONSTRAINT must_link_detection_id_1_fkey
    FOREIGN KEY (detection_id_1) REFERENCES person_detection(id) ON DELETE CASCADE;
ALTER TABLE must_link ADD CONSTRAINT must_link_detection_id_2_fkey
    FOREIGN KEY (detection_id_2) REFERENCES person_detection(id) ON DELETE CASCADE;
ALTER TABLE must_link ADD CONSTRAINT must_link_detection_id_1_detection_id_2_key
    UNIQUE (detection_id_1, detection_id_2);
ALTER TABLE must_link ADD CONSTRAINT must_link_check
    CHECK (detection_id_1 < detection_id_2);

-- 4e. cannot_link: rename face_id_1/2 to detection_id_1/2
ALTER TABLE cannot_link DROP CONSTRAINT IF EXISTS cannot_link_face_id_1_fkey;
ALTER TABLE cannot_link DROP CONSTRAINT IF EXISTS cannot_link_face_id_2_fkey;
ALTER TABLE cannot_link DROP CONSTRAINT IF EXISTS cannot_link_face_id_1_face_id_2_key;
ALTER TABLE cannot_link DROP CONSTRAINT IF EXISTS cannot_link_check;
ALTER TABLE cannot_link RENAME COLUMN face_id_1 TO detection_id_1;
ALTER TABLE cannot_link RENAME COLUMN face_id_2 TO detection_id_2;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_detection_id_1_fkey
    FOREIGN KEY (detection_id_1) REFERENCES person_detection(id) ON DELETE CASCADE;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_detection_id_2_fkey
    FOREIGN KEY (detection_id_2) REFERENCES person_detection(id) ON DELETE CASCADE;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_detection_id_1_detection_id_2_key
    UNIQUE (detection_id_1, detection_id_2);
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_check
    CHECK (detection_id_1 < detection_id_2);

-- 4f. face_match_candidate: rename face_id to detection_id
ALTER TABLE face_match_candidate DROP CONSTRAINT IF EXISTS face_match_candidate_face_id_fkey;
ALTER TABLE face_match_candidate RENAME COLUMN face_id TO detection_id;
ALTER TABLE face_match_candidate ADD CONSTRAINT face_match_candidate_detection_id_fkey
    FOREIGN KEY (detection_id) REFERENCES person_detection(id) ON DELETE CASCADE;

-- ============================================================================
-- 5. Update indexes for renamed columns
-- ============================================================================

-- Drop old indexes
DROP INDEX IF EXISTS idx_must_link_face1;
DROP INDEX IF EXISTS idx_must_link_face2;
DROP INDEX IF EXISTS idx_cannot_link_face1;
DROP INDEX IF EXISTS idx_cannot_link_face2;
DROP INDEX IF EXISTS idx_face_match_candidate_face;

-- Create new indexes
CREATE INDEX IF NOT EXISTS idx_must_link_detection1 ON must_link(detection_id_1);
CREATE INDEX IF NOT EXISTS idx_must_link_detection2 ON must_link(detection_id_2);
CREATE INDEX IF NOT EXISTS idx_cannot_link_detection1 ON cannot_link(detection_id_1);
CREATE INDEX IF NOT EXISTS idx_cannot_link_detection2 ON cannot_link(detection_id_2);
CREATE INDEX IF NOT EXISTS idx_face_match_candidate_detection ON face_match_candidate(detection_id);

-- ============================================================================
-- 6. Add age/gender columns to person table
-- ============================================================================

ALTER TABLE person ADD COLUMN IF NOT EXISTS estimated_birth_year integer;
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_year_stddev real;
ALTER TABLE person ADD COLUMN IF NOT EXISTS gender char(1) CHECK (gender IN ('M', 'F', 'U'));
ALTER TABLE person ADD COLUMN IF NOT EXISTS gender_confidence real;
ALTER TABLE person ADD COLUMN IF NOT EXISTS age_gender_sample_count integer DEFAULT 0;
ALTER TABLE person ADD COLUMN IF NOT EXISTS age_gender_updated_at timestamptz;

-- ============================================================================
-- 7. Update helper functions for renamed columns
-- ============================================================================

-- Drop existing functions first (PostgreSQL doesn't allow renaming parameters in place)
DROP FUNCTION IF EXISTS add_must_link(bigint, bigint, text);
DROP FUNCTION IF EXISTS add_cannot_link(bigint, bigint, text);
DROP FUNCTION IF EXISTS check_constraint_violations();

-- Function to add a must-link constraint with canonical ordering
CREATE OR REPLACE FUNCTION add_must_link(p_detection_id_1 bigint, p_detection_id_2 bigint, p_created_by text DEFAULT 'human')
RETURNS bigint AS $$
DECLARE
    v_id bigint;
    v_detection_1 bigint;
    v_detection_2 bigint;
BEGIN
    -- Ensure canonical ordering
    IF p_detection_id_1 < p_detection_id_2 THEN
        v_detection_1 := p_detection_id_1;
        v_detection_2 := p_detection_id_2;
    ELSE
        v_detection_1 := p_detection_id_2;
        v_detection_2 := p_detection_id_1;
    END IF;

    INSERT INTO must_link (detection_id_1, detection_id_2, created_by)
    VALUES (v_detection_1, v_detection_2, p_created_by)
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to add a cannot-link constraint with canonical ordering
CREATE OR REPLACE FUNCTION add_cannot_link(p_detection_id_1 bigint, p_detection_id_2 bigint, p_created_by text DEFAULT 'human')
RETURNS bigint AS $$
DECLARE
    v_id bigint;
    v_detection_1 bigint;
    v_detection_2 bigint;
BEGIN
    -- Ensure canonical ordering
    IF p_detection_id_1 < p_detection_id_2 THEN
        v_detection_1 := p_detection_id_1;
        v_detection_2 := p_detection_id_2;
    ELSE
        v_detection_1 := p_detection_id_2;
        v_detection_2 := p_detection_id_1;
    END IF;

    INSERT INTO cannot_link (detection_id_1, detection_id_2, created_by)
    VALUES (v_detection_1, v_detection_2, p_created_by)
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check if a constraint violation exists (updated for person_detection)
CREATE OR REPLACE FUNCTION check_constraint_violations()
RETURNS TABLE(constraint_id bigint, cluster_id bigint, detection_id_1 bigint, detection_id_2 bigint) AS $$
BEGIN
    RETURN QUERY
    SELECT cl.id, pd1.cluster_id, cl.detection_id_1, cl.detection_id_2
    FROM cannot_link cl
    JOIN person_detection pd1 ON cl.detection_id_1 = pd1.id
    JOIN person_detection pd2 ON cl.detection_id_2 = pd2.id
    WHERE pd1.cluster_id = pd2.cluster_id
      AND pd1.cluster_id IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 8. Reset person_detection sequence to continue from max existing id
-- ============================================================================

SELECT setval('person_detection_id_seq', COALESCE((SELECT MAX(id) FROM person_detection), 0) + 1, false);

-- ============================================================================
-- 9. Drop old face table and its indexes
-- ============================================================================

DROP INDEX IF EXISTS idx_face_photo_id;
DROP INDEX IF EXISTS idx_face_person_id;
DROP INDEX IF EXISTS idx_face_confidence;
DROP INDEX IF EXISTS idx_face_cluster_status;
DROP INDEX IF EXISTS idx_face_cluster_id;
DROP INDEX IF EXISTS idx_face_unassigned;

DROP TABLE IF EXISTS face CASCADE;

-- ============================================================================
-- 10. Record migration
-- ============================================================================

INSERT INTO schema_migrations (version, description)
VALUES ('005', 'Add person_detection table replacing face table with age/gender support')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ============================================================================
-- Post-migration notes:
-- ============================================================================
--
-- After running this migration:
--
-- 1. The `face` table has been replaced by `person_detection`
-- 2. All existing face data has been migrated with face_* bbox fields populated
-- 3. New body_* bbox fields are available for body detection
-- 4. Age/gender fields (age_estimate, gender, gender_confidence, mivolo_output) are ready
-- 5. The person table now has aggregated age/gender columns
-- 6. All foreign key references have been updated:
--    - face_embedding.face_id -> face_embedding.person_detection_id
--    - cluster.representative_face_id -> cluster.representative_detection_id
--    - cluster.medoid_face_id -> cluster.medoid_detection_id
--    - must_link.face_id_1/2 -> must_link.detection_id_1/2
--    - cannot_link.face_id_1/2 -> cannot_link.detection_id_1/2
--    - face_match_candidate.face_id -> face_match_candidate.detection_id
--
-- To verify the migration:
--   SELECT version, applied_at, description FROM schema_migrations WHERE version = '005';
--   SELECT COUNT(*) FROM person_detection;
--   \d person_detection
--   \d person
--
-- To rollback (if needed), run: migrations/005_add_person_detection_rollback.sql
