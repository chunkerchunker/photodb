BEGIN;

-- Create user/collection tables
CREATE TABLE IF NOT EXISTS app_user(
    id bigserial PRIMARY KEY,
    username text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    first_name text NOT NULL,
    last_name text NOT NULL,
    default_collection_id bigint,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS collection(
    id bigserial PRIMARY KEY,
    owner_user_id bigint NOT NULL,
    name text NOT NULL,
    created_at timestamptz DEFAULT now(),
    FOREIGN KEY (owner_user_id) REFERENCES app_user(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS collection_member(
    collection_id bigint NOT NULL,
    user_id bigint NOT NULL,
    created_at timestamptz DEFAULT now(),
    PRIMARY KEY (collection_id, user_id),
    FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES app_user(id) ON DELETE CASCADE
);

ALTER TABLE app_user ADD COLUMN IF NOT EXISTS default_collection_id bigint;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'app_user_default_collection_fk') THEN
        ALTER TABLE app_user
            ADD CONSTRAINT app_user_default_collection_fk
            FOREIGN KEY (default_collection_id) REFERENCES collection(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Add collection_id columns
ALTER TABLE photo ADD COLUMN IF NOT EXISTS collection_id bigint;
ALTER TABLE metadata ADD COLUMN IF NOT EXISTS collection_id bigint;
ALTER TABLE person ADD COLUMN IF NOT EXISTS collection_id bigint;
ALTER TABLE person_detection ADD COLUMN IF NOT EXISTS collection_id bigint;
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS collection_id bigint;
ALTER TABLE face_match_candidate ADD COLUMN IF NOT EXISTS collection_id bigint;
ALTER TABLE must_link ADD COLUMN IF NOT EXISTS collection_id bigint;
ALTER TABLE cannot_link ADD COLUMN IF NOT EXISTS collection_id bigint;
ALTER TABLE cluster_cannot_link ADD COLUMN IF NOT EXISTS collection_id bigint;

-- Seed default user and collection if none exist
INSERT INTO app_user (username, password_hash, first_name, last_name)
SELECT 'default', 'changeme', 'Default', 'User'
WHERE NOT EXISTS (SELECT 1 FROM app_user);

INSERT INTO collection (owner_user_id, name)
SELECT u.id, 'Default Collection'
FROM app_user u
WHERE u.username = 'default'
  AND NOT EXISTS (SELECT 1 FROM collection);

UPDATE app_user u
SET default_collection_id = c.id
FROM collection c
WHERE u.username = 'default'
  AND u.default_collection_id IS NULL;

INSERT INTO collection_member (collection_id, user_id)
SELECT c.id, u.id
FROM collection c
JOIN app_user u ON u.username = 'default'
WHERE NOT EXISTS (
    SELECT 1 FROM collection_member cm
    WHERE cm.collection_id = c.id AND cm.user_id = u.id
);

-- Backfill collection_id values
WITH default_collection AS (
    SELECT id FROM collection ORDER BY id LIMIT 1
)
UPDATE photo p
SET collection_id = (SELECT id FROM default_collection)
WHERE p.collection_id IS NULL;

UPDATE metadata m
SET collection_id = p.collection_id
FROM photo p
WHERE m.photo_id = p.id AND m.collection_id IS NULL;

UPDATE person_detection pd
SET collection_id = p.collection_id
FROM photo p
WHERE pd.photo_id = p.id AND pd.collection_id IS NULL;

UPDATE cluster c
SET collection_id = sub.collection_id
FROM (
    SELECT pd.cluster_id, MIN(pd.collection_id) AS collection_id
    FROM person_detection pd
    WHERE pd.cluster_id IS NOT NULL
    GROUP BY pd.cluster_id
) sub
WHERE c.id = sub.cluster_id AND c.collection_id IS NULL;

WITH default_collection AS (
    SELECT id FROM collection ORDER BY id LIMIT 1
)
UPDATE cluster c
SET collection_id = (SELECT id FROM default_collection)
WHERE c.collection_id IS NULL;

UPDATE person p
SET collection_id = sub.collection_id
FROM (
    SELECT c.person_id, MIN(c.collection_id) AS collection_id
    FROM cluster c
    WHERE c.person_id IS NOT NULL
    GROUP BY c.person_id
) sub
WHERE p.id = sub.person_id AND p.collection_id IS NULL;

WITH default_collection AS (
    SELECT id FROM collection ORDER BY id LIMIT 1
)
UPDATE person p
SET collection_id = (SELECT id FROM default_collection)
WHERE p.collection_id IS NULL;

UPDATE face_match_candidate fmc
SET collection_id = pd.collection_id
FROM person_detection pd
WHERE fmc.detection_id = pd.id AND fmc.collection_id IS NULL;

UPDATE must_link ml
SET collection_id = pd.collection_id
FROM person_detection pd
WHERE ml.detection_id_1 = pd.id AND ml.collection_id IS NULL;

UPDATE cannot_link cl
SET collection_id = pd.collection_id
FROM person_detection pd
WHERE cl.detection_id_1 = pd.id AND cl.collection_id IS NULL;

UPDATE cluster_cannot_link ccl
SET collection_id = c.collection_id
FROM cluster c
WHERE ccl.cluster_id_1 = c.id AND ccl.collection_id IS NULL;

WITH default_collection AS (
    SELECT id FROM collection ORDER BY id LIMIT 1
)
UPDATE face_match_candidate fmc
SET collection_id = (SELECT id FROM default_collection)
WHERE fmc.collection_id IS NULL;

WITH default_collection AS (
    SELECT id FROM collection ORDER BY id LIMIT 1
)
UPDATE must_link ml
SET collection_id = (SELECT id FROM default_collection)
WHERE ml.collection_id IS NULL;

WITH default_collection AS (
    SELECT id FROM collection ORDER BY id LIMIT 1
)
UPDATE cannot_link cl
SET collection_id = (SELECT id FROM default_collection)
WHERE cl.collection_id IS NULL;

WITH default_collection AS (
    SELECT id FROM collection ORDER BY id LIMIT 1
)
UPDATE cluster_cannot_link ccl
SET collection_id = (SELECT id FROM default_collection)
WHERE ccl.collection_id IS NULL;

-- Add foreign keys for new collection_id columns
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'photo_collection_fk') THEN
        ALTER TABLE photo
            ADD CONSTRAINT photo_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'metadata_collection_fk') THEN
        ALTER TABLE metadata
            ADD CONSTRAINT metadata_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'person_collection_fk') THEN
        ALTER TABLE person
            ADD CONSTRAINT person_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'person_detection_collection_fk') THEN
        ALTER TABLE person_detection
            ADD CONSTRAINT person_detection_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'cluster_collection_fk') THEN
        ALTER TABLE cluster
            ADD CONSTRAINT cluster_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'face_match_candidate_collection_fk') THEN
        ALTER TABLE face_match_candidate
            ADD CONSTRAINT face_match_candidate_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'must_link_collection_fk') THEN
        ALTER TABLE must_link
            ADD CONSTRAINT must_link_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'cannot_link_collection_fk') THEN
        ALTER TABLE cannot_link
            ADD CONSTRAINT cannot_link_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'cluster_cannot_link_collection_fk') THEN
        ALTER TABLE cluster_cannot_link
            ADD CONSTRAINT cluster_cannot_link_collection_fk
            FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE;
    END IF;
END $$;

-- Enforce NOT NULL after backfill
ALTER TABLE photo ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE metadata ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE person ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE person_detection ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE cluster ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE face_match_candidate ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE must_link ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE cannot_link ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE cluster_cannot_link ALTER COLUMN collection_id SET NOT NULL;

-- Replace uniqueness on photo filename/normalized_path with collection-scoped uniqueness
ALTER TABLE photo DROP CONSTRAINT IF EXISTS photo_filename_key;
ALTER TABLE photo DROP CONSTRAINT IF EXISTS photo_normalized_path_key;

CREATE UNIQUE INDEX IF NOT EXISTS idx_photo_collection_filename_unique
    ON photo(collection_id, filename);
CREATE UNIQUE INDEX IF NOT EXISTS idx_photo_collection_normalized_path_unique
    ON photo(collection_id, normalized_path);

-- Indexes for collection-scoped performance
CREATE INDEX IF NOT EXISTS idx_photo_collection_id ON photo(collection_id, id);
CREATE INDEX IF NOT EXISTS idx_metadata_collection_captured_at ON metadata(collection_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_metadata_collection_year_month
    ON metadata (collection_id, EXTRACT(YEAR FROM captured_at AT TIME ZONE 'UTC'), EXTRACT(MONTH FROM captured_at AT TIME ZONE 'UTC'));
CREATE INDEX IF NOT EXISTS idx_metadata_collection_location ON metadata(collection_id, latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_person_detection_collection_id ON person_detection(collection_id);
CREATE INDEX IF NOT EXISTS idx_person_collection_first_name ON person(collection_id, first_name);
CREATE INDEX IF NOT EXISTS idx_person_collection_last_name ON person(collection_id, last_name);
CREATE INDEX IF NOT EXISTS idx_cluster_collection_visible
    ON cluster(collection_id, face_count DESC, id)
    WHERE face_count > 0 AND (hidden = false OR hidden IS NULL);
CREATE INDEX IF NOT EXISTS idx_cluster_collection_hidden_listing
    ON cluster(collection_id, face_count DESC, id)
    WHERE face_count > 0 AND hidden = true;
CREATE INDEX IF NOT EXISTS idx_face_match_candidate_collection ON face_match_candidate(collection_id);
CREATE INDEX IF NOT EXISTS idx_must_link_collection ON must_link(collection_id);
CREATE INDEX IF NOT EXISTS idx_cannot_link_collection ON cannot_link(collection_id);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_collection ON cluster_cannot_link(collection_id);

-- Update constraint helper functions to include collection_id
CREATE OR REPLACE FUNCTION add_must_link(
    p_detection_id_1 bigint,
    p_detection_id_2 bigint,
    p_collection_id bigint,
    p_created_by text DEFAULT 'human'
) RETURNS bigint AS $$
DECLARE
    v_id bigint;
    v_detection_1 bigint;
    v_detection_2 bigint;
BEGIN
    IF p_detection_id_1 < p_detection_id_2 THEN
        v_detection_1 := p_detection_id_1;
        v_detection_2 := p_detection_id_2;
    ELSE
        v_detection_1 := p_detection_id_2;
        v_detection_2 := p_detection_id_1;
    END IF;

    INSERT INTO must_link (detection_id_1, detection_id_2, collection_id, created_by)
    VALUES (v_detection_1, v_detection_2, p_collection_id, p_created_by)
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION add_cannot_link(
    p_detection_id_1 bigint,
    p_detection_id_2 bigint,
    p_collection_id bigint,
    p_created_by text DEFAULT 'human'
) RETURNS bigint AS $$
DECLARE
    v_id bigint;
    v_detection_1 bigint;
    v_detection_2 bigint;
BEGIN
    IF p_detection_id_1 < p_detection_id_2 THEN
        v_detection_1 := p_detection_id_1;
        v_detection_2 := p_detection_id_2;
    ELSE
        v_detection_1 := p_detection_id_2;
        v_detection_2 := p_detection_id_1;
    END IF;

    INSERT INTO cannot_link (detection_id_1, detection_id_2, collection_id, created_by)
    VALUES (v_detection_1, v_detection_2, p_collection_id, p_created_by)
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

INSERT INTO schema_migrations (version, description)
VALUES ('009', 'Add users and collections with collection-scoped data')
ON CONFLICT (version) DO NOTHING;

COMMIT;
