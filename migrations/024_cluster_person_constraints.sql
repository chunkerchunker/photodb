-- Migration 024: Cluster-Person Must-Link and Cannot-Link Constraints
--
-- Two tables that record user intent when assigning/removing clusters from persons:
-- - cluster_person_must_link: cluster MUST be linked to this person (user assigned it)
-- - cluster_person_cannot_link: cluster MUST NOT be linked to this person (user removed it)

-- A cluster MUST be linked to this person (created when user assigns cluster to person)
CREATE TABLE IF NOT EXISTS cluster_person_must_link (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER NOT NULL REFERENCES cluster(id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (cluster_id)  -- a cluster can only be must-linked to one person
);

-- A cluster MUST NOT be linked to this person (created when user removes cluster from person)
CREATE TABLE IF NOT EXISTS cluster_person_cannot_link (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER NOT NULL REFERENCES cluster(id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (cluster_id, person_id)  -- one constraint per cluster-person pair
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_cluster_person_must_link_person ON cluster_person_must_link(person_id);
CREATE INDEX IF NOT EXISTS idx_cluster_person_must_link_collection ON cluster_person_must_link(collection_id);
CREATE INDEX IF NOT EXISTS idx_cluster_person_cannot_link_person ON cluster_person_cannot_link(person_id);
CREATE INDEX IF NOT EXISTS idx_cluster_person_cannot_link_collection ON cluster_person_cannot_link(collection_id);
