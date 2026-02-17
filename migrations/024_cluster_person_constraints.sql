-- Migration 024: Cluster-Person Cannot-Link Constraints
--
-- Records when a user explicitly removes a cluster from a person,
-- so auto-association won't re-link them.

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
CREATE INDEX IF NOT EXISTS idx_cluster_person_cannot_link_person ON cluster_person_cannot_link(person_id);
CREATE INDEX IF NOT EXISTS idx_cluster_person_cannot_link_collection ON cluster_person_cannot_link(collection_id);
