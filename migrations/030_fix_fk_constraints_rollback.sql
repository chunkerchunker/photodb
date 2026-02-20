-- Rollback Migration 030: Revert FK constraint fixes

BEGIN;

-- 1. Re-add the inline FK on cluster.person_id (restores the duplicate)
ALTER TABLE "cluster"
    ADD CONSTRAINT cluster_person_id_fkey
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL;

-- 2. Remove the llm_analysis.batch_id FK
ALTER TABLE llm_analysis DROP CONSTRAINT IF EXISTS llm_analysis_batch_id_fk;

-- 3. Revert cluster_person_cannot_link column types back to INTEGER
ALTER TABLE cluster_person_cannot_link
    ALTER COLUMN id TYPE integer,
    ALTER COLUMN cluster_id TYPE integer,
    ALTER COLUMN person_id TYPE integer,
    ALTER COLUMN collection_id TYPE integer;

COMMIT;
