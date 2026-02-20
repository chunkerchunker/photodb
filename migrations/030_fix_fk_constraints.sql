-- Migration 030: Fix missing and duplicate FK constraints
--
-- 1. Remove duplicate FK on cluster.person_id (inline REFERENCES + table FOREIGN KEY)
-- 2. Add missing FK on llm_analysis.batch_id -> batch_job.provider_batch_id
-- 3. Fix cluster_person_cannot_link column types (INTEGER -> bigint)

BEGIN;

-- 1. Remove the duplicate inline FK on cluster.person_id
-- Keep the table-level FOREIGN KEY constraint, drop the auto-generated inline one.
-- The inline constraint name is auto-generated as "cluster_person_id_fkey".
ALTER TABLE "cluster" DROP CONSTRAINT IF EXISTS cluster_person_id_fkey;

-- 2. Add missing FK: llm_analysis.batch_id -> batch_job.provider_batch_id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'llm_analysis_batch_id_fk'
    ) THEN
        ALTER TABLE llm_analysis
            ADD CONSTRAINT llm_analysis_batch_id_fk
            FOREIGN KEY (batch_id) REFERENCES batch_job(provider_batch_id) ON DELETE SET NULL;
    END IF;
END;
$$;

-- 3. Fix cluster_person_cannot_link column types (INTEGER -> bigint for consistency)
ALTER TABLE cluster_person_cannot_link
    ALTER COLUMN cluster_id TYPE bigint,
    ALTER COLUMN person_id TYPE bigint,
    ALTER COLUMN collection_id TYPE bigint;

-- Also fix the id column (SERIAL -> bigserial is just INTEGER -> bigint + sequence)
ALTER TABLE cluster_person_cannot_link
    ALTER COLUMN id TYPE bigint;

COMMIT;
