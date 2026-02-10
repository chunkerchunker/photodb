-- Migration: Add person_id to collection_member
-- Links a collection member (user) to their person record in the collection

-- Add the person_id column
ALTER TABLE collection_member
ADD COLUMN IF NOT EXISTS person_id bigint;

-- Add foreign key constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'collection_member_person_id_fkey'
    ) THEN
        ALTER TABLE collection_member
        ADD CONSTRAINT collection_member_person_id_fkey
        FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL;
    END IF;
END;
$$;

-- Add index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_collection_member_person_id ON collection_member(person_id);
