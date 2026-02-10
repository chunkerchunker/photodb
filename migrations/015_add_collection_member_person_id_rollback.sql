-- Rollback: Remove person_id from collection_member

-- Drop index
DROP INDEX IF EXISTS idx_collection_member_person_id;

-- Drop foreign key constraint
ALTER TABLE collection_member
DROP CONSTRAINT IF EXISTS collection_member_person_id_fkey;

-- Drop column
ALTER TABLE collection_member
DROP COLUMN IF EXISTS person_id;
