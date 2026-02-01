-- Rollback Migration 004: Merge first_name and last_name back to name

-- Drop new indexes
DROP INDEX IF EXISTS idx_person_first_name;
DROP INDEX IF EXISTS idx_person_last_name;

-- Drop last_name column
ALTER TABLE person DROP COLUMN last_name;

-- Rename first_name back to name
ALTER TABLE person RENAME COLUMN first_name TO name;

-- Recreate original index
CREATE INDEX idx_person_name ON person(name);
