-- Migration 004: Split person.name into first_name and last_name
-- Assumes existing name values are first names

-- Rename name to first_name
ALTER TABLE person RENAME COLUMN name TO first_name;

-- Add last_name column (nullable)
ALTER TABLE person ADD COLUMN last_name text;

-- Drop old index and create new ones
DROP INDEX IF EXISTS idx_person_name;
CREATE INDEX idx_person_first_name ON person(first_name);
CREATE INDEX idx_person_last_name ON person(last_name);
