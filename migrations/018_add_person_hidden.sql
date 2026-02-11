-- Migration: Add hidden flag to person table
-- This allows hiding people similar to how clusters can be hidden

-- Add hidden column to person table
ALTER TABLE person ADD COLUMN IF NOT EXISTS hidden boolean DEFAULT false;

-- Index for efficient hidden person queries
CREATE INDEX IF NOT EXISTS idx_person_hidden ON person(hidden) WHERE hidden = true;

-- Composite index for hidden person listing with collection filter
CREATE INDEX IF NOT EXISTS idx_person_collection_hidden
ON person(collection_id)
WHERE hidden = true;
