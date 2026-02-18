-- Migration 029: Add alternate_names array to person table
-- Stores alternate names (e.g., names in other languages)

ALTER TABLE person ADD COLUMN alternate_names text[] DEFAULT '{}';
