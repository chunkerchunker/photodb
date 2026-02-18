-- Rollback migration 029: Remove alternate_names from person table

ALTER TABLE person DROP COLUMN IF EXISTS alternate_names;
