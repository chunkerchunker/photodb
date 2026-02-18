-- Rollback migration 028: Remove optional name fields from person table

ALTER TABLE person DROP COLUMN IF EXISTS middle_name;
ALTER TABLE person DROP COLUMN IF EXISTS maiden_name;
ALTER TABLE person DROP COLUMN IF EXISTS preferred_name;
ALTER TABLE person DROP COLUMN IF EXISTS suffix;
