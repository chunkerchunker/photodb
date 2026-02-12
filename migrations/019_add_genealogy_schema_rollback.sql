-- Rollback Migration 019: Add Genealogy Schema
-- Description: Removes genealogical relationship modeling from person entities
-- Created: 2026-02-11

BEGIN;

-- ============================================================================
-- Part 1: Drop Functions (reverse order of creation)
-- ============================================================================

DROP FUNCTION IF EXISTS get_unrelated_persons(BIGINT);
DROP FUNCTION IF EXISTS are_persons_unrelated(BIGINT, BIGINT);
DROP FUNCTION IF EXISTS propagate_birth_year_constraints(INT, INT);
DROP FUNCTION IF EXISTS get_family_tree(BIGINT, INT, BOOLEAN);
DROP FUNCTION IF EXISTS refresh_genealogy_closures();
DROP FUNCTION IF EXISTS refresh_birth_order_closure();
DROP FUNCTION IF EXISTS refresh_ancestor_closure();
DROP FUNCTION IF EXISTS link_siblings(BIGINT, BIGINT, TEXT);
DROP FUNCTION IF EXISTS create_placeholder_parent(BIGINT, TEXT);
DROP FUNCTION IF EXISTS person_display_name(person);

-- ============================================================================
-- Part 2: Drop Views
-- ============================================================================

DROP VIEW IF EXISTS person_known;
DROP VIEW IF EXISTS person_siblings;

-- ============================================================================
-- Part 3: Drop Closure Tables
-- ============================================================================

DROP TABLE IF EXISTS person_birth_order_closure;
DROP TABLE IF EXISTS person_ancestor_closure;

-- ============================================================================
-- Part 4: Drop Relationship Tables
-- ============================================================================

DROP TABLE IF EXISTS person_not_related;
DROP TABLE IF EXISTS person_birth_order;
DROP TABLE IF EXISTS person_partnership;
DROP TABLE IF EXISTS person_parent;

-- ============================================================================
-- Part 5: Revert Person Table Modifications
-- ============================================================================

-- Drop indexes
DROP INDEX IF EXISTS idx_person_placeholder;

-- Drop constraints
ALTER TABLE person DROP CONSTRAINT IF EXISTS person_birth_date_year_check;
ALTER TABLE person DROP CONSTRAINT IF EXISTS person_birth_year_source_check;
ALTER TABLE person DROP CONSTRAINT IF EXISTS person_gender_check;

-- Drop columns
ALTER TABLE person DROP COLUMN IF EXISTS birth_date;
ALTER TABLE person DROP COLUMN IF EXISTS birth_year_source;
ALTER TABLE person DROP COLUMN IF EXISTS birth_year_max;
ALTER TABLE person DROP COLUMN IF EXISTS birth_year_min;
ALTER TABLE person DROP COLUMN IF EXISTS gender;
ALTER TABLE person DROP COLUMN IF EXISTS placeholder_description;
ALTER TABLE person DROP COLUMN IF EXISTS is_placeholder;

-- Note: We do NOT restore NOT NULL constraints on first_name/last_name
-- because there may now be data that relies on them being nullable.
-- If you need to restore the NOT NULL constraints, first ensure all
-- person records have non-null names, then run:
--   ALTER TABLE person ALTER COLUMN first_name SET NOT NULL;
--   ALTER TABLE person ALTER COLUMN last_name SET NOT NULL;

-- ============================================================================
-- Part 6: Remove Migration Record
-- ============================================================================

DELETE FROM schema_migrations WHERE version = '019';

COMMIT;

-- ============================================================================
-- Post-rollback notes:
-- ============================================================================
--
-- This rollback removes all genealogy features added in migration 019.
--
-- WARNING: All genealogical relationship data will be permanently deleted:
-- - person_parent relationships
-- - person_partnership relationships
-- - person_birth_order relationships
-- - person_not_related relationships
-- - person_ancestor_closure data
-- - person_birth_order_closure data
--
-- The first_name and last_name columns remain nullable after rollback.
-- If you need to restore NOT NULL constraints, manually update any null
-- values first, then run:
--   UPDATE person SET first_name = 'Unknown' WHERE first_name IS NULL;
--   UPDATE person SET last_name = 'Unknown' WHERE last_name IS NULL;
--   ALTER TABLE person ALTER COLUMN first_name SET NOT NULL;
--   ALTER TABLE person ALTER COLUMN last_name SET NOT NULL;
--
-- Placeholder persons created during the genealogy feature period will
-- remain in the person table but without the is_placeholder flag.
-- You may want to delete these manually:
--   DELETE FROM person WHERE first_name IS NULL AND last_name IS NULL;
