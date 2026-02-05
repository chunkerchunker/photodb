BEGIN;

-- Drop collection-scoped indexes
DROP INDEX IF EXISTS idx_cluster_cannot_link_collection;
DROP INDEX IF EXISTS idx_cannot_link_collection;
DROP INDEX IF EXISTS idx_must_link_collection;
DROP INDEX IF EXISTS idx_face_match_candidate_collection;
DROP INDEX IF EXISTS idx_cluster_collection_hidden_listing;
DROP INDEX IF EXISTS idx_cluster_collection_visible;
DROP INDEX IF EXISTS idx_person_collection_last_name;
DROP INDEX IF EXISTS idx_person_collection_first_name;
DROP INDEX IF EXISTS idx_person_detection_collection_id;
DROP INDEX IF EXISTS idx_metadata_collection_location;
DROP INDEX IF EXISTS idx_metadata_collection_year_month;
DROP INDEX IF EXISTS idx_metadata_collection_captured_at;
DROP INDEX IF EXISTS idx_photo_collection_id;
DROP INDEX IF EXISTS idx_photo_collection_normalized_path_unique;
DROP INDEX IF EXISTS idx_photo_collection_filename_unique;

-- Remove collection_id columns (drops dependent constraints)
ALTER TABLE cluster_cannot_link DROP COLUMN IF EXISTS collection_id;
ALTER TABLE cannot_link DROP COLUMN IF EXISTS collection_id;
ALTER TABLE must_link DROP COLUMN IF EXISTS collection_id;
ALTER TABLE face_match_candidate DROP COLUMN IF EXISTS collection_id;
ALTER TABLE cluster DROP COLUMN IF EXISTS collection_id;
ALTER TABLE person_detection DROP COLUMN IF EXISTS collection_id;
ALTER TABLE person DROP COLUMN IF EXISTS collection_id;
ALTER TABLE metadata DROP COLUMN IF EXISTS collection_id;
ALTER TABLE photo DROP COLUMN IF EXISTS collection_id;

-- Restore original uniqueness on photo fields
ALTER TABLE photo ADD CONSTRAINT photo_filename_key UNIQUE (filename);
ALTER TABLE photo ADD CONSTRAINT photo_normalized_path_key UNIQUE (normalized_path);

-- Drop default collection FK
ALTER TABLE app_user DROP CONSTRAINT IF EXISTS app_user_default_collection_fk;
ALTER TABLE app_user DROP COLUMN IF EXISTS default_collection_id;

-- Drop collection tables
DROP TABLE IF EXISTS collection_member;
DROP TABLE IF EXISTS collection;
DROP TABLE IF EXISTS app_user;

DELETE FROM schema_migrations WHERE version = '009';

COMMIT;
