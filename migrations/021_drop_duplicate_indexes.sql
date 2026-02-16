-- Drop duplicate indexes that are covered by composite indexes
-- These single-column indexes are redundant because PostgreSQL can use the
-- leftmost column(s) of a composite index to satisfy queries on those columns.
--
-- Recommended by pgHero duplicate index analysis.

BEGIN;

-- album: idx_album_collection_id covered by idx_album_collection_name (collection_id, name)
DROP INDEX IF EXISTS idx_album_collection_id;

-- cannot_link: idx_cannot_link_detection1 covered by unique constraint (detection_id_1, detection_id_2)
DROP INDEX IF EXISTS idx_cannot_link_detection1;

-- cluster_cannot_link: idx_cluster_cannot_link_c1 covered by unique constraint (cluster_id_1, cluster_id_2)
DROP INDEX IF EXISTS idx_cluster_cannot_link_c1;

-- detection_tag: idx_detection_tag_detection covered by unique constraint (detection_id, prompt_id)
DROP INDEX IF EXISTS idx_detection_tag_detection;

-- photo_album: idx_photo_album_photo_id covered by primary key (photo_id, album_id)
DROP INDEX IF EXISTS idx_photo_album_photo_id;

-- photo_tag: idx_photo_tag_photo covered by unique constraint (photo_id, prompt_id)
DROP INDEX IF EXISTS idx_photo_tag_photo;

COMMIT;
