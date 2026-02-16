-- Rollback: Recreate duplicate indexes if needed

BEGIN;

CREATE INDEX IF NOT EXISTS idx_album_collection_id ON album (collection_id);
CREATE INDEX IF NOT EXISTS idx_cannot_link_detection1 ON cannot_link (detection_id_1);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c1 ON cluster_cannot_link (cluster_id_1);
CREATE INDEX IF NOT EXISTS idx_detection_tag_detection ON detection_tag (detection_id);
CREATE INDEX IF NOT EXISTS idx_photo_album_photo_id ON photo_album (photo_id);
CREATE INDEX IF NOT EXISTS idx_photo_tag_photo ON photo_tag (photo_id);

COMMIT;
