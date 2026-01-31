-- Rollback: Remove face_count_at_last_medoid column

ALTER TABLE "cluster" DROP COLUMN IF EXISTS face_count_at_last_medoid;
