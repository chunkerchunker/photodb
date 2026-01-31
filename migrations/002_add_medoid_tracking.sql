-- Add face_count_at_last_medoid to track when medoid was last computed
-- Used for threshold-based inline medoid recomputation

ALTER TABLE "cluster" ADD COLUMN IF NOT EXISTS face_count_at_last_medoid bigint DEFAULT 0;

-- Initialize existing clusters with current face_count
UPDATE "cluster" SET face_count_at_last_medoid = face_count WHERE face_count_at_last_medoid = 0;
