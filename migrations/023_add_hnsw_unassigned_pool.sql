-- Migration 023: Upgrade face_embedding vector index from IVFFlat to HNSW
--
-- The find_similar_unassigned_detections query has been restructured to do
-- vector search first (ORDER BY + LIMIT on face_embedding alone), then
-- join person_detection for pool filters. This lets pgvector use the index.
--
-- HNSW is preferred over IVFFlat here because:
-- 1. No training step needed (the embedding table grows as photos are processed)
-- 2. Better recall without tuning lists/probes parameters
DROP INDEX IF EXISTS face_embedding_idx;
CREATE INDEX IF NOT EXISTS face_embedding_hnsw_idx
    ON face_embedding USING hnsw(embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
