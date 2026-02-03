-- Rollback Migration 008: Remove scalability indexes

DROP INDEX IF EXISTS idx_cluster_visible;
DROP INDEX IF EXISTS idx_cluster_hidden_listing;
DROP INDEX IF EXISTS idx_person_detection_unassigned_pool;
DROP INDEX IF EXISTS idx_processing_status_completed;
