-- Migration: Add hidden column to cluster table
-- This allows users to hide clusters (ignored people) from the main UI

-- Add hidden column
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS hidden boolean DEFAULT false;

-- Index for efficient filtering of hidden clusters
CREATE INDEX IF NOT EXISTS idx_cluster_hidden ON cluster(hidden) WHERE hidden = true;
