-- Add auto_created flag to person table
-- Indicates the person was automatically created by auto_associate_clusters
ALTER TABLE person ADD COLUMN IF NOT EXISTS auto_created boolean DEFAULT false;

-- Backfill: mark existing persons named "Unknown" with no last name as auto-created
UPDATE person SET auto_created = true
WHERE first_name = 'Unknown' AND last_name IS NULL;
