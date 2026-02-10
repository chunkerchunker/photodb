-- Migration: Add is_admin flag to app_user
-- Allows marking users as system administrators

ALTER TABLE app_user ADD COLUMN IF NOT EXISTS is_admin boolean DEFAULT false;

-- Set existing NULL values to false
UPDATE app_user SET is_admin = false WHERE is_admin IS NULL;

-- Add NOT NULL constraint now that all values are set
ALTER TABLE app_user ALTER COLUMN is_admin SET NOT NULL;

-- Index for efficient admin user lookups
CREATE INDEX IF NOT EXISTS idx_app_user_is_admin ON app_user(is_admin) WHERE is_admin = true;
