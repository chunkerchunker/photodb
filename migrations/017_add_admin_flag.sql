-- Migration: Add is_admin flag to app_user
-- Allows marking users as system administrators

ALTER TABLE app_user ADD COLUMN IF NOT EXISTS is_admin boolean DEFAULT false;

-- Index for efficient admin user lookups
CREATE INDEX IF NOT EXISTS idx_app_user_is_admin ON app_user(is_admin) WHERE is_admin = true;
