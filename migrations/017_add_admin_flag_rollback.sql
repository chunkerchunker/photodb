-- Rollback: Remove is_admin flag from app_user

DROP INDEX IF EXISTS idx_app_user_is_admin;
ALTER TABLE app_user DROP COLUMN IF EXISTS is_admin;
