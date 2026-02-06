BEGIN;

DROP INDEX IF EXISTS idx_app_session_expires_at;
DROP INDEX IF EXISTS idx_app_session_user_id;
DROP TABLE IF EXISTS app_session;

DELETE FROM schema_migrations WHERE version = '010';

COMMIT;
