BEGIN;

DROP INDEX IF EXISTS idx_app_session_expires_at;
DROP INDEX IF EXISTS idx_app_session_user_id;
DROP TABLE IF EXISTS app_session;

INSERT INTO schema_migrations (version, description)
VALUES ('010', 'Drop app sessions (cookie-only auth)')
ON CONFLICT (version) DO NOTHING;

COMMIT;
