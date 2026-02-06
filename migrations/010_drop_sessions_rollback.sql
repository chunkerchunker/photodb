BEGIN;

CREATE TABLE IF NOT EXISTS app_session(
    id bigserial PRIMARY KEY,
    user_id bigint NOT NULL,
    token text NOT NULL UNIQUE,
    created_at timestamptz DEFAULT now(),
    expires_at timestamptz NOT NULL,
    FOREIGN KEY (user_id) REFERENCES app_user(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_app_session_user_id ON app_session(user_id);
CREATE INDEX IF NOT EXISTS idx_app_session_expires_at ON app_session(expires_at);

DELETE FROM schema_migrations WHERE version = '010';

COMMIT;
