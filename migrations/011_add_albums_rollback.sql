-- Rollback: Remove Albums

BEGIN;

DROP TABLE IF EXISTS photo_album;
DROP TABLE IF EXISTS album;

DELETE FROM schema_migrations WHERE version = '011';

COMMIT;
