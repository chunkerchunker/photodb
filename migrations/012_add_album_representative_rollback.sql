-- Rollback: Remove representative_photo_id from album table

ALTER TABLE album DROP CONSTRAINT IF EXISTS album_representative_photo_fk;
ALTER TABLE album DROP COLUMN IF EXISTS representative_photo_id;
