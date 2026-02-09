-- Migration: Add representative_photo_id to album table
-- This allows albums to have a cover/thumbnail photo

ALTER TABLE album
ADD COLUMN IF NOT EXISTS representative_photo_id bigint;

-- Add foreign key constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'album_representative_photo_fk'
    ) THEN
        ALTER TABLE album
            ADD CONSTRAINT album_representative_photo_fk
            FOREIGN KEY (representative_photo_id) REFERENCES photo(id) ON DELETE SET NULL;
    END IF;
END;
$$;
