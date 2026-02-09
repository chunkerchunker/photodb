-- Migration: Add Albums
-- Albums are arbitrary groupings of photos within a collection.
-- A photo can belong to multiple albums (many-to-many relationship).

BEGIN;

-- =============================================================================
-- Part 1: Create album table
-- =============================================================================

CREATE TABLE IF NOT EXISTS album (
    id bigserial PRIMARY KEY,
    collection_id bigint NOT NULL,
    name text NOT NULL,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE
);

-- =============================================================================
-- Part 2: Create photo_album junction table
-- =============================================================================

CREATE TABLE IF NOT EXISTS photo_album (
    photo_id bigint NOT NULL,
    album_id bigint NOT NULL,
    added_at timestamptz DEFAULT now(),
    PRIMARY KEY (photo_id, album_id),
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
    FOREIGN KEY (album_id) REFERENCES album(id) ON DELETE CASCADE
);

-- =============================================================================
-- Part 3: Indexes for performance
-- =============================================================================

-- Find albums by collection
CREATE INDEX IF NOT EXISTS idx_album_collection_id ON album(collection_id);

-- Find albums by name within a collection
CREATE INDEX IF NOT EXISTS idx_album_collection_name ON album(collection_id, name);

-- Find all photos in an album
CREATE INDEX IF NOT EXISTS idx_photo_album_album_id ON photo_album(album_id);

-- Find all albums a photo belongs to
CREATE INDEX IF NOT EXISTS idx_photo_album_photo_id ON photo_album(photo_id);

-- =============================================================================
-- Part 4: Record migration
-- =============================================================================

INSERT INTO schema_migrations (version, description)
VALUES ('011', 'Add albums for grouping photos within collections')
ON CONFLICT (version) DO NOTHING;

COMMIT;
