-- Photos table: Core photo records
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,  -- UUID
    filename TEXT NOT NULL UNIQUE,  -- Relative path from INGEST_PATH
    normalized_path TEXT NOT NULL,  -- Path to normalized image in IMG_PATH
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metadata table: Extracted photo metadata
CREATE TABLE IF NOT EXISTS metadata (
    photo_id TEXT PRIMARY KEY,
    captured_at TIMESTAMP,  -- When photo was taken
    latitude REAL,
    longitude REAL,
    extra JSONB,  -- All EXIF/TIFF/IFD metadata as JSONB (PostgreSQL native JSON)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

-- Processing status table: Track processing stages
CREATE TABLE IF NOT EXISTS processing_status (
    photo_id TEXT NOT NULL,
    stage TEXT NOT NULL,  -- 'normalize', 'metadata', etc.
    status TEXT NOT NULL,  -- 'pending', 'processing', 'completed', 'failed'
    processed_at TIMESTAMP,
    error_message TEXT,
    PRIMARY KEY (photo_id, stage),
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_photos_filename ON photos(filename);
CREATE INDEX IF NOT EXISTS idx_metadata_captured_at ON metadata(captured_at);
CREATE INDEX IF NOT EXISTS idx_metadata_location ON metadata(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_status(status, stage);

-- PostgreSQL-specific: GIN index for JSONB search
CREATE INDEX IF NOT EXISTS idx_metadata_extra ON metadata USING GIN (extra);