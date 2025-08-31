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

-- LLM Analysis table: Store LLM-based photo analysis results
CREATE TABLE IF NOT EXISTS llm_analysis (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id TEXT NOT NULL UNIQUE,
    
    -- LLM processing metadata
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    batch_id VARCHAR(255), -- Provider batch job ID
    
    -- Analysis results (JSON structure matching analyze_photo.md prompt)
    analysis JSONB NOT NULL,
    
    -- Extracted key fields for indexing/querying
    description TEXT, -- Main scene description
    objects TEXT[], -- Array of identified objects
    people_count INTEGER, -- Number of people detected
    location_description TEXT, -- Described location if not in EXIF
    emotional_tone VARCHAR(50), -- Happy, sad, neutral, etc.
    
    -- Processing metadata
    confidence_score DECIMAL(3,2), -- Overall confidence 0.00-1.00
    processing_duration_ms INTEGER,
    error_message TEXT, -- If processing failed
    
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

-- Batch Jobs table: Track LLM batch processing jobs
CREATE TABLE IF NOT EXISTS batch_jobs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_batch_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL, -- submitted, processing, completed, failed
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    photo_count INTEGER NOT NULL,
    processed_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    photo_ids TEXT[] NOT NULL, -- Array of photo IDs in the batch
    error_message TEXT
);

-- Indexes for LLM analysis performance
CREATE INDEX IF NOT EXISTS idx_llm_analysis_objects ON llm_analysis USING GIN (objects);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_emotional_tone ON llm_analysis(emotional_tone);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_people_count ON llm_analysis(people_count);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_processed_at ON llm_analysis(processed_at);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_batch_id ON llm_analysis(batch_id);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_analysis ON llm_analysis USING GIN (analysis);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_submitted_at ON batch_jobs(submitted_at);