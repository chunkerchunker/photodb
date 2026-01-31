CREATE EXTENSION IF NOT EXISTS vector;

-- Photo table: Core photo records
CREATE TABLE IF NOT EXISTS photo(
    id bigserial PRIMARY KEY,
    filename text NOT NULL UNIQUE, -- Relative path from INGEST_PATH
    normalized_path text UNIQUE, -- Path to normalized image in IMG_PATH
    width integer, -- Original image width in pixels
    height integer, -- Original image height in pixels
    normalized_width integer, -- Normalized image width in pixels
    normalized_height integer, -- Normalized image height in pixels
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Metadata table: Extracted photo metadata
CREATE TABLE IF NOT EXISTS metadata(
    photo_id bigint PRIMARY KEY,
    captured_at timestamptz, -- When photo was taken
    latitude real,
    longitude real,
    extra jsonb, -- All EXIF/TIFF/IFD metadata as JSONB (PostgreSQL native JSON)
    created_at timestamptz DEFAULT now(),
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

-- Processing status table: Track processing stages
CREATE TABLE IF NOT EXISTS processing_status(
    photo_id bigint NOT NULL,
    stage text NOT NULL, -- 'normalize', 'metadata', etc.
    status text NOT NULL, -- 'pending', 'processing', 'completed', 'failed'
    processed_at timestamptz,
    error_message text,
    PRIMARY KEY (photo_id, stage),
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_photo_filename ON photo(filename);

CREATE INDEX IF NOT EXISTS idx_metadata_captured_at ON metadata(captured_at);

CREATE INDEX IF NOT EXISTS idx_metadata_location ON metadata(latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_status(status, stage);

-- PostgreSQL-specific: GIN index for JSONB search
CREATE INDEX IF NOT EXISTS idx_metadata_extra ON metadata USING GIN(extra);

-- LLM Analysis table: Store LLM-based photo analysis results
CREATE TABLE IF NOT EXISTS llm_analysis(
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL UNIQUE,
    -- LLM processing metadata
    model_name varchar(100) NOT NULL,
    model_version varchar(50),
    processed_at timestamp with time zone DEFAULT NOW(),
    batch_id varchar(255), -- Provider batch job ID
    -- Analysis results (JSON structure matching analyze_photo.md prompt)
    analysis jsonb NOT NULL,
    -- Extracted key fields for indexing/querying
    description text, -- Main scene description
    objects text[], -- Array of identified objects
    people_count integer, -- Number of people detected
    location_description text, -- Described location if not in EXIF
    emotional_tone varchar(50), -- Happy, sad, neutral, etc.
    -- Processing metadata
    confidence_score DECIMAL(3, 2), -- Overall confidence 0.00-1.00
    processing_duration_ms integer,
    -- Token usage tracking (per photo)
    input_tokens integer,
    output_tokens integer,
    cache_creation_tokens integer,
    cache_read_tokens integer,
    error_message text, -- If processing failed
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

-- Batch Job table: Track LLM batch processing jobs
CREATE TABLE IF NOT EXISTS batch_job(
    id bigserial PRIMARY KEY,
    provider_batch_id varchar(255) UNIQUE NOT NULL,
    status varchar(20) NOT NULL, -- submitted, processing, completed, failed
    submitted_at timestamp with time zone DEFAULT NOW(),
    completed_at timestamp with time zone,
    photo_count integer NOT NULL,
    processed_count integer DEFAULT 0,
    failed_count integer DEFAULT 0,
    photo_ids bigint[] NOT NULL, -- Array of photo IDs in the batch
    -- Token usage tracking
    total_input_tokens integer DEFAULT 0,
    total_output_tokens integer DEFAULT 0,
    total_cache_creation_tokens integer DEFAULT 0,
    total_cache_read_tokens integer DEFAULT 0,
    -- Cost tracking (in USD cents for precision)
    estimated_cost_cents integer DEFAULT 0,
    actual_cost_cents integer DEFAULT 0,
    -- Additional metadata
    model_name varchar(100),
    batch_discount_applied boolean DEFAULT TRUE, -- Batch API gives 50% discount
    error_message text
);

-- Indexes for LLM analysis performance
CREATE INDEX IF NOT EXISTS idx_llm_analysis_objects ON llm_analysis USING GIN(objects);

CREATE INDEX IF NOT EXISTS idx_llm_analysis_emotional_tone ON llm_analysis(emotional_tone);

CREATE INDEX IF NOT EXISTS idx_llm_analysis_people_count ON llm_analysis(people_count);

CREATE INDEX IF NOT EXISTS idx_llm_analysis_processed_at ON llm_analysis(processed_at);

CREATE INDEX IF NOT EXISTS idx_llm_analysis_batch_id ON llm_analysis(batch_id);

CREATE INDEX IF NOT EXISTS idx_llm_analysis_analysis ON llm_analysis USING GIN(analysis);

CREATE INDEX IF NOT EXISTS idx_batch_job_status ON batch_job(status);

CREATE INDEX IF NOT EXISTS idx_batch_job_submitted_at ON batch_job(submitted_at);

-- People table: Named individuals that can appear in photos (whether detected or manually assigned)
CREATE TABLE IF NOT EXISTS person(
    id bigserial PRIMARY KEY,
    name text NOT NULL,
    created_at timestamp with time zone DEFAULT NOW(),
    updated_at timestamp with time zone DEFAULT NOW()
);

-- Faces table: Detected faces in photos with bounding boxes
CREATE TABLE IF NOT EXISTS face(
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL,
    -- Bounding box coordinates (normalized 0.0-1.0 or pixel values)
    bbox_x real NOT NULL, -- X coordinate of top-left corner
    bbox_y real NOT NULL, -- Y coordinate of top-left corner
    bbox_width real NOT NULL, -- Width of bounding box
    bbox_height real NOT NULL, -- Height of bounding box
    confidence DECIMAL(3, 2) NOT NULL DEFAULT 0, -- Face detection confidence 0.00-1.00
    -- Detection metadata
    person_id bigint,
    -- Clustering fields
    cluster_status text CHECK (cluster_status IN ('auto', 'pending', 'manual', 'unassigned', 'constrained')) DEFAULT NULL,
    cluster_id bigint REFERENCES "cluster"(id) ON DELETE SET NULL,
    cluster_confidence DECIMAL(3, 2) DEFAULT 0, -- Cluster assignment confidence 0.00-1.00
    -- Unassigned pool tracking
    unassigned_since timestamptz DEFAULT NULL, -- When face was added to unassigned pool
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL
);

-- Indexes for face detection performance
CREATE INDEX IF NOT EXISTS idx_face_photo_id ON face(photo_id);

CREATE INDEX IF NOT EXISTS idx_face_person_id ON face(person_id);

CREATE INDEX IF NOT EXISTS idx_face_confidence ON face(confidence);

CREATE INDEX IF NOT EXISTS idx_face_cluster_status ON face(cluster_status);

CREATE INDEX IF NOT EXISTS idx_face_cluster_id ON face(cluster_id);

CREATE INDEX IF NOT EXISTS idx_person_name ON person(name);

-- Face-level embeddings (for clustering & recognition)
CREATE TABLE IF NOT EXISTS face_embedding(
    face_id bigint PRIMARY KEY REFERENCES face(id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL
);

CREATE INDEX IF NOT EXISTS face_embedding_idx ON face_embedding USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Group of faces belonging to the same person
CREATE TABLE IF NOT EXISTS "cluster"(
    id bigserial PRIMARY KEY,
    face_count bigint DEFAULT 0,
    face_count_at_last_medoid bigint DEFAULT 0, -- For threshold-based medoid recomputation
    representative_face_id bigint REFERENCES face(id) ON DELETE SET NULL,
    centroid VECTOR(512),
    medoid_face_id bigint REFERENCES face(id) ON DELETE SET NULL,
    person_id bigint REFERENCES person(id) ON DELETE SET NULL,
    -- Verification status to protect human-verified clusters
    verified boolean DEFAULT false,
    verified_at timestamptz DEFAULT NULL,
    verified_by text DEFAULT NULL,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL
);

-- Tracks potential cluster assignments requiring review
CREATE TABLE IF NOT EXISTS face_match_candidate(
    candidate_id bigserial PRIMARY KEY,
    face_id bigint REFERENCES face(id) ON DELETE CASCADE,
    cluster_id bigint REFERENCES "cluster"(id) ON DELETE CASCADE,
    similarity float NOT NULL,
    status text CHECK (status IN ('pending', 'accepted', 'rejected')) DEFAULT 'pending',
    created_at timestamp DEFAULT now()
);

-- Indexes for clustering performance
CREATE INDEX IF NOT EXISTS idx_cluster_centroid ON "cluster" USING ivfflat(centroid vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_face_match_candidate_face ON face_match_candidate(face_id);

CREATE INDEX IF NOT EXISTS idx_face_match_candidate_status ON face_match_candidate(status);

CREATE INDEX IF NOT EXISTS idx_face_unassigned ON face(unassigned_since) WHERE cluster_id IS NULL;

-- Must-link constraint: forces faces to be in the same cluster
CREATE TABLE IF NOT EXISTS must_link(
    id bigserial PRIMARY KEY,
    face_id_1 bigint NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    face_id_2 bigint NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    created_by text DEFAULT 'human', -- 'human' or 'system'
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(face_id_1, face_id_2),
    CHECK (face_id_1 < face_id_2) -- Canonical ordering to prevent duplicates
);

-- Cannot-link constraint: prevents faces from being in the same cluster
CREATE TABLE IF NOT EXISTS cannot_link(
    id bigserial PRIMARY KEY,
    face_id_1 bigint NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    face_id_2 bigint NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    created_by text DEFAULT 'human',
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(face_id_1, face_id_2),
    CHECK (face_id_1 < face_id_2)
);

-- Cluster-level cannot-link for efficiency (prevents merging)
CREATE TABLE IF NOT EXISTS cluster_cannot_link(
    id bigserial PRIMARY KEY,
    cluster_id_1 bigint NOT NULL REFERENCES "cluster"(id) ON DELETE CASCADE,
    cluster_id_2 bigint NOT NULL REFERENCES "cluster"(id) ON DELETE CASCADE,
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(cluster_id_1, cluster_id_2),
    CHECK (cluster_id_1 < cluster_id_2)
);

-- Indexes for constraint lookups
CREATE INDEX IF NOT EXISTS idx_must_link_face1 ON must_link(face_id_1);
CREATE INDEX IF NOT EXISTS idx_must_link_face2 ON must_link(face_id_2);
CREATE INDEX IF NOT EXISTS idx_cannot_link_face1 ON cannot_link(face_id_1);
CREATE INDEX IF NOT EXISTS idx_cannot_link_face2 ON cannot_link(face_id_2);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c1 ON cluster_cannot_link(cluster_id_1);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c2 ON cluster_cannot_link(cluster_id_2);
CREATE INDEX IF NOT EXISTS idx_cluster_verified ON "cluster"(verified) WHERE verified = true;

