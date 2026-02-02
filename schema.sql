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
    first_name text NOT NULL,
    last_name text,
    -- Age/gender aggregation from detections
    estimated_birth_year integer,
    birth_year_stddev real,
    gender char(1) CHECK (gender IN ('M', 'F', 'U')),
    gender_confidence real,
    age_gender_sample_count integer DEFAULT 0,
    age_gender_updated_at timestamptz,
    created_at timestamp with time zone DEFAULT NOW(),
    updated_at timestamp with time zone DEFAULT NOW()
);

-- Person Detection table: Detected faces/bodies in photos with bounding boxes and age/gender
CREATE TABLE IF NOT EXISTS person_detection(
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL,

    -- Face bounding box (nullable - may have body-only detection)
    face_bbox_x real,
    face_bbox_y real,
    face_bbox_width real,
    face_bbox_height real,
    face_confidence real,

    -- Body bounding box (nullable - may have face-only detection)
    body_bbox_x real,
    body_bbox_y real,
    body_bbox_width real,
    body_bbox_height real,
    body_confidence real,

    -- Age/gender estimation
    age_estimate real,
    gender char(1) CHECK (gender IN ('M', 'F', 'U')),
    gender_confidence real,
    mivolo_output jsonb,

    -- Clustering fields
    person_id bigint,
    cluster_status text CHECK (cluster_status IN ('auto', 'pending', 'manual', 'unassigned', 'constrained')) DEFAULT NULL,
    cluster_id bigint,
    cluster_confidence real DEFAULT 0,

    -- Unassigned pool tracking
    unassigned_since timestamptz DEFAULT NULL,

    -- Detector metadata
    detector_model text,
    detector_version text,

    -- Timestamps
    created_at timestamptz DEFAULT now(),

    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL,
    FOREIGN KEY (cluster_id) REFERENCES "cluster"(id) ON DELETE SET NULL
);

-- Indexes for person detection performance
CREATE INDEX IF NOT EXISTS idx_person_detection_photo_id ON person_detection(photo_id);

CREATE INDEX IF NOT EXISTS idx_person_detection_person_id ON person_detection(person_id);

CREATE INDEX IF NOT EXISTS idx_person_detection_face_confidence ON person_detection(face_confidence);

CREATE INDEX IF NOT EXISTS idx_person_detection_cluster_status ON person_detection(cluster_status);

CREATE INDEX IF NOT EXISTS idx_person_detection_cluster_id ON person_detection(cluster_id);

CREATE INDEX IF NOT EXISTS idx_person_detection_unassigned ON person_detection(unassigned_since) WHERE cluster_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_person_detection_gender ON person_detection(gender);

CREATE INDEX IF NOT EXISTS idx_person_detection_age ON person_detection(age_estimate);

CREATE INDEX IF NOT EXISTS idx_person_first_name ON person(first_name);
CREATE INDEX IF NOT EXISTS idx_person_last_name ON person(last_name);

-- Face-level embeddings (for clustering & recognition)
CREATE TABLE IF NOT EXISTS face_embedding(
    person_detection_id bigint PRIMARY KEY REFERENCES person_detection(id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL
);

CREATE INDEX IF NOT EXISTS face_embedding_idx ON face_embedding USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Group of faces belonging to the same person
CREATE TABLE IF NOT EXISTS "cluster"(
    id bigserial PRIMARY KEY,
    face_count bigint DEFAULT 0,
    face_count_at_last_medoid bigint DEFAULT 0, -- For threshold-based medoid recomputation
    representative_detection_id bigint REFERENCES person_detection(id) ON DELETE SET NULL,
    centroid VECTOR(512),
    medoid_detection_id bigint REFERENCES person_detection(id) ON DELETE SET NULL,
    person_id bigint REFERENCES person(id) ON DELETE SET NULL,
    -- Verification status to protect human-verified clusters
    verified boolean DEFAULT false,
    verified_at timestamptz DEFAULT NULL,
    verified_by text DEFAULT NULL,
    -- Hidden clusters (ignored people)
    hidden boolean DEFAULT false,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL
);

-- Tracks potential cluster assignments requiring review
CREATE TABLE IF NOT EXISTS face_match_candidate(
    candidate_id bigserial PRIMARY KEY,
    detection_id bigint REFERENCES person_detection(id) ON DELETE CASCADE,
    cluster_id bigint REFERENCES "cluster"(id) ON DELETE CASCADE,
    similarity float NOT NULL,
    status text CHECK (status IN ('pending', 'accepted', 'rejected')) DEFAULT 'pending',
    created_at timestamp DEFAULT now()
);

-- Indexes for clustering performance
CREATE INDEX IF NOT EXISTS idx_cluster_centroid ON "cluster" USING ivfflat(centroid vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_face_match_candidate_detection ON face_match_candidate(detection_id);

CREATE INDEX IF NOT EXISTS idx_face_match_candidate_status ON face_match_candidate(status);

-- Must-link constraint: forces detections to be in the same cluster
CREATE TABLE IF NOT EXISTS must_link(
    id bigserial PRIMARY KEY,
    detection_id_1 bigint NOT NULL REFERENCES person_detection(id) ON DELETE CASCADE,
    detection_id_2 bigint NOT NULL REFERENCES person_detection(id) ON DELETE CASCADE,
    created_by text DEFAULT 'human', -- 'human' or 'system'
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(detection_id_1, detection_id_2),
    CHECK (detection_id_1 < detection_id_2) -- Canonical ordering to prevent duplicates
);

-- Cannot-link constraint: prevents detections from being in the same cluster
CREATE TABLE IF NOT EXISTS cannot_link(
    id bigserial PRIMARY KEY,
    detection_id_1 bigint NOT NULL REFERENCES person_detection(id) ON DELETE CASCADE,
    detection_id_2 bigint NOT NULL REFERENCES person_detection(id) ON DELETE CASCADE,
    created_by text DEFAULT 'human',
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(detection_id_1, detection_id_2),
    CHECK (detection_id_1 < detection_id_2)
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
CREATE INDEX IF NOT EXISTS idx_must_link_detection1 ON must_link(detection_id_1);
CREATE INDEX IF NOT EXISTS idx_must_link_detection2 ON must_link(detection_id_2);
CREATE INDEX IF NOT EXISTS idx_cannot_link_detection1 ON cannot_link(detection_id_1);
CREATE INDEX IF NOT EXISTS idx_cannot_link_detection2 ON cannot_link(detection_id_2);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c1 ON cluster_cannot_link(cluster_id_1);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c2 ON cluster_cannot_link(cluster_id_2);
CREATE INDEX IF NOT EXISTS idx_cluster_verified ON "cluster"(verified) WHERE verified = true;

-- Helper functions for constraint management

-- Function to add a must-link constraint with canonical ordering
CREATE OR REPLACE FUNCTION add_must_link(p_detection_id_1 bigint, p_detection_id_2 bigint, p_created_by text DEFAULT 'human')
RETURNS bigint AS $$
DECLARE
    v_id bigint;
    v_detection_1 bigint;
    v_detection_2 bigint;
BEGIN
    -- Ensure canonical ordering
    IF p_detection_id_1 < p_detection_id_2 THEN
        v_detection_1 := p_detection_id_1;
        v_detection_2 := p_detection_id_2;
    ELSE
        v_detection_1 := p_detection_id_2;
        v_detection_2 := p_detection_id_1;
    END IF;

    INSERT INTO must_link (detection_id_1, detection_id_2, created_by)
    VALUES (v_detection_1, v_detection_2, p_created_by)
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to add a cannot-link constraint with canonical ordering
CREATE OR REPLACE FUNCTION add_cannot_link(p_detection_id_1 bigint, p_detection_id_2 bigint, p_created_by text DEFAULT 'human')
RETURNS bigint AS $$
DECLARE
    v_id bigint;
    v_detection_1 bigint;
    v_detection_2 bigint;
BEGIN
    -- Ensure canonical ordering
    IF p_detection_id_1 < p_detection_id_2 THEN
        v_detection_1 := p_detection_id_1;
        v_detection_2 := p_detection_id_2;
    ELSE
        v_detection_1 := p_detection_id_2;
        v_detection_2 := p_detection_id_1;
    END IF;

    INSERT INTO cannot_link (detection_id_1, detection_id_2, created_by)
    VALUES (v_detection_1, v_detection_2, p_created_by)
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check if a constraint violation exists
CREATE OR REPLACE FUNCTION check_constraint_violations()
RETURNS TABLE(constraint_id bigint, cluster_id bigint, detection_id_1 bigint, detection_id_2 bigint) AS $$
BEGIN
    RETURN QUERY
    SELECT cl.id, pd1.cluster_id, cl.detection_id_1, cl.detection_id_2
    FROM cannot_link cl
    JOIN person_detection pd1 ON cl.detection_id_1 = pd1.id
    JOIN person_detection pd2 ON cl.detection_id_2 = pd2.id
    WHERE pd1.cluster_id = pd2.cluster_id
      AND pd1.cluster_id IS NOT NULL;
END;
$$ LANGUAGE plpgsql;
