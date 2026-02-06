CREATE EXTENSION IF NOT EXISTS vector;

-- Users table: Application users (minimal auth)
CREATE TABLE IF NOT EXISTS app_user(
    id bigserial PRIMARY KEY,
    username text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    first_name text NOT NULL,
    last_name text NOT NULL,
    default_collection_id bigint,
    created_at timestamptz DEFAULT now()
);

-- Collections table: Named groups of photos owned by users
CREATE TABLE IF NOT EXISTS collection(
    id bigserial PRIMARY KEY,
    owner_user_id bigint NOT NULL,
    name text NOT NULL,
    created_at timestamptz DEFAULT now(),
    FOREIGN KEY (owner_user_id) REFERENCES app_user(id) ON DELETE CASCADE
);

-- Collection membership table: Explicit access rows (owner included)
CREATE TABLE IF NOT EXISTS collection_member(
    collection_id bigint NOT NULL,
    user_id bigint NOT NULL,
    created_at timestamptz DEFAULT now(),
    PRIMARY KEY (collection_id, user_id),
    FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES app_user(id) ON DELETE CASCADE
);

-- Add default collection FK after collection exists (nullable to avoid cyclic bootstrapping)
ALTER TABLE app_user
    ADD CONSTRAINT app_user_default_collection_fk
    FOREIGN KEY (default_collection_id) REFERENCES collection(id) ON DELETE SET NULL;

-- Photo table: Core photo records
CREATE TABLE IF NOT EXISTS photo(
    id bigserial PRIMARY KEY,
    collection_id bigint NOT NULL,
    filename text NOT NULL, -- Relative path from INGEST_PATH
    normalized_path text, -- Path to normalized image in IMG_PATH
    width integer, -- Original image width in pixels
    height integer, -- Original image height in pixels
    normalized_width integer, -- Normalized image width in pixels
    normalized_height integer, -- Normalized image height in pixels
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE,
    UNIQUE (collection_id, filename),
    UNIQUE (collection_id, normalized_path)
);

-- Metadata table: Extracted photo metadata
CREATE TABLE IF NOT EXISTS metadata(
    photo_id bigint PRIMARY KEY,
    collection_id bigint NOT NULL,
    captured_at timestamptz, -- When photo was taken
    latitude real,
    longitude real,
    extra jsonb, -- All EXIF/TIFF/IFD metadata as JSONB (PostgreSQL native JSON)
    created_at timestamptz DEFAULT now(),
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
    FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE
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
-- Note: photo.filename has UNIQUE constraint which creates an implicit index

CREATE INDEX IF NOT EXISTS idx_photo_collection_id ON photo(collection_id, id);

CREATE INDEX IF NOT EXISTS idx_metadata_captured_at ON metadata(captured_at);
CREATE INDEX IF NOT EXISTS idx_metadata_collection_captured_at ON metadata(collection_id, captured_at);

-- Expression index for date-based queries using EXTRACT(YEAR/MONTH FROM captured_at)
-- Note: Must use AT TIME ZONE 'UTC' for EXTRACT to be immutable on timestamptz
CREATE INDEX IF NOT EXISTS idx_metadata_year_month
ON metadata (EXTRACT(YEAR FROM captured_at AT TIME ZONE 'UTC'), EXTRACT(MONTH FROM captured_at AT TIME ZONE 'UTC'));
CREATE INDEX IF NOT EXISTS idx_metadata_collection_year_month
ON metadata (collection_id, EXTRACT(YEAR FROM captured_at AT TIME ZONE 'UTC'), EXTRACT(MONTH FROM captured_at AT TIME ZONE 'UTC'));

CREATE INDEX IF NOT EXISTS idx_metadata_location ON metadata(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_metadata_collection_location ON metadata(collection_id, latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_status(status, stage);

-- Index for queries that filter by stage first (e.g., getUnprocessedPhotos)
CREATE INDEX IF NOT EXISTS idx_processing_status_stage_status ON processing_status(stage, status);

-- Partial index for efficient "find unprocessed" queries using NOT EXISTS pattern
CREATE INDEX IF NOT EXISTS idx_processing_status_completed
ON processing_status(photo_id)
WHERE status = 'completed';

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
    collection_id bigint NOT NULL,
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
    updated_at timestamp with time zone DEFAULT NOW(),
    FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE
);

-- Person Detection table: Detected faces/bodies in photos with bounding boxes and age/gender
CREATE TABLE IF NOT EXISTS person_detection(
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL,
    collection_id bigint NOT NULL,

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
    FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL,
    FOREIGN KEY (cluster_id) REFERENCES "cluster"(id) ON DELETE SET NULL
);

-- Indexes for person detection performance
CREATE INDEX IF NOT EXISTS idx_person_detection_photo_id ON person_detection(photo_id);
CREATE INDEX IF NOT EXISTS idx_person_detection_collection_id ON person_detection(collection_id);

CREATE INDEX IF NOT EXISTS idx_person_detection_person_id ON person_detection(person_id);

CREATE INDEX IF NOT EXISTS idx_person_detection_face_confidence ON person_detection(face_confidence);

CREATE INDEX IF NOT EXISTS idx_person_detection_cluster_status ON person_detection(cluster_status);

CREATE INDEX IF NOT EXISTS idx_person_detection_cluster_id ON person_detection(cluster_id);

CREATE INDEX IF NOT EXISTS idx_person_detection_unassigned ON person_detection(unassigned_since) WHERE cluster_id IS NULL;

-- Composite index for unassigned detection pool queries (clustering seed selection)
CREATE INDEX IF NOT EXISTS idx_person_detection_unassigned_pool
ON person_detection(face_confidence DESC, face_bbox_width, face_bbox_height)
WHERE cluster_id IS NULL AND cluster_status = 'unassigned';

CREATE INDEX IF NOT EXISTS idx_person_detection_gender ON person_detection(gender);

CREATE INDEX IF NOT EXISTS idx_person_detection_age ON person_detection(age_estimate);

CREATE INDEX IF NOT EXISTS idx_person_first_name ON person(first_name);
CREATE INDEX IF NOT EXISTS idx_person_last_name ON person(last_name);
CREATE INDEX IF NOT EXISTS idx_person_collection_first_name ON person(collection_id, first_name);
CREATE INDEX IF NOT EXISTS idx_person_collection_last_name ON person(collection_id, last_name);

-- Face-level embeddings (for clustering & recognition)
CREATE TABLE IF NOT EXISTS face_embedding(
    person_detection_id bigint PRIMARY KEY REFERENCES person_detection(id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL
);

CREATE INDEX IF NOT EXISTS face_embedding_idx ON face_embedding USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Group of faces belonging to the same person
CREATE TABLE IF NOT EXISTS "cluster"(
    id bigserial PRIMARY KEY,
    collection_id bigint NOT NULL,
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
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL,
    FOREIGN KEY (collection_id) REFERENCES collection(id) ON DELETE CASCADE
);

-- Tracks potential cluster assignments requiring review
CREATE TABLE IF NOT EXISTS face_match_candidate(
    candidate_id bigserial PRIMARY KEY,
    detection_id bigint REFERENCES person_detection(id) ON DELETE CASCADE,
    cluster_id bigint REFERENCES "cluster"(id) ON DELETE CASCADE,
    collection_id bigint NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    similarity float NOT NULL,
    status text CHECK (status IN ('pending', 'accepted', 'rejected')) DEFAULT 'pending',
    created_at timestamp DEFAULT now()
);

-- Indexes for clustering performance
CREATE INDEX IF NOT EXISTS idx_cluster_centroid ON "cluster" USING ivfflat(centroid vector_cosine_ops) WITH (lists = 100);

-- Partial indexes for cluster listing queries (visible vs hidden)
CREATE INDEX IF NOT EXISTS idx_cluster_visible
ON cluster(face_count DESC, id)
WHERE face_count > 0 AND (hidden = false OR hidden IS NULL);
CREATE INDEX IF NOT EXISTS idx_cluster_collection_visible
ON cluster(collection_id, face_count DESC, id)
WHERE face_count > 0 AND (hidden = false OR hidden IS NULL);

-- Simple filter index for hidden column (from migration 003)
CREATE INDEX IF NOT EXISTS idx_cluster_hidden ON cluster(hidden) WHERE hidden = true;

-- Composite index for hidden cluster listing with ORDER BY (from migration 008)
CREATE INDEX IF NOT EXISTS idx_cluster_hidden_listing
ON cluster(face_count DESC, id)
WHERE face_count > 0 AND hidden = true;
CREATE INDEX IF NOT EXISTS idx_cluster_collection_hidden_listing
ON cluster(collection_id, face_count DESC, id)
WHERE face_count > 0 AND hidden = true;

CREATE INDEX IF NOT EXISTS idx_face_match_candidate_detection ON face_match_candidate(detection_id);
CREATE INDEX IF NOT EXISTS idx_face_match_candidate_collection ON face_match_candidate(collection_id);

CREATE INDEX IF NOT EXISTS idx_face_match_candidate_status ON face_match_candidate(status);

-- Must-link constraint: forces detections to be in the same cluster
CREATE TABLE IF NOT EXISTS must_link(
    id bigserial PRIMARY KEY,
    detection_id_1 bigint NOT NULL REFERENCES person_detection(id) ON DELETE CASCADE,
    detection_id_2 bigint NOT NULL REFERENCES person_detection(id) ON DELETE CASCADE,
    collection_id bigint NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
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
    collection_id bigint NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
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
    collection_id bigint NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    created_at timestamptz DEFAULT NOW(),
    UNIQUE(cluster_id_1, cluster_id_2),
    CHECK (cluster_id_1 < cluster_id_2)
);

-- Indexes for constraint lookups
CREATE INDEX IF NOT EXISTS idx_must_link_detection1 ON must_link(detection_id_1);
CREATE INDEX IF NOT EXISTS idx_must_link_detection2 ON must_link(detection_id_2);
CREATE INDEX IF NOT EXISTS idx_must_link_collection ON must_link(collection_id);
CREATE INDEX IF NOT EXISTS idx_cannot_link_detection1 ON cannot_link(detection_id_1);
CREATE INDEX IF NOT EXISTS idx_cannot_link_detection2 ON cannot_link(detection_id_2);
CREATE INDEX IF NOT EXISTS idx_cannot_link_collection ON cannot_link(collection_id);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c1 ON cluster_cannot_link(cluster_id_1);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_c2 ON cluster_cannot_link(cluster_id_2);
CREATE INDEX IF NOT EXISTS idx_cluster_cannot_link_collection ON cluster_cannot_link(collection_id);
CREATE INDEX IF NOT EXISTS idx_cluster_verified ON "cluster"(verified) WHERE verified = true;

-- Helper functions for constraint management

-- Function to add a must-link constraint with canonical ordering
CREATE OR REPLACE FUNCTION add_must_link(p_detection_id_1 bigint, p_detection_id_2 bigint, p_collection_id bigint, p_created_by text DEFAULT 'human')
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

    INSERT INTO must_link (detection_id_1, detection_id_2, collection_id, created_by)
    VALUES (v_detection_1, v_detection_2, p_collection_id, p_created_by)
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to add a cannot-link constraint with canonical ordering
CREATE OR REPLACE FUNCTION add_cannot_link(p_detection_id_1 bigint, p_detection_id_2 bigint, p_collection_id bigint, p_created_by text DEFAULT 'human')
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

    INSERT INTO cannot_link (detection_id_1, detection_id_2, collection_id, created_by)
    VALUES (v_detection_1, v_detection_2, p_collection_id, p_created_by)
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

-- ============================================================================
-- Prompt-based analysis system
-- Supports 1000+ configurable prompts with precomputed embeddings
-- ============================================================================

-- Prompt categories organize prompts into logical groups
CREATE TABLE IF NOT EXISTS prompt_category (
    id serial PRIMARY KEY,
    name text UNIQUE NOT NULL,          -- 'face_emotion', 'scene_mood', 'scene_setting'
    target text NOT NULL                -- 'face' or 'scene'
        CHECK (target IN ('face', 'scene')),
    selection_mode text NOT NULL        -- 'single' (pick best) or 'multi' (all above threshold)
        CHECK (selection_mode IN ('single', 'multi')),
    min_confidence real DEFAULT 0.1,    -- minimum confidence to include in results
    max_results int DEFAULT 5,          -- max results for 'multi' mode
    description text,
    display_order int DEFAULT 0,        -- for UI ordering
    is_active boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Individual prompts with precomputed text embeddings
CREATE TABLE IF NOT EXISTS prompt_embedding (
    id serial PRIMARY KEY,
    category_id int NOT NULL REFERENCES prompt_category(id) ON DELETE CASCADE,
    label text NOT NULL,                -- 'happy', 'beach_sunset', etc.
    prompt_text text NOT NULL,          -- 'a photo of a happy smiling person'
    embedding vector(512),              -- precomputed MobileCLIP text embedding
    model_name text NOT NULL,           -- 'MobileCLIP-S2' (for cache invalidation)
    model_version text,
    display_name text,                  -- 'Happy' (for UI, nullable = use label)
    parent_label text,                  -- optional hierarchy: 'outdoor' -> 'beach'
    confidence_boost real DEFAULT 0.0,  -- adjust confidence for rare/common labels
    metadata jsonb,                     -- additional data (synonyms, examples, etc.)
    is_active boolean DEFAULT true,
    embedding_computed_at timestamptz,  -- track when embedding was computed
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(category_id, label)
);

-- Indexes for prompt queries
CREATE INDEX IF NOT EXISTS idx_prompt_embedding_category
    ON prompt_embedding(category_id) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_prompt_embedding_model
    ON prompt_embedding(model_name);
CREATE INDEX IF NOT EXISTS idx_prompt_embedding_parent
    ON prompt_embedding(parent_label) WHERE parent_label IS NOT NULL;

-- Vector index for similarity search (if needed for prompt discovery)
CREATE INDEX IF NOT EXISTS idx_prompt_embedding_vector
    ON prompt_embedding USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- Analysis output storage (model-agnostic)
-- ============================================================================

CREATE TABLE IF NOT EXISTS analysis_output (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL,
    model_type text NOT NULL,           -- 'classifier', 'tagger', 'detector'
    model_name text NOT NULL,           -- 'apple_vision_classify', 'mobileclip', etc.
    model_version text,
    output jsonb NOT NULL,              -- raw model output
    processing_time_ms integer,
    device text,                        -- 'cpu', 'mps', 'cuda', 'ane'
    created_at timestamptz DEFAULT now(),
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analysis_output_photo_model
    ON analysis_output(photo_id, model_type, model_name);
CREATE INDEX IF NOT EXISTS idx_analysis_output_output
    ON analysis_output USING GIN(output);

-- ============================================================================
-- Photo tags: Multi-label results from prompt-based classification
-- ============================================================================

CREATE TABLE IF NOT EXISTS photo_tag (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    prompt_id int NOT NULL REFERENCES prompt_embedding(id) ON DELETE CASCADE,
    confidence real NOT NULL,
    rank_in_category int,               -- 1 = top match in category
    analysis_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL,
    created_at timestamptz DEFAULT now(),
    UNIQUE(photo_id, prompt_id)
);

CREATE INDEX IF NOT EXISTS idx_photo_tag_photo ON photo_tag(photo_id);
CREATE INDEX IF NOT EXISTS idx_photo_tag_prompt ON photo_tag(prompt_id);
CREATE INDEX IF NOT EXISTS idx_photo_tag_confidence ON photo_tag(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_photo_tag_high_confidence
    ON photo_tag(photo_id, confidence) WHERE confidence > 0.5;

-- ============================================================================
-- Detection tags: Per-detection (face) tags
-- ============================================================================

CREATE TABLE IF NOT EXISTS detection_tag (
    id bigserial PRIMARY KEY,
    detection_id bigint NOT NULL REFERENCES person_detection(id) ON DELETE CASCADE,
    prompt_id int NOT NULL REFERENCES prompt_embedding(id) ON DELETE CASCADE,
    confidence real NOT NULL,
    rank_in_category int,
    analysis_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL,
    created_at timestamptz DEFAULT now(),
    UNIQUE(detection_id, prompt_id)
);

CREATE INDEX IF NOT EXISTS idx_detection_tag_detection ON detection_tag(detection_id);
CREATE INDEX IF NOT EXISTS idx_detection_tag_prompt ON detection_tag(prompt_id);
CREATE INDEX IF NOT EXISTS idx_detection_tag_confidence ON detection_tag(confidence DESC);

-- ============================================================================
-- Scene analysis: Photo-level Apple Vision taxonomy
-- ============================================================================

CREATE TABLE IF NOT EXISTS scene_analysis (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL UNIQUE,

    -- Apple Vision taxonomy (VNClassifyImageRequest)
    taxonomy_labels text[],             -- Top labels from Vision framework
    taxonomy_confidences real[],
    taxonomy_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL,

    -- MobileCLIP analysis metadata
    mobileclip_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL,

    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),

    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scene_analysis_labels
    ON scene_analysis USING GIN(taxonomy_labels);

-- ============================================================================
-- Model registry: Track available models
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_registry (
    id serial PRIMARY KEY,
    name text UNIQUE NOT NULL,
    display_name text NOT NULL,
    model_type text NOT NULL,
    capabilities text[] NOT NULL,
    embedding_dimension int,
    config jsonb,
    is_active boolean DEFAULT true,
    created_at timestamptz DEFAULT now()
);

INSERT INTO model_registry (name, display_name, model_type, capabilities, embedding_dimension, config) VALUES
    ('yolo_person_face', 'YOLO Person+Face', 'detector',
     ARRAY['face_detection', 'body_detection'], NULL,
     '{"model": "yolov8x_person_face"}'),
    ('mivolo', 'MiVOLO', 'estimator',
     ARRAY['age_estimation', 'gender_estimation'], NULL, '{}'),
    ('insightface_buffalo', 'InsightFace Buffalo', 'embedder',
     ARRAY['face_embedding'], 512,
     '{"model": "buffalo_l"}'),
    ('apple_vision_classify', 'Apple Vision Classify', 'classifier',
     ARRAY['scene_taxonomy'], NULL,
     '{"classes": 1303}'),
    ('mobileclip_s2', 'MobileCLIP-S2', 'tagger',
     ARRAY['zero_shot', 'scene_tagging', 'face_tagging'], 512,
     '{"source": "datacompdr"}')
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- Seed initial prompt categories
-- ============================================================================

INSERT INTO prompt_category (name, target, selection_mode, min_confidence, description, display_order) VALUES
    -- Face categories
    ('face_emotion', 'face', 'single', 0.15, 'Primary emotional expression', 10),
    ('face_expression', 'face', 'multi', 0.2, 'Facial expression details', 20),
    ('face_gaze', 'face', 'single', 0.2, 'Where the person is looking', 30),

    -- Scene categories
    ('scene_mood', 'scene', 'single', 0.15, 'Overall emotional mood of scene', 100),
    ('scene_setting', 'scene', 'multi', 0.1, 'Physical location/environment', 110),
    ('scene_activity', 'scene', 'multi', 0.15, 'Activities happening in scene', 120),
    ('scene_time', 'scene', 'single', 0.2, 'Time of day', 130),
    ('scene_weather', 'scene', 'single', 0.2, 'Weather conditions', 140),
    ('scene_social', 'scene', 'single', 0.15, 'Social context', 150)
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- Helper views for easy tag queries
-- ============================================================================

CREATE OR REPLACE VIEW photo_tags_view AS
SELECT
    pt.photo_id,
    pc.name as category,
    pc.target,
    pe.label,
    pe.display_name,
    pt.confidence,
    pt.rank_in_category
FROM photo_tag pt
JOIN prompt_embedding pe ON pt.prompt_id = pe.id
JOIN prompt_category pc ON pe.category_id = pc.id
WHERE pe.is_active AND pc.is_active
ORDER BY pt.photo_id, pc.display_order, pt.rank_in_category;

CREATE OR REPLACE VIEW detection_tags_view AS
SELECT
    dt.detection_id,
    pd.photo_id,
    pc.name as category,
    pe.label,
    pe.display_name,
    dt.confidence,
    dt.rank_in_category
FROM detection_tag dt
JOIN person_detection pd ON dt.detection_id = pd.id
JOIN prompt_embedding pe ON dt.prompt_id = pe.id
JOIN prompt_category pc ON pe.category_id = pc.id
WHERE pe.is_active AND pc.is_active
ORDER BY dt.detection_id, pc.display_order, dt.rank_in_category;
