-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

BEGIN;

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
-- Helper view for easy tag queries
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

-- Schema migration tracking
INSERT INTO schema_migrations (version, description)
VALUES ('006', 'Add prompt-based analysis system with embeddings and tagging')
ON CONFLICT (version) DO NOTHING;

COMMIT;
