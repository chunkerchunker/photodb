# Scene Taxonomy & Sentiment Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add scene taxonomy classification (Apple Vision) and generalizable prompt-based sentiment/tagging analysis (MobileCLIP) to the existing YOLO+MiVOLO detection pipeline.

**Architecture:** Keep existing YOLO→MiVOLO→InsightFace pipeline. Add: (1) Apple Vision `VNClassifyImageRequest` for scene taxonomy, (2) MobileCLIP with configurable prompt embeddings stored in database for zero-shot classification. Prompts are organized into categories (face_emotion, scene_mood, scene_setting, etc.) with precomputed text embeddings for efficient inference over 1000+ prompts.

**Tech Stack:** PyObjC (`pyobjc-framework-Vision`), MobileCLIP via `open_clip`, pgvector for prompt embeddings, PostgreSQL with JSONB.

---

## Architecture Overview

```
Photo (normalized)
  │
  ├─► YOLO ─────────────────► Face/Body boxes ─┬─► MiVOLO ──────► Age/Gender
  │   (existing)                               │
  │                                            ├─► InsightFace ─► Embeddings
  │                                            │
  │                                            └─► MobileCLIP ──► Face tags
  │                                                (per face)     (from prompt_embedding)
  │
  ├─► VNClassifyImageRequest ─► Scene taxonomy (1303 labels)
  │   (Apple Vision)
  │
  └─► MobileCLIP ─────────────► Scene tags
      (full image)               (from prompt_embedding)
                                 - mood: joyful, peaceful, somber...
                                 - setting: indoor, outdoor, beach...
                                 - activity: celebration, work, travel...
```

## Prompt Embedding System

```
prompt_category                    prompt_embedding
┌─────────────────────┐           ┌────────────────────────────────┐
│ id: 1               │           │ id: 1                          │
│ name: face_emotion  │◄──────────│ category_id: 1                 │
│ target: face        │           │ label: happy                   │
│ selection_mode:     │           │ prompt_text: "a photo of a     │
│   single            │           │   happy smiling person"        │
└─────────────────────┘           │ embedding: vector(512)         │
                                  │ model_name: MobileCLIP-S2      │
┌─────────────────────┐           └────────────────────────────────┘
│ id: 2               │
│ name: scene_setting │           ┌────────────────────────────────┐
│ target: scene       │◄──────────│ id: 50                         │
│ selection_mode:     │           │ category_id: 2                 │
│   multi             │           │ label: beach                   │
└─────────────────────┘           │ prompt_text: "a photo taken    │
                                  │   at a beach with ocean"       │
        ▼                         │ embedding: vector(512)         │
                                  └────────────────────────────────┘
   Runtime: PromptCache
   - Loads embeddings into GPU tensor
   - Single matmul for 1000+ prompts
   - ~0.5ms per category
```

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add new dependencies**

```toml
# In dependencies list, add:
    "pyobjc-framework-Vision>=10.0",
    "pyobjc-framework-Quartz>=10.0",
    "open-clip-torch>=2.24.0",
```

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Successfully installs packages

**Step 3: Verify imports**

Run: `uv run python -c "import Vision; import open_clip; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pyobjc-framework-Vision and open-clip-torch"
```

---

## Task 2: Create Database Migration

**Files:**
- Create: `migrations/006_add_prompt_based_analysis.sql`

**Step 1: Write migration SQL**

```sql
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
```

**Step 2: Apply migration**

Run: `psql $DATABASE_URL -f migrations/006_add_prompt_based_analysis.sql`
Expected: Tables created successfully

**Step 3: Verify schema**

Run: `psql $DATABASE_URL -c "\d prompt_category" && psql $DATABASE_URL -c "\d prompt_embedding"`
Expected: Shows expected columns

**Step 4: Commit**

```bash
git add migrations/006_add_prompt_based_analysis.sql
git commit -m "db: add prompt-based analysis schema with categories and embeddings"
```

---

## Task 3: Add Database Models

**Files:**
- Modify: `src/photodb/database/models.py`

**Step 1: Add new dataclasses**

Add after the existing models:

```python
@dataclass
class PromptCategory:
    """Category for organizing prompts (face_emotion, scene_setting, etc.)."""

    id: Optional[int]
    name: str
    target: str  # 'face' or 'scene'
    selection_mode: str  # 'single' or 'multi'
    min_confidence: float
    max_results: int
    description: Optional[str]
    display_order: int
    is_active: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        name: str,
        target: str,
        selection_mode: str = "single",
        min_confidence: float = 0.1,
        max_results: int = 5,
        description: Optional[str] = None,
        display_order: int = 0,
    ) -> "PromptCategory":
        now = datetime.now(timezone.utc)
        return cls(
            id=None,
            name=name,
            target=target,
            selection_mode=selection_mode,
            min_confidence=min_confidence,
            max_results=max_results,
            description=description,
            display_order=display_order,
            is_active=True,
            created_at=now,
            updated_at=now,
        )


@dataclass
class PromptEmbedding:
    """A prompt with precomputed text embedding for zero-shot classification."""

    id: Optional[int]
    category_id: int
    label: str
    prompt_text: str
    embedding: Optional[List[float]]  # 512-dim vector
    model_name: str
    model_version: Optional[str]
    display_name: Optional[str]
    parent_label: Optional[str]
    confidence_boost: float
    metadata: Optional[Dict[str, Any]]
    is_active: bool
    embedding_computed_at: Optional[datetime]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        category_id: int,
        label: str,
        prompt_text: str,
        model_name: str,
        embedding: Optional[List[float]] = None,
        display_name: Optional[str] = None,
        parent_label: Optional[str] = None,
        confidence_boost: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PromptEmbedding":
        now = datetime.now(timezone.utc)
        return cls(
            id=None,
            category_id=category_id,
            label=label,
            prompt_text=prompt_text,
            embedding=embedding,
            model_name=model_name,
            model_version=None,
            display_name=display_name,
            parent_label=parent_label,
            confidence_boost=confidence_boost,
            metadata=metadata,
            is_active=True,
            embedding_computed_at=now if embedding else None,
            created_at=now,
            updated_at=now,
        )


@dataclass
class PhotoTag:
    """A tag assigned to a photo from prompt-based classification."""

    id: Optional[int]
    photo_id: int
    prompt_id: int
    confidence: float
    rank_in_category: Optional[int]
    analysis_output_id: Optional[int]
    created_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        photo_id: int,
        prompt_id: int,
        confidence: float,
        rank_in_category: Optional[int] = None,
        analysis_output_id: Optional[int] = None,
    ) -> "PhotoTag":
        return cls(
            id=None,
            photo_id=photo_id,
            prompt_id=prompt_id,
            confidence=confidence,
            rank_in_category=rank_in_category,
            analysis_output_id=analysis_output_id,
            created_at=datetime.now(timezone.utc),
        )


@dataclass
class DetectionTag:
    """A tag assigned to a face detection from prompt-based classification."""

    id: Optional[int]
    detection_id: int
    prompt_id: int
    confidence: float
    rank_in_category: Optional[int]
    analysis_output_id: Optional[int]
    created_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        detection_id: int,
        prompt_id: int,
        confidence: float,
        rank_in_category: Optional[int] = None,
        analysis_output_id: Optional[int] = None,
    ) -> "DetectionTag":
        return cls(
            id=None,
            detection_id=detection_id,
            prompt_id=prompt_id,
            confidence=confidence,
            rank_in_category=rank_in_category,
            analysis_output_id=analysis_output_id,
            created_at=datetime.now(timezone.utc),
        )


@dataclass
class AnalysisOutput:
    """Model-agnostic storage for raw analysis outputs."""

    id: Optional[int]
    photo_id: int
    model_type: str
    model_name: str
    model_version: Optional[str]
    output: Dict[str, Any]
    processing_time_ms: Optional[int]
    device: Optional[str]
    created_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        photo_id: int,
        model_type: str,
        model_name: str,
        output: Dict[str, Any],
        model_version: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        device: Optional[str] = None,
    ) -> "AnalysisOutput":
        return cls(
            id=None,
            photo_id=photo_id,
            model_type=model_type,
            model_name=model_name,
            model_version=model_version,
            output=output,
            processing_time_ms=processing_time_ms,
            device=device,
            created_at=datetime.now(timezone.utc),
        )


@dataclass
class SceneAnalysis:
    """Photo-level scene analysis results."""

    id: Optional[int]
    photo_id: int
    taxonomy_labels: Optional[List[str]]
    taxonomy_confidences: Optional[List[float]]
    taxonomy_output_id: Optional[int]
    mobileclip_output_id: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        photo_id: int,
        taxonomy_labels: Optional[List[str]] = None,
        taxonomy_confidences: Optional[List[float]] = None,
        taxonomy_output_id: Optional[int] = None,
        mobileclip_output_id: Optional[int] = None,
    ) -> "SceneAnalysis":
        now = datetime.now(timezone.utc)
        return cls(
            id=None,
            photo_id=photo_id,
            taxonomy_labels=taxonomy_labels,
            taxonomy_confidences=taxonomy_confidences,
            taxonomy_output_id=taxonomy_output_id,
            mobileclip_output_id=mobileclip_output_id,
            created_at=now,
            updated_at=now,
        )
```

**Step 2: Commit**

```bash
git add src/photodb/database/models.py
git commit -m "models: add PromptCategory, PromptEmbedding, PhotoTag, DetectionTag"
```

---

## Task 4: Add Repository Methods

**Files:**
- Modify: `src/photodb/database/pg_repository.py`

**Step 1: Add prompt category methods**

```python
def get_prompt_categories(
    self, target: Optional[str] = None, active_only: bool = True
) -> List[PromptCategory]:
    """Get prompt categories, optionally filtered by target."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            query = "SELECT * FROM prompt_category WHERE 1=1"
            params = []

            if active_only:
                query += " AND is_active = true"
            if target:
                query += " AND target = %s"
                params.append(target)

            query += " ORDER BY display_order"
            cur.execute(query, params)

            return [self._row_to_prompt_category(row) for row in cur.fetchall()]

def get_prompt_category_by_name(self, name: str) -> Optional[PromptCategory]:
    """Get a prompt category by name."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM prompt_category WHERE name = %s", (name,))
            row = cur.fetchone()
            return self._row_to_prompt_category(row) if row else None

def _row_to_prompt_category(self, row) -> PromptCategory:
    return PromptCategory(
        id=row[0], name=row[1], target=row[2], selection_mode=row[3],
        min_confidence=row[4], max_results=row[5], description=row[6],
        display_order=row[7], is_active=row[8], created_at=row[9], updated_at=row[10],
    )
```

**Step 2: Add prompt embedding methods**

```python
def get_prompts_by_category(
    self, category_id: int, active_only: bool = True, with_embeddings: bool = True
) -> List[PromptEmbedding]:
    """Get all prompts for a category."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cols = "*" if with_embeddings else "id, category_id, label, prompt_text, NULL as embedding, model_name, model_version, display_name, parent_label, confidence_boost, metadata, is_active, embedding_computed_at, created_at, updated_at"
            query = f"SELECT {cols} FROM prompt_embedding WHERE category_id = %s"
            params = [category_id]

            if active_only:
                query += " AND is_active = true"

            query += " ORDER BY label"
            cur.execute(query, params)

            return [self._row_to_prompt_embedding(row) for row in cur.fetchall()]

def get_prompts_needing_embedding(self, model_name: str) -> List[PromptEmbedding]:
    """Get prompts that need embedding computation for a model."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM prompt_embedding
                WHERE is_active = true
                AND (embedding IS NULL OR model_name != %s)
                ORDER BY category_id, label
                """,
                (model_name,),
            )
            return [self._row_to_prompt_embedding(row) for row in cur.fetchall()]

def upsert_prompt_embedding(self, prompt: PromptEmbedding) -> PromptEmbedding:
    """Insert or update a prompt embedding."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prompt_embedding
                (category_id, label, prompt_text, embedding, model_name, model_version,
                 display_name, parent_label, confidence_boost, metadata, is_active,
                 embedding_computed_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (category_id, label) DO UPDATE SET
                    prompt_text = EXCLUDED.prompt_text,
                    embedding = EXCLUDED.embedding,
                    model_name = EXCLUDED.model_name,
                    model_version = EXCLUDED.model_version,
                    display_name = EXCLUDED.display_name,
                    parent_label = EXCLUDED.parent_label,
                    confidence_boost = EXCLUDED.confidence_boost,
                    metadata = EXCLUDED.metadata,
                    embedding_computed_at = EXCLUDED.embedding_computed_at,
                    updated_at = EXCLUDED.updated_at
                RETURNING id
                """,
                (
                    prompt.category_id, prompt.label, prompt.prompt_text,
                    prompt.embedding, prompt.model_name, prompt.model_version,
                    prompt.display_name, prompt.parent_label, prompt.confidence_boost,
                    json.dumps(prompt.metadata) if prompt.metadata else None,
                    prompt.is_active, prompt.embedding_computed_at,
                    prompt.created_at, prompt.updated_at,
                ),
            )
            prompt.id = cur.fetchone()[0]
            conn.commit()
    return prompt

def update_prompt_embedding_vector(
    self, prompt_id: int, embedding: List[float], model_name: str, model_version: Optional[str] = None
) -> None:
    """Update just the embedding vector for a prompt."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE prompt_embedding
                SET embedding = %s, model_name = %s, model_version = %s,
                    embedding_computed_at = now(), updated_at = now()
                WHERE id = %s
                """,
                (embedding, model_name, model_version, prompt_id),
            )
            conn.commit()

def _row_to_prompt_embedding(self, row) -> PromptEmbedding:
    return PromptEmbedding(
        id=row[0], category_id=row[1], label=row[2], prompt_text=row[3],
        embedding=list(row[4]) if row[4] else None,
        model_name=row[5], model_version=row[6], display_name=row[7],
        parent_label=row[8], confidence_boost=row[9],
        metadata=row[10] if isinstance(row[10], dict) else json.loads(row[10]) if row[10] else None,
        is_active=row[11], embedding_computed_at=row[12],
        created_at=row[13], updated_at=row[14],
    )
```

**Step 3: Add photo/detection tag methods**

```python
def bulk_upsert_photo_tags(self, tags: List[PhotoTag]) -> None:
    """Bulk insert/update photo tags."""
    if not tags:
        return
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            # Delete existing tags for this photo in affected categories
            photo_id = tags[0].photo_id
            prompt_ids = [t.prompt_id for t in tags]
            cur.execute(
                """
                DELETE FROM photo_tag
                WHERE photo_id = %s AND prompt_id = ANY(%s)
                """,
                (photo_id, prompt_ids),
            )

            # Insert new tags
            for tag in tags:
                cur.execute(
                    """
                    INSERT INTO photo_tag
                    (photo_id, prompt_id, confidence, rank_in_category, analysis_output_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (photo_id, prompt_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        rank_in_category = EXCLUDED.rank_in_category,
                        analysis_output_id = EXCLUDED.analysis_output_id
                    """,
                    (tag.photo_id, tag.prompt_id, tag.confidence,
                     tag.rank_in_category, tag.analysis_output_id, tag.created_at),
                )
            conn.commit()

def bulk_upsert_detection_tags(self, tags: List[DetectionTag]) -> None:
    """Bulk insert/update detection tags."""
    if not tags:
        return
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            for tag in tags:
                cur.execute(
                    """
                    INSERT INTO detection_tag
                    (detection_id, prompt_id, confidence, rank_in_category, analysis_output_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (detection_id, prompt_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        rank_in_category = EXCLUDED.rank_in_category,
                        analysis_output_id = EXCLUDED.analysis_output_id
                    """,
                    (tag.detection_id, tag.prompt_id, tag.confidence,
                     tag.rank_in_category, tag.analysis_output_id, tag.created_at),
                )
            conn.commit()

def get_photo_tags(self, photo_id: int) -> List[Dict[str, Any]]:
    """Get all tags for a photo with category info."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM photo_tags_view WHERE photo_id = %s
                """,
                (photo_id,),
            )
            return [
                {
                    "photo_id": row[0], "category": row[1], "target": row[2],
                    "label": row[3], "display_name": row[4],
                    "confidence": row[5], "rank": row[6],
                }
                for row in cur.fetchall()
            ]
```

**Step 4: Add analysis output and scene analysis methods**

```python
def create_analysis_output(self, output: AnalysisOutput) -> AnalysisOutput:
    """Insert analysis output record."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analysis_output
                (photo_id, model_type, model_name, model_version, output,
                 processing_time_ms, device, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    output.photo_id, output.model_type, output.model_name,
                    output.model_version, json.dumps(output.output),
                    output.processing_time_ms, output.device, output.created_at,
                ),
            )
            output.id = cur.fetchone()[0]
            conn.commit()
    return output

def upsert_scene_analysis(self, analysis: SceneAnalysis) -> SceneAnalysis:
    """Insert or update scene analysis."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scene_analysis
                (photo_id, taxonomy_labels, taxonomy_confidences,
                 taxonomy_output_id, mobileclip_output_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (photo_id) DO UPDATE SET
                    taxonomy_labels = COALESCE(EXCLUDED.taxonomy_labels, scene_analysis.taxonomy_labels),
                    taxonomy_confidences = COALESCE(EXCLUDED.taxonomy_confidences, scene_analysis.taxonomy_confidences),
                    taxonomy_output_id = COALESCE(EXCLUDED.taxonomy_output_id, scene_analysis.taxonomy_output_id),
                    mobileclip_output_id = COALESCE(EXCLUDED.mobileclip_output_id, scene_analysis.mobileclip_output_id),
                    updated_at = EXCLUDED.updated_at
                RETURNING id
                """,
                (
                    analysis.photo_id, analysis.taxonomy_labels, analysis.taxonomy_confidences,
                    analysis.taxonomy_output_id, analysis.mobileclip_output_id,
                    analysis.created_at, analysis.updated_at,
                ),
            )
            analysis.id = cur.fetchone()[0]
            conn.commit()
    return analysis
```

**Step 5: Commit**

```bash
git add src/photodb/database/pg_repository.py
git commit -m "repo: add methods for prompts, tags, and scene analysis"
```

---

## Task 5: Create Apple Vision Scene Classifier

**Files:**
- Create: `src/photodb/utils/apple_vision_classifier.py`
- Test: `tests/test_apple_vision_classifier.py`

**Step 1: Write failing test**

```python
# tests/test_apple_vision_classifier.py
"""Tests for Apple Vision scene classifier."""
import sys
import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Apple Vision only available on macOS"
)


class TestAppleVisionClassifier:
    def test_classify_returns_labels(self, test_image_path):
        from photodb.utils.apple_vision_classifier import AppleVisionClassifier
        classifier = AppleVisionClassifier()
        result = classifier.classify(str(test_image_path))

        assert result["status"] in ("success", "error")
        if result["status"] == "success":
            assert len(result["classifications"]) > 0
            assert "identifier" in result["classifications"][0]
            assert "confidence" in result["classifications"][0]

    def test_classify_returns_top_k(self, test_image_path):
        from photodb.utils.apple_vision_classifier import AppleVisionClassifier
        classifier = AppleVisionClassifier()
        result = classifier.classify(str(test_image_path), top_k=5)

        if result["status"] == "success":
            assert len(result["classifications"]) <= 5


@pytest.fixture
def test_image_path():
    return Path(__file__).parent.parent / "test_photos" / "test.jpg"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_apple_vision_classifier.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement AppleVisionClassifier**

```python
# src/photodb/utils/apple_vision_classifier.py
"""Apple Vision scene classifier using VNClassifyImageRequest."""
import logging
import sys
import time
from typing import Any, Dict

if sys.platform != "darwin":
    raise ImportError("Apple Vision only available on macOS")

import Quartz
import Vision
from Foundation import NSURL

logger = logging.getLogger(__name__)


class AppleVisionClassifier:
    """Classify scene content using Apple Vision Framework (1303 labels)."""

    def __init__(self):
        logger.info("AppleVisionClassifier initialized")

    def classify(
        self, image_path: str, top_k: int = 10, min_confidence: float = 0.01
    ) -> Dict[str, Any]:
        """Classify scene content in an image."""
        start_time = time.time()

        try:
            image_url = NSURL.fileURLWithPath_(image_path)
            ci_image = Quartz.CIImage.imageWithContentsOfURL_(image_url)

            if ci_image is None:
                return {
                    "status": "error",
                    "classifications": [],
                    "error": f"Failed to load image: {image_path}",
                }

            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                ci_image, None
            )
            request = Vision.VNClassifyImageRequest.alloc().init()
            success, error = handler.performRequests_error_([request], None)

            if not success:
                return {
                    "status": "error",
                    "classifications": [],
                    "error": str(error) if error else "Classification failed",
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                }

            results = request.results() or []
            classifications = []

            for observation in results:
                conf = float(observation.confidence())
                if conf >= min_confidence:
                    classifications.append({
                        "identifier": str(observation.identifier()),
                        "confidence": conf,
                    })

            classifications.sort(key=lambda x: x["confidence"], reverse=True)
            classifications = classifications[:top_k]

            return {
                "status": "success",
                "classifications": classifications,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "status": "error",
                "classifications": [],
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_apple_vision_classifier.py -v`
Expected: PASS (on macOS)

**Step 5: Commit**

```bash
git add src/photodb/utils/apple_vision_classifier.py tests/test_apple_vision_classifier.py
git commit -m "feat: add AppleVisionClassifier for scene taxonomy"
```

---

## Task 6: Create Prompt Cache

**Files:**
- Create: `src/photodb/utils/prompt_cache.py`
- Test: `tests/test_prompt_cache.py`

**Step 1: Write failing test**

```python
# tests/test_prompt_cache.py
"""Tests for prompt embedding cache."""
import pytest
from unittest.mock import MagicMock
import torch


class TestPromptCache:
    def test_load_category_embeddings(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        # Mock repository
        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1, name="test", target="scene", selection_mode="single",
            min_confidence=0.1, max_results=5, description=None,
            display_order=0, is_active=True, created_at=None, updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=1, category_id=1, label="happy", prompt_text="happy scene",
                embedding=[0.1] * 512, model_name="test", model_version=None,
                display_name=None, parent_label=None, confidence_boost=0.0,
                metadata=None, is_active=True, embedding_computed_at=None,
                created_at=None, updated_at=None,
            ),
            PromptEmbedding(
                id=2, category_id=1, label="sad", prompt_text="sad scene",
                embedding=[0.2] * 512, model_name="test", model_version=None,
                display_name=None, parent_label=None, confidence_boost=0.0,
                metadata=None, is_active=True, embedding_computed_at=None,
                created_at=None, updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)
        labels, embeddings, category = cache.get_category("test")

        assert labels == ["happy", "sad"]
        assert embeddings.shape == (2, 512)
        assert category.selection_mode == "single"

    def test_classify_returns_scores(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1, name="test", target="scene", selection_mode="single",
            min_confidence=0.1, max_results=5, description=None,
            display_order=0, is_active=True, created_at=None, updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=1, category_id=1, label="a", prompt_text="a",
                embedding=[1.0] + [0.0] * 511, model_name="test", model_version=None,
                display_name=None, parent_label=None, confidence_boost=0.0,
                metadata=None, is_active=True, embedding_computed_at=None,
                created_at=None, updated_at=None,
            ),
            PromptEmbedding(
                id=2, category_id=1, label="b", prompt_text="b",
                embedding=[0.0] * 512, model_name="test", model_version=None,
                display_name=None, parent_label=None, confidence_boost=0.0,
                metadata=None, is_active=True, embedding_computed_at=None,
                created_at=None, updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)

        # Image embedding similar to "a"
        image_embedding = torch.tensor([[1.0] + [0.0] * 511])
        results = cache.classify(image_embedding, "test")

        assert "a" in results
        assert "b" in results
        assert results["a"] > results["b"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_cache.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement PromptCache**

```python
# src/photodb/utils/prompt_cache.py
"""
Prompt embedding cache for efficient zero-shot classification.

Caches prompt embeddings in GPU/CPU tensors for fast similarity computation.
Supports 1000+ prompts per category with single matrix multiplication.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..database.models import PromptCategory, PromptEmbedding

logger = logging.getLogger(__name__)


class PromptCache:
    """Cache prompt embeddings for fast classification.

    Usage:
        cache = PromptCache(repository)
        labels, embeddings, category = cache.get_category("scene_mood")
        scores = cache.classify(image_embedding, "scene_mood")
    """

    def __init__(self, repository, device: Optional[str] = None):
        """
        Initialize prompt cache.

        Args:
            repository: Database repository for loading prompts.
            device: Device for tensors ('cpu', 'mps', 'cuda'). Auto-detects if None.
        """
        self._repository = repository
        self._cache: Dict[str, Tuple[List[str], List[int], torch.Tensor, PromptCategory]] = {}

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        logger.info(f"PromptCache initialized on {device}")

    def get_category(
        self, category_name: str
    ) -> Tuple[List[str], torch.Tensor, PromptCategory]:
        """
        Get cached embeddings for a category.

        Args:
            category_name: Name of the prompt category.

        Returns:
            Tuple of (labels, embeddings_tensor, category).
        """
        if category_name not in self._cache:
            self._load_category(category_name)

        labels, prompt_ids, embeddings, category = self._cache[category_name]
        return labels, embeddings, category

    def get_prompt_ids(self, category_name: str) -> List[int]:
        """Get prompt IDs for a category (for creating tags)."""
        if category_name not in self._cache:
            self._load_category(category_name)
        return self._cache[category_name][1]

    def _load_category(self, category_name: str) -> None:
        """Load a category's prompts into cache."""
        category = self._repository.get_prompt_category_by_name(category_name)
        if not category:
            raise ValueError(f"Unknown prompt category: {category_name}")

        prompts = self._repository.get_prompts_by_category(category.id, active_only=True)
        if not prompts:
            raise ValueError(f"No prompts found for category: {category_name}")

        # Filter prompts with embeddings
        prompts_with_embeddings = [p for p in prompts if p.embedding is not None]
        if not prompts_with_embeddings:
            raise ValueError(f"No computed embeddings for category: {category_name}")

        labels = [p.label for p in prompts_with_embeddings]
        prompt_ids = [p.id for p in prompts_with_embeddings]

        # Stack embeddings into tensor
        embeddings = torch.stack([
            torch.tensor(p.embedding, dtype=torch.float32)
            for p in prompts_with_embeddings
        ]).to(self._device)

        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        self._cache[category_name] = (labels, prompt_ids, embeddings, category)
        logger.info(f"Cached {len(labels)} prompts for category '{category_name}'")

    def classify(
        self,
        image_embedding: torch.Tensor,
        category_name: str,
        temperature: float = 100.0,
    ) -> Dict[str, float]:
        """
        Classify image against all prompts in a category.

        Args:
            image_embedding: Image embedding tensor (1, 512) or (512,).
            category_name: Name of the prompt category.
            temperature: Softmax temperature (higher = sharper distribution).

        Returns:
            Dict mapping labels to confidence scores.
        """
        labels, text_embeddings, category = self.get_category(category_name)

        # Ensure image embedding is 2D and normalized
        if image_embedding.dim() == 1:
            image_embedding = image_embedding.unsqueeze(0)
        image_embedding = image_embedding.to(self._device)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Compute cosine similarity via matrix multiplication
        similarities = image_embedding @ text_embeddings.T  # (1, num_prompts)

        # Apply softmax for probability distribution
        scores = torch.softmax(similarities * temperature, dim=-1).squeeze()

        return {label: float(score) for label, score in zip(labels, scores)}

    def classify_multi(
        self,
        image_embedding: torch.Tensor,
        category_name: str,
    ) -> List[Tuple[str, float, int]]:
        """
        Classify for multi-select categories, returning results above threshold.

        Args:
            image_embedding: Image embedding tensor.
            category_name: Name of the prompt category.

        Returns:
            List of (label, confidence, prompt_id) tuples sorted by confidence.
        """
        labels, text_embeddings, category = self.get_category(category_name)
        prompt_ids = self.get_prompt_ids(category_name)

        if image_embedding.dim() == 1:
            image_embedding = image_embedding.unsqueeze(0)
        image_embedding = image_embedding.to(self._device)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Raw cosine similarities (not softmaxed for multi-label)
        similarities = (image_embedding @ text_embeddings.T).squeeze()

        # Convert to 0-1 range: (similarity + 1) / 2
        confidences = (similarities + 1) / 2

        # Filter by threshold and max results
        results = []
        for i, (label, conf) in enumerate(zip(labels, confidences)):
            conf_val = float(conf)
            if conf_val >= category.min_confidence:
                results.append((label, conf_val, prompt_ids[i]))

        # Sort by confidence and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:category.max_results]

    def invalidate(self, category_name: Optional[str] = None) -> None:
        """Invalidate cache for a category or all categories."""
        if category_name:
            self._cache.pop(category_name, None)
        else:
            self._cache.clear()
        logger.info(f"Cache invalidated: {category_name or 'all'}")
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_prompt_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/utils/prompt_cache.py tests/test_prompt_cache.py
git commit -m "feat: add PromptCache for efficient prompt-based classification"
```

---

## Task 7: Create MobileCLIP Analyzer

**Files:**
- Create: `src/photodb/utils/mobileclip_analyzer.py`
- Test: `tests/test_mobileclip_analyzer.py`

**Step 1: Write failing test**

```python
# tests/test_mobileclip_analyzer.py
"""Tests for MobileCLIP analyzer."""
import pytest
from pathlib import Path
from PIL import Image
import numpy as np


class TestMobileCLIPAnalyzer:
    def test_encode_image(self, test_image_path):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        embedding = analyzer.encode_image(str(test_image_path))

        assert embedding.shape == (1, 512)

    def test_encode_text(self):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        embedding = analyzer.encode_text("a happy scene")

        assert embedding.shape == (1, 512)

    def test_encode_texts_batch(self):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        texts = ["happy", "sad", "neutral"]
        embeddings = analyzer.encode_texts(texts)

        assert embeddings.shape == (3, 512)

    def test_encode_face_crop(self, face_crop):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        embedding = analyzer.encode_face(face_crop)

        assert embedding.shape == (1, 512)


@pytest.fixture
def test_image_path():
    return Path(__file__).parent.parent / "test_photos" / "test.jpg"


@pytest.fixture
def face_crop():
    return Image.fromarray(np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8))
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mobileclip_analyzer.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement MobileCLIPAnalyzer**

```python
# src/photodb/utils/mobileclip_analyzer.py
"""
MobileCLIP analyzer for image and text encoding.

Provides efficient encoding for zero-shot classification using prompt embeddings.
"""
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None
_preprocess = None
_tokenizer = None
_device = None


def _load_model():
    """Lazy-load MobileCLIP model."""
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        return _model, _preprocess, _tokenizer, _device

    import open_clip

    logger.info("Loading MobileCLIP-S2 model...")
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP-S2", pretrained="datacompdr"
    )
    _tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
    _model.eval()

    # Select device
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    _model = _model.to(_device)
    logger.info(f"MobileCLIP-S2 loaded on {_device}")

    return _model, _preprocess, _tokenizer, _device


class MobileCLIPAnalyzer:
    """MobileCLIP image and text encoder.

    Usage:
        analyzer = MobileCLIPAnalyzer()
        image_emb = analyzer.encode_image("photo.jpg")
        text_emb = analyzer.encode_text("a happy scene")
    """

    MODEL_NAME = "MobileCLIP-S2"
    EMBEDDING_DIM = 512

    def __init__(self):
        """Initialize analyzer (model loaded lazily on first use)."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is None:
            self._model, self._preprocess, self._tokenizer, self._device = _load_model()

    @property
    def device(self) -> str:
        """Get the device being used."""
        self._ensure_loaded()
        return self._device

    def encode_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Encode an image file to embedding.

        Args:
            image_path: Path to image file.

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        self._ensure_loaded()

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding

    def encode_face(
        self,
        face_image: Image.Image,
    ) -> torch.Tensor:
        """
        Encode a face crop to embedding.

        Args:
            face_image: PIL Image of cropped face.

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        self._ensure_loaded()

        if face_image.mode != "RGB":
            face_image = face_image.convert("RGB")

        image_tensor = self._preprocess(face_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding

    def encode_face_from_bbox(
        self,
        image_path: Union[str, Path],
        bbox: dict,
    ) -> torch.Tensor:
        """
        Crop face from image and encode.

        Args:
            image_path: Path to full image.
            bbox: Bounding box with x1, y1, x2, y2.

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        image = Image.open(image_path).convert("RGB")

        x1 = max(0, int(bbox["x1"]))
        y1 = max(0, int(bbox["y1"]))
        x2 = min(image.width, int(bbox["x2"]))
        y2 = min(image.height, int(bbox["y2"]))

        face_crop = image.crop((x1, y1, x2, y2))
        return self.encode_face(face_crop)

    def encode_faces_batch(
        self,
        image_path: Union[str, Path],
        bboxes: List[dict],
    ) -> torch.Tensor:
        """
        Batch encode multiple face crops from one image.

        Args:
            image_path: Path to full image.
            bboxes: List of bounding boxes.

        Returns:
            Normalized embeddings tensor of shape (N, 512).
        """
        if not bboxes:
            return torch.empty(0, self.EMBEDDING_DIM)

        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")

        face_tensors = []
        for bbox in bboxes:
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(image.width, int(bbox["x2"]))
            y2 = min(image.height, int(bbox["y2"]))
            face_crop = image.crop((x1, y1, x2, y2))
            face_tensors.append(self._preprocess(face_crop))

        batch = torch.stack(face_tensors).to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to embedding.

        Args:
            text: Text string.

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        self._ensure_loaded()

        tokens = self._tokenizer([text]).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Batch encode multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            Normalized embeddings tensor of shape (N, 512).
        """
        if not texts:
            return torch.empty(0, self.EMBEDDING_DIM)

        self._ensure_loaded()

        tokens = self._tokenizer(texts).to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mobileclip_analyzer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/utils/mobileclip_analyzer.py tests/test_mobileclip_analyzer.py
git commit -m "feat: add MobileCLIPAnalyzer for image and text encoding"
```

---

## Task 8: Create Prompt Seeder Script

**Files:**
- Create: `scripts/seed_prompts.py`

**Step 1: Create seed script**

```python
#!/usr/bin/env python3
"""
Seed initial prompts and compute embeddings.

Usage:
    uv run python scripts/seed_prompts.py
    uv run python scripts/seed_prompts.py --recompute-embeddings
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Prompt definitions
# =============================================================================

FACE_EMOTION_PROMPTS = [
    ("happy", "a photo of a happy smiling joyful person"),
    ("sad", "a photo of a sad unhappy melancholic person"),
    ("angry", "a photo of an angry frustrated irritated person"),
    ("surprised", "a photo of a surprised shocked astonished person"),
    ("fearful", "a photo of a fearful scared anxious person"),
    ("disgusted", "a photo of a disgusted repulsed person"),
    ("neutral", "a photo of a person with neutral calm expression"),
    ("confused", "a photo of a confused puzzled person"),
    ("excited", "a photo of an excited enthusiastic thrilled person"),
    ("proud", "a photo of a proud confident person"),
    ("embarrassed", "a photo of an embarrassed shy person"),
    ("contempt", "a photo of a person showing contempt or disdain"),
]

FACE_GAZE_PROMPTS = [
    ("looking_at_camera", "a photo of a person looking directly at the camera"),
    ("looking_away", "a photo of a person looking away from the camera"),
    ("looking_down", "a photo of a person looking downward"),
    ("looking_up", "a photo of a person looking upward"),
    ("eyes_closed", "a photo of a person with eyes closed"),
]

SCENE_MOOD_PROMPTS = [
    ("joyful", "a joyful happy celebratory cheerful scene"),
    ("peaceful", "a peaceful calm serene tranquil scene"),
    ("somber", "a somber sad melancholic gloomy scene"),
    ("tense", "a tense dramatic intense suspenseful scene"),
    ("energetic", "an energetic exciting dynamic lively scene"),
    ("romantic", "a romantic loving intimate scene"),
    ("mysterious", "a mysterious intriguing enigmatic scene"),
    ("nostalgic", "a nostalgic wistful sentimental scene"),
    ("neutral", "an ordinary everyday neutral mundane scene"),
]

SCENE_SETTING_PROMPTS = [
    # Indoor
    ("indoor_home", "a photo taken inside a home or apartment"),
    ("indoor_office", "a photo taken in an office or workplace"),
    ("indoor_restaurant", "a photo taken in a restaurant or cafe"),
    ("indoor_store", "a photo taken in a store or shopping mall"),
    ("indoor_school", "a photo taken in a school or classroom"),
    ("indoor_gym", "a photo taken in a gym or fitness center"),
    ("indoor_museum", "a photo taken in a museum or gallery"),
    ("indoor_hospital", "a photo taken in a hospital or medical facility"),

    # Outdoor natural
    ("outdoor_beach", "a photo taken at a beach with sand and ocean"),
    ("outdoor_mountain", "a photo taken in mountains or hills"),
    ("outdoor_forest", "a photo taken in a forest or woods"),
    ("outdoor_park", "a photo taken in a park or garden"),
    ("outdoor_lake", "a photo taken at a lake or river"),
    ("outdoor_desert", "a photo taken in a desert landscape"),
    ("outdoor_field", "a photo taken in an open field or meadow"),

    # Outdoor urban
    ("outdoor_city", "a photo taken in a city with buildings"),
    ("outdoor_street", "a photo taken on a street or road"),
    ("outdoor_parking", "a photo taken in a parking lot"),

    # Transportation
    ("in_car", "a photo taken inside a car or vehicle"),
    ("in_airplane", "a photo taken inside an airplane"),
    ("at_airport", "a photo taken at an airport"),
]

SCENE_ACTIVITY_PROMPTS = [
    ("celebration", "a celebration party birthday or festive event"),
    ("wedding", "a wedding ceremony or reception"),
    ("graduation", "a graduation ceremony"),
    ("travel", "travel vacation or tourism"),
    ("sports", "sports or athletic activity"),
    ("dining", "eating food or dining together"),
    ("working", "working or professional activity"),
    ("relaxing", "relaxing or leisure activity"),
    ("playing", "playing games or recreational activity"),
    ("concert", "a concert or live music performance"),
    ("meeting", "a meeting or gathering"),
    ("studying", "studying or educational activity"),
]

SCENE_TIME_PROMPTS = [
    ("daytime", "a photo taken during daytime with daylight"),
    ("sunset", "a photo taken during sunset or golden hour"),
    ("sunrise", "a photo taken during sunrise or dawn"),
    ("night", "a photo taken at night or evening"),
    ("overcast", "a photo taken on a cloudy overcast day"),
]

SCENE_WEATHER_PROMPTS = [
    ("sunny", "a photo taken on a sunny clear day"),
    ("cloudy", "a photo taken on a cloudy day"),
    ("rainy", "a photo taken in rain or wet weather"),
    ("snowy", "a photo taken in snow or winter weather"),
    ("foggy", "a photo taken in fog or mist"),
]

SCENE_SOCIAL_PROMPTS = [
    ("solo", "a photo of one person alone"),
    ("couple", "a photo of a couple or two people together"),
    ("small_group", "a photo of a small group of 3-5 people"),
    ("large_group", "a photo of a large group or crowd of people"),
    ("family", "a photo of a family with adults and children"),
    ("no_people", "a photo with no people visible"),
]

PROMPT_SETS = {
    "face_emotion": FACE_EMOTION_PROMPTS,
    "face_gaze": FACE_GAZE_PROMPTS,
    "scene_mood": SCENE_MOOD_PROMPTS,
    "scene_setting": SCENE_SETTING_PROMPTS,
    "scene_activity": SCENE_ACTIVITY_PROMPTS,
    "scene_time": SCENE_TIME_PROMPTS,
    "scene_weather": SCENE_WEATHER_PROMPTS,
    "scene_social": SCENE_SOCIAL_PROMPTS,
}


def seed_prompts(repository, recompute: bool = False):
    """Seed prompts into database and compute embeddings."""
    from photodb.database.models import PromptEmbedding
    from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

    analyzer = MobileCLIPAnalyzer()
    model_name = analyzer.MODEL_NAME

    total_created = 0
    total_updated = 0

    for category_name, prompts in PROMPT_SETS.items():
        category = repository.get_prompt_category_by_name(category_name)
        if not category:
            logger.warning(f"Category '{category_name}' not found, skipping")
            continue

        logger.info(f"Processing category: {category_name} ({len(prompts)} prompts)")

        for label, prompt_text in prompts:
            # Check if exists
            existing = repository.get_prompts_by_category(category.id, with_embeddings=False)
            existing_labels = {p.label for p in existing}

            needs_embedding = recompute or label not in existing_labels

            if needs_embedding:
                # Compute embedding
                embedding = analyzer.encode_text(prompt_text)
                embedding_list = embedding.cpu().squeeze().tolist()

                prompt = PromptEmbedding.create(
                    category_id=category.id,
                    label=label,
                    prompt_text=prompt_text,
                    model_name=model_name,
                    embedding=embedding_list,
                )
                repository.upsert_prompt_embedding(prompt)

                if label in existing_labels:
                    total_updated += 1
                else:
                    total_created += 1

    logger.info(f"Done: {total_created} created, {total_updated} updated")


def main():
    parser = argparse.ArgumentParser(description="Seed prompts and compute embeddings")
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Recompute embeddings for all prompts",
    )
    args = parser.parse_args()

    from photodb.database.pg_connection import get_connection_pool
    from photodb.database.pg_repository import PgRepository

    pool = get_connection_pool()
    repository = PgRepository(pool)

    try:
        seed_prompts(repository, recompute=args.recompute_embeddings)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
```

**Step 2: Make executable and test**

Run: `chmod +x scripts/seed_prompts.py`

**Step 3: Commit**

```bash
git add scripts/seed_prompts.py
git commit -m "scripts: add prompt seeder with initial prompt definitions"
```

---

## Task 9: Create Scene Analysis Stage

**Files:**
- Create: `src/photodb/stages/scene_analysis.py`
- Test: `tests/test_scene_analysis_stage.py`

**Step 1: Write failing test**

```python
# tests/test_scene_analysis_stage.py
"""Tests for scene analysis stage."""
import sys
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


class TestSceneAnalysisStage:
    def test_stage_name(self):
        from photodb.stages.scene_analysis import SceneAnalysisStage
        mock_repo = MagicMock()
        stage = SceneAnalysisStage(mock_repo, {"IMG_PATH": "/tmp"})
        assert stage.stage_name == "scene_analysis"

    def test_process_creates_tags(self, mock_repository, mock_photo):
        from photodb.stages.scene_analysis import SceneAnalysisStage

        # Mock the dependencies
        with patch("photodb.stages.scene_analysis.MobileCLIPAnalyzer") as MockAnalyzer:
            with patch("photodb.stages.scene_analysis.PromptCache") as MockCache:
                import torch

                mock_analyzer = MockAnalyzer.return_value
                mock_analyzer.encode_image.return_value = torch.randn(1, 512)
                mock_analyzer.encode_faces_batch.return_value = torch.randn(0, 512)

                mock_cache = MockCache.return_value
                mock_cache.classify.return_value = {"joyful": 0.8, "peaceful": 0.1}
                mock_cache.classify_multi.return_value = [("beach", 0.7, 1)]
                mock_cache.get_prompt_ids.return_value = [1, 2]

                from photodb.database.models import PromptCategory
                mock_cache.get_category.return_value = (
                    ["joyful", "peaceful"],
                    torch.randn(2, 512),
                    PromptCategory(
                        id=1, name="scene_mood", target="scene", selection_mode="single",
                        min_confidence=0.1, max_results=5, description=None,
                        display_order=0, is_active=True, created_at=None, updated_at=None,
                    ),
                )

                stage = SceneAnalysisStage(mock_repository, {"IMG_PATH": "/tmp"})

                from pathlib import Path
                result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                assert result is True
                assert mock_repository.bulk_upsert_photo_tags.called


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.get_detections_for_photo.return_value = []
    repo.get_prompt_categories.return_value = []
    return repo


@pytest.fixture
def mock_photo():
    from photodb.database.models import Photo
    return Photo(
        id=1, filename="test.jpg", normalized_path="2024/01/test.jpg",
        width=1920, height=1080, normalized_width=1920, normalized_height=1080,
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
```

**Step 2: Implement SceneAnalysisStage**

```python
# src/photodb/stages/scene_analysis.py
"""
Scene analysis stage: Taxonomy and prompt-based tagging.

Uses Apple Vision for scene taxonomy (macOS) and MobileCLIP with
configurable prompts for scene and face tagging.
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .base import BaseStage
from ..database.models import (
    Photo, AnalysisOutput, SceneAnalysis, PhotoTag, DetectionTag,
)
from ..utils.mobileclip_analyzer import MobileCLIPAnalyzer
from ..utils.prompt_cache import PromptCache

logger = logging.getLogger(__name__)

_apple_vision_available = sys.platform == "darwin"
if _apple_vision_available:
    from ..utils.apple_vision_classifier import AppleVisionClassifier


class SceneAnalysisStage(BaseStage):
    """Stage for scene taxonomy and prompt-based tagging."""

    stage_name = "scene_analysis"

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)

        self.analyzer = MobileCLIPAnalyzer()
        self.prompt_cache = PromptCache(repository, device=self.analyzer.device)

        if _apple_vision_available:
            self.apple_classifier = AppleVisionClassifier()
        else:
            self.apple_classifier = None
            logger.warning("Apple Vision not available (not macOS)")

        # Load category configs
        self.scene_categories = repository.get_prompt_categories(target="scene")
        self.face_categories = repository.get_prompt_categories(target="face")

        logger.info(
            f"SceneAnalysisStage initialized: "
            f"{len(self.scene_categories)} scene categories, "
            f"{len(self.face_categories)} face categories"
        )

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process scene taxonomy and tagging for a photo."""
        try:
            if not photo.normalized_path:
                logger.warning(f"No normalized path for photo {photo.id}")
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            # 1. Apple Vision taxonomy (macOS only)
            taxonomy_output_id = None
            taxonomy_labels = None
            taxonomy_confidences = None

            if self.apple_classifier:
                taxonomy_result = self.apple_classifier.classify(str(normalized_path), top_k=15)

                taxonomy_output = AnalysisOutput.create(
                    photo_id=photo.id,
                    model_type="classifier",
                    model_name="apple_vision_classify",
                    output=taxonomy_result,
                    processing_time_ms=taxonomy_result.get("processing_time_ms"),
                    device="ane",
                )
                self.repository.create_analysis_output(taxonomy_output)
                taxonomy_output_id = taxonomy_output.id

                if taxonomy_result["status"] == "success":
                    taxonomy_labels = [c["identifier"] for c in taxonomy_result["classifications"]]
                    taxonomy_confidences = [c["confidence"] for c in taxonomy_result["classifications"]]

            # 2. Encode image with MobileCLIP
            image_embedding = self.analyzer.encode_image(str(normalized_path))

            # 3. Classify against all scene categories
            all_photo_tags: List[PhotoTag] = []
            scene_results = {}

            for category in self.scene_categories:
                try:
                    if category.selection_mode == "single":
                        scores = self.prompt_cache.classify(image_embedding, category.name)
                        # Get top result
                        top_label = max(scores, key=scores.get)
                        top_score = scores[top_label]

                        if top_score >= category.min_confidence:
                            prompt_ids = self.prompt_cache.get_prompt_ids(category.name)
                            labels, _, _ = self.prompt_cache.get_category(category.name)
                            idx = labels.index(top_label)

                            all_photo_tags.append(PhotoTag.create(
                                photo_id=photo.id,
                                prompt_id=prompt_ids[idx],
                                confidence=top_score,
                                rank_in_category=1,
                            ))
                        scene_results[category.name] = scores
                    else:
                        # Multi-select
                        results = self.prompt_cache.classify_multi(image_embedding, category.name)
                        for rank, (label, conf, prompt_id) in enumerate(results, 1):
                            all_photo_tags.append(PhotoTag.create(
                                photo_id=photo.id,
                                prompt_id=prompt_id,
                                confidence=conf,
                                rank_in_category=rank,
                            ))
                        scene_results[category.name] = {r[0]: r[1] for r in results}
                except Exception as e:
                    logger.warning(f"Failed to classify category {category.name}: {e}")

            # 4. Store scene analysis output
            mobileclip_output = AnalysisOutput.create(
                photo_id=photo.id,
                model_type="tagger",
                model_name="mobileclip",
                output={"scene": scene_results},
                device=self.analyzer.device,
            )
            self.repository.create_analysis_output(mobileclip_output)

            # 5. Save photo tags
            if all_photo_tags:
                self.repository.bulk_upsert_photo_tags(all_photo_tags)

            # 6. Save scene analysis record
            scene_analysis = SceneAnalysis.create(
                photo_id=photo.id,
                taxonomy_labels=taxonomy_labels,
                taxonomy_confidences=taxonomy_confidences,
                taxonomy_output_id=taxonomy_output_id,
                mobileclip_output_id=mobileclip_output.id,
            )
            self.repository.upsert_scene_analysis(scene_analysis)

            # 7. Process face tags if detections exist
            detections = self.repository.get_detections_for_photo(photo.id)
            face_detections = [d for d in detections if d.has_face()]

            if face_detections and self.face_categories:
                self._process_face_tags(photo.id, normalized_path, face_detections)

            logger.info(
                f"Scene analysis complete for {file_path}: "
                f"{len(all_photo_tags)} scene tags, "
                f"{len(face_detections)} faces"
            )
            return True

        except Exception as e:
            logger.error(f"Scene analysis failed for {file_path}: {e}")
            return False

    def _process_face_tags(self, photo_id: int, image_path: Path, detections) -> None:
        """Tag each face detection with face categories."""
        # Build bboxes
        bboxes = []
        for det in detections:
            bboxes.append({
                "x1": det.face_bbox_x,
                "y1": det.face_bbox_y,
                "x2": det.face_bbox_x + det.face_bbox_width,
                "y2": det.face_bbox_y + det.face_bbox_height,
            })

        # Batch encode faces
        face_embeddings = self.analyzer.encode_faces_batch(str(image_path), bboxes)

        if face_embeddings.shape[0] == 0:
            return

        # Store face analysis output
        face_results = {}
        all_detection_tags: List[DetectionTag] = []

        for i, (det, face_emb) in enumerate(zip(detections, face_embeddings)):
            face_emb = face_emb.unsqueeze(0)
            face_results[det.id] = {}

            for category in self.face_categories:
                try:
                    if category.selection_mode == "single":
                        scores = self.prompt_cache.classify(face_emb, category.name)
                        top_label = max(scores, key=scores.get)
                        top_score = scores[top_label]

                        if top_score >= category.min_confidence:
                            prompt_ids = self.prompt_cache.get_prompt_ids(category.name)
                            labels, _, _ = self.prompt_cache.get_category(category.name)
                            idx = labels.index(top_label)

                            all_detection_tags.append(DetectionTag.create(
                                detection_id=det.id,
                                prompt_id=prompt_ids[idx],
                                confidence=top_score,
                                rank_in_category=1,
                            ))
                        face_results[det.id][category.name] = scores
                    else:
                        results = self.prompt_cache.classify_multi(face_emb, category.name)
                        for rank, (label, conf, prompt_id) in enumerate(results, 1):
                            all_detection_tags.append(DetectionTag.create(
                                detection_id=det.id,
                                prompt_id=prompt_id,
                                confidence=conf,
                                rank_in_category=rank,
                            ))
                        face_results[det.id][category.name] = {r[0]: r[1] for r in results}
                except Exception as e:
                    logger.warning(f"Failed face category {category.name}: {e}")

        # Save tags
        if all_detection_tags:
            self.repository.bulk_upsert_detection_tags(all_detection_tags)

        # Store output
        face_output = AnalysisOutput.create(
            photo_id=photo_id,
            model_type="tagger",
            model_name="mobileclip",
            output={"faces": face_results},
            device=self.analyzer.device,
        )
        self.repository.create_analysis_output(face_output)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_scene_analysis_stage.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/photodb/stages/scene_analysis.py tests/test_scene_analysis_stage.py
git commit -m "feat: add SceneAnalysisStage with prompt-based tagging"
```

---

## Task 10: Register Stage and Update Documentation

**Files:**
- Modify: `src/photodb/cli_local.py`
- Modify: `CLAUDE.md`

**Step 1: Register stage in CLI**

Add import and register `scene_analysis` stage in the CLI.

**Step 2: Update CLAUDE.md**

Add section documenting:
- Scene analysis stage configuration
- Prompt management
- How to add custom prompts
- Running the seed script

**Step 3: Commit**

```bash
git add src/photodb/cli_local.py CLAUDE.md
git commit -m "cli/docs: register scene_analysis stage and document prompt system"
```

---

## Summary

This plan adds a generalizable prompt-based analysis system:

### New Tables

| Table | Purpose |
|-------|---------|
| `prompt_category` | Organize prompts (face_emotion, scene_mood, etc.) |
| `prompt_embedding` | Prompts with precomputed 512-dim vectors |
| `photo_tag` | Multi-label results for photos |
| `detection_tag` | Multi-label results for face detections |
| `analysis_output` | Raw model outputs (generalizable) |
| `scene_analysis` | Photo-level Apple Vision taxonomy |

### New Components

| Component | Purpose |
|-----------|---------|
| `AppleVisionClassifier` | Scene taxonomy (1303 labels, macOS) |
| `MobileCLIPAnalyzer` | Image/text encoding |
| `PromptCache` | Efficient cached prompt classification |
| `SceneAnalysisStage` | Orchestrates everything |
| `scripts/seed_prompts.py` | Populate initial prompts |

### Prompt Categories (Initial)

| Category | Target | Mode | Prompts |
|----------|--------|------|---------|
| face_emotion | face | single | 12 |
| face_gaze | face | single | 5 |
| scene_mood | scene | single | 9 |
| scene_setting | scene | multi | 22 |
| scene_activity | scene | multi | 12 |
| scene_time | scene | single | 5 |
| scene_weather | scene | single | 5 |
| scene_social | scene | single | 6 |

### Scalability

- 1000+ prompts per category: Single matrix multiply ~0.5ms
- Embeddings cached in GPU tensor
- Easily extensible via database (no code changes)

---

Plan saved. Ready for execution?

**1. Subagent-Driven (this session)**
**2. Parallel Session (separate)**
