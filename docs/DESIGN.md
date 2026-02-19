# PhotoDB Design Document

PhotoDB is a personal photo indexing pipeline that processes photos through multiple stages to create a searchable, navigable database. It combines local processing (file normalization, metadata extraction, person detection, age/gender estimation, clustering) with remote LLM-based enrichment.

## Architecture Overview

```
Photo Files (raw)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  LOCAL PROCESSING (process-local CLI)               │
│  ├─ Stage 1: Normalize - Convert to PNG, resize     │
│  ├─ Stage 2: Metadata - Extract EXIF data           │
│  ├─ Stage 3: Detection - YOLO face+body detection   │
│  ├─ Stage 4: Age/Gender - MiVOLO estimation         │
│  └─ Stage 5: Scene Analysis - Taxonomy + tagging    │
└─────────────────────────────────────────────────────┘
       │
       ▼ (PostgreSQL + pgvector)
┌─────────────────────────────────────────────────────┐
│  CLUSTERING (scripts/bootstrap_clusters.py)         │
│  └─ HDBSCAN bootstrap on all embeddings             │
│     Run after batch photo imports, not per-photo    │
└─────────────────────────────────────────────────────┘
       │
       ▼ (PostgreSQL + pgvector)
┌─────────────────────────────────────────────────────┐
│  REMOTE PROCESSING (enrich-photos CLI)              │
│  └─ Stage R1: Enrich - LLM semantic analysis        │
│     ├─ Anthropic Batch API                          │
│     └─ AWS Bedrock Batch Inference                  │
└─────────────────────────────────────────────────────┘
       │
       ▼ (PostgreSQL)
┌─────────────────────────────────────────────────────┐
│  MAINTENANCE (photodb-maintenance CLI)              │
│  ├─ Daily: Recompute centroids, cleanup             │
│  └─ Weekly: Update medoids, merge similar clusters  │
└─────────────────────────────────────────────────────┘
```

### Key Design Principles

- **Horizontal scalability**: Designed for 500+ concurrent workers
- **Idempotent stages**: All stages can be rerun safely with `--force`
- **Hybrid processing**: Local stages for CPU/I/O tasks, remote for LLM analysis
- **PostgreSQL with pgvector**: Native vector operations for face clustering
- **Batch processing**: Efficient LLM batch APIs for cost savings (50% discount)

## CLI Commands

### process-local

Local photo processing with parallel workers.

```bash
# Basic usage
uv run process-local /path/to/photos

# Parallel processing (recommended)
uv run process-local /path/to/photos --parallel 500

# Specific stage only
uv run process-local /path/to/photos --stage normalize
uv run process-local /path/to/photos --stage metadata
uv run process-local /path/to/photos --stage detection
uv run process-local /path/to/photos --stage age_gender
uv run process-local /path/to/photos --stage scene_analysis

# Clustering (run separately after batch imports)
uv run python scripts/bootstrap_clusters.py --dry-run
uv run python scripts/bootstrap_clusters.py

# Force reprocessing
uv run process-local /path/to/photos --force

# Dry run
uv run process-local /path/to/photos --dry-run

# Limit photos
uv run process-local /path/to/photos --max-photos 1000
```

**Source**: `src/photodb/cli_local.py`

### enrich-photos

LLM-based photo enrichment with batch processing.

```bash
# Process directory (batch mode default)
uv run enrich-photos /path/to/photos

# Check batch status
uv run enrich-photos --check-batches
uv run enrich-photos --check-batches --wait

# Retry failed
uv run enrich-photos /path/to/photos --retry-failed

# Disable batch mode
uv run enrich-photos /path/to/photos --no-batch
```

**Source**: `src/photodb/cli_enrich.py`

### photodb-maintenance

Periodic cluster optimization and system health.

```bash
# Scheduled maintenance
uv run photodb-maintenance daily
uv run photodb-maintenance weekly

# Individual tasks
uv run photodb-maintenance recompute-centroids
uv run photodb-maintenance update-medoids
uv run photodb-maintenance merge-similar --similarity-threshold 0.25
uv run photodb-maintenance cleanup-empty
uv run photodb-maintenance update-stats

# Clustering staleness check
uv run photodb-maintenance check-staleness

# Health check
uv run photodb-maintenance health

# JSON output
uv run photodb-maintenance daily --json
```

**Source**: `src/photodb/cli_maintenance.py`

## Processing Pipeline Stages

### Stage 1: Normalize

**File**: `src/photodb/stages/normalize.py`

Converts images to standardized PNG format with proper orientation.

**Process**:

1. Read image (supports HEIC, JPEG, PNG, WebP, TIFF, BMP, GIF)
2. Apply EXIF orientation correction
3. Resize if `RESIZE_SCALE` configured
4. Save as optimized PNG to `IMG_PATH/{uuid}.png`
5. Store dimensions in database

**Key features**:

- HEIC/HEIF support via `pillow-heif`
- Decompression bomb protection (max 179M pixels)
- Color mode conversion (RGBA/P → RGB with white background)
- File handles closed immediately to prevent resource exhaustion

**Output**: PNG file + database record with dimensions

### Stage 2: Metadata

**File**: `src/photodb/stages/metadata.py`

Extracts EXIF/TIFF/IFD metadata from original images.

**Process**:

1. Extract EXIF using piexif (primary) or PIL (fallback)
2. Parse datetime from DateTimeOriginal/DateTimeDigitized/DateTime
3. Convert GPS coordinates to decimal degrees
4. Parse camera info (make, model, lens, ISO, aperture, etc.)
5. Store in `metadata` table with JSONB `extra` column

**Key features**:

- IFDRational → float conversion
- Filename-based date inference as fallback
- Null byte stripping (PostgreSQL compatibility)
- ON CONFLICT for safe updates

**Output**: Metadata record with captured_at, GPS, and full EXIF

### Stage 3: Detection

**File**: `src/photodb/stages/detection.py`

Detects faces and bodies using YOLOv8x, extracts face embeddings for clustering.

**Process**:

1. Load normalized image
2. Run YOLOv8x person_face model (detects both faces and bodies)
3. Match faces to bodies based on spatial containment
4. Filter by confidence threshold (default: 0.5)
5. Extract 512-dimensional InsightFace embeddings for faces (ArcFace buffalo_l)
6. Store person_detection records with face and body bounding boxes
7. Save face embeddings in pgvector format

**Key features**:

- YOLOv8x person_face model for unified face+body detection
- Face-to-body matching via spatial containment algorithm
- **Batch inference via BatchCoordinator** when `--parallel > 1` (PyTorch MPS)
- PyTorch with auto device detection (MPS/CUDA/CPU)
- `DETECTION_FORCE_CPU=true` for CPU fallback
- L2-normalized embeddings for cosine similarity
- Supports face-only, body-only, and face+body detections

**Output**: PersonDetection records + pgvector embeddings (for faces)

### Stage 4: Age/Gender

**File**: `src/photodb/stages/age_gender.py`

Estimates age and gender using MiVOLO model on existing detections.

**Process**:

1. Load existing person_detection records for photo
2. For each detection with face or body bbox:
   - Run MiVOLO prediction with available bboxes
   - MiVOLO performs better with both face AND body
3. Update detection records with age_estimate, gender, gender_confidence
4. Store full MiVOLO output in mivolo_output JSONB field

**Key features**:

- MiVOLO d1 model (pth.tar format, face+body)
- Uses both face and body bboxes for improved accuracy
- Graceful degradation if MiVOLO not installed
- `MIVOLO_FORCE_CPU=true` for CPU fallback
- Gender stored as CHAR(1): 'M', 'F', or 'U' (unknown)

**Output**: Updated PersonDetection records with age/gender data

### Stage 5: Scene Analysis

**File**: `src/photodb/stages/scene_analysis.py`

Classifies photos and detected faces using Apple Vision scene taxonomy and MobileCLIP zero-shot tagging.

**Process**:

1. **Apple Vision taxonomy (macOS only)**: Classify the image against Apple's built-in 1303-label scene taxonomy, returning top-k labels with confidence scores. Uses the Neural Engine for fast inference.
2. **Image embedding**: Encode the normalized image with MobileCLIP-S2.
3. **Scene tagging**: Classify the image embedding against all scene prompt categories (mood, setting, activity, time, weather, social). Each category supports either single-select (top-1) or multi-select mode with per-category confidence thresholds.
4. **Face tagging**: For each detected face, crop and encode the face region with MobileCLIP, then classify against face prompt categories (emotion, expression, gaze).
5. **Persist results**: Store `AnalysisOutput` records for both taxonomy and MobileCLIP outputs, `PhotoTag` records for scene tags, `DetectionTag` records for face tags, and a `SceneAnalysis` summary record linking everything together.

**Prompt-based classification**:

Prompts are stored in the database as pre-computed embeddings organized into categories. Classification works by computing cosine similarity between the image/face embedding and each prompt embedding in a category, then selecting the best match(es).

Categories are configured with:
- `selection_mode`: `single` (top-1 winner) or `multi` (all above threshold)
- `min_confidence`: Per-category threshold for tag inclusion
- `target`: `scene` (applied to whole image) or `face` (applied to face crops)

See [`docs/prompt_strategy.md`](prompt_strategy.md) for details on prompt ensembling, template sets, and embedding computation.

**Key features**:

- Apple Vision runs via `pyobjc` with `objc.autorelease_pool()` for thread safety
- MobileCLIP embeddings are cached per-category in `PromptCache` for fast repeated classification
- **Batch inference via BatchCoordinator** when `--parallel > 1`: separate coordinators for image and face embeddings
- Both classifiers warm up at stage initialization to avoid cold-start latency

**Output**: SceneAnalysis record + PhotoTag records (scene) + DetectionTag records (faces) + AnalysisOutput records (raw model outputs)

### Clustering (separate from pipeline)

**File**: `src/photodb/stages/clustering.py`
**Script**: `scripts/bootstrap_clusters.py`

Hierarchical density-based clustering using HDBSCAN with two-tier incremental assignment for person identity grouping.

Clustering is **not** part of the normal `process-local` pipeline. HDBSCAN operates on the full set of embeddings to build a global cluster hierarchy, so it should be run after importing batches of photos rather than per-photo. Incremental assignment (for ad-hoc single photo additions) is supported but there is no automated flow for it yet.

**Two-Phase Approach**:

1. **Bootstrap Phase**: HDBSCAN runs on all embeddings to build the cluster hierarchy
2. **Incremental Phase**: New faces are assigned to existing clusters using a two-tier strategy

**Incremental Assignment Strategy** (two-tier):

1. **Tier 1: approximate_predict** - HDBSCAN's built-in incremental classifier
   - Uses the condensed tree from bootstrap phase
   - Classifies new points based on hierarchical structure
   - Fast and hierarchically consistent
2. **Tier 2: Epsilon-ball fallback** - Distance-based assignment if approximate_predict returns -1 (outlier)
   - Checks distance to each cluster centroid
   - Assigns if distance < cluster's epsilon threshold
   - Epsilon derived from lambda_birth via condensed tree

**Lambda-derived Epsilon** (per-cluster):

Each cluster's epsilon is calculated as `1 / lambda_birth`, where:

- `lambda_birth`: The λ value at which the cluster was born in HDBSCAN's condensed tree
- Higher λ (smaller epsilon) = denser, tighter cluster
- Lower λ (larger epsilon) = sparser, looser cluster

This makes epsilon hierarchically consistent with HDBSCAN's density-based structure.

**HDBSCAN Run Persistence** (`hdbscan_run` table):

Each bootstrap run is persisted with:

- `condensed_tree`: HDBSCAN condensed tree as JSONB (serialized via `to_pandas().to_dict()`)
- `clusterer_state`: Pickled HDBSCAN clusterer object as BYTEA (for `approximate_predict`)
- `label_to_cluster_id`: Mapping from HDBSCAN label to database cluster ID (JSONB)
- `embedding_count`, `cluster_count`, `noise_count`: Run statistics
- `min_cluster_size`, `min_samples`: Parameters used for this run
- `is_active`: Only one active run per collection (partial unique index)
- Clusters reference their parent run via `hdbscan_run_id`

**Cluster Hierarchy Metadata**:

Each cluster stores HDBSCAN hierarchy information:

- `lambda_birth`: Birth λ from condensed tree (used to calculate epsilon)
- `persistence`: Lifetime of cluster in hierarchy (lambda_death - lambda_birth)
- Higher persistence = more stable cluster

**Detection-level Metadata**:

Each person_detection stores:

- `lambda_val`: λ value at assignment (from HDBSCAN)
- `outlier_score`: HDBSCAN's GLOSH outlier score (0-1, higher = more outlier-like)

**Staleness Detection**:

Clusters can become stale if:

- Face embeddings change significantly
- Bootstrap parameters change (min_cluster_size, min_samples)
- Many new faces added without re-bootstrapping

Check staleness with:

```bash
uv run photodb-maintenance check-staleness
```

**Concurrency Safety**:

- Row-level locking with `SELECT FOR UPDATE` during cluster modifications
- Atomic face assignment checks prevent race conditions between workers
- Empty clusters automatically deleted when all faces reassigned

**Constraint System**:

- **Cannot-link**: Prevents faces from sharing cluster (checked during incremental assignment)
- Constraints are applied after HDBSCAN bootstrap completes

**Configuration**:

- `HDBSCAN_MIN_CLUSTER_SIZE`: Minimum faces to form a cluster (default: 3)
- `HDBSCAN_MIN_SAMPLES`: Core point requirement for HDBSCAN (default: 2)
- `CLUSTERING_THRESHOLD`: Fallback distance threshold for clusters without epsilon (default: 0.45)

**Centroid update formula** (incremental, during assignment):

```text
new_centroid = (old_centroid * face_count + embedding) / (face_count + 1)
```

All embeddings and centroids are L2-normalized to unit vectors for cosine similarity.

**Output**: Hierarchical cluster assignments + condensed tree persistence + lambda/persistence metadata

### Cluster Auto-Association (Person Grouping)

**File**: `src/photodb/utils/maintenance.py` (`auto_associate_clusters`)

Automatically groups clusters into shared person records based on centroid similarity. This runs as part of weekly maintenance to link clusters that likely represent the same individual.

**Algorithm**:

1. **Pair discovery**: Uses pgvector's `<=>` cosine distance operator to find cluster pairs within a configurable threshold (default: `PERSON_ASSOCIATION_THRESHOLD=0.8`). Hidden clusters and pairs blocked by cannot-link constraints are excluded.
2. **Complete-linkage grouping**: Builds groups of similar clusters using complete-linkage (not single-linkage) to prevent chaining. A cluster joins a group only if it is within threshold of **every** existing member. Pairs are processed in distance order (closest first).
3. **Cannot-link filtering**: Before linking, clusters with `cluster_person_cannot_link` constraints to the group's candidate person are removed.
4. **Person assignment**: For each group of 2+ clusters:
   - **No existing person**: Creates a new auto-created person (`first_name='Unknown'`, `auto_created=true`) and links all clusters
   - **One existing person**: Links unlinked clusters to the existing person
   - **Multiple existing persons**: Merges into the best person (priority: verified clusters > most clusters > highest ID), then links remaining unlinked clusters

**Complete-linkage vs single-linkage**:

Single-linkage would chain clusters A→B→C even if A and C are dissimilar. Complete-linkage requires all pairs within a group to be within threshold, producing tighter, more reliable groupings.

```text
Single-linkage (avoided):     Complete-linkage (used):
A -- B -- C -- D              A -- B    C -- D
(chain, A↔D may be far)      (A↔B close, C↔D close, but groups separate)
```

**Person merge priority** (when a group spans multiple persons):

```text
1. Has verified clusters (boolean, highest priority)
2. Number of clusters linked to the person
3. Person ID (tiebreaker, higher = newer)
```

The losing person is deleted and all its clusters are moved to the winning person via `merge_persons`.

**Per-collection scoping**: Auto-association runs independently per collection to avoid mixing identities across collections.

**Configuration**:

- `PERSON_ASSOCIATION_THRESHOLD`: Maximum cosine distance between cluster centroids to consider them the same person (default: `0.8`)

**CLI usage**:

```bash
# Run as part of weekly maintenance
uv run photodb-maintenance weekly

# Run standalone
uv run photodb-maintenance auto-associate
uv run photodb-maintenance auto-associate --threshold 0.6
uv run photodb-maintenance auto-associate --dry-run
```

**Output**: Person records created/merged + cluster-to-person linkages

### Stage R1: Enrich

**File**: `src/photodb/stages/enrich.py`

LLM-based semantic analysis of photos.

**Process**:

1. Load system prompt from `prompts/system_prompt.md`
2. Build message with base64-encoded image + EXIF context
3. Submit to LLM (single or batch mode)
4. Parse structured response using Instructor + Pydantic
5. Store analysis in `llm_analysis` table

**Supported providers**:

- **Anthropic**: instructor.from_anthropic + Batch API
- **AWS Bedrock**: Native batch inference API via S3

**Output**: Structured analysis (description, objects, people_count, emotional_tone, tags)

## Database Schema

**Location**: `schema.sql`

### Core Tables

```sql
-- Photo record
photo (
    id, filename, normalized_path,
    width, height, normalized_width, normalized_height,
    created_at, updated_at
)

-- Extracted metadata
metadata (
    photo_id, captured_at, latitude, longitude,
    extra JSONB, created_at
)

-- Stage tracking
processing_status (
    photo_id, stage, status, processed_at, error_message
)
-- status: 'pending' | 'processing' | 'completed' | 'failed'
-- stage: 'normalize' | 'metadata' | 'detection' | 'age_gender' | 'clustering' | 'enrich'
```

### LLM Analysis Tables

```sql
-- Stored enrichment results
llm_analysis (
    id, photo_id, model_name, model_version, processed_at, batch_id,
    analysis JSONB, description, objects[], people_count,
    location_description, emotional_tone, confidence_score,
    processing_duration_ms, input_tokens, output_tokens,
    cache_creation_tokens, cache_read_tokens, error_message
)

-- Batch job tracking
batch_job (
    id, provider_batch_id, status, submitted_at, completed_at,
    photo_count, processed_count, failed_count, photo_ids[],
    total_input_tokens, total_output_tokens,
    estimated_cost_cents, actual_cost_cents,
    model_name, batch_discount_applied, error_message
)
```

### Person Detection & Clustering Tables

```sql
-- Named individuals (with aggregated age/gender)
person (
    id, first_name, last_name,
    estimated_birth_year, birth_year_stddev,  -- Aggregated from detections
    gender, gender_confidence,                 -- Aggregated from detections
    age_gender_sample_count, age_gender_updated_at,
    created_at, updated_at
)

-- Detected faces and bodies (replaces old face table)
person_detection (
    id, photo_id,
    -- Face bounding box (nullable - may have body-only detection)
    face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height, face_confidence,
    -- Body bounding box (nullable - may have face-only detection)
    body_bbox_x, body_bbox_y, body_bbox_width, body_bbox_height, body_confidence,
    -- Age/gender estimation (from MiVOLO)
    age_estimate, gender, gender_confidence, mivolo_output JSONB,
    -- Clustering
    person_id, cluster_status, cluster_id, cluster_confidence,
    unassigned_since,  -- When added to unassigned pool
    lambda_val,        -- Lambda value from HDBSCAN assignment
    outlier_score,     -- GLOSH outlier score (0-1, higher = more outlier-like)
    -- Detector metadata
    detector_model, detector_version,
    created_at
)
-- cluster_status: 'auto' | 'pending' | 'manual' | 'unassigned' | 'constrained'
-- gender: 'M' | 'F' | 'U' (unknown)

-- Face embeddings (pgvector) - only for detections with faces
face_embedding (
    person_detection_id, embedding VECTOR(512)
)
-- IVFFlat index for similarity search

-- Identity clusters
cluster (
    id, face_count, representative_detection_id,
    centroid VECTOR(512), medoid_detection_id,
    face_count_at_last_medoid,  -- Tracks when to recompute medoid
    person_id, verified, verified_at, verified_by,  -- Cluster verification
    hdbscan_run_id,    -- Reference to HDBSCAN bootstrap run
    lambda_birth,      -- Birth lambda from condensed tree (for epsilon calculation)
    persistence,       -- Cluster lifetime in hierarchy (lambda_death - lambda_birth)
    created_at, updated_at
)

-- HDBSCAN bootstrap runs
hdbscan_run (
    id, collection_id, created_at,
    embedding_count, cluster_count, noise_count,
    min_cluster_size, min_samples,
    condensed_tree JSONB,          -- Condensed tree (serialized via to_pandas().to_dict())
    label_to_cluster_id JSONB,     -- HDBSCAN label -> database cluster_id mapping
    clusterer_state BYTEA,         -- Pickled clusterer (for approximate_predict)
    is_active BOOLEAN              -- Only one active per collection
)

-- Ambiguous detection matches for review
face_match_candidate (
    candidate_id, detection_id, cluster_id, similarity,
    status, created_at
)
-- status: 'pending' | 'accepted' | 'rejected'
```

### Constraint Tables

```sql
-- Must-link: forces detections into same cluster
must_link (
    id, detection_id_1, detection_id_2, created_by, created_at
)
-- CHECK: detection_id_1 < detection_id_2 (canonical ordering)

-- Cannot-link: prevents detections from sharing cluster
cannot_link (
    id, detection_id_1, detection_id_2, created_by, created_at
)

-- Cluster-level cannot-link (prevents cluster merging)
cluster_cannot_link (
    id, cluster_id_1, cluster_id_2, created_at
)
```

### Required Indexes

```sql
-- Vector similarity search
CREATE INDEX idx_face_embedding ON face_embedding
    USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_cluster_centroid ON cluster
    USING ivfflat(centroid vector_cosine_ops) WITH (lists = 100);

-- Common queries
CREATE INDEX idx_person_detection_cluster_id ON person_detection(cluster_id);
CREATE INDEX idx_person_detection_cluster_status ON person_detection(cluster_status);
CREATE INDEX idx_person_detection_photo_id ON person_detection(photo_id);
CREATE INDEX idx_person_detection_gender ON person_detection(gender);
CREATE INDEX idx_person_detection_age ON person_detection(age_estimate);
CREATE INDEX idx_processing_status ON processing_status(status, stage);
```

## LLM Integration

### Provider: Anthropic

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-3-5-sonnet-20241022  # optional
```

**Features**:

- Single photo: Synchronous instructor call
- Batch: instructor.batch.BatchProcessor with Anthropic Batch API
- 50% cost discount with batch processing
- Prompt caching support for repeated EXIF patterns

### Provider: AWS Bedrock

```bash
export LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
export BEDROCK_BATCH_S3_BUCKET=your-bucket
export BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::ACCOUNT:role/BedrockBatchRole
```

**Features**:

- Single photo: boto3 bedrock-runtime InvokeModel
- Batch: S3-based input/output with batch inference jobs
- Requires IAM role with S3 access for Bedrock service

See [BEDROCK_SETUP.md](BEDROCK_SETUP.md) for detailed AWS configuration.

### Photo Analysis Response Schema

**File**: `src/photodb/models/photo_analysis.py`

```python
class PhotoAnalysisResponse:
    description: str              # 2-4 sentence summary
    quality: QualityAssessment    # score 0.0-1.0 + notes
    time: TimeAssessment          # season, time_of_day, date_estimate
    location: LocationAssessment  # environment, hypotheses
    people: PeopleAssessment      # count, face info
    activities: ActivitiesAssessment
    objects: List[DetectedObject]
    text_in_image: List[TextBlock]
    tags: List[str]
```

## Maintenance Tasks

**File**: `src/photodb/utils/maintenance.py`

### Daily Tasks

1. **Recompute centroids**: Recalculate cluster centroids from member faces
2. **Cleanup empty clusters**: Remove clusters with no assigned faces
3. **Update statistics**: Refresh face_count and other cluster stats
4. **Propagate must-link constraints**: Transitive closure (A~B, B~C → A~C)
5. **Merge must-linked clusters**: Merge clusters connected by must-link faces
6. **Check constraint violations**: Find cannot-link faces in same cluster

### Weekly Tasks

All daily tasks, plus:

1. **Update medoids**: Find face closest to each cluster centroid
2. **Merge similar clusters**: Combine clusters with centroid similarity > threshold (respects verified clusters and cannot-link constraints)
3. **Cleanup unassigned pool**: Create singleton clusters for old unassigned faces

### Health Checks

```python
{
    "total_clusters": int,
    "empty_clusters": int,
    "clusters_without_centroids": int,
    "clusters_without_medoids": int,
    "avg_cluster_size": float,
    "min_cluster_size": int,
    "max_cluster_size": int,
    "total_faces": int,
    "unclustered_faces": int,
    "must_link_count": int,
    "cannot_link_count": int,
    "verified_clusters": int,
    "unassigned_pool_size": int
}
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://localhost/photodb

# Paths
INGEST_PATH=./photos/raw           # Source photos
IMG_PATH=./photos/processed        # Normalized output
LOG_FILE=./logs/photodb.log
BATCH_REQUESTS_PATH=./batch_requests

# Processing
RESIZE_SCALE=1.0                   # Image resize multiplier

# Detection Stage (YOLO)
DETECTION_MODEL_PATH=models/yolov8x_person_face.pt
DETECTION_MIN_CONFIDENCE=0.5       # Detection confidence threshold
DETECTION_FORCE_CPU=false          # Force CPU for detection
DETECTION_PREFER_COREML=false      # Use CoreML instead of PyTorch MPS (macOS)
YOLO_BATCH_ENABLED=true            # Enable YOLO batch inference via BatchCoordinator

# Batch ML Inference (applies when --parallel > 1)
BATCH_COORDINATOR_ENABLED=true     # Enable/disable batch coordinators
BATCH_COORDINATOR_MAX_SIZE=32      # Max items per batch
BATCH_COORDINATOR_MAX_WAIT_MS=50   # Max wait for batch formation (ms)

# Age/Gender Stage (MiVOLO)
MIVOLO_MODEL_PATH=models/mivolo_d1.pth.tar
MIVOLO_FORCE_CPU=false             # Force CPU for MiVOLO

# Clustering (HDBSCAN + incremental assignment)
HDBSCAN_MIN_CLUSTER_SIZE=3         # Minimum faces to form a cluster
HDBSCAN_MIN_SAMPLES=2              # Core point requirement for HDBSCAN
CLUSTERING_THRESHOLD=0.45          # Fallback distance threshold for clusters without epsilon

# LLM - Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-sonnet-20241022

# LLM - Bedrock
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
AWS_REGION=us-east-1
AWS_PROFILE=                       # optional
BEDROCK_BATCH_S3_BUCKET=bucket
BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::...

# Batch Processing
BATCH_SIZE=100                     # Photos per batch
MIN_BATCH_SIZE=10                  # Skip smaller batches
BATCH_CHECK_INTERVAL=300           # Seconds between status checks

# Logging
LOG_LEVEL=INFO
```

## Batch ML Inference

### BatchCoordinator

**File**: `src/photodb/utils/batch_coordinator.py`

When processing photos in parallel (`--parallel N`), individual worker threads would each make separate GPU calls for ML inference. The `BatchCoordinator` collects these requests from worker threads, batches them together, runs a single batched inference call, and distributes results back via `Future` objects. This amortizes per-call overhead (GPU kernel launch, model dispatch) across many inputs.

```
Worker Thread 1 ──submit(img1)──→ ┌──────────────────┐
Worker Thread 2 ──submit(img2)──→ │  BatchCoordinator │──batch([img1,img2,img3])──→ GPU
Worker Thread 3 ──submit(img3)──→ │  (daemon thread)  │
                                  └──────────────────┘
                  future.result() ←── split results ←── batched output
```

**How it works**:

1. A background daemon thread blocks on a queue waiting for items
2. When the first item arrives, it collects more items up to `max_batch_size` or `max_wait_ms`
3. Items are combined into a batch (tensor via `torch.cat`, list via concatenation, or scalar list)
4. The `inference_fn` runs on the batch
5. Results are split back (via `torch.split` with tracked per-item sizes) and distributed to callers' futures
6. On macOS, inference is wrapped in `objc.autorelease_pool()` to drain Metal/MPS ObjC objects

**Three coordinators** are created when `--parallel > 1`:

| Coordinator | Inference Function | Batch Mode | Used By |
|---|---|---|---|
| `yolo` | `PersonDetector.run_yolo` | scalar (list of PIL images) | Detection stage |
| `clip_image` | `MobileCLIPAnalyzer.batch_encode` | tensor (`torch.cat` on `(1,C,H,W)` tensors) | Scene analysis (image embeddings) |
| `clip_face` | `MobileCLIPAnalyzer.batch_encode` | tensor (`torch.cat` on `(1,C,H,W)` tensors) | Scene analysis (face crop embeddings) |

**Configuration** (in `config.py`):

- `BATCH_COORDINATOR_ENABLED`: Enable/disable batch coordinators (default: `True`)
- `BATCH_COORDINATOR_MAX_SIZE`: Maximum batch size (default: `32`)
- `BATCH_COORDINATOR_MAX_WAIT_MS`: Max wait time for batch formation (default: `50`)
- `YOLO_BATCH_ENABLED`: Enable YOLO batch inference (default: `True`)
- `DETECTION_PREFER_COREML`: Use CoreML instead of PyTorch MPS (default: `False`). CoreML is incompatible with batch coordinators — the guard prevents creating YOLO batch coordinator when CoreML is active.

### Pipeline Stage Ordering

Stages are ordered for optimal batch utilization:

```
detection → scene_analysis → age_gender
```

**Why this order matters**: Detection produces a burst of work that feeds directly into scene_analysis's batch coordinators. If age_gender (which is serialized with a thread lock) ran between them, it would throttle the burst into a trickle, producing small, inefficient batches for scene_analysis. Putting the serial stage last means it doesn't matter that it's slow — there's nothing after it waiting for batches.

**Init order** is separate from processing order:

```
age_gender → scene_analysis → detection (init)
detection → scene_analysis → age_gender (processing)
```

Detection (YOLO) must initialize last because CoreML model loading (when used as fallback) corrupts Metal/CoreGraphics state, causing SIGSEGV in any model loaded after it.

### Performance Results (Apple M1 Max)

| Configuration | 83 photos | Notes |
|---|---|---|
| CoreML YOLO + MobileCLIP on CPU | 159s | Old default |
| PyTorch YOLO on MPS + MobileCLIP on MPS (batch) | 33s | **4.8x faster** — current default |

MobileCLIP on MPS vs CPU alone is 7x faster for scene_analysis.

### Recommended --parallel Values

| Pipeline | Recommended | Rationale |
|---|---|---|
| Full pipeline (all stages) | 40–50 | Enough to keep batch coordinators fed; age_gender serialization doesn't block other workers |
| Detection + scene_analysis only | 40–60 | Both stages batch well |
| Normalize + metadata only | 100–200 | I/O bound, no GPU contention |

### macOS Native Library Ordering

On Apple Silicon, loading certain native libraries before PyTorch MPS initialization corrupts Metal/CoreGraphics state, causing SIGSEGV. Known problematic libraries:

- **CoreML** (via coremltools/ultralytics): Fixed by loading YOLO model last in init order
- **libvips** (via pyvips/cffi): Fixed by lazy-importing pyvips inside ImageHandler methods

The general pattern: any native library that initializes Metal or CoreGraphics must load **after** PyTorch MPS. Use lazy imports for these libraries.

## Performance Considerations

### Parallel Processing

- **Workers**: `--parallel N` creates ThreadPoolExecutor with N workers
- **Connection pool**: Sized dynamically (max = min(parallel, 50))
- **PostgreSQL limit**: Default max_connections=100, reserve 50 for other services

### Database Optimization

- **pgvector IVFFlat**: Lists=100 for fast similarity search
- **ON CONFLICT**: Prevents transaction rollbacks on duplicates
- **JSONB GIN index**: Efficient nested metadata queries
- **Row locking**: `SELECT FOR UPDATE` during cluster modifications

### Image Processing

- **Lazy discovery**: Generator-based file traversal
- **HEIF support**: pillow-heif for Apple formats
- **Resource limits**: MAX_PIXELS=179M, file handle management

### LLM Cost Optimization

- **Batch API**: 50% discount on Anthropic
- **Token tracking**: Per-photo and per-batch usage
- **MIN_BATCH_SIZE**: Skip small batches to maximize efficiency

## Project Structure

```text
photodb/
├── src/photodb/
│   ├── cli_local.py           # process-local CLI
│   ├── cli_enrich.py          # enrich-photos CLI
│   ├── cli_maintenance.py     # photodb-maintenance CLI
│   ├── config.py              # Configuration management
│   ├── async_batch_monitor.py # Async batch monitoring
│   ├── database/
│   │   ├── connection.py      # Connection and pool management
│   │   ├── repository.py      # Data access layer
│   │   └── models.py          # Dataclass models
│   ├── processors/
│   │   ├── base_processor.py  # Common processing logic
│   │   ├── local_processor.py # Local stage orchestration
│   │   └── batch_processor.py # Batch enrichment orchestration
│   ├── stages/
│   │   ├── base.py            # BaseStage class
│   │   ├── normalize.py       # Stage 1
│   │   ├── metadata.py        # Stage 2
│   │   ├── detection.py       # Stage 3 (YOLO face+body)
│   │   ├── age_gender.py      # Stage 4 (MiVOLO)
│   │   ├── clustering.py      # Stage 5
│   │   └── enrich.py          # Stage R1
│   ├── models/
│   │   └── photo_analysis.py  # Pydantic LLM response schema
│   └── utils/
│       ├── batch_coordinator.py     # ML inference batching
│       ├── exif.py                  # EXIF extraction
│       ├── image.py                 # Image handling
│       ├── mobileclip_analyzer.py   # MobileCLIP-S2 encoder
│       ├── person_detector.py       # YOLO + InsightFace
│       ├── age_gender_aggregator.py # Person-level aggregation
│       ├── embedding_extractor.py   # InsightFace face embeddings
│       ├── timm_compat.py           # timm 0.8→1.0 shim for MiVOLO
│       └── maintenance.py           # Cluster maintenance
├── prompts/
│   └── system_prompt.md       # LLM system prompt
├── schema.sql                 # Database schema
├── scripts/
│   ├── bootstrap_clusters.py  # HDBSCAN bootstrap clustering
│   └── download_models.sh     # Download YOLO + MiVOLO models
├── migrations/                # Database migrations
├── tests/                     # Test suite
└── docs/
    ├── DESIGN.md              # This document
    └── BEDROCK_SETUP.md       # AWS Bedrock configuration
```

## Model Setup

Download required models for detection and age/gender estimation:

```bash
./scripts/download_models.sh
```

This downloads:

- `yolov8x_person_face.pt` (137 MB) - YOLO face+body detector
- `mivolo_d1.pth.tar` (~330 MB) - MiVOLO age/gender model (face+body)

## Future Enhancements

- Person association UI for resolving `face_match_candidate` records
- Cluster quality metrics and monitoring dashboard
- Active learning to improve clustering threshold
- Multi-modal clustering using photo metadata (time, location proximity)
- Full-text search on LLM descriptions
- Geographic clustering and location inference
- Timeline reconstruction using person age estimates across photos
