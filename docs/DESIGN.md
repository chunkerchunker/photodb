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
│  └─ Stage 5: Clustering - Group detections by ID    │
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
uv run process-local /path/to/photos --stage clustering

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
5. Extract 512-dimensional FaceNet embeddings for faces (InceptionResnetV1)
6. Store person_detection records with face and body bounding boxes
7. Save face embeddings in pgvector format

**Key features**:
- YOLOv8x person_face model for unified face+body detection
- Face-to-body matching via spatial containment algorithm
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

### Stage 5: Clustering

**File**: `src/photodb/stages/clustering.py`

Constrained incremental clustering for person identity grouping based on face embeddings.

**Algorithm** (per-detection with constraints):

1. **Check must-link constraints** - if detection has must-link to clustered detection, assign directly
2. **Find K nearest neighbors** (detections, not clusters) using pgvector
3. **Calculate core distance confidence** - mean distance to K neighbors
4. **Filter by cannot-link constraints** - remove forbidden clusters
5. **Apply decision rules**:
   - **No valid matches**: Add to unassigned pool
   - **Single valid cluster**: Assign (stricter threshold for verified clusters)
   - **Multiple valid clusters**: Mark for manual review
6. **Unassigned pool**: When enough similar unassigned detections accumulate, form new cluster

**Pool Cluster Formation** (prevents chaining of dissimilar detections):

1. Calculate centroid from all similar unassigned detections
2. Filter to only detections within `POOL_CLUSTERING_THRESHOLD` of centroid
3. Recalculate centroid with filtered detections only
4. Find medoid (detection closest to centroid)
5. Assign detections with confidence based on distance to centroid

**Inline Medoid Recomputation**:

When a cluster grows by `MEDOID_RECOMPUTE_THRESHOLD` (default 25%) since last medoid computation, the medoid is recomputed inline during detection assignment. This uses `face_count_at_last_medoid` to track when recomputation is needed.

**Concurrency Safety**:

- Row-level locking with `SELECT FOR UPDATE` during cluster modifications
- Atomic face assignment checks prevent race conditions between workers
- Empty clusters automatically deleted when all faces reassigned

**Constraint System**:
- **Must-link**: Forces faces into same cluster (human override)
- **Cannot-link**: Prevents faces from sharing cluster (human override)
- **Verified clusters**: Protected from auto-merge, use stricter assignment threshold

**Configuration**:
- `CLUSTERING_THRESHOLD`: Cosine distance threshold for assigning to existing clusters (default: 0.45)
- `POOL_CLUSTERING_THRESHOLD`: Stricter threshold for forming clusters from unassigned pool (default: 70% of CLUSTERING_THRESHOLD)
- `CLUSTERING_K_NEIGHBORS`: K nearest neighbors to check (default: 5)
- `UNASSIGNED_CLUSTER_THRESHOLD`: Faces needed to form pool cluster (default: 5)
- `VERIFIED_THRESHOLD_MULTIPLIER`: Stricter threshold for verified clusters (default: 0.8)
- `MEDOID_RECOMPUTE_THRESHOLD`: Growth ratio to trigger inline medoid recomputation (default: 0.25)

**Centroid update formula** (incremental, during assignment):
```
new_centroid = (old_centroid * face_count + embedding) / (face_count + 1)
```

All embeddings and centroids are L2-normalized to unit vectors for cosine similarity.

**Output**: Face cluster assignments + cluster centroids + constraint records

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
    created_at, updated_at
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

# Age/Gender Stage (MiVOLO)
MIVOLO_MODEL_PATH=models/mivolo_d1.pth.tar
MIVOLO_FORCE_CPU=false             # Force CPU for MiVOLO

# Clustering (constrained incremental)
CLUSTERING_THRESHOLD=0.45          # Face similarity threshold for existing clusters
POOL_CLUSTERING_THRESHOLD=0.315    # Stricter threshold for pool clustering (default: 70% of main)
CLUSTERING_K_NEIGHBORS=5           # K nearest neighbors to check
UNASSIGNED_CLUSTER_THRESHOLD=5     # Faces needed to form pool cluster
VERIFIED_THRESHOLD_MULTIPLIER=0.8  # Stricter threshold for verified clusters
MEDOID_RECOMPUTE_THRESHOLD=0.25    # Growth ratio to trigger inline medoid recomputation

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

```
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
│       ├── exif.py            # EXIF extraction
│       ├── image.py           # Image handling
│       ├── person_detector.py # YOLO + FaceNet
│       ├── age_gender_aggregator.py  # Person-level aggregation
│       └── maintenance.py     # Cluster maintenance
├── prompts/
│   └── system_prompt.md       # LLM system prompt
├── schema.sql                 # Database schema
├── scripts/
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
