# PhotoDB Design Document

PhotoDB is a personal photo indexing pipeline that processes photos through multiple stages to create a searchable, navigable database. It combines local processing (file normalization, metadata extraction, face detection, clustering) with remote LLM-based enrichment.

## Architecture Overview

```
Photo Files (raw)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  LOCAL PROCESSING (process-local CLI)               │
│  ├─ Stage 1: Normalize - Convert to PNG, resize     │
│  ├─ Stage 2: Metadata - Extract EXIF data           │
│  ├─ Stage 3: Faces - Detect faces, extract embed.   │
│  └─ Stage 4: Clustering - Group faces by identity   │
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
uv run process-local /path/to/photos --stage faces
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

### Stage 3: Faces

**File**: `src/photodb/stages/faces.py`

Detects faces and extracts embeddings for clustering.

**Process**:
1. Load normalized image
2. Detect faces using MTCNN (Multi-task Cascaded CNNs)
3. Filter by confidence threshold (default: 0.85)
4. Extract 512-dimensional FaceNet embeddings (InceptionResnetV1)
5. Store face records with bounding boxes
6. Save embeddings in pgvector format

**Key features**:
- PyTorch with auto device detection (MPS/CUDA/CPU)
- `FACE_DETECTION_FORCE_CPU=true` for CPU fallback
- MTCNN parameters: margin=20, min_face_size=20
- L2-normalized embeddings for cosine similarity

**Output**: Face records + pgvector embeddings

### Stage 4: Clustering

**File**: `src/photodb/stages/clustering.py`

Groups faces into identity clusters using embedding similarity.

**Algorithm** (per-face incremental):

1. **Query unclustered faces** with embeddings
2. **Find nearest clusters** using pgvector IVFFlat index (KNN search)
3. **Apply decision rules**:
   - **Rule A** (no match < threshold): Create new cluster
   - **Rule B** (single match): Assign to cluster, update centroid
   - **Rule C** (multiple matches): Mark for manual review

**Configuration**:
- `CLUSTERING_THRESHOLD`: Cosine distance threshold (default: 0.45)

**Centroid update formula**:
```
new_centroid = (old_centroid * face_count + embedding) / (face_count + 1)
```

**Output**: Face cluster assignments + cluster centroids

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
-- stage: 'normalize' | 'metadata' | 'faces' | 'clustering' | 'enrich'
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

### Face Detection & Clustering Tables

```sql
-- Named individuals
person (id, name, created_at, updated_at)

-- Detected faces
face (
    id, photo_id, bbox_x, bbox_y, bbox_width, bbox_height,
    confidence, person_id, cluster_status, cluster_id, cluster_confidence
)
-- cluster_status: 'auto' | 'pending' | 'manual'

-- Face embeddings (pgvector)
face_embedding (
    face_id, embedding VECTOR(512)
)
-- IVFFlat index for similarity search

-- Identity clusters
cluster (
    id, face_count, representative_face_id,
    centroid VECTOR(512), medoid_face_id,
    person_id, created_at, updated_at
)

-- Ambiguous face matches for review
face_match_candidate (
    candidate_id, face_id, cluster_id, similarity,
    status, created_at
)
-- status: 'pending' | 'accepted' | 'rejected'
```

### Required Indexes

```sql
-- Vector similarity search
CREATE INDEX idx_face_embedding ON face_embedding
    USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_cluster_centroid ON cluster
    USING ivfflat(centroid vector_cosine_ops) WITH (lists = 100);

-- Common queries
CREATE INDEX idx_face_cluster_id ON face(cluster_id);
CREATE INDEX idx_face_cluster_status ON face(cluster_status);
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

### Weekly Tasks

All daily tasks, plus:

4. **Update medoids**: Find face closest to each cluster centroid
5. **Merge similar clusters**: Combine clusters with centroid similarity > threshold

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
    "unclustered_faces": int
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
FACE_MIN_CONFIDENCE=0.85           # Face detection threshold
FACE_DETECTION_FORCE_CPU=false     # Force CPU for face detection
CLUSTERING_THRESHOLD=0.45          # Face similarity threshold

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
│   │   ├── faces.py           # Stage 3
│   │   ├── clustering.py      # Stage 4
│   │   └── enrich.py          # Stage R1
│   ├── models/
│   │   └── photo_analysis.py  # Pydantic LLM response schema
│   └── utils/
│       ├── exif.py            # EXIF extraction
│       ├── image.py           # Image handling
│       ├── face_extractor.py  # MTCNN + FaceNet
│       └── maintenance.py     # Cluster maintenance
├── prompts/
│   └── system_prompt.md       # LLM system prompt
├── schema.sql                 # Database schema
├── tests/                     # Test suite
└── docs/
    ├── DESIGN.md              # This document
    └── BEDROCK_SETUP.md       # AWS Bedrock configuration
```

## Future Enhancements

- Person association UI for resolving `face_match_candidate` records
- Cluster quality metrics and monitoring dashboard
- Active learning to improve clustering threshold
- Multi-modal clustering using photo metadata (time, location proximity)
- Full-text search on LLM descriptions
- Geographic clustering and location inference
