# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

PhotoDB is a personal photo indexing pipeline built with Python and PostgreSQL. It processes photos through multiple stages in a parallel, distributed architecture designed for high throughput with hundreds of concurrent workers.

Detailed processing design in `docs/DESIGN.md`.

### Key Components

- **CLI**: Two separate executables
  - `process-local` (`src/photodb/cli_local.py`): Local photo processing (normalize, metadata extraction)
  - `enrich-photos` (`src/photodb/cli_enrich.py`): Remote LLM-based enrichment with batch processing
- **Processors (`src/photodb/processors.py`)**: Orchestrates parallel photo processing using ThreadPoolExecutor
- **Stages (`src/photodb/stages/`)**: Processing pipeline stages (normalize, metadata, detection, age_gender, clustering, scene_analysis, enrich)
  - All stages inherit from `BaseStage` (`src/photodb/stages/base.py`)
  - Each stage handles a specific aspect: file normalization, metadata extraction, person/face detection, age/gender estimation, face clustering, scene/sentiment analysis, enrichment
- **Database Layer**: PostgreSQL-based with connection pooling
  - Models (`src/photodb/database/models.py`): Photo, ProcessingStatus, Metadata entities
  - Repository (`src/photodb/database/pg_repository.py`): Data access layer
  - Connection Pool (`src/photodb/database/pg_connection.py`): Manages concurrent database connections
- **Utilities**: EXIF processing, image handling, validation, logging

### Processing Pipeline

Photos flow through stages sequentially but can be processed in parallel. The pipeline is split into local and remote processing:

**Local Processing (`process-local`):**

1. **Normalize**: File organization and path standardization
2. **Metadata**: EXIF extraction and metadata parsing
3. **Detection**: YOLO-based face and body detection using YOLOv8x person_face model
4. **Age/Gender**: MiVOLO-based age and gender estimation from detected faces
5. **Clustering**: Face embedding extraction and clustering for person identification
6. **Scene Analysis**: Apple Vision scene taxonomy (macOS) and prompt-based tagging with MobileCLIP

**Remote Processing (`enrich-photos`):**
7. **Enrich**: LLM-based analysis and enrichment using batch processing

Each stage tracks its processing status per photo, allowing for granular recovery and reprocessing.

Each stage should be idempotent and support force reprocessing

All paths should handle both absolute and relative configurations

## Development Commands

### Package Management

This project uses `uv` as the package manager:

```bash
uv sync                    # Install dependencies
uv add <package>           # Add dependency
uv run <command>           # Run command in project environment
```

### Running the Application

#### Local Photo Processing

```bash
# Basic usage
uv run process-local /path/to/photos

# With parallel processing (recommended for local stages)
uv run process-local /path/to/photos --parallel 500

# Specific stage only (normalize, metadata, detection, age_gender, clustering, scene_analysis)
uv run process-local /path/to/photos --stage metadata

# Run detection stage only (face/body detection)
uv run process-local /path/to/photos --stage detection

# Run age/gender estimation only
uv run process-local /path/to/photos --stage age_gender

# Run scene analysis (taxonomy + prompt tagging)
uv run process-local /path/to/photos --stage scene_analysis

# Force reprocessing
uv run process-local /path/to/photos --force

# Dry run to see what would be processed
uv run process-local /path/to/photos --dry-run
```

#### Remote Enrichment Processing

```bash
# Basic usage (processes in batches)
uv run enrich-photos /path/to/photos

# Force reprocessing
uv run enrich-photos /path/to/photos --force

# Check status of running batches
uv run enrich-photos --check-batches

# Check and wait for batch completion
uv run enrich-photos --check-batches --wait

# Retry failed enrichment
uv run enrich-photos /path/to/photos --retry-failed

# Disable batch mode (process one at a time)
uv run enrich-photos /path/to/photos --no-batch
```

### Testing

```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_*.py    # Run specific test file
uv run pytest -v                # Verbose output
uv run pytest --cov             # With coverage
```

### Linting, Formatting, Type Checking

```bash
uv run ruff format              # Format code (line length: 100)
uv run ruff check               # Lint code
uv run ruff check --fix         # Auto-fix linting issues
uv run pyright check            # Type check code
```

### Database Setup

PostgreSQL is required. See `POSTGRESQL_SETUP.md` for detailed setup instructions.

```bash
# Create database
createdb photodb

# Set environment variable
export DATABASE_URL="postgresql://localhost/photodb"

# Or add to .env file
echo "DATABASE_URL=postgresql://localhost/photodb" >> .env
```

#### Database Migrations

For existing databases, run migrations to add new tables:

```bash
# Add person detection tables (required for detection and age_gender stages)
psql $DATABASE_URL -f migrations/005_add_person_detection.sql
```

### Model Setup

The detection and age/gender stages require pre-trained model files. Download them using:

```bash
./scripts/download_models.sh
```

This downloads:
- YOLOv8x person_face model for face/body detection
- MiVOLO model for age/gender estimation

Models are saved to the `models/` directory by default.

## Configuration

The application uses environment variables and optional config files:

- `DATABASE_URL`: PostgreSQL connection string (required)
- `INGEST_PATH`: Default path for photo ingestion (default: `./photos/raw`)
- `IMG_PATH`: Processed photos output path (default: `./photos/processed`)
- `LOG_LEVEL`: Logging verbosity (default: `INFO`)
- `LOG_FILE`: Log file path (default: `./logs/photodb.log`)
- `LLM_PROVIDER`: LLM provider - `anthropic` (default) or `bedrock` for AWS Bedrock
- `LLM_API_KEY` or `ANTHROPIC_API_KEY`: API key for Anthropic (not needed for Bedrock)
- `BEDROCK_MODEL_ID`: Bedrock model ID (default: `anthropic.claude-3-5-sonnet-20241022-v2:0`)
- `AWS_REGION`: AWS region for Bedrock (default: `us-east-1`)
- `AWS_PROFILE`: Optional AWS profile name for Bedrock
- `BATCH_SIZE`: Number of photos per LLM batch (default: `100`)
- `MIN_BATCH_SIZE`: Minimum batch size for enrich processing (default: `10`) - batches smaller than this will be skipped
- `MIN_FACE_SIZE_PX`: Minimum face size in pixels for clustering (default: `50`) - faces smaller than this are excluded from clustering
- `MIN_FACE_CONFIDENCE`: Minimum face detection confidence for clustering (default: `0.9`) - faces with lower confidence are excluded from clustering

### Detection Stage Configuration

- `DETECTION_MODEL_PATH`: Path to YOLOv8x person_face model (default: `models/yolov8x_person_face.pt`)
- `DETECTION_FORCE_CPU`: Force CPU mode for detection (default: `false`)
- `DETECTION_MIN_CONFIDENCE`: Minimum detection confidence threshold (default: `0.5`)

**CoreML Support (macOS):** On macOS, the detector automatically uses CoreML (`.mlpackage`) if available, providing 5x faster inference via the Neural Engine. CoreML is also thread-safe, enabling parallel processing. The download script automatically exports the CoreML model on macOS.

### Age/Gender Stage Configuration

- `MIVOLO_MODEL_PATH`: Path to MiVOLO checkpoint (default: `models/mivolo_d1.pth.tar`)
- `MIVOLO_FORCE_CPU`: Force CPU mode for MiVOLO inference (default: `false`)

**Thread Safety:** MiVOLO inference is serialized with a lock. Testing showed that without serialization:
1. MiVOLO's internal YOLO detector lazy-initializes on first use, causing race conditions when multiple threads call `recognize()` simultaneously
2. Even after initialization, concurrent inference produces inconsistent results (same image returns different prediction counts)

*Tested with mivolo 0.6.0.dev0 (git HEAD) on 2026-02-01. Future versions may fix these issues.*

**timm Compatibility:** MiVOLO was written for timm 0.8.x but we use timm 1.0.x for MobileCLIP-S2's FastViT backbone. A compatibility shim (`src/photodb/utils/timm_compat.py`) patches:
1. `remap_checkpoint()` → `remap_state_dict()` (API renamed in timm 0.9+)
2. `split_model_name_tag()` (removed in timm 0.9+)
3. `MiVOLOModel.__init__()` (VOLO class added `pos_drop_rate` parameter in timm 0.9+)

Always import `timm_compat` before importing mivolo to apply the patches.

### Face Embedding Configuration

- `EMBEDDING_MODEL_NAME`: InsightFace model pack name (default: `buffalo_l`)
- `EMBEDDING_MODEL_ROOT`: Custom model directory (default: `~/.insightface/models`)

**Hardware Acceleration:**
- **macOS:** Uses CoreML via ONNX Runtime for Neural Engine acceleration (thread-safe)
- **CUDA:** Uses CUDA via ONNX Runtime for GPU acceleration
- **CPU:** Falls back to CPU if no accelerators available

**Model Location:** InsightFace models auto-download to `~/.insightface/models/` on first use.

### Clustering Stage Configuration

The clustering stage uses a hybrid HDBSCAN → Incremental DBSCAN approach:
- **Bootstrap phase**: HDBSCAN identifies stable clusters and core points from existing embeddings
- **Incremental phase**: New faces are assigned using per-cluster epsilon-ball queries

**Configuration:**
- `HDBSCAN_MIN_CLUSTER_SIZE`: Minimum faces to form a cluster (default: `3`)
- `HDBSCAN_MIN_SAMPLES`: Core point requirement for HDBSCAN (default: `2`)
- `EPSILON_PERCENTILE`: Percentile of core point distances for epsilon calculation (default: `90`)
- `CLUSTERING_THRESHOLD`: Fallback distance threshold for clusters without epsilon (default: `0.45`)

**Bootstrap Clustering:**
To run HDBSCAN bootstrap on existing embeddings:
```bash
uv run python scripts/migrate_to_hdbscan.py --dry-run  # Preview changes
uv run python scripts/migrate_to_hdbscan.py            # Run migration
```

**Database Migration:**
For existing databases, run the migration for HDBSCAN support:
```bash
psql $DATABASE_URL -f migrations/010_hdbscan_clustering.sql
```

**How it works:**
1. HDBSCAN runs on all embeddings to identify density-based clusters
2. Core points (high-probability cluster members) are marked with `is_core = true`
3. Per-cluster epsilon is calculated as the 90th percentile of core point distances
4. New faces are assigned if distance to any cluster centroid < that cluster's epsilon
5. Manual assignments (`cluster_status = 'manual'`) and verified clusters are preserved

**Metal/MPS GPU Acceleration (Apple Silicon):**
On Apple Silicon Macs, HDBSCAN bootstrap uses Metal/MPS for ~8x faster clustering:
- GPU-accelerated k-NN search via PyTorch MPS
- Sparse graph reduces MST complexity from O(n²) to O(n×k)
- Automatically falls back to CPU if MPS unavailable
- Validated with ARI > 0.97 (nearly identical results to CPU)

**Person-Cluster Relationship:**
- A Person can have multiple clusters (e.g., same person at different ages)
- Clusters are linked to a Person via `cluster.person_id`
- Users assign clusters to persons directly via the web UI
- Cannot-link constraints prevent faces from being assigned to forbidden clusters

**Constraints:**
- `cannot_link`: Prevents a face from being assigned to a cluster (checked during incremental clustering)
- No must-link table - identity linking is done directly via person_id

### Scene Analysis Stage Configuration

- `CLIP_MODEL_NAME`: CLIP model to use (default: `MobileCLIP-S2`)
- `CLIP_PRETRAINED`: Pretrained weights source (default: `datacompdr`)

The scene analysis stage provides two features:

1. **Apple Vision Scene Taxonomy (macOS only):** Uses Apple's Vision framework to classify scenes with 1303 built-in labels. Returns top-k labels with confidence scores.

2. **Prompt-based Tagging:** Uses MobileCLIP to match images and face crops against configurable text prompts for semantic tagging.

#### Prompt Management

Prompts are stored in the database and organized into categories:
- **Face categories:** `face_emotion`, `face_expression`, `face_gaze`
- **Scene categories:** `scene_mood`, `scene_setting`, `scene_activity`, `scene_time`, `scene_weather`, `scene_social`

To seed initial prompts:
```bash
uv run python scripts/seed_prompts.py
```

To recompute embeddings after model changes:
```bash
uv run python scripts/seed_prompts.py --recompute-embeddings
```

To add custom prompts, insert rows into the `prompt_embedding` table with the appropriate `category_id` and run the seed script with `--recompute-embeddings` to compute the embeddings.

#### Database Migration

For existing databases, run the migration to add scene analysis tables:
```bash
psql $DATABASE_URL -f migrations/006_add_scene_analysis.sql
```

### Free-threaded Python

**Not currently usable** due to two blockers:

1. **opencv-python**: MiVOLO depends on opencv-python, which has no wheels for Python 3.13t (free-threaded). This is a build/packaging issue that may be resolved in future opencv releases.

2. **MiVOLO thread safety**: Even if opencv were available, MiVOLO inference must be serialized (see above), so free-threaded Python wouldn't improve throughput for the age/gender stage.

Use standard Python 3.13 for now. The detection stage uses CoreML which is already thread-safe and parallel.

Note: InsightFace (face embeddings) uses ONNX Runtime which is thread-safe, so it would
benefit from free-threaded Python once the opencv blocker is resolved.

## Performance Considerations

- **Parallel Processing**: Designed for 500+ concurrent workers
- **Connection Pooling**: Automatically sized (2x workers, max 200 connections)
- **File Semaphore**: Limits concurrent file operations to prevent "too many open files"
- **PostgreSQL Benefits**: No write locks, true concurrency, JSONB support for metadata

## Testing Strategy

Tests are organized by component:

- `test_database.py`: Database operations and models
- `test_metadata.py`: EXIF and metadata extraction
- `test_normalize.py`: File normalization logic
- `test_image.py`: Image processing utilities
- `test_validation.py`: Input validation
- `test_*.py`: Additional component tests

When adding new features, follow the existing test patterns and ensure all stages can be tested independently.
