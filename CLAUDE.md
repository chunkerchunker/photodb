# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

PhotoDB is a personal photo indexing pipeline built with Python and PostgreSQL. It processes photos through multiple stages in a parallel, distributed architecture designed for high throughput with hundreds of concurrent workers.

See `docs/DESIGN.md` for detailed processing pipeline design, database schema, and algorithm documentation.

### Key Components

- **CLI**: Three executables
  - `process-local` (`src/photodb/cli_local.py`): Local photo processing (normalize, metadata, detection, age/gender, scene analysis)
  - `enrich-photos` (`src/photodb/cli_enrich.py`): Remote LLM-based enrichment with batch processing
  - `photodb-maintenance` (`src/photodb/cli_maintenance.py`): Periodic cluster optimization and system health
- **Processors (`src/photodb/processors/`)**: Orchestrates parallel photo processing using ThreadPoolExecutor
  - `base_processor.py`: Common processing logic
  - `local_processor.py`: Local stage orchestration
  - `batch_processor.py`: Batch enrichment orchestration
- **Stages (`src/photodb/stages/`)**: Processing pipeline stages (normalize, metadata, detection, age_gender, scene_analysis, enrich) plus standalone clustering
  - All stages inherit from `BaseStage` (`src/photodb/stages/base.py`)
  - Clustering is run separately via `scripts/bootstrap_clusters.py` after batch imports
- **Database Layer**: PostgreSQL with pgvector, connection pooling
  - Models (`src/photodb/database/models.py`): Photo, ProcessingStatus, Metadata entities
  - Repository (`src/photodb/database/repository.py`): Data access layer
  - Connection Pool (`src/photodb/database/connection.py`): Manages concurrent database connections
- **Web UI** (`web/`): React Router v7 + Vite + Tailwind frontend for browsing and managing photos
- **Utilities**: EXIF processing, image handling, validation, logging

### Processing Pipeline

Photos flow through stages sequentially but can be processed in parallel:

1. **Normalize**: File organization and path standardization
2. **Metadata**: EXIF extraction and metadata parsing
3. **Detection**: YOLO face/body detection (CoreML on macOS)
4. **Age/Gender**: MiVOLO estimation from detected faces
5. **Scene Analysis**: Apple Vision taxonomy + MobileCLIP prompt tagging
6. **Clustering**: HDBSCAN face clustering (run separately after batch imports)
7. **Cluster Grouping**: Auto-association of similar clusters into persons (weekly maintenance)
8. **Enrich**: LLM-based analysis via batch processing

Each stage is idempotent, tracks processing status per photo, and supports force reprocessing.

All paths should handle both absolute and relative configurations.

## Development Commands

### Package Management

This project uses `uv` as the package manager:

```bash
uv sync                    # Install dependencies
uv add <package>           # Add dependency
uv run <command>           # Run command in project environment
```

### Running the Application

Common tasks have `just` shortcuts (see `.justfile`):

```bash
just local /path/to/photos --parallel 4    # Run local pipeline
just cluster <collection_id>               # Run HDBSCAN bootstrap clustering
just group-clusters <collection_id>        # Auto-associate clusters to persons
just daily                                 # Daily maintenance
just weekly                                # Weekly maintenance
just capture-import --order-id 123         # Import from Capture system
```

#### Local Photo Processing

```bash
uv run process-local /path/to/photos
uv run process-local /path/to/photos --parallel 5
uv run process-local /path/to/photos --stage metadata   # specific stage
uv run process-local /path/to/photos --force             # force reprocessing
uv run process-local /path/to/photos --dry-run
```

#### Remote Enrichment Processing

```bash
uv run enrich-photos /path/to/photos
uv run enrich-photos --check-batches --wait
uv run enrich-photos /path/to/photos --retry-failed
uv run enrich-photos /path/to/photos --no-batch
```

#### Maintenance

```bash
uv run photodb-maintenance daily
uv run photodb-maintenance weekly
uv run photodb-maintenance auto-associate --collection-id 1
uv run photodb-maintenance check-staleness
uv run photodb-maintenance health
```

### Testing

```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_*.py    # Run specific test file
uv run pytest -v                 # Verbose output
uv run pytest --cov              # With coverage
```

### Linting, Formatting, Type Checking

```bash
uv run ruff format              # Format code (line length: 100)
uv run ruff check               # Lint code
uv run ruff check --fix         # Auto-fix linting issues
uv run pyright check            # Type check code
cd web && pnpm check            # Web frontend (Biome)
```

### Database Setup

PostgreSQL is required. See `POSTGRESQL_SETUP.md` for detailed setup instructions.

```bash
createdb photodb
export DATABASE_URL="postgresql://localhost/photodb"
```

#### Database Migrations

```bash
psql $DATABASE_URL -f migrations/005_add_person_detection.sql
psql $DATABASE_URL -f migrations/006_add_scene_analysis.sql
psql $DATABASE_URL -f migrations/020_hdbscan_hierarchy.sql
```

### Model Setup

```bash
./scripts/download_models.sh    # Downloads YOLOv8x + MiVOLO to models/
```

## Configuration

Environment variables (see `docs/DESIGN.md` for full list with defaults):

- `DATABASE_URL`: PostgreSQL connection string (required)
- `INGEST_PATH`: Source photos path (default: `./photos/raw`)
- `IMG_PATH`: Processed photos output path (default: `./photos/processed`)
- `LOG_LEVEL`: Logging verbosity (default: `INFO`)
- `LOG_FILE`: Log file path (default: `./logs/photodb.log`)

### Detection

- `DETECTION_MODEL_PATH`: Path to YOLOv8x person_face model (default: `models/yolov8x_person_face.pt`)
- `DETECTION_FORCE_CPU`: Force CPU mode (default: `false`)
- `DETECTION_MIN_CONFIDENCE`: Confidence threshold (default: `0.5`)

### Age/Gender (MiVOLO)

- `MIVOLO_MODEL_PATH`: Path to MiVOLO checkpoint (default: `models/mivolo_d1.pth.tar`)
- `MIVOLO_FORCE_CPU`: Force CPU mode (default: `false`)

### Face Embeddings (InsightFace)

- `EMBEDDING_MODEL_NAME`: InsightFace model pack (default: `buffalo_l`)
- `EMBEDDING_MODEL_ROOT`: Custom model directory (default: `~/.insightface/models`)

### Clustering

- `HDBSCAN_MIN_CLUSTER_SIZE`: Minimum faces to form a cluster (default: `3`)
- `HDBSCAN_MIN_SAMPLES`: Core point requirement (default: `2`)
- `CLUSTERING_THRESHOLD`: Fallback distance threshold (default: `0.45`)
- `PERSON_ASSOCIATION_THRESHOLD`: Cosine distance for cluster auto-grouping (default: `0.8`)

### Scene Analysis

- `CLIP_MODEL_NAME`: CLIP model (default: `MobileCLIP-S2`)
- `CLIP_PRETRAINED`: Pretrained weights (default: `datacompdr`)

Seed prompts with `uv run python scripts/seed_prompts.py`. Recompute embeddings after model changes with `--recompute-embeddings`.

### LLM Enrichment

- `LLM_PROVIDER`: `anthropic` (default) or `bedrock`
- `ANTHROPIC_API_KEY`: API key for Anthropic
- `BEDROCK_MODEL_ID`: Bedrock model ID (default: `anthropic.claude-3-5-sonnet-20241022-v2:0`)
- `AWS_REGION`: AWS region for Bedrock (default: `us-east-1`)
- `BATCH_SIZE`: Photos per LLM batch (default: `100`)
- `MIN_BATCH_SIZE`: Minimum batch size (default: `10`)

## Important Gotchas

### timm Compatibility

MiVOLO was written for timm 0.8.x but we use timm 1.0.x for MobileCLIP-S2's FastViT backbone. A compatibility shim (`src/photodb/utils/timm_compat.py`) patches:

1. `remap_checkpoint()` → `remap_state_dict()` (API renamed in timm 0.9+)
2. `split_model_name_tag()` (removed in timm 0.9+)
3. `MiVOLOModel.__init__()` (VOLO class added `pos_drop_rate` parameter in timm 0.9+)

Always import `timm_compat` before importing mivolo to apply the patches.

### Free-threaded Python

**Not currently usable** due to one blocker:

1. **opencv-python**: No wheels for Python 3.13t (free-threaded)

Use standard Python 3.13. InsightFace (ONNX Runtime), MiVOLO, and CoreML detection are already thread-safe.

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
