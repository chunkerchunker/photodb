# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

PhotoDB is a personal photo indexing pipeline built with Python and PostgreSQL. It processes photos through multiple stages in a parallel, distributed architecture designed for high throughput with hundreds of concurrent workers.

### Key Components

- **CLI**: Three separate executables
  - `process-local` (`src/photodb/cli_local.py`): Local photo processing (normalize, metadata extraction)
  - `enrich-photos` (`src/photodb/cli_enrich.py`): Remote LLM-based enrichment with batch processing
  - `photodb-web` (`src/photodb/cli_web.py`): Web server for browsing photos
- **Processors (`src/photodb/processors.py`)**: Orchestrates parallel photo processing using ThreadPoolExecutor
- **Stages (`src/photodb/stages/`)**: Processing pipeline stages (normalize, metadata, enrich, stats)
  - All stages inherit from `BaseStage` (`src/photodb/stages/base.py`)
  - Each stage handles a specific aspect: file normalization, metadata extraction, enrichment, statistics
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

**Remote Processing (`enrich-photos`):**
3. **Enrich**: LLM-based analysis and enrichment using batch processing

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

# Specific stage only (normalize or metadata)
uv run process-local /path/to/photos --stage metadata

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

#### Web Server

```bash
# Start the photo browsing web server
uv run photodb-web

# With custom port and debug mode
uv run photodb-web --port 8080 --debug
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
