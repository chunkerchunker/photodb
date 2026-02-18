# PhotoDB

Personal photo indexing pipeline built with Python and PostgreSQL. Processes photos through detection, face recognition, clustering, scene analysis, and LLM enrichment stages.

## Requirements

- macOS on Apple Silicon (uses Vision, CoreML, and Metal APIs)
- [Homebrew](https://brew.sh)

## Quick Start

```bash
git clone <repo-url> && cd photodb

# Create .env with your paths
cat > .env << 'EOF'
DATABASE_URL=postgresql://localhost/photodb
INGEST_PATH=/path/to/your/photos
IMG_PATH=/path/to/processed/output
EOF

./scripts/bootstrap.sh
```

The bootstrap script handles the rest: Homebrew packages, Python environment, database setup, ML model downloads, prompt seeding, and web frontend dependencies.

## Usage

### Process photos

```bash
# Run full local pipeline, up to clustering
just local /path/to/photos --parallel 4

# Run a specific stage
just local /path/to/photos --stage detection

# run basic clustering
just cluster

# run cluster grouping
just group-clusters

# optional LLM enrichment (batch mode via AWS Bedrock)
uv run enrich-photos /path/to/photos
```

### Web UI

```bash
cd web && pnpm dev        # Development server at http://localhost:5173
cd web && pnpm build      # Production build
```

Default login: username `default`, password `changeme`. The password is automatically upgraded to a scrypt hash on first login. To set a password directly:

```bash
uv run python scripts/update_user_password.py --user-id 1 --password "yourpassword"
```

### Import from Capture

Import an order from the Capture photography system. Creates a new user, collection, albums, and photo entries in PhotoDB.

```bash
# Preview what would be imported
just capture-import --order-id 123 --dry-run

# Import with default settings
just capture-import --order-id 123
```

Configure via `.env`: `CAPTURE_DATABASE_URL` (defaults to `postgresql://localhost/capture`) and `CAPTURE_BASE_PATH` (defaults to `/Volumes/media/Pictures/capture`).

### Tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check         # Lint
uv run ruff format        # Format
cd web && pnpm check      # Web frontend (Biome)
```

## Architecture

Photos flow through stages sequentially, processed in parallel across workers:

1. **Normalize** — file organization and path standardization
2. **Metadata** — EXIF extraction and parsing
3. **Detection** — YOLO face/body detection (CoreML on macOS)
4. **Age/Gender** — MiVOLO estimation from detected faces
5. **Scene Analysis** — Apple Vision taxonomy + MobileCLIP prompt tagging
6. **Clustering** — HDBSCAN face clustering with Metal-accelerated k-NN (run separately via `scripts/bootstrap_clusters.py`)
7. **Cluster Grouping** -- Auto-grouping of similar clusters into persons (run separately via `scripts/photodb-maintenance auto-associate`)
8. **Enrich** — LLM-based analysis via AWS Bedrock batch processing

The web frontend (React Router v7 + Vite + Tailwind) connects directly to the same PostgreSQL database for browsing and managing photos.

See `CLAUDE.md` for detailed architecture documentation.
