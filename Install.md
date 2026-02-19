# PhotoDB Manual Installation

**See [README](README.md) for automated installation. This is only for manual installs.**

## 1. Install system dependencies

```bash
brew bundle
```

This installs from the `Brewfile`: git, just, node, pnpm, uv, PostgreSQL 18, pgvector, jq.

## 2. Install Python dependencies

```bash
uv sync
```

## 3. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set DATABASE_URL, INGEST_PATH, and IMG_PATH
```

## 4. Set up the database

```bash
createdb photodb
psql $DATABASE_URL -f schema.sql

# Apply all migrations
for f in migrations/[0-9]*.sql; do
    [[ "$f" == *rollback* ]] && continue
    psql $DATABASE_URL -f "$f"
done
```

## 5. Download ML models

```bash
./scripts/download_models.sh
```

Downloads YOLOv8x person/face detector, MiVOLO age/gender model, InsightFace embeddings, and exports CoreML models on macOS (~500 MB total).

## 6. Seed prompt embeddings

```bash
uv run python scripts/seed_prompts.py
```

## 7. Install web frontend

```bash
cd web && pnpm install
```
