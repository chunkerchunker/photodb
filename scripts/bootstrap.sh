#!/bin/bash
# Bootstrap a fresh PhotoDB development environment on macOS (Apple Silicon).
# Requires: Homebrew (https://brew.sh)
#
# Usage: ./scripts/bootstrap.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
DIM='\033[2m'
RESET='\033[0m'

step() { echo -e "\n${GREEN}==> $1${RESET}"; }
info() { echo -e "${DIM}    $1${RESET}"; }
fail() { echo -e "${RED}ERROR: $1${RESET}" >&2; exit 1; }

# ---------- pre-flight checks ----------

[[ "$OSTYPE" == darwin* ]] || fail "This project requires macOS (uses Vision, CoreML, Metal APIs)"
command -v brew &>/dev/null || fail "Homebrew not found. Install from https://brew.sh"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------- Homebrew dependencies ----------

step "Installing Homebrew dependencies"
brew bundle --file=Brewfile

# ---------- Python environment ----------

step "Installing Python dependencies"
uv sync
info "Python $(uv run python --version | awk '{print $2}')"

# ---------- .env ----------

if [ ! -f .env ]; then
    step "Creating .env from .env.example"
    cp .env.example .env
    info "Edit .env to configure DATABASE_URL and paths"
else
    info ".env already exists, skipping"
fi

# ---------- PostgreSQL ----------

step "Setting up database"

# Source .env to get DATABASE_URL
set -a; source .env; set +a

DB_NAME=$(echo "$DATABASE_URL" | sed 's|.*/||' | sed 's|\?.*||')
if psql "$DATABASE_URL" -c '\q' 2>/dev/null; then
    info "Database '$DB_NAME' already exists"
else
    info "Creating database '$DB_NAME'"
    createdb "$DB_NAME" 2>/dev/null || fail "Could not create database '$DB_NAME'. Is PostgreSQL running?"
fi

info "Applying schema"
psql "$DATABASE_URL" -f schema.sql -q

info "Running migrations"
for f in migrations/[0-9]*.sql; do
    [[ "$f" == *rollback* ]] && continue
    psql "$DATABASE_URL" -f "$f" -q 2>/dev/null || true
done

# ---------- ML models ----------

step "Downloading ML models"
info "This may take a few minutes on first run (~500 MB total)"
bash scripts/download_models.sh

# ---------- Seed data ----------

step "Seeding prompt embeddings"
uv run python scripts/seed_prompts.py

# ---------- Web frontend ----------

step "Installing web frontend dependencies"
cd web
pnpm install
cd "$REPO_ROOT"

# ---------- done ----------

step "Bootstrap complete"
echo ""
echo "  Python pipeline:  uv run process-local /path/to/photos"
echo "  Web frontend:     cd web && pnpm dev"
echo "  Run tests:        uv run pytest"
echo ""
