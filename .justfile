@help:
  just -l

@cloc:
  cloc . --exclude-dir=node_modules,.venv,__pycache__,.worktrees,build --include-ext=py,ts

@daily *args:
  uv run photodb-maintenance daily {{args}}

@weekly *args:
  uv run photodb-maintenance weekly {{args}}

@local +args:
  uv run process-local {{args}}

@cluster collection_id *args:
  uv run python scripts/bootstrap_clusters.py --collection-id {{collection_id}} {{args}}

@group-clusters collection_id *args:
  uv run photodb-maintenance auto-associate --collection-id {{collection_id}} {{args}}

@reset-clusters collection_id *args:
  uv run python scripts/reset_clustering.py --collection-id {{collection_id}} {{args}}

@pghero:
  docker run -e DATABASE_URL=postgres://andrewchoi:PXpYgUwiAGpOgSAd@host.docker.internal:5432/photodb -p 8080:8080 --rm ankane/pghero

@capture-import *args:
  uv run python scripts/import_capture_order.py {{args}}
