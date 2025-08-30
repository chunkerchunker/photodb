# Task 01: Project Setup and Environment

## Objective

Set up the initial project structure, configure UV package manager, and establish the development environment for the PhotoDB pipeline.

## Dependencies

- None (this is the first task)

## Deliverables

### 1. Project Structure

Create the following directory structure:

```
photodb/
├── src/
│   └── photodb/
│       ├── __init__.py
│       ├── cli.py
│       ├── database/
│       │   ├── __init__.py
│       │   └── models.py
│       ├── stages/
│       │   ├── __init__.py
│       │   ├── normalize.py
│       │   └── metadata.py
│       └── utils/
│           ├── __init__.py
│           └── image.py
├── tests/
│   ├── __init__.py
│   ├── test_normalize.py
│   └── test_metadata.py
├── pyproject.toml
├── .env.example
└── .gitignore
```

### 2. UV Configuration (pyproject.toml)

```toml
[project]
name = "photodb"
version = "0.1.0"
description = "Personal photo indexing pipeline"
requires-python = ">=3.11"
dependencies = [
    "pillow>=10.0.0",
    "pillow-heif>=0.13.0",
    "click>=8.1.0",
    "python-dotenv>=1.0.0",
    "exif>=1.6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
process-photos = "photodb.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.black]
line-length = 100
target-version = ["py311"]
```

### 3. Environment Configuration (.env.example)

```
# Database path
DB_PATH=./data/photos.db

# Base path for ingested photos (source)
INGEST_PATH=./photos/raw

# Path for normalized photos (processed)
IMG_PATH=./photos/processed

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/photodb.log
```

### 4. Git Configuration (.gitignore)

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# UV
.venv/

# Project specific
data/
photos/
logs/
.env
*.db
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store
```

### 5. Initial Package Structure

#### src/photodb/__init__.py

```python
__version__ = "0.1.0"
```

#### src/photodb/cli.py

```python
import click
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--force', is_flag=True, help='Force reprocessing of photos')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def main(path: str, force: bool, verbose: bool):
    """Process photos from PATH (file or directory)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Processing photos from: {path}")
    
    # TODO: Implement processing logic
    click.echo(f"Processing: {path}")
    if force:
        click.echo("Force mode enabled")

if __name__ == "__main__":
    main()
```

## Implementation Steps

1. __Initialize UV project__

   ```bash
   uv init photodb
   cd photodb
   ```

2. __Create directory structure__
   - Set up all directories as specified above
   - Create empty __init__.py files

3. __Configure pyproject.toml__
   - Add all dependencies
   - Configure development tools
   - Set up entry points

4. __Install dependencies__

   ```bash
   uv sync
   uv pip install -e .
   ```

5. __Set up environment__
   - Copy .env.example to .env
   - Configure local paths

6. __Verify installation__

   ```bash
   uv run process-photos --help
   ```

## Testing Checklist

- [ ] UV project initializes successfully
- [ ] All dependencies install without conflicts
- [ ] CLI entry point is accessible
- [ ] Environment variables load correctly
- [ ] Basic CLI help works
- [ ] Project structure matches specification

## Notes

- Use Python 3.11+ for better performance and type hints
- UV provides faster dependency resolution than pip
- Consider adding pre-commit hooks for code quality
