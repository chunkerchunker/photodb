import click
import logging
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Load .env BEFORE importing modules that read environment variables at module level
load_dotenv(os.getenv("ENV_FILE", "./.env"))

from .database.connection import ConnectionPool  # noqa: E402
from .database.repository import PhotoRepository  # noqa: E402
from .processors import LocalProcessor  # noqa: E402
from .utils.logging import setup_logging  # noqa: E402


@click.command()
@click.argument("path", type=click.Path(exists=False), required=False, default=None)
@click.option("--force", is_flag=True, help="Force reprocessing of already processed photos")
@click.option(
    "--stage",
    type=click.Choice(
        [
            "all",
            "normalize",
            "metadata",
            "detection",
            "age_gender",
            "clustering",
            "scene_analysis",
        ]
    ),
    default="all",
    help="Specific stage to run",
)
@click.option(
    "--exclude",
    type=click.Choice(
        ["normalize", "metadata", "detection", "age_gender", "clustering", "scene_analysis"]
    ),
    multiple=True,
    help="Stages to exclude (can be specified multiple times)",
)
@click.option("--recursive/--no-recursive", default=True, help="Process directories recursively")
@click.option("--pattern", default="*", help='File pattern to match (e.g., "*.jpg")')
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", is_flag=True, help="Reduce logging to warnings and above only")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without doing it")
@click.option("--parallel", type=int, default=1, help="Number of parallel workers (default: 1)")
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option(
    "--max-photos", type=int, help="Maximum number of photos to process (excluding skipped ones)"
)
@click.option(
    "--force-directory-scan",
    is_flag=True,
    help="Force scanning directory for files (default for normalize stage)",
)
@click.option(
    "--skip-directory-scan",
    is_flag=True,
    help="Force skipping directory scan and using database instead (default for non-normalize stages)",
)
@click.option(
    "--collection-id",
    type=int,
    help="Collection ID to use (overrides COLLECTION_ID env var)",
)
def main(
    path: Optional[str],
    force: bool,
    stage: str,
    exclude: tuple,
    recursive: bool,
    pattern: str,
    verbose: bool,
    quiet: bool,
    dry_run: bool,
    parallel: int,
    config: Optional[str],
    max_photos: Optional[int],
    force_directory_scan: bool,
    skip_directory_scan: bool,
    collection_id: Optional[int],
):
    """
    Process photos locally from PATH (file or directory).

    This tool handles local processing stages (normalize, metadata extraction, and face detection)
    with support for parallel processing.

    PATH can be:
    - A single image file
    - A directory containing images
    - A relative path from INGEST_PATH
    - Omitted (when using --skip-directory-scan) to process all photos in the collection

    Examples:
        process-local /path/to/photo.jpg
        process-local /path/to/directory
        process-local . --recursive --pattern "*.heic" --parallel 500
        process-local /path/to/directory --stage faces --parallel 100
        process-local --skip-directory-scan --stage clustering
    """
    if quiet:
        log_level = logging.WARNING
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = setup_logging(log_level)

    try:
        config_data = load_configuration(config)

        # Validate that path is provided when scanning directories
        if path is None and not skip_directory_scan:
            logger.error("PATH is required when scanning directories. Use --skip-directory-scan to process from database.")
            sys.exit(1)

        # Create PostgreSQL connection pool
        # Limit connections to avoid exceeding PostgreSQL's max_connections (typically 100)
        max_connections = min(parallel, 50)  # Use at most 50 connections
        min_connections = min(2, max_connections)  # Ensure min_conn <= max_conn
        with ConnectionPool(
            connection_string=config_data.get("DATABASE_URL"),
            min_conn=min_connections,
            max_conn=max_connections,
        ) as connection_pool:
            logger.info(
                f"Created connection pool with max {max_connections} connections for {parallel} workers"
            )
            # CLI option overrides config/env
            effective_collection_id = (
                collection_id
                if collection_id is not None
                else int(config_data.get("COLLECTION_ID", 1))
            )
            # Update config so stages created via constructor get the correct value
            config_data["COLLECTION_ID"] = effective_collection_id
            repository = PhotoRepository(connection_pool, collection_id=effective_collection_id)

            # Resolve path if provided
            input_path = None
            if path is not None:
                input_path = resolve_path(
                    path, config_data["INGEST_PATH"], skip_disk_check=skip_directory_scan
                )
                # When skip_directory_scan is enabled, don't access disk at all
                if not skip_directory_scan and not input_path.exists():
                    logger.error(f"Path does not exist: {input_path}")
                    sys.exit(1)

            # Create local processor for parallel processing
            # Pass stage to constructor so only required ML models are loaded
            with LocalProcessor(
                repository=repository,
                config=config_data,
                collection_id=effective_collection_id,
                force=force,
                dry_run=dry_run,
                parallel=parallel,
                max_photos=max_photos,
                stage=stage,
                exclude=list(exclude),
                force_directory_scan=force_directory_scan,
                skip_directory_scan=skip_directory_scan,
            ) as processor:
                # When skip_directory_scan is enabled, assume directory (no disk access)
                if input_path is not None and not skip_directory_scan and input_path.is_file():
                    logger.info(f"Processing single file: {input_path}")
                    result = processor.process_file(input_path)
                else:
                    if input_path is None:
                        logger.info("Processing all photos in collection")
                    else:
                        logger.info(f"Processing directory: {input_path}")
                    result = processor.process_directory(
                        input_path, recursive=recursive, pattern=pattern
                    )

            report_results(result, logger)

            sys.exit(0 if result.success else 1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def load_configuration(config_path: Optional[str]) -> dict:
    """Load configuration from environment and optional config file."""
    config = {
        "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://localhost/photodb"),
        "INGEST_PATH": os.getenv("INGEST_PATH", "./photos/raw"),
        "IMG_PATH": os.getenv("IMG_PATH", "./photos/processed"),
        "COLLECTION_ID": int(os.getenv("COLLECTION_ID", "1")),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "LOG_FILE": os.getenv("LOG_FILE", "./logs/photodb.log"),
        "RESIZE_SCALE": float(os.getenv("RESIZE_SCALE", "1.0")),
    }

    if config_path:
        import json

        with open(config_path, "r") as f:
            file_config = json.load(f)
            config.update(file_config)

    return config


def resolve_path(path: str, base_path: str, skip_disk_check: bool = False) -> Path:
    """Resolve path relative to base path if needed.

    Args:
        path: The path to resolve
        base_path: Base path to resolve relative paths against
        skip_disk_check: If True, don't access disk to check existence
    """
    p = Path(path)
    if p.is_absolute():
        return p

    if skip_disk_check:
        # When skipping disk checks, prefer resolving against base_path
        base = Path(base_path)
        return base / p

    if p.exists():
        return p.resolve()

    base = Path(base_path)
    resolved = base / p
    if resolved.exists():
        return resolved.resolve()

    return p


def report_results(result, logger):
    """Report processing results."""
    logger.info("=" * 50)
    logger.info("Processing Complete")
    logger.info(f"Total files found: {result.total_files}")
    logger.info(f"Successfully processed: {result.processed}")
    logger.info(f"Skipped (already processed): {result.skipped}")
    logger.info(f"Failed: {result.failed}")

    if result.failed_files:
        logger.warning("Failed files:")
        for file, error in result.failed_files:
            logger.warning(f"  - {file}: {error}")


if __name__ == "__main__":
    main()
