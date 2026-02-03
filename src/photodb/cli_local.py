import click
import logging
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

from .database.connection import ConnectionPool
from .database.repository import PhotoRepository
from .processors import LocalProcessor
from .utils.logging import setup_logging

load_dotenv(os.getenv("ENV_FILE", "./.env"))


@click.command()
@click.argument("path", type=click.Path(exists=True))
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
            "faces",
        ]
    ),
    default="all",
    help="Specific stage to run (faces is a legacy alias for detection)",
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
def main(
    path: str,
    force: bool,
    stage: str,
    recursive: bool,
    pattern: str,
    verbose: bool,
    quiet: bool,
    dry_run: bool,
    parallel: int,
    config: Optional[str],
    max_photos: Optional[int],
):
    """
    Process photos locally from PATH (file or directory).

    This tool handles local processing stages (normalize, metadata extraction, and face detection)
    with support for parallel processing.

    PATH can be:
    - A single image file
    - A directory containing images
    - A relative path from INGEST_PATH

    Examples:
        process-local /path/to/photo.jpg
        process-local /path/to/directory
        process-local . --recursive --pattern "*.heic" --parallel 500
        process-local /path/to/directory --stage faces --parallel 100
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
            repository = PhotoRepository(connection_pool)

            input_path = resolve_path(path, config_data["INGEST_PATH"])

            if not input_path.exists():
                logger.error(f"Path does not exist: {input_path}")
                sys.exit(1)

            # Create local processor for parallel processing
            with LocalProcessor(
                repository=repository,
                config=config_data,
                force=force,
                dry_run=dry_run,
                parallel=parallel,
                max_photos=max_photos,
            ) as processor:
                if input_path.is_file():
                    logger.info(f"Processing single file: {input_path}")
                    result = processor.process_file(input_path, stage)
                else:
                    logger.info(f"Processing directory: {input_path}")
                    result = processor.process_directory(
                        input_path, stage=stage, recursive=recursive, pattern=pattern
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


def resolve_path(path: str, base_path: str) -> Path:
    """Resolve path relative to base path if needed."""
    p = Path(path)
    if p.is_absolute():
        return p

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
