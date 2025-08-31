import click
import logging
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

from .database.connection import ConnectionPool
from .database.repository import PhotoRepository
from .processors import PhotoProcessor
from .async_batch_monitor import AsyncBatchMonitor
from .stages.enrich import EnrichStage
from .utils.logging import setup_logging

load_dotenv()


@click.command()
@click.argument("path", type=click.Path(exists=True), required=False)
@click.option("--force", is_flag=True, help="Force reprocessing of already processed photos")
@click.option(
    "--stage",
    type=click.Choice(["all", "normalize", "metadata", "enrich"]),
    default="all",
    help="Specific stage to run",
)
@click.option("--recursive/--no-recursive", default=True, help="Process directories recursively")
@click.option("--pattern", default="*", help='File pattern to match (e.g., "*.jpg")')
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without doing it")
@click.option("--parallel", type=int, default=1, help="Number of parallel workers (default: 1)")
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option(
    "--max-photos", type=int, help="Maximum number of photos to process (excluding skipped ones)"
)
@click.option("--check-batches", is_flag=True, help="Check status of running LLM analysis batches")
@click.option(
    "--retry-failed", is_flag=True, help="Retry failed LLM analysis (use with --stage enrich)"
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Number of photos per batch for LLM processing (default: 100)",
)
@click.option("--no-batch", is_flag=True, help="Disable batch processing for enrich stage")
@click.option(
    "--no-async", is_flag=True, help="Disable async batch monitoring (use synchronous processing)"
)
def main(
    path: Optional[str],
    force: bool,
    stage: str,
    recursive: bool,
    pattern: str,
    verbose: bool,
    dry_run: bool,
    parallel: int,
    config: Optional[str],
    max_photos: Optional[int],
    check_batches: bool,
    retry_failed: bool,
    batch_size: int,
    no_batch: bool,
    no_async: bool,
):
    """
    Process photos from PATH (file or directory).

    PATH can be:
    - A single image file
    - A directory containing images
    - A relative path from INGEST_PATH

    Examples:
        process-photos /path/to/photo.jpg
        process-photos /path/to/directory
        process-photos . --recursive --pattern "*.heic"
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logging(log_level)

    try:
        config_data = load_configuration(config)

        # Create PostgreSQL connection pool
        # Limit connections to avoid exceeding PostgreSQL's max_connections (typically 100)
        max_connections = min(parallel, 50)  # Use at most 50 connections
        connection_pool = ConnectionPool(
            connection_string=config_data.get("DATABASE_URL"), min_conn=2, max_conn=max_connections
        )
        logger.info(
            f"Created connection pool with max {max_connections} connections for {parallel} workers"
        )
        repository = PhotoRepository(connection_pool)

        # Handle batch checking mode
        if check_batches:
            # Get active batch jobs from repository
            active_batches = repository.get_active_batch_jobs()

            if not active_batches:
                logger.info("No active batch jobs found")
                sys.exit(0)

            logger.info(f"Found {len(active_batches)} active batch job(s)")

            # Create EnrichStage to use its monitor_batch method
            enrich_stage = EnrichStage(repository, config_data)

            # Check status of each batch
            completed_count = 0
            processing_count = 0
            failed_count = 0

            for batch_job in active_batches:
                batch_id = batch_job.provider_batch_id
                status_info = enrich_stage.monitor_batch(batch_id)

                if status_info:
                    status = status_info.get("status", "unknown")
                    logger.info(f"Batch {batch_id}:")
                    logger.info(f"  Status: {status}")
                    logger.info(f"  Photos: {status_info.get('photo_count', 0)}")
                    logger.info(f"  Processed: {status_info.get('processed_count', 0)}")
                    logger.info(f"  Failed: {status_info.get('failed_count', 0)}")

                    if status == "completed":
                        completed_count += 1
                    elif status in ["submitted", "processing"]:
                        processing_count += 1
                    elif status == "failed":
                        failed_count += 1
                else:
                    logger.warning(f"Could not get status for batch {batch_id}")

            logger.info("\nBatch Status Summary:")
            logger.info(f"  Total batches: {len(active_batches)}")
            logger.info(f"  Processing: {processing_count}")
            logger.info(f"  Completed: {completed_count}")
            logger.info(f"  Failed: {failed_count}")

            # Clean up stale batches using async monitor
            import asyncio

            async def cleanup():
                monitor = AsyncBatchMonitor(repository, config_data)
                return await monitor.cleanup_stale_batches()

            cleaned = asyncio.run(cleanup())
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} stale batches")

            sys.exit(0)

        if not path:
            logger.error("PATH argument is required when not using --check-batches")
            sys.exit(1)

        # Type narrowing: path is guaranteed to be str after the check above
        assert path is not None
        input_path = resolve_path(path, config_data["INGEST_PATH"])

        if not input_path.exists():
            logger.error(f"Path does not exist: {input_path}")
            sys.exit(1)

        # Determine batch mode settings
        use_batch_mode = not no_batch and (stage == "enrich" or (stage == "all" and not dry_run))

        if use_batch_mode:
            logger.info(f"Using batch mode with batch size: {batch_size}")
        elif stage == "enrich":
            logger.info("Batch mode disabled for enrich stage")

        processor = PhotoProcessor(
            repository=repository,
            config=config_data,
            force=force or retry_failed,
            dry_run=dry_run,
            parallel=parallel,
            max_photos=max_photos,
            batch_mode=use_batch_mode,
            batch_size=batch_size,
            async_batch=not no_async,
        )

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
        # LLM Configuration
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "anthropic"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "100")),
        "BATCH_CHECK_INTERVAL": int(os.getenv("BATCH_CHECK_INTERVAL", "300")),
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
