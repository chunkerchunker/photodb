import click
import logging
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

from .database.connection import ConnectionPool
from .database.repository import PhotoRepository
from .processors import BatchProcessor
from .async_batch_monitor import AsyncBatchMonitor
from .stages.enrich import EnrichStage
from .utils.logging import setup_logging
from .utils.batch import wait_for_batch_completion

load_dotenv(os.getenv("ENV_FILE", "./.env"))


@click.command()
@click.argument("path", type=click.Path(exists=True), required=False)
@click.option("--force", is_flag=True, help="Force reprocessing of already enriched photos")
@click.option("--recursive/--no-recursive", default=True, help="Process directories recursively")
@click.option("--pattern", default="*", help='File pattern to match (e.g., "*.jpg")')
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", is_flag=True, help="Reduce logging to warnings and above only")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without doing it")
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option(
    "--max-photos", type=int, help="Maximum number of photos to process (excluding skipped ones)"
)
@click.option("--check-batches", is_flag=True, help="Check status of running LLM analysis batches")
@click.option(
    "--retry-failed", is_flag=True, help="Retry failed LLM analysis"
)
@click.option("--no-batch", is_flag=True, help="Disable batch processing (process one at a time)")
@click.option(
    "--no-async", is_flag=True, help="Disable async batch monitoring (use synchronous processing)"
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for batch completion when checking batches (use with --check-batches)",
)
def main(
    path: Optional[str],
    force: bool,
    recursive: bool,
    pattern: str,
    verbose: bool,
    quiet: bool,
    dry_run: bool,
    config: Optional[str],
    max_photos: Optional[int],
    check_batches: bool,
    retry_failed: bool,
    no_batch: bool,
    no_async: bool,
    wait: bool,
):
    """
    Enrich photos with LLM analysis from PATH (file or directory).
    
    This tool handles the enrichment stage with LLM batch processing support.
    
    When called without PATH and with --check-batches, it monitors batch status.

    PATH can be:
    - A single image file
    - A directory containing images
    - A relative path from INGEST_PATH

    Examples:
        enrich-photos /path/to/photo.jpg
        enrich-photos /path/to/directory
        enrich-photos . --recursive --pattern "*.heic"
        enrich-photos --check-batches
        enrich-photos --check-batches --wait
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
        # For enrich, we don't need as many connections since we're not doing parallel processing
        with ConnectionPool(
            connection_string=config_data.get("DATABASE_URL"), min_conn=2, max_conn=10
        ) as connection_pool:
            logger.info("Created connection pool for enrich processing")
            repository = PhotoRepository(connection_pool)

            # Handle batch checking mode
            if check_batches:
                handle_batch_checking(repository, config_data, wait, logger)
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
            use_batch_mode = not no_batch and not dry_run

            if use_batch_mode:
                logger.info(f"Using batch mode with batch size: {config_data['BATCH_SIZE']}")
            else:
                logger.info("Batch mode disabled for enrich stage")

            # Create batch processor for enrich stage
            processor = BatchProcessor(
                repository=repository,
                config=config_data,
                force=force or retry_failed,
                dry_run=dry_run,
                max_photos=max_photos,
                batch_mode=use_batch_mode,
                async_batch=not no_async,
            )

            if input_path.is_file():
                logger.info(f"Processing single file: {input_path}")
                result = processor.process_file(input_path, "enrich")
            else:
                logger.info(f"Processing directory: {input_path}")
                result = processor.process_directory(
                    input_path, stage="enrich", recursive=recursive, pattern=pattern
                )

            report_results(result, logger)

            sys.exit(0 if result.success else 1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def handle_batch_checking(repository, config_data, wait, logger):
    """Handle batch status checking and monitoring."""
    # Get active batch jobs from repository
    active_batches = repository.get_active_batch_jobs()

    if not active_batches:
        logger.info("No active batch jobs found")
        return

    logger.info(f"Found {len(active_batches)} active batch job(s)")

    # Create EnrichStage to use its monitor_batch method
    enrich_stage = EnrichStage(repository, config_data)
    batch_ids = [batch_job.provider_batch_id for batch_job in active_batches]

    if wait:
        # Use shared batch waiting logic
        result = wait_for_batch_completion(batch_ids, enrich_stage, logger=logger)

        if result["all_completed"]:
            logger.info(f"All {len(batch_ids)} batches completed successfully")
            if result["failed_count"] > 0:
                logger.warning(
                    f"Total failed items across all batches: {result['failed_count']}"
                )
        else:
            logger.warning("Some batches did not complete or timed out")
            if result["timed_out"]:
                logger.warning("Operation timed out after waiting")
    else:
        # Original one-time status check logic
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


def load_configuration(config_path: Optional[str]) -> dict:
    """Load configuration from environment and optional config file."""
    config = {
        "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://localhost/photodb"),
        "INGEST_PATH": os.getenv("INGEST_PATH", "./photos/raw"),
        "IMG_PATH": os.getenv("IMG_PATH", "./photos/processed"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "LOG_FILE": os.getenv("LOG_FILE", "./logs/photodb.log"),
        # LLM Configuration
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "anthropic"),  # "anthropic" or "bedrock"
        "LLM_MODEL": os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        # Bedrock-specific configuration
        "BEDROCK_MODEL_ID": os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        ),
        "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
        "AWS_PROFILE": os.getenv("AWS_PROFILE"),  # Optional: use specific AWS profile
        "BEDROCK_BATCH_S3_BUCKET": os.getenv(
            "BEDROCK_BATCH_S3_BUCKET"
        ),  # S3 bucket for batch processing
        "BEDROCK_BATCH_ROLE_ARN": os.getenv(
            "BEDROCK_BATCH_ROLE_ARN"
        ),  # IAM role ARN for batch processing
        "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "100")),
        "MIN_BATCH_SIZE": int(
            os.getenv("MIN_BATCH_SIZE", "10")
        ),  # Minimum batch size for enrich processing
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