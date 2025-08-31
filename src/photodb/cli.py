import click
import logging
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

from .database.pg_connection import PostgresConnectionPool
from .database.pg_repository import PostgresPhotoRepository
from .processors import PhotoProcessor
from .utils.logging import setup_logging

load_dotenv()

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--force', is_flag=True, help='Force reprocessing of already processed photos')
@click.option('--stage', type=click.Choice(['all', 'normalize', 'metadata']), 
              default='all', help='Specific stage to run')
@click.option('--recursive/--no-recursive', default=True, 
              help='Process directories recursively')
@click.option('--pattern', default='*', help='File pattern to match (e.g., "*.jpg")')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--dry-run', is_flag=True, help='Show what would be processed without doing it')
@click.option('--parallel', type=int, default=1, 
              help='Number of parallel workers (default: 1)')
@click.option('--config', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--max-photos', type=int, 
              help='Maximum number of photos to process (excluding skipped ones)')
def main(
    path: str,
    force: bool,
    stage: str,
    recursive: bool,
    pattern: str,
    verbose: bool,
    dry_run: bool,
    parallel: int,
    config: Optional[str],
    max_photos: Optional[int]
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
        
        input_path = resolve_path(path, config_data['ingest_path'])
        
        if not input_path.exists():
            logger.error(f"Path does not exist: {input_path}")
            sys.exit(1)
        
        # Create PostgreSQL connection pool
        # Limit connections to avoid exceeding PostgreSQL's max_connections (typically 100)
        max_connections = min(parallel, 50)  # Use at most 50 connections
        connection_pool = PostgresConnectionPool(
            connection_string=config_data.get('database_url'),
            min_conn=2,
            max_conn=max_connections
        )
        logger.info(f"Created connection pool with max {max_connections} connections for {parallel} workers")
        repository = PostgresPhotoRepository(connection_pool)
        
        processor = PhotoProcessor(
            repository=repository,
            config=config_data,
            force=force,
            dry_run=dry_run,
            parallel=parallel,
            max_photos=max_photos
        )
        
        if input_path.is_file():
            logger.info(f"Processing single file: {input_path}")
            result = processor.process_file(input_path, stage)
        else:
            logger.info(f"Processing directory: {input_path}")
            result = processor.process_directory(
                input_path, 
                stage=stage,
                recursive=recursive,
                pattern=pattern
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
        'database_url': os.getenv('DATABASE_URL', 'postgresql://localhost/photodb'),
        'ingest_path': os.getenv('INGEST_PATH', './photos/raw'),
        'img_path': os.getenv('IMG_PATH', './photos/processed'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_file': os.getenv('LOG_FILE', './logs/photodb.log'),
    }
    
    if config_path:
        import json
        with open(config_path, 'r') as f:
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