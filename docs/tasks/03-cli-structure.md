# Task 03: CLI Structure and Argument Parsing

## Objective
Implement the complete command-line interface for the `process_photos` command, including argument parsing, configuration loading, and orchestration of the processing pipeline.

## Dependencies
- Task 01: Project Setup (entry point configuration)
- Task 02: Database Setup (for initializing connections)

## Deliverables

### 1. Main CLI Module (src/photodb/cli.py)
Complete implementation of the CLI with all required features:

```python
import click
import logging
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

from .database.connection import DatabaseConnection
from .database.repository import PhotoRepository
from .processors import PhotoProcessor
from .utils.logging import setup_logging

# Load environment variables
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
def main(
    path: str,
    force: bool,
    stage: str,
    recursive: bool,
    pattern: str,
    verbose: bool,
    dry_run: bool,
    parallel: int,
    config: Optional[str]
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
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logging(log_level)
    
    try:
        # Load configuration
        config_data = load_configuration(config)
        
        # Resolve input path
        input_path = resolve_path(path, config_data['ingest_path'])
        
        if not input_path.exists():
            logger.error(f"Path does not exist: {input_path}")
            sys.exit(1)
        
        # Initialize database
        db_connection = DatabaseConnection(config_data['db_path'])
        repository = PhotoRepository(db_connection)
        
        # Create processor
        processor = PhotoProcessor(
            repository=repository,
            config=config_data,
            force=force,
            dry_run=dry_run,
            parallel=parallel
        )
        
        # Process based on input type
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
        
        # Report results
        report_results(result, logger)
        
        # Exit with appropriate code
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
        'db_path': os.getenv('DB_PATH', './data/photos.db'),
        'ingest_path': os.getenv('INGEST_PATH', './photos/raw'),
        'img_path': os.getenv('IMG_PATH', './photos/processed'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_file': os.getenv('LOG_FILE', './logs/photodb.log'),
    }
    
    if config_path:
        # Load additional config from file (JSON/YAML)
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
    
    # Try relative to current directory first
    if p.exists():
        return p.resolve()
    
    # Try relative to ingest path
    base = Path(base_path)
    resolved = base / p
    if resolved.exists():
        return resolved.resolve()
    
    # Return original path (will fail existence check)
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
```

### 2. Logging Configuration (src/photodb/utils/logging.py)
```python
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the application."""
    # Create logger
    logger = logging.getLogger('photodb')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if configured)
    log_file = os.getenv('LOG_FILE')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10_485_760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(f'photodb.{name}')
```

### 3. Processing Orchestrator (src/photodb/processors.py)
```python
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .database.repository import PhotoRepository
from .stages.normalize import NormalizeStage
from .stages.metadata import MetadataStage

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    total_files: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    failed_files: List[tuple] = None
    success: bool = True
    
    def __post_init__(self):
        if self.failed_files is None:
            self.failed_files = []

class PhotoProcessor:
    def __init__(self, repository: PhotoRepository, config: dict, 
                 force: bool = False, dry_run: bool = False, 
                 parallel: int = 1):
        self.repository = repository
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.parallel = max(1, parallel)
        
        # Initialize stages
        self.stages = {
            'normalize': NormalizeStage(repository, config),
            'metadata': MetadataStage(repository, config)
        }
    
    def process_file(self, file_path: Path, stage: str = 'all') -> ProcessingResult:
        """Process a single file through specified stages."""
        result = ProcessingResult(total_files=1)
        
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would process: {file_path}")
                result.skipped = 1
                return result
            
            # Determine which stages to run
            stages_to_run = self._get_stages(stage)
            
            # Process through each stage
            for stage_name in stages_to_run:
                stage_obj = self.stages[stage_name]
                
                if stage_obj.should_process(file_path, self.force):
                    logger.debug(f"Running {stage_name} on {file_path}")
                    stage_obj.process(file_path)
                else:
                    logger.debug(f"Skipping {stage_name} for {file_path} (already processed)")
            
            result.processed = 1
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            result.failed = 1
            result.failed_files.append((str(file_path), str(e)))
            result.success = False
        
        return result
    
    def process_directory(self, directory: Path, stage: str = 'all',
                         recursive: bool = True, 
                         pattern: str = '*') -> ProcessingResult:
        """Process all matching files in a directory."""
        # Find all matching files
        files = self._find_files(directory, recursive, pattern)
        
        result = ProcessingResult(total_files=len(files))
        
        if not files:
            logger.warning(f"No matching files found in {directory}")
            return result
        
        logger.info(f"Found {len(files)} files to process")
        
        if self.parallel > 1:
            result = self._process_parallel(files, stage)
        else:
            result = self._process_sequential(files, stage)
        
        return result
    
    def _find_files(self, directory: Path, recursive: bool, 
                    pattern: str) -> List[Path]:
        """Find all matching image files in directory."""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', 
                              '.bmp', '.tiff', '.webp'}
        
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)
        
        # Filter to supported image files
        image_files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        return sorted(image_files)
    
    def _get_stages(self, stage: str) -> List[str]:
        """Get list of stages to run."""
        if stage == 'all':
            return ['normalize', 'metadata']
        return [stage]
    
    def _process_sequential(self, files: List[Path], stage: str) -> ProcessingResult:
        """Process files sequentially."""
        result = ProcessingResult(total_files=len(files))
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing {i}/{len(files)}: {file_path.name}")
            file_result = self.process_file(file_path, stage)
            
            result.processed += file_result.processed
            result.skipped += file_result.skipped
            result.failed += file_result.failed
            result.failed_files.extend(file_result.failed_files)
        
        result.success = result.failed == 0
        return result
    
    def _process_parallel(self, files: List[Path], stage: str) -> ProcessingResult:
        """Process files in parallel using thread pool."""
        result = ProcessingResult(total_files=len(files))
        
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = {
                executor.submit(self.process_file, file_path, stage): file_path
                for file_path in files
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    file_result = future.result()
                    result.processed += file_result.processed
                    result.skipped += file_result.skipped
                    result.failed += file_result.failed
                    result.failed_files.extend(file_result.failed_files)
                except Exception as e:
                    logger.error(f"Parallel processing error for {file_path}: {e}")
                    result.failed += 1
                    result.failed_files.append((str(file_path), str(e)))
        
        result.success = result.failed == 0
        return result
```

## Implementation Steps

1. **Implement CLI module**
   - Set up Click decorators
   - Add all command options
   - Implement help text

2. **Create logging utilities**
   - Configure console and file handlers
   - Set up log rotation
   - Add module-specific loggers

3. **Build processing orchestrator**
   - Implement file discovery
   - Create sequential processing
   - Add parallel processing support

4. **Add configuration management**
   - Load from environment
   - Support config files
   - Validate settings

5. **Implement error handling**
   - Graceful interruption
   - Detailed error reporting
   - Exit codes

## Testing Checklist

- [ ] CLI help text displays correctly
- [ ] Single file processing works
- [ ] Directory processing finds all files
- [ ] Recursive flag works properly
- [ ] Pattern matching filters correctly
- [ ] Force flag triggers reprocessing
- [ ] Dry run doesn't modify anything
- [ ] Parallel processing works
- [ ] Error handling and reporting
- [ ] Logging to console and file

## Notes

- Consider adding progress bars for better UX
- Add support for glob patterns in file paths
- Implement signal handlers for graceful shutdown
- Consider adding a --stats flag for detailed statistics