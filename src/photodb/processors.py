from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .database.pg_repository import PostgresPhotoRepository
from .database.pg_connection import PostgresConnectionPool
from .stages.normalize import NormalizeStage
from .stages.metadata import MetadataStage

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    total_files: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    failed_files: List[tuple] = field(default_factory=list)
    success: bool = True

class PhotoProcessor:
    def __init__(self, repository, config: dict, 
                 force: bool = False, dry_run: bool = False, 
                 parallel: int = 1):
        self.repository = repository
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.parallel = max(1, parallel)
        
        # Semaphore to limit concurrent file operations (prevents "too many open files")
        # Allow at most 50 concurrent file operations regardless of thread count
        self.file_semaphore = threading.Semaphore(min(50, parallel))
        
        # Create connection pool for parallel processing
        if parallel > 1:
            # Limit to 50 connections to stay well under PostgreSQL's default limit of 100
            connection_string = config.get('database_url', 'postgresql://localhost/photodb')
            max_connections = min(parallel, 50)
            self.connection_pool = PostgresConnectionPool(
                connection_string=connection_string,
                min_conn=2,
                max_conn=max_connections
            )
        else:
            self.connection_pool = None
        
        self.stages = {
            'normalize': NormalizeStage(repository, config),
            'metadata': MetadataStage(repository, config)
        }
    
    def _process_single_file(self, file_path: Path, stage: str, stages_dict: dict) -> ProcessingResult:
        """Process a single file with provided stages dictionary."""
        result = ProcessingResult(total_files=1)
        
        # Use semaphore to limit concurrent file operations
        logger.debug(f"Waiting for file semaphore for {file_path}")
        with self.file_semaphore:
            logger.debug(f"Acquired file semaphore for {file_path}")
            try:
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would process: {file_path}")
                    result.skipped = 1
                    return result
                
                stages_to_run = self._get_stages(stage)
                
                # Check if any stages need processing
                needs_processing = False
                for stage_name in stages_to_run:
                    stage_obj = stages_dict[stage_name]
                    if stage_obj.should_process(file_path, self.force):
                        needs_processing = True
                        break
                
                if not needs_processing:
                    logger.debug(f"Skipping {file_path} (all stages already processed)")
                    result.skipped = 1
                    return result
                
                # Process stages that need processing
                stages_processed = 0
                all_stages_succeeded = True
                
                for stage_name in stages_to_run:
                    stage_obj = stages_dict[stage_name]
                    
                    if stage_obj.should_process(file_path, self.force):
                        logger.debug(f"Running {stage_name} on {file_path}")
                        try:
                            stage_obj.process(file_path)
                            
                            # Check if the stage actually succeeded by looking at processing status
                            photo = stage_obj.repository.get_photo_by_filename(str(file_path))
                            if photo:
                                stage_status = stage_obj.repository.get_processing_status(photo.id, stage_obj.stage_name)
                                if stage_status and stage_status.status == 'failed':
                                    all_stages_succeeded = False
                                    error_msg = stage_status.error_message or "Stage processing failed"
                                    logger.error(f"Stage {stage_name} failed for {file_path}: {error_msg}")
                                elif stage_status and stage_status.status == 'completed':
                                    stages_processed += 1
                            else:
                                all_stages_succeeded = False
                                logger.error(f"Could not find photo record for {file_path} after processing")
                            
                        except Exception as e:
                            all_stages_succeeded = False
                            logger.error(f"Stage {stage_name} threw exception for {file_path}: {e}")
                            raise  # Re-raise to be caught by outer exception handler
                    else:
                        logger.debug(f"Skipping {stage_name} for {file_path} (already processed)")
                
                if not all_stages_succeeded:
                    result.failed = 1
                    result.failed_files.append((str(file_path), "One or more stages failed"))
                    result.success = False
                elif stages_processed > 0:
                    result.processed = 1
                else:
                    result.skipped = 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                result.failed = 1
                result.failed_files.append((str(file_path), str(e)))
                result.success = False
            
            return result
    
    def process_file(self, file_path: Path, stage: str = 'all') -> ProcessingResult:
        """Process a single file through specified stages."""
        return self._process_single_file(file_path, stage, self.stages)
    
    def process_directory(self, directory: Path, stage: str = 'all',
                         recursive: bool = True, 
                         pattern: str = '*') -> ProcessingResult:
        """Process all matching files in a directory."""
        if self.parallel > 1:
            result = self._process_streaming_parallel(directory, recursive, pattern, stage)
        else:
            files = self._find_files(directory, recursive, pattern)
            result = ProcessingResult(total_files=len(files))
            
            if not files:
                logger.warning(f"No matching files found in {directory}")
                return result
            
            logger.info(f"Found {len(files)} files to process")
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
        """Process files in parallel using thread pool with pooled connections."""
        result = ProcessingResult(total_files=len(files))
        
        def process_with_pooled_repo(file_path: Path) -> ProcessingResult:
            """Process file with a repository using the connection pool."""
            if self.connection_pool:
                # Create a temporary repository using the PostgreSQL connection pool
                pooled_repo = PostgresPhotoRepository(self.connection_pool)
                
                # Create stages with the pooled repository
                pooled_stages = {
                    'normalize': NormalizeStage(pooled_repo, self.config),
                    'metadata': MetadataStage(pooled_repo, self.config)
                }
                
                # Process with pooled stages
                return self._process_single_file(file_path, stage, pooled_stages)
            else:
                return self.process_file(file_path, stage)
        
        # With GIL disabled, ThreadPoolExecutor provides true parallelism
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = {
                executor.submit(process_with_pooled_repo, file_path): file_path
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

    def _process_streaming_parallel(self, directory: Path, recursive: bool, 
                                   pattern: str, stage: str) -> ProcessingResult:
        """Process files as they are discovered using streaming parallel approach."""
        result = ProcessingResult()
        supported_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', 
                              '.bmp', '.tiff', '.webp'}
        
        def process_with_pooled_repo(file_path: Path) -> ProcessingResult:
            """Process file with a repository using the connection pool."""
            if self.connection_pool:
                # Create a temporary repository using the PostgreSQL connection pool
                pooled_repo = PostgresPhotoRepository(self.connection_pool)
                
                # Create stages with the pooled repository
                pooled_stages = {
                    'normalize': NormalizeStage(pooled_repo, self.config),
                    'metadata': MetadataStage(pooled_repo, self.config)
                }
                
                # Process with pooled stages
                return self._process_single_file(file_path, stage, pooled_stages)
            else:
                return self.process_file(file_path, stage)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = {}
            
            # Stream files and submit them for processing as they're found
            if recursive:
                file_iter = directory.rglob(pattern)
            else:
                file_iter = directory.glob(pattern)
            
            for file_path in file_iter:
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    result.total_files += 1
                    future = executor.submit(process_with_pooled_repo, file_path)
                    futures[future] = file_path
            
            if result.total_files == 0:
                logger.warning(f"No matching files found in {directory}")
                return result
            
            logger.info(f"Found {result.total_files} files to process (streaming)")
            
            # Process completed futures as they finish
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    file_result = future.result()
                    result.processed += file_result.processed
                    result.skipped += file_result.skipped
                    result.failed += file_result.failed
                    result.failed_files.extend(file_result.failed_files)
                except Exception as e:
                    logger.error(f"Streaming parallel processing error for {file_path}: {e}")
                    result.failed += 1
                    result.failed_files.append((str(file_path), str(e)))
        
        result.success = result.failed == 0
        return result

