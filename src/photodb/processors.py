from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
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
    failed_files: List[tuple] = field(default_factory=list)
    success: bool = True

class PhotoProcessor:
    def __init__(self, repository: PhotoRepository, config: dict, 
                 force: bool = False, dry_run: bool = False, 
                 parallel: int = 1):
        self.repository = repository
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.parallel = max(1, parallel)
        
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
            
            stages_to_run = self._get_stages(stage)
            
            # Check if any stages need processing
            needs_processing = False
            for stage_name in stages_to_run:
                stage_obj = self.stages[stage_name]
                if stage_obj.should_process(file_path, self.force):
                    needs_processing = True
                    break
            
            if not needs_processing:
                logger.debug(f"Skipping {file_path} (all stages already processed)")
                result.skipped = 1
                return result
            
            # Process stages that need processing
            stages_processed = 0
            for stage_name in stages_to_run:
                stage_obj = self.stages[stage_name]
                
                if stage_obj.should_process(file_path, self.force):
                    logger.debug(f"Running {stage_name} on {file_path}")
                    stage_obj.process(file_path)
                    stages_processed += 1
                else:
                    logger.debug(f"Skipping {stage_name} for {file_path} (already processed)")
            
            if stages_processed > 0:
                result.processed = 1
            else:
                result.skipped = 1
            
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