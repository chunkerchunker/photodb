# Task 06: Stage 1 - Photo Normalization

## Objective
Implement Stage 1 of the processing pipeline: normalizing photos by resizing them according to aspect ratio constraints, converting to PNG format, and storing them with UUID-based naming.

## Dependencies
- Task 02: Database Setup (for storing photo records)
- Task 04: Image Format Handling (for image operations)
- Task 05: File Discovery (for processing integration)

## Deliverables

### 1. Normalization Stage Implementation (src/photodb/stages/normalize.py)
```python
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import uuid
import logging
import os
from datetime import datetime

from ..database.repository import PhotoRepository
from ..database.models import Photo, ProcessingStatus
from ..utils.image import ImageHandler
from ..utils.validation import ImageValidator

logger = logging.getLogger(__name__)

class NormalizeStage:
    """Stage 1: Normalize photos to standard size and format."""
    
    STAGE_NAME = 'normalize'
    
    # Use ImageHandler's MAX_DIMENSIONS from utils.image
    
    def __init__(self, repository: PhotoRepository, config: dict):
        self.repository = repository
        self.config = config
        self.output_path = Path(config.get('img_path', './photos/processed'))
        self.ingest_path = Path(config.get('ingest_path', './photos/raw'))
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """
        Check if file needs processing.
        
        Args:
            file_path: Path to check
            force: Force reprocessing even if already done
            
        Returns:
            True if processing needed
        """
        if force:
            return True
        
        # Get relative path for database lookup
        rel_path = self._get_relative_path(file_path)
        
        # Check if photo exists in database
        photo = self.repository.get_photo_by_filename(str(rel_path))
        if not photo:
            return True
        
        # Check processing status
        status = self.repository.get_processing_status(photo.id, self.STAGE_NAME)
        if not status or status.status != 'completed':
            return True
        
        # Check if normalized file still exists
        normalized_path = Path(photo.normalized_path)
        if not normalized_path.exists():
            logger.warning(f"Normalized file missing for {file_path}, will reprocess")
            return True
        
        return False
    
    def process(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single photo through normalization.
        
        Args:
            file_path: Path to photo to process
            
        Returns:
            Dict with processing results
        """
        logger.info(f"Normalizing photo: {file_path}")
        
        # Get or create photo record
        rel_path = self._get_relative_path(file_path)
        photo = self.repository.get_photo_by_filename(str(rel_path))
        
        if not photo:
            # Create new photo record with UUID
            photo_id = str(uuid.uuid4())
            normalized_filename = f"{photo_id}.png"
            normalized_path = self.output_path / normalized_filename
            
            photo = Photo.create(
                filename=str(rel_path),
                normalized_path=str(normalized_path)
            )
            photo.id = photo_id
        else:
            photo_id = photo.id
            normalized_path = Path(photo.normalized_path)
        
        # Update processing status to 'processing'
        self.repository.update_processing_status(
            ProcessingStatus(
                photo_id=photo_id,
                stage=self.STAGE_NAME,
                status='processing',
                processed_at=datetime.now(),
                error_message=None
            )
        )
        
        try:
            # Validate file
            if not ImageValidator.validate_file(file_path):
                raise ValueError(f"Invalid image file: {file_path}")
            
            # Open image
            image = ImageHandler.open_image(file_path)
            original_size = (image.width, image.height)
            
            logger.debug(f"Original size: {original_size[0]}x{original_size[1]}")
            
            # Calculate resize dimensions using ImageHandler
            new_size = ImageHandler.calculate_resize_dimensions(original_size, {})
            
            # Resize if needed
            if new_size and new_size != original_size:
                logger.debug(f"Resizing to: {new_size[0]}x{new_size[1]}")
                image = ImageHandler.resize_image(image, new_size)
                was_resized = True
            else:
                logger.debug("No resize needed")
                was_resized = False
            
            # Save as PNG
            ImageHandler.save_as_png(image, normalized_path, optimize=True)
            
            # Create or update photo record
            if not self.repository.get_photo_by_id(photo_id):
                self.repository.create_photo(photo)
            else:
                photo.updated_at = datetime.now()
                self.repository.update_photo(photo)
            
            # Update processing status to 'completed'
            self.repository.update_processing_status(
                ProcessingStatus(
                    photo_id=photo_id,
                    stage=self.STAGE_NAME,
                    status='completed',
                    processed_at=datetime.now(),
                    error_message=None
                )
            )
            
            result = {
                'success': True,
                'photo_id': photo_id,
                'original_size': original_size,
                'new_size': new_size if was_resized else original_size,
                'was_resized': was_resized,
                'output_path': str(normalized_path)
            }
            
            logger.info(f"Successfully normalized {file_path} -> {normalized_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to normalize {file_path}: {e}")
            
            # Update processing status to 'failed'
            self.repository.update_processing_status(
                ProcessingStatus(
                    photo_id=photo_id,
                    stage=self.STAGE_NAME,
                    status='failed',
                    processed_at=datetime.now(),
                    error_message=str(e)
                )
            )
            
            return {
                'success': False,
                'photo_id': photo_id,
                'error': str(e)
            }
    
    def _get_relative_path(self, file_path: Path) -> Path:
        """Get path relative to ingest path."""
        try:
            return file_path.relative_to(self.ingest_path)
        except ValueError:
            # File is outside ingest path, use absolute path
            return file_path
    
    # Resize calculation methods removed - use ImageHandler.calculate_resize_dimensions() instead
```

### 2. Batch Normalization Processor (src/photodb/stages/batch_normalize.py)
```python
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .normalize import NormalizeStage

logger = logging.getLogger(__name__)

class BatchNormalizer:
    """Process multiple photos through normalization stage."""
    
    def __init__(self, normalize_stage: NormalizeStage):
        self.stage = normalize_stage
    
    def process_batch(
        self,
        file_paths: List[Path],
        parallel: int = 1,
        stop_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple files through normalization.
        
        Args:
            file_paths: List of paths to process
            parallel: Number of parallel workers
            stop_on_error: Stop processing on first error
            
        Returns:
            Dict with batch processing results
        """
        results = {
            'total': len(file_paths),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'results': []
        }
        
        if parallel > 1:
            results = self._process_parallel(
                file_paths, parallel, stop_on_error, results
            )
        else:
            results = self._process_sequential(
                file_paths, stop_on_error, results
            )
        
        return results
    
    def _process_sequential(
        self,
        file_paths: List[Path],
        stop_on_error: bool,
        results: Dict
    ) -> Dict:
        """Process files sequentially."""
        for file_path in file_paths:
            try:
                if not self.stage.should_process(file_path):
                    logger.debug(f"Skipping already processed: {file_path}")
                    results['skipped'] += 1
                    continue
                
                result = self.stage.process(file_path)
                results['results'].append(result)
                
                if result['success']:
                    results['processed'] += 1
                else:
                    results['failed'] += 1
                    if stop_on_error:
                        break
                        
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {e}")
                results['failed'] += 1
                if stop_on_error:
                    break
        
        return results
    
    def _process_parallel(
        self,
        file_paths: List[Path],
        parallel: int,
        stop_on_error: bool,
        results: Dict
    ) -> Dict:
        """Process files in parallel."""
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            # Submit all tasks
            future_to_path = {}
            for file_path in file_paths:
                if not self.stage.should_process(file_path):
                    results['skipped'] += 1
                    continue
                
                future = executor.submit(self.stage.process, file_path)
                future_to_path[future] = file_path
            
            # Collect results
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result = future.result()
                    results['results'].append(result)
                    
                    if result['success']:
                        results['processed'] += 1
                    else:
                        results['failed'] += 1
                        if stop_on_error:
                            # Cancel remaining tasks
                            for f in future_to_path:
                                f.cancel()
                            break
                            
                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path}: {e}")
                    results['failed'] += 1
                    if stop_on_error:
                        for f in future_to_path:
                            f.cancel()
                        break
        
        return results
```

### 3. Stage Integration (__init__.py)
```python
from .normalize import NormalizeStage
from .batch_normalize import BatchNormalizer

__all__ = ['NormalizeStage', 'BatchNormalizer']
```

## Implementation Steps

1. __Implement NormalizeStage class__
   - UUID generation for photos
   - Database record management using existing ImageHandler utilities
   - Processing status tracking

2. __Create batch processor__
   - Sequential processing
   - Parallel processing with thread pool
   - Error handling and recovery
   - Progress tracking

3. __Add status tracking__
   - Processing status updates
   - Error recording
   - Skip detection for processed files

4. __Integration with existing utilities__
   - Use ImageValidator for file validation
   - Use ImageHandler for image operations
   - Leverage existing resize and conversion logic

5. __Write tests__
   - Test database record creation
   - Test processing status tracking
   - Test batch processing workflows
   - Test error handling

## Testing Checklist

- [ ] Photos are assigned unique UUIDs
- [ ] Database records are created/updated correctly
- [ ] Processing status is tracked accurately
- [ ] Already processed files are skipped
- [ ] Force flag triggers reprocessing
- [ ] Parallel processing works correctly
- [ ] Errors are handled gracefully
- [ ] Integration with ImageHandler utilities works
- [ ] Batch processing handles all file types

## Notes

- Leverages existing ImageHandler utilities from Task 04
- Focus on workflow and database management rather than image operations
- ImageHandler already provides aspect ratio detection, resizing, and PNG conversion
- Consider adding progress reporting for batch operations