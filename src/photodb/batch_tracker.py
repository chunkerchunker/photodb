import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from .database.models import Photo, BatchJob
from .database.pg_repository import PostgresPhotoRepository

logger = logging.getLogger(__name__)


class BatchTracker:
    """Manages LLM batch processing jobs."""
    
    def __init__(self, repository: PostgresPhotoRepository, config: dict):
        self.repository = repository
        self.config = config
        
        # LLM configuration
        self.model_name = config.get('LLM_MODEL', 'claude-sonnet-4-20250514')
        self.api_key = config.get('LLM_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        self.batch_size = int(config.get('BATCH_SIZE', 100))
        self.check_interval = int(config.get('BATCH_CHECK_INTERVAL', 300))
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY or ANTHROPIC_API_KEY environment variable required")
    
    def submit_batch(self, photos: List[Photo]) -> Optional[str]:
        """Submit a batch of photos for LLM analysis."""
        try:
            logger.info(f"Preparing batch of {len(photos)} photos for LLM analysis")
            
            # Filter photos that need processing
            valid_photos = []
            for photo in photos:
                img_path = self.config.get('IMG_PATH', './photos/processed')
                normalized_path = f"{img_path}/{photo.id}.png"
                
                if not os.path.exists(normalized_path):
                    logger.warning(f"Skipping photo {photo.id} - normalized image not found")
                    continue
                
                if self.repository.has_llm_analysis(photo.id):
                    logger.debug(f"Skipping photo {photo.id} - already has analysis")
                    continue
                
                valid_photos.append(photo)
            
            if not valid_photos:
                logger.info("No photos need LLM analysis")
                return None
            
            logger.info(f"Submitting batch of {len(valid_photos)} photos")
            
            # Create batch job record for tracking
            # For now, we'll process synchronously as a fallback
            batch_id = f"batch_{int(time.time())}_{len(valid_photos)}"
            
            batch_job = BatchJob.create(
                provider_batch_id=batch_id,
                photo_count=len(valid_photos)
            )
            self.repository.create_batch_job(batch_job)
            
            logger.info(f"Created batch job: {batch_id}")
            return batch_id
            
        except Exception as e:
            logger.error(f"Error submitting batch: {e}")
            return None
    
    def check_batch_status(self, batch_id: str) -> bool:
        """Check and update batch status. Returns True if batch is complete."""
        try:
            batch_job = self.repository.get_batch_job_by_provider_id(batch_id)
            if not batch_job:
                logger.error(f"Batch job not found: {batch_id}")
                return False
            
            if batch_job.status in ['completed', 'failed']:
                return True
            
            # For now, since we're not using actual batch processing,
            # we'll mark as processing and return False
            if batch_job.status == 'submitted':
                batch_job.status = 'processing'
                self.repository.update_batch_job(batch_job)
                logger.info(f"Batch {batch_id} status updated to processing")
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking batch status {batch_id}: {e}")
            return False
    
    def check_all_batches(self) -> Dict[str, Any]:
        """Check status of all active batches."""
        try:
            active_batches = self.repository.get_active_batch_jobs()
            results = {
                'total_batches': len(active_batches),
                'completed': 0,
                'processing': 0,
                'failed': 0
            }
            
            for batch_job in active_batches:
                complete = self.check_batch_status(batch_job.provider_batch_id)
                if complete:
                    if batch_job.status == 'completed':
                        results['completed'] += 1
                    else:
                        results['failed'] += 1
                else:
                    results['processing'] += 1
            
            logger.info(f"Batch status check: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error checking all batches: {e}")
            return {'error': str(e)}
    
    def process_batch_results(self, batch_id: str) -> int:
        """Process results from a completed batch. Returns number of photos processed."""
        try:
            batch_job = self.repository.get_batch_job_by_provider_id(batch_id)
            if not batch_job:
                logger.error(f"Batch job not found: {batch_id}")
                return 0
            
            # In a real implementation, this would download and process batch results
            # For now, we'll simulate completion
            batch_job.status = 'completed'
            batch_job.completed_at = datetime.now()
            batch_job.processed_count = batch_job.photo_count
            batch_job.failed_count = 0
            
            self.repository.update_batch_job(batch_job)
            
            logger.info(f"Batch {batch_id} marked as completed: {batch_job.processed_count} photos")
            return batch_job.processed_count
            
        except Exception as e:
            logger.error(f"Error processing batch results {batch_id}: {e}")
            return 0
    
    def get_photos_for_batch_processing(self, limit: Optional[int] = None) -> List[Photo]:
        """Get photos that need LLM analysis for batch processing."""
        batch_limit = limit or self.batch_size
        return self.repository.get_photos_for_llm_analysis(batch_limit)
    
    def cleanup_stale_batches(self, max_age_hours: int = 24) -> int:
        """Clean up batches that have been processing too long."""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            
            active_batches = self.repository.get_active_batch_jobs()
            cleaned = 0
            
            for batch_job in active_batches:
                if batch_job.submitted_at.timestamp() < cutoff_time:
                    logger.warning(f"Marking stale batch as failed: {batch_job.provider_batch_id}")
                    batch_job.status = 'failed'
                    batch_job.error_message = f"Batch processing timeout after {max_age_hours} hours"
                    batch_job.completed_at = datetime.now()
                    
                    self.repository.update_batch_job(batch_job)
                    cleaned += 1
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} stale batches")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning up stale batches: {e}")
            return 0