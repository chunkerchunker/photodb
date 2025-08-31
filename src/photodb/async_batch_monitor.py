import asyncio
import time
from typing import List, Dict, Any
import logging
from datetime import datetime

from .database.repository import PhotoRepository

logger = logging.getLogger(__name__)


class AsyncBatchMonitor:
    """Async monitor for batch processing jobs."""

    def __init__(self, repository: PhotoRepository, config: dict):
        self.repository = repository
        self.check_interval = int(config.get("BATCH_CHECK_INTERVAL", 300))  # 5 minutes
        self.max_retries = 3
        self._monitoring = False
        self._monitoring_tasks = []

    async def submit_and_monitor_batches(
        self, enrich_stage, photo_batches: List[List]
    ) -> Dict[str, Any]:
        """Submit multiple batches and monitor them asynchronously."""
        try:
            logger.info(f"Submitting and monitoring {len(photo_batches)} batches")

            # Submit all batches
            batch_ids = []
            for i, batch in enumerate(photo_batches, 1):
                batch_id = enrich_stage.process_batch(batch)
                if batch_id:
                    batch_ids.append(batch_id)
                    logger.info(f"Submitted batch {i}/{len(photo_batches)}: {batch_id}")
                else:
                    logger.error(f"Failed to submit batch {i}")

            if not batch_ids:
                return {"success": False, "error": "No batches were successfully submitted"}

            # Monitor all batches concurrently
            results = await self._monitor_batches_concurrently(enrich_stage, batch_ids)

            # Aggregate results
            total_processed = sum(r.get("processed_count", 0) for r in results.values())
            total_failed = sum(r.get("failed_count", 0) for r in results.values())
            total_photos = sum(r.get("photo_count", 0) for r in results.values())

            return {
                "success": True,
                "batch_count": len(batch_ids),
                "total_photos": total_photos,
                "processed_count": total_processed,
                "failed_count": total_failed,
                "batch_results": results,
            }

        except Exception as e:
            logger.error(f"Async batch monitoring failed: {e}")
            return {"success": False, "error": str(e)}

    async def _monitor_batches_concurrently(
        self, enrich_stage, batch_ids: List[str]
    ) -> Dict[str, Any]:
        """Monitor multiple batches concurrently."""
        tasks = []
        for batch_id in batch_ids:
            task = asyncio.create_task(self._monitor_single_batch(enrich_stage, batch_id))
            tasks.append(task)

        # Wait for all monitoring tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        batch_results = {}
        for i, result in enumerate(results):
            batch_id = batch_ids[i]
            if isinstance(result, Exception):
                logger.error(f"Error monitoring batch {batch_id}: {result}")
                batch_results[batch_id] = {"error": str(result)}
            else:
                batch_results[batch_id] = result

        return batch_results

    async def _monitor_single_batch(self, enrich_stage, batch_id: str) -> Dict[str, Any]:
        """Monitor a single batch until completion."""
        retries = 0
        start_time = time.time()

        while retries < self.max_retries:
            try:
                # Check batch status
                batch_status = enrich_stage.monitor_batch(batch_id)

                if not batch_status:
                    logger.warning(f"Batch {batch_id} not found, attempt {retries + 1}")
                    retries += 1
                    await asyncio.sleep(self.check_interval)
                    continue

                status = batch_status["status"]

                if status in ["completed", "failed"]:
                    duration = time.time() - start_time
                    logger.info(f"Batch {batch_id} {status} after {duration:.1f}s")
                    return {**batch_status, "monitoring_duration": duration}
                elif status in ["submitted", "processing"]:
                    # Still processing, wait and check again
                    logger.debug(f"Batch {batch_id} still {status}, waiting...")
                    await asyncio.sleep(self.check_interval)
                else:
                    logger.warning(f"Unknown batch status: {status}")
                    retries += 1
                    await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error checking batch {batch_id}: {e}")
                retries += 1
                if retries < self.max_retries:
                    await asyncio.sleep(self.check_interval)

        # Max retries exceeded
        return {
            "batch_id": batch_id,
            "status": "timeout",
            "error": "Monitoring timeout after max retries",
        }

    async def start_background_monitoring(self, enrich_stage):
        """Start background monitoring of active batches."""
        if self._monitoring:
            logger.warning("Background monitoring already running")
            return

        self._monitoring = True
        logger.info("Starting background batch monitoring")

        while self._monitoring:
            try:
                # Get all active batches
                active_batches = self.repository.get_active_batch_jobs()

                if active_batches:
                    logger.info(f"Monitoring {len(active_batches)} active batches")

                    # Create monitoring tasks for active batches
                    tasks = []
                    for batch_job in active_batches:
                        task = asyncio.create_task(
                            self._monitor_single_batch(enrich_stage, batch_job.provider_batch_id)
                        )
                        tasks.append(task)

                    # Wait for all monitoring tasks with timeout
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=self.check_interval * 2,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Background monitoring iteration timed out")

                # Wait before next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(self.check_interval)

    def stop_background_monitoring(self):
        """Stop background monitoring."""
        logger.info("Stopping background batch monitoring")
        self._monitoring = False

        # Cancel any running monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()

        self._monitoring_tasks.clear()

    async def cleanup_stale_batches(self, max_age_hours: int = 24) -> int:
        """Clean up stale batches asynchronously."""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

            active_batches = self.repository.get_active_batch_jobs()
            cleaned = 0

            for batch_job in active_batches:
                if batch_job.submitted_at.timestamp() < cutoff_time:
                    logger.warning(f"Marking stale batch as failed: {batch_job.provider_batch_id}")

                    batch_job.status = "failed"
                    batch_job.error_message = (
                        f"Batch processing timeout after {max_age_hours} hours"
                    )
                    batch_job.completed_at = datetime.now()

                    self.repository.update_batch_job(batch_job)
                    cleaned += 1

                    # Small delay to avoid overwhelming the database
                    await asyncio.sleep(0.1)

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} stale batches")

            return cleaned

        except Exception as e:
            logger.error(f"Error cleaning up stale batches: {e}")
            return 0


# Convenience function to run async batch processing
def run_async_batch_processing(repository, config, enrich_stage, photo_batches):
    """Run async batch processing in a new event loop."""

    async def _run():
        monitor = AsyncBatchMonitor(repository, config)
        return await monitor.submit_and_monitor_batches(enrich_stage, photo_batches)

    # Create and run event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
