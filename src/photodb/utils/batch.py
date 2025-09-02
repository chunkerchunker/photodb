"""Batch processing utilities for PhotoDB."""

import logging
import time
from typing import List, Optional, Dict, Any


def wait_for_batch_completion(
    batch_ids: List[str],
    enrich_stage,
    max_wait_time: int = 3600,
    check_interval: int = 10,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Wait for batch processing to complete.

    Args:
        batch_ids: List of batch IDs to monitor
        enrich_stage: EnrichStage instance with monitor_batch method
        max_wait_time: Maximum time to wait in seconds (default: 1 hour)
        check_interval: How often to check batch status in seconds (default: 10)
        logger: Logger instance for output

    Returns:
        Dict containing:
        - all_completed: bool - whether all batches completed successfully
        - failed_count: int - total number of failed items across all batches
        - timed_out: bool - whether the operation timed out
    """
    if not logger:
        logger = logging.getLogger(__name__)

    if not batch_ids:
        return {"all_completed": True, "failed_count": 0, "timed_out": False}

    logger.info(f"Monitoring {len(batch_ids)} batch(es) for completion")

    all_completed = False
    total_failed_count = 0
    start_time = time.time()

    while not all_completed and (time.time() - start_time) < max_wait_time:
        all_completed = True
        current_failed_count = 0

        for batch_id in batch_ids:
            status = enrich_stage.monitor_batch(batch_id)
            if status:
                if status["status"] not in ["completed", "failed"]:
                    all_completed = False
                else:
                    # Track failed count from this batch
                    current_failed_count += status.get("failed_count", 0)
            else:
                logger.warning(f"Could not get status for batch {batch_id}")
                all_completed = False

        # Update total failed count
        total_failed_count = current_failed_count

        if not all_completed:
            logger.info("Waiting for batches to complete...")
            time.sleep(check_interval)

    timed_out = not all_completed

    if all_completed:
        logger.info("All batches completed successfully")
    else:
        logger.warning("Batch processing timed out or some batches did not complete")

    return {
        "all_completed": all_completed,
        "failed_count": total_failed_count,
        "timed_out": timed_out,
    }
