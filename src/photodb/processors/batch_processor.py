from pathlib import Path
from typing import Optional
import logging

from .base_processor import BaseProcessor, ProcessingResult
from ..stages.enrich import EnrichStage
from ..utils.batch import wait_for_batch_completion

logger = logging.getLogger(__name__)


class BatchProcessor(BaseProcessor):
    """Processor for batch LLM enrichment operations."""

    def __init__(
        self,
        repository,
        config: dict,
        force: bool = False,
        dry_run: bool = False,
        max_photos: Optional[int] = None,
        batch_mode: bool = True,
        async_batch: bool = True,
    ):
        super().__init__(repository, config, force, dry_run, max_photos)
        self.batch_mode = batch_mode
        self.batch_size = config.get("BATCH_SIZE", 100)
        self.min_batch_size = config.get("MIN_BATCH_SIZE", 10)
        self.async_batch = async_batch

        # Only create enrich stage
        self.enrich_stage = EnrichStage(repository, config)

    def process_file(self, file_path: Path, stage: str = "enrich") -> ProcessingResult:
        """Process a single file through enrich stage."""
        if stage not in ["enrich", "all"]:
            raise ValueError(f"BatchProcessor only supports 'enrich' stage, got '{stage}'")

        result = ProcessingResult(total_files=1)

        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would enrich: {file_path}")
                result.skipped = 1
                return result

            # Check if enrichment is needed
            if not self.enrich_stage.should_process(file_path, self.force):
                logger.debug(f"Skipping {file_path} (already enriched)")
                result.skipped = 1
                return result

            # Process single file
            logger.debug(f"Running enrich on {file_path}")
            self.enrich_stage.process(file_path)

            # Check if it succeeded
            photo = self.repository.get_photo_by_filename(str(file_path))
            if photo:
                stage_status = self.repository.get_processing_status(
                    photo.id, self.enrich_stage.stage_name
                )
                if stage_status and stage_status.status == "failed":
                    error_msg = stage_status.error_message or "Enrich processing failed"
                    logger.error(f"Enrich failed for {file_path}: {error_msg}")
                    result.failed = 1
                    result.failed_files.append((str(file_path), error_msg))
                    result.success = False
                elif stage_status and stage_status.status == "completed":
                    result.processed = 1
            else:
                logger.error(f"Could not find photo record for {file_path} after processing")
                result.failed = 1
                result.failed_files.append((str(file_path), "Photo record not found"))
                result.success = False

        except Exception as e:
            logger.error(f"Failed to enrich {file_path}: {e}")
            result.failed = 1
            result.failed_files.append((str(file_path), str(e)))
            result.success = False

        return result

    def process_directory(
        self, directory: Path, stage: str = "enrich", recursive: bool = True, pattern: str = "*"
    ) -> ProcessingResult:
        """Process all matching files in a directory with batch processing."""
        if stage not in ["enrich", "all"]:
            raise ValueError(f"BatchProcessor only supports 'enrich' stage, got '{stage}'")

        result = ProcessingResult(total_files=0)

        # Use the generator to iterate through files efficiently
        file_generator = self._find_files_generator(directory, recursive, pattern)

        # If we're in sequential mode (no batching), process files one by one
        if not self.batch_mode or self.dry_run:
            logger.info("Processing files in sequential mode")
            for file_path in file_generator:
                result.total_files += 1
                
                # Check if we've hit the max_photos limit
                if self.max_photos is not None and result.processed >= self.max_photos:
                    logger.info(f"Reached maximum photo limit ({self.max_photos}), stopping processing")
                    break

                logger.info(f"Processing {result.total_files}: {file_path.name}")
                file_result = self.process_file(file_path, "enrich")

                result.processed += file_result.processed
                result.skipped += file_result.skipped
                result.failed += file_result.failed
                result.failed_files.extend(file_result.failed_files)
            
            if result.total_files == 0:
                logger.warning(f"No matching files found in {directory}")
            
            result.success = result.failed == 0
            return result

        # Batch mode processing
        logger.info(f"Processing enrich stage in batches (batch_size={self.batch_size})")

        # Collect photos that need enrichment
        photos_to_enrich = []
        files_checked = 0
        processed_photo_paths = set()

        for file_path in file_generator:
            files_checked += 1

            # Stop collecting if we've hit the limit
            if self.max_photos is not None and len(processed_photo_paths) >= self.max_photos:
                logger.info(f"Reached maximum photo limit ({self.max_photos}), skipping remaining")
                break

            photo = self.repository.get_photo_by_filename(str(file_path))
            if photo and self.enrich_stage.should_process(file_path, self.force):
                photos_to_enrich.append(photo)
                processed_photo_paths.add(str(file_path))
            elif photo:
                logger.debug(f"Skipping {file_path} (already enriched)")

        result.total_files = files_checked

        if not photos_to_enrich:
            logger.info(f"Checked {files_checked} files, no photos need enrichment")
            result.success = True
            return result

        logger.info(f"Found {len(photos_to_enrich)} photos needing enrichment")

        # Process in batches
        batch_ids = []
        total_submitted = 0

        for i in range(0, len(photos_to_enrich), self.batch_size):
            batch = photos_to_enrich[i : i + self.batch_size]

            # Skip batches that are smaller than min_batch_size
            if len(batch) < self.min_batch_size:
                if i == 0:
                    # This is the first batch, so we don't have enough photos total
                    logger.warning(
                        f"Only {len(photos_to_enrich)} photos need enrichment, "
                        f"but minimum batch size is {self.min_batch_size}. "
                        f"Skipping enrich processing - these photos will be processed later "
                        f"when more photos are available."
                    )
                else:
                    # This is a final partial batch
                    logger.warning(
                        f"Skipping final batch of {len(batch)} photos "
                        f"(less than minimum batch size of {self.min_batch_size}). "
                        f"These photos will be processed later."
                    )
                break

            logger.info(f"Submitting batch {i // self.batch_size + 1} with {len(batch)} photos")

            batch_id = self.enrich_stage.process_batch(batch)
            if batch_id:
                batch_ids.append(batch_id)
                total_submitted += len(batch)
                logger.info(f"Batch submitted with ID: {batch_id}")
            else:
                logger.error(f"Failed to submit batch {i // self.batch_size + 1}. Stopping.")
                result.failed += len(batch)
                # Break early, in case we've hit quotas or other issues
                break

        # Monitor batch completion if async
        if self.async_batch and batch_ids:
            logger.info(f"Monitoring {len(batch_ids)} batch(es) for completion...")
            batch_result = wait_for_batch_completion(batch_ids, self.enrich_stage, logger=logger)

            # Update counts based on batch results
            result.processed = total_submitted - batch_result["failed_count"]
            result.failed = batch_result["failed_count"]

            if not batch_result["all_completed"]:
                logger.warning("Some batches did not complete successfully")
                if batch_result["timed_out"]:
                    logger.warning("Batch monitoring timed out")
        else:
            result.processed = total_submitted

        result.success = result.failed == 0

        logger.info(
            f"Total: checked {files_checked} files, submitted {total_submitted} photos for enrichment"
        )

        return result