from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .database.repository import PhotoRepository
from .database.connection import ConnectionPool
from .stages.normalize import NormalizeStage
from .stages.metadata import MetadataStage
from .stages.enrich import EnrichStage

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
    def __init__(
        self,
        repository,
        config: dict,
        force: bool = False,
        dry_run: bool = False,
        parallel: int = 1,
        max_photos: Optional[int] = None,
        batch_mode: bool = False,
        batch_size: int = 100,
        async_batch: bool = True,
    ):
        self.repository = repository
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.parallel = max(1, parallel)
        self.max_photos = max_photos
        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.async_batch = async_batch

        # Semaphore to limit concurrent file operations (prevents "too many open files")
        # Allow at most 50 concurrent file operations regardless of thread count
        self.file_semaphore = threading.Semaphore(min(50, parallel))

        # Create connection pool for parallel processing
        if parallel > 1:
            # Limit to 50 connections to stay well under PostgreSQL's default limit of 100
            connection_string = config.get("DATABASE_URL", "postgresql://localhost/photodb")
            max_connections = min(parallel, 50)
            self.connection_pool = ConnectionPool(
                connection_string=connection_string, min_conn=2, max_conn=max_connections
            )
        else:
            self.connection_pool = None

        self.stages = {
            "normalize": NormalizeStage(repository, config),
            "metadata": MetadataStage(repository, config),
            "enrich": EnrichStage(repository, config),
        }

    def _process_single_file(
        self, file_path: Path, stage: str, stages_dict: dict
    ) -> ProcessingResult:
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
                                stage_status = stage_obj.repository.get_processing_status(
                                    photo.id, stage_obj.stage_name
                                )
                                if stage_status and stage_status.status == "failed":
                                    all_stages_succeeded = False
                                    error_msg = (
                                        stage_status.error_message or "Stage processing failed"
                                    )
                                    logger.error(
                                        f"Stage {stage_name} failed for {file_path}: {error_msg}"
                                    )
                                elif stage_status and stage_status.status == "completed":
                                    stages_processed += 1
                            else:
                                all_stages_succeeded = False
                                logger.error(
                                    f"Could not find photo record for {file_path} after processing"
                                )

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

    def process_file(self, file_path: Path, stage: str = "all") -> ProcessingResult:
        """Process a single file through specified stages."""
        return self._process_single_file(file_path, stage, self.stages)

    def process_directory(
        self, directory: Path, stage: str = "all", recursive: bool = True, pattern: str = "*"
    ) -> ProcessingResult:
        """Process all matching files in a directory."""
        # Check if we should use batch processing for enrich stage
        if self.batch_mode and (stage == "enrich" or stage == "all"):
            return self._process_directory_batch_mode(directory, stage, recursive, pattern)

        # Use existing processing logic
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

    def _find_files(self, directory: Path, recursive: bool, pattern: str) -> List[Path]:
        """Find all matching image files in directory."""
        supported_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tiff", ".webp"}

        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        image_files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]

        return sorted(image_files)

    def _get_stages(self, stage: str) -> List[str]:
        """Get list of stages to run."""
        if stage == "all":
            return ["normalize", "metadata", "enrich"]
        return [stage]

    def _process_sequential(self, files: List[Path], stage: str) -> ProcessingResult:
        """Process files sequentially."""
        result = ProcessingResult(total_files=len(files))

        for i, file_path in enumerate(files, 1):
            # Check if we've hit the max_photos limit
            if self.max_photos is not None and result.processed >= self.max_photos:
                logger.info(f"Reached maximum photo limit ({self.max_photos}), stopping processing")
                break

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
                pooled_repo = PhotoRepository(self.connection_pool)

                # Create stages with the pooled repository
                stages = self._get_stages(stage)
                pooled_stages = {
                    "normalize": NormalizeStage(pooled_repo, self.config)
                    if "normalize" in stages
                    else None,
                    "metadata": MetadataStage(pooled_repo, self.config)
                    if "metadata" in stages
                    else None,
                    "enrich": EnrichStage(pooled_repo, self.config) if "enrich" in stages else None,
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

                    # Check if we've hit the max_photos limit after each completed task
                    if self.max_photos is not None and result.processed >= self.max_photos:
                        logger.info(
                            f"Reached maximum photo limit ({self.max_photos}), stopping processing"
                        )
                        # Cancel remaining futures
                        for remaining_future in futures:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break

                except Exception as e:
                    logger.error(f"Parallel processing error for {file_path}: {e}")
                    result.failed += 1
                    result.failed_files.append((str(file_path), str(e)))

        result.success = result.failed == 0
        return result

    def _process_streaming_parallel(
        self, directory: Path, recursive: bool, pattern: str, stage: str
    ) -> ProcessingResult:
        """Process files as they are discovered using streaming parallel approach."""
        result = ProcessingResult()
        supported_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tiff", ".webp"}

        def process_with_pooled_repo(file_path: Path) -> ProcessingResult:
            """Process file with a repository using the connection pool."""
            if self.connection_pool:
                # Create a temporary repository using the PostgreSQL connection pool
                pooled_repo = PhotoRepository(self.connection_pool)

                # Create stages with the pooled repository
                stages = self._get_stages(stage)
                pooled_stages = {
                    "normalize": NormalizeStage(pooled_repo, self.config)
                    if "normalize" in stages
                    else None,
                    "metadata": MetadataStage(pooled_repo, self.config)
                    if "metadata" in stages
                    else None,
                    "enrich": EnrichStage(pooled_repo, self.config) if "enrich" in stages else None,
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

                    # Check if we've hit the max_photos limit after each completed task
                    if self.max_photos is not None and result.processed >= self.max_photos:
                        logger.info(
                            f"Reached maximum photo limit ({self.max_photos}), stopping processing"
                        )
                        # Cancel remaining futures
                        for remaining_future in futures:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break

                except Exception as e:
                    logger.error(f"Streaming parallel processing error for {file_path}: {e}")
                    result.failed += 1
                    result.failed_files.append((str(file_path), str(e)))

        result.success = result.failed == 0
        return result

    def _process_directory_batch_mode(
        self, directory: Path, stage: str, recursive: bool, pattern: str
    ) -> ProcessingResult:
        """Process directory in batch mode for improved performance."""
        files = self._find_files(directory, recursive, pattern)

        # Apply max_photos limit if specified
        if self.max_photos is not None and len(files) > self.max_photos:
            logger.info(f"Limiting to {self.max_photos} photos (found {len(files)})")
            files = files[: self.max_photos]

        result = ProcessingResult(total_files=len(files))

        if not files:
            logger.warning(f"No matching files found in {directory}")
            return result

        logger.info(f"Found {len(files)} files to process in batch mode")

        # Process non-enrich stages first (normalize and metadata)
        stages_to_run = self._get_stages(stage)
        enrich_stage = self.stages.get("enrich")

        # If we need to run normalize or metadata, do those first
        if "normalize" in stages_to_run or "metadata" in stages_to_run:
            logger.info("Processing normalize/metadata stages before batch enrichment")

            # Process normalize and metadata stages individually
            for file_path in files:
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would process: {file_path}")
                    result.skipped += 1
                    continue

                # Process normalize stage if needed
                if "normalize" in stages_to_run:
                    normalize_stage = self.stages["normalize"]
                    if normalize_stage.should_process(file_path, self.force):
                        try:
                            normalize_stage.process(file_path)
                            result.processed += 1
                        except Exception as e:
                            logger.error(f"Normalize failed for {file_path}: {e}")
                            result.failed += 1
                            result.failed_files.append((str(file_path), str(e)))

                # Process metadata stage if needed
                if "metadata" in stages_to_run:
                    metadata_stage = self.stages["metadata"]
                    if metadata_stage.should_process(file_path, self.force):
                        try:
                            metadata_stage.process(file_path)
                            result.processed += 1
                        except Exception as e:
                            logger.error(f"Metadata failed for {file_path}: {e}")
                            result.failed += 1
                            result.failed_files.append((str(file_path), str(e)))

        # Now process enrich stage in batches
        if "enrich" in stages_to_run and enrich_stage:
            logger.info(f"Processing enrich stage in batches (batch_size={self.batch_size})")

            # Collect photos that need enrichment
            photos_to_enrich = []
            for file_path in files:
                photo = self.repository.get_photo_by_filename(str(file_path))
                if photo and enrich_stage.should_process(file_path, self.force):
                    photos_to_enrich.append(photo)

            if photos_to_enrich:
                logger.info(f"Found {len(photos_to_enrich)} photos needing enrichment")

                # Process in batches
                batch_ids = []
                for i in range(0, len(photos_to_enrich), self.batch_size):
                    batch = photos_to_enrich[i : i + self.batch_size]
                    logger.info(
                        f"Submitting batch {i // self.batch_size + 1} with {len(batch)} photos"
                    )

                    batch_id = enrich_stage.process_batch(batch)
                    if batch_id:
                        batch_ids.append(batch_id)
                        logger.info(f"Batch submitted with ID: {batch_id}")
                    else:
                        logger.error(f"Failed to submit batch {i // self.batch_size + 1}")
                        result.failed += len(batch)

                # Monitor batch completion if async
                if self.async_batch and batch_ids:
                    logger.info(f"Monitoring {len(batch_ids)} batch(es) for completion")

                    import time

                    all_completed = False
                    max_wait_time = 3600  # 1 hour max wait
                    start_time = time.time()

                    while not all_completed and (time.time() - start_time) < max_wait_time:
                        all_completed = True
                        for batch_id in batch_ids:
                            status = enrich_stage.monitor_batch(batch_id)
                            if status:
                                if status["status"] not in ["completed", "failed"]:
                                    all_completed = False
                                else:
                                    # Update results based on batch status
                                    result.processed += status.get("processed_count", 0)
                                    result.failed += status.get("failed_count", 0)
                            else:
                                logger.warning(f"Could not get status for batch {batch_id}")

                        if not all_completed:
                            logger.info("Waiting for batches to complete...")
                            time.sleep(10)  # Check every 10 seconds

                    if all_completed:
                        logger.info("All batches completed successfully")
                    else:
                        logger.warning(
                            "Batch processing timed out or some batches did not complete"
                        )
                else:
                    # For synchronous batch processing or if no batches were submitted
                    result.processed += len(photos_to_enrich)
            else:
                logger.info("No photos need enrichment")
                result.skipped += len(files)

        result.success = result.failed == 0
        return result
