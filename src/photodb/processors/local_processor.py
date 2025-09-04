from pathlib import Path
from typing import List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_processor import BaseProcessor, ProcessingResult
from ..database.repository import PhotoRepository
from ..database.connection import ConnectionPool
from ..stages.normalize import NormalizeStage
from ..stages.metadata import MetadataStage

logger = logging.getLogger(__name__)


class LocalProcessor(BaseProcessor):
    """Processor for local file operations with parallel processing support."""

    def __init__(
        self,
        repository,
        config: dict,
        force: bool = False,
        dry_run: bool = False,
        parallel: int = 1,
        max_photos: Optional[int] = None,
    ):
        super().__init__(repository, config, force, dry_run, max_photos)
        self.parallel = max(1, parallel)

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

        # Only create normalize and metadata stages
        self.stages = {
            "normalize": NormalizeStage(repository, config),
            "metadata": MetadataStage(repository, config),
        }

    def _get_stages(self, stage: str) -> List[str]:
        """Get list of stages to run (limited to normalize and metadata)."""
        if stage == "all":
            return ["normalize", "metadata"]
        elif stage in ["normalize", "metadata"]:
            return [stage]
        else:
            raise ValueError(f"Invalid stage for LocalProcessor: {stage}")

    def _process_single_file(
        self, file_path: Path, stage: str, stages_dict: dict
    ) -> ProcessingResult:
        """Process a single file with provided stages dictionary."""
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
                                error_msg = stage_status.error_message or "Stage processing failed"
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
        result = ProcessingResult()
        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".heic",
            ".heif",
            ".bmp",
            ".tiff",
            ".webp",
        }

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

            logger.info(f"Found {result.total_files} files to process")

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
