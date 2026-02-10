from pathlib import Path
from typing import List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_processor import BaseProcessor, ProcessingResult
from ..database.repository import PhotoRepository
from ..database.connection import ConnectionPool
from ..stages.normalize import NormalizeStage
from ..stages.metadata import MetadataStage
from ..stages.detection import DetectionStage
from ..stages.age_gender import AgeGenderStage
from ..stages.clustering import ClusteringStage
from ..stages.scene_analysis import SceneAnalysisStage

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
        stage: str = "all",
        exclude: Optional[List[str]] = None,
        force_directory_scan: bool = False,
        skip_directory_scan: bool = False,
    ):
        super().__init__(repository, config, force, dry_run, max_photos)
        self.parallel = max(1, parallel)
        self.stage = stage
        self.exclude = exclude or []
        self.force_directory_scan = force_directory_scan
        self.skip_directory_scan = skip_directory_scan

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

        # Only create stages that will be used (avoids loading large ML models unnecessarily)
        stages_to_create = self._get_stages(stage)
        self.stages: dict = {}

        # Cache shared ML models for thread reuse (only if stage is loaded)
        self._shared_detector = None
        self._shared_mivolo = None
        self._shared_scene_analyzer = None
        self._shared_prompt_cache = None
        self._shared_apple_classifier = None

        if self.exclude:
            logger.info(f"Excluding stages: {', '.join(self.exclude)}")
        logger.info(f"Loading stages: {', '.join(stages_to_create)}")

        if "normalize" in stages_to_create:
            self.stages["normalize"] = NormalizeStage(repository, config)
        if "metadata" in stages_to_create:
            self.stages["metadata"] = MetadataStage(repository, config)
        if "detection" in stages_to_create:
            self.stages["detection"] = DetectionStage(repository, config)
            self._shared_detector = self.stages["detection"].detector
        if "age_gender" in stages_to_create:
            self.stages["age_gender"] = AgeGenderStage(repository, config)
            self._shared_mivolo = self.stages["age_gender"].predictor
        if "clustering" in stages_to_create:
            self.stages["clustering"] = ClusteringStage(repository, config)
        if "scene_analysis" in stages_to_create:
            self.stages["scene_analysis"] = SceneAnalysisStage(repository, config)
            self._shared_scene_analyzer = self.stages["scene_analysis"].analyzer
            self._shared_prompt_cache = self.stages["scene_analysis"].prompt_cache
            self._shared_apple_classifier = self.stages["scene_analysis"].apple_classifier

    def close(self):
        """Clean up resources, including connection pool."""
        if self.connection_pool:
            self.connection_pool.close_all()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        self.close()

    def _get_stages(self, stage: str) -> List[str]:
        """Get list of stages to run (normalize, metadata, detection, age_gender, clustering, scene_analysis)."""
        all_stages = [
            "normalize",
            "metadata",
            "detection",
            "age_gender",
            "clustering",
            "scene_analysis",
        ]

        if stage == "all":
            stages = all_stages
        elif stage in all_stages:
            stages = [stage]
        else:
            raise ValueError(f"Invalid stage for LocalProcessor: {stage}")

        # Filter out excluded stages
        if self.exclude:
            stages = [s for s in stages if s not in self.exclude]

        return stages

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
                        # Set force flag on stage object for clustering logic
                        stage_obj.force = self.force
                        stage_obj.process(file_path)

                        # Check if the stage actually succeeded by looking at processing status
                        photo = stage_obj.repository.get_photo_by_orig_path(
                            str(file_path),
                            collection_id=int(self.config.get("COLLECTION_ID", 1)),
                        )
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

    def process_file(self, file_path: Path, stage: Optional[str] = None) -> ProcessingResult:
        """Process a single file through specified stages."""
        stage = stage or self.stage
        return self._process_single_file(file_path, stage, self.stages)

    def process_directory(
        self,
        directory: Path,
        stage: Optional[str] = None,
        recursive: bool = True,
        pattern: str = "*",
    ) -> ProcessingResult:
        """Process all matching files in a directory."""
        stage = stage or self.stage
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
                pooled_repo = PhotoRepository(
                    self.connection_pool, collection_id=int(self.config.get("COLLECTION_ID", 1))
                )

                # Create stages with the pooled repository
                # IMPORTANT: Reuse shared ML models to avoid expensive reloading
                stages_list = self._get_stages(stage)
                pooled_stages = {}

                if "normalize" in stages_list:
                    pooled_stages["normalize"] = NormalizeStage(pooled_repo, self.config)
                if "metadata" in stages_list:
                    pooled_stages["metadata"] = MetadataStage(pooled_repo, self.config)
                if "detection" in stages_list:
                    assert self._shared_detector is not None  # Loaded in __init__
                    detection_stage = DetectionStage.__new__(DetectionStage)
                    detection_stage.repository = pooled_repo
                    detection_stage.config = self.config
                    detection_stage.stage_name = "detection"
                    detection_stage.detector = self._shared_detector  # Reuse shared model
                    pooled_stages["detection"] = detection_stage
                if "age_gender" in stages_list:
                    assert self._shared_mivolo is not None  # Loaded in __init__
                    age_gender_stage = AgeGenderStage.__new__(AgeGenderStage)
                    age_gender_stage.repository = pooled_repo
                    age_gender_stage.config = self.config
                    age_gender_stage.stage_name = "age_gender"
                    age_gender_stage.predictor = self._shared_mivolo  # Reuse shared model
                    pooled_stages["age_gender"] = age_gender_stage
                if "clustering" in stages_list:
                    pooled_stages["clustering"] = ClusteringStage(pooled_repo, self.config)
                if "scene_analysis" in stages_list:
                    assert self._shared_scene_analyzer is not None  # Loaded in __init__
                    assert self._shared_prompt_cache is not None  # Loaded in __init__
                    scene_stage = SceneAnalysisStage.__new__(SceneAnalysisStage)
                    scene_stage.repository = pooled_repo
                    scene_stage.config = self.config
                    scene_stage.stage_name = "scene_analysis"
                    scene_stage.analyzer = self._shared_scene_analyzer
                    scene_stage.prompt_cache = self._shared_prompt_cache
                    scene_stage.apple_classifier = self._shared_apple_classifier
                    scene_stage.scene_categories = self.stages["scene_analysis"].scene_categories
                    scene_stage.face_categories = self.stages["scene_analysis"].face_categories
                    pooled_stages["scene_analysis"] = scene_stage

                # Process with pooled stages
                return self._process_single_file(file_path, stage, pooled_stages)
            else:
                return self.process_file(file_path, stage)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = {}

            # Stream files and submit them for processing as they're found
            should_scan = False
            if self.force_directory_scan:
                should_scan = True
            elif self.skip_directory_scan:
                should_scan = False
            elif "normalize" in self._get_stages(stage):
                should_scan = True

            if should_scan:
                logger.info("Scanning directory for files...")
                if recursive:
                    file_iter = directory.rglob(pattern)
                else:
                    file_iter = directory.glob(pattern)
            else:
                logger.info(f"Querying database for photos in {directory}...")
                photos = self.repository.get_photos_by_directory(str(directory))
                # Create generator that matches the pattern (consistent with glob)
                file_iter = (
                    Path(p.orig_path)
                    for p in photos
                    if Path(p.orig_path).match(pattern)
                )

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
