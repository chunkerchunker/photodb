from pathlib import Path
from typing import List, Optional
from collections import defaultdict
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_processor import BaseProcessor, ProcessingResult
from ..database.repository import PhotoRepository
from ..database.connection import ConnectionPool
from ..stages.normalize import NormalizeStage
from ..stages.metadata import MetadataStage
from ..stages.detection import DetectionStage
from ..stages.age_gender import AgeGenderStage
from ..stages.scene_analysis import SceneAnalysisStage
from ..utils.batch_coordinator import BatchCoordinator
from .. import config as defaults

logger = logging.getLogger(__name__)


class LocalProcessor(BaseProcessor):
    """Processor for local file operations with parallel processing support."""

    def __init__(
        self,
        repository,
        config: dict,
        collection_id: int,
        force: bool = False,
        dry_run: bool = False,
        parallel: int = 1,
        max_photos: Optional[int] = None,
        stage: str = "all",
        exclude: Optional[List[str]] = None,
        force_directory_scan: bool = False,
        skip_directory_scan: bool = False,
        progress_interval: float = 10.0,
        force_progress: bool = False,
    ):
        super().__init__(repository, config, force, dry_run, max_photos)
        self.collection_id = collection_id
        self.parallel = max(1, parallel)
        self.stage = stage
        self.exclude = exclude or []
        self.force_directory_scan = force_directory_scan
        self.skip_directory_scan = skip_directory_scan
        self.progress_interval = progress_interval
        self.force_progress = force_progress

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
        # IMPORTANT: detection (CoreML YOLO) must be initialized LAST.
        # Loading a CoreML model via coremltools/ultralytics corrupts internal state
        # such that any subsequent torch.load / model creation causes SIGSEGV.
        # (coremltools 9.0, PyTorch 2.8, Apple Silicon)
        if "age_gender" in stages_to_create:
            self.stages["age_gender"] = AgeGenderStage(repository, config)
            self._shared_mivolo = self.stages["age_gender"].predictor
        if "scene_analysis" in stages_to_create:
            self.stages["scene_analysis"] = SceneAnalysisStage(repository, config)
            self._shared_scene_analyzer = self.stages["scene_analysis"].analyzer
            self._shared_prompt_cache = self.stages["scene_analysis"].prompt_cache
            self._shared_apple_classifier = self.stages["scene_analysis"].apple_classifier
        if "detection" in stages_to_create:
            self.stages["detection"] = DetectionStage(repository, config)
            self._shared_detector = self.stages["detection"].detector

        # Create batch coordinators for ML inference when parallel > 1
        self._batch_coordinators: list[BatchCoordinator] = []
        self._yolo_coordinator = None
        self._clip_image_coordinator = None
        self._clip_face_coordinator = None

        if parallel > 1 and defaults.BATCH_COORDINATOR_ENABLED:
            max_size = defaults.BATCH_COORDINATOR_MAX_SIZE
            wait_ms = defaults.BATCH_COORDINATOR_MAX_WAIT_MS

            if (
                "detection" in stages_to_create
                and self._shared_detector is not None
                and defaults.YOLO_BATCH_ENABLED
            ):
                self._yolo_coordinator = BatchCoordinator(
                    inference_fn=self._shared_detector.run_yolo,
                    max_batch_size=max_size,
                    max_wait_ms=wait_ms,
                )
                self._batch_coordinators.append(self._yolo_coordinator)
                self.stages["detection"].yolo_batch_coordinator = self._yolo_coordinator
                logger.info("YOLO batch coordinator enabled")

            if "scene_analysis" in stages_to_create and self._shared_scene_analyzer is not None:
                self._clip_image_coordinator = BatchCoordinator(
                    inference_fn=self._shared_scene_analyzer.batch_encode,
                    max_batch_size=max_size,
                    max_wait_ms=wait_ms,
                )
                self._clip_face_coordinator = BatchCoordinator(
                    inference_fn=self._shared_scene_analyzer.batch_encode,
                    max_batch_size=max_size,
                    max_wait_ms=wait_ms,
                )
                self._batch_coordinators.extend(
                    [self._clip_image_coordinator, self._clip_face_coordinator]
                )
                self.stages["scene_analysis"].image_batch_coordinator = (
                    self._clip_image_coordinator
                )
                self.stages["scene_analysis"].face_batch_coordinator = (
                    self._clip_face_coordinator
                )
                logger.info("MobileCLIP batch coordinators enabled (image + face)")

    def close(self):
        """Clean up resources, including batch coordinators and connection pool."""
        for coordinator in self._batch_coordinators:
            coordinator.close()
        self._batch_coordinators.clear()
        if self.connection_pool:
            self.connection_pool.close_all()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        self.close()

    def _get_stages(self, stage: str) -> List[str]:
        """Get list of stages to run (normalize, metadata, detection, age_gender, scene_analysis)."""
        all_stages = [
            "normalize",
            "metadata",
            "detection",
            "age_gender",
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
                        stage_start = time.monotonic()
                        stage_obj.process(file_path)
                        stage_end = time.monotonic()
                        result.stage_timings[stage_name] = (stage_start, stage_end)

                        # Check if the stage actually succeeded by looking at processing status
                        photo = stage_obj.repository.get_photo_by_orig_path(
                            str(file_path),
                            collection_id=self.collection_id,
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
                                # Track face counts for detection-related stages
                                if stage_name in (
                                    "detection", "age_gender", "scene_analysis",
                                ):
                                    detections = (
                                        stage_obj.repository.get_detections_for_photo(
                                            photo.id
                                        )
                                    )
                                    result.stage_face_counts[stage_name] = len(
                                        detections
                                    )
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
        directory: Optional[Path],
        stage: Optional[str] = None,
        recursive: bool = True,
        pattern: str = "*",
    ) -> ProcessingResult:
        """Process all matching files in a directory or all photos in collection.

        Args:
            directory: Directory to process. If None (only valid when skip_directory_scan
                       is True), processes all photos in the collection.
            stage: Processing stage(s) to run.
            recursive: Whether to process subdirectories recursively.
            pattern: File pattern to match (e.g., "*.jpg").
        """
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
                    self.connection_pool, collection_id=self.collection_id
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
                    detection_stage.collection_id = self.collection_id
                    detection_stage.detector = self._shared_detector  # Reuse shared model
                    detection_stage.faces_output_dir = self.stages["detection"].faces_output_dir
                    detection_stage.yolo_batch_coordinator = self._yolo_coordinator
                    pooled_stages["detection"] = detection_stage
                if "age_gender" in stages_list:
                    assert self._shared_mivolo is not None  # Loaded in __init__
                    age_gender_stage = AgeGenderStage.__new__(AgeGenderStage)
                    age_gender_stage.repository = pooled_repo
                    age_gender_stage.config = self.config
                    age_gender_stage.stage_name = "age_gender"
                    age_gender_stage.collection_id = self.collection_id
                    age_gender_stage.predictor = self._shared_mivolo  # Reuse shared model
                    pooled_stages["age_gender"] = age_gender_stage
                if "scene_analysis" in stages_list:
                    assert self._shared_scene_analyzer is not None  # Loaded in __init__
                    assert self._shared_prompt_cache is not None  # Loaded in __init__
                    scene_stage = SceneAnalysisStage.__new__(SceneAnalysisStage)
                    scene_stage.repository = pooled_repo
                    scene_stage.config = self.config
                    scene_stage.stage_name = "scene_analysis"
                    scene_stage.collection_id = self.collection_id
                    scene_stage.analyzer = self._shared_scene_analyzer
                    scene_stage.prompt_cache = self._shared_prompt_cache
                    scene_stage.apple_classifier = self._shared_apple_classifier
                    scene_stage.scene_categories = self.stages["scene_analysis"].scene_categories
                    scene_stage.face_categories = self.stages["scene_analysis"].face_categories
                    scene_stage.image_batch_coordinator = self._clip_image_coordinator
                    scene_stage.face_batch_coordinator = self._clip_face_coordinator
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
                if directory is None:
                    raise ValueError("directory is required when scanning (not using database)")
                logger.info("Scanning directory for files...")
                if recursive:
                    file_iter = directory.rglob(pattern)
                else:
                    file_iter = directory.glob(pattern)
                # Collect matching files upfront so we can log the count before processing
                files_to_process = [
                    f for f in file_iter
                    if f.is_file() and f.suffix.lower() in supported_extensions
                ]
            else:
                directory_str = str(directory) if directory is not None else None
                if directory_str:
                    logger.info(f"Querying database for photos in {directory_str}...")
                else:
                    logger.info("Querying database for all photos in collection...")
                photos = self.repository.get_photos_by_directory(directory_str)
                logger.info(f"Found {len(photos)} photos in database")
                files_to_process = [
                    Path(p.orig_path) for p in photos
                    if Path(p.orig_path).match(pattern)
                    and Path(p.orig_path).suffix.lower() in supported_extensions
                ]

            result.total_files = len(files_to_process)

            if result.total_files == 0:
                location = str(directory) if directory else "collection"
                logger.warning(f"No matching files found in {location}")
                return result

            logger.info(f"Expecting to process {result.total_files} files")

            for file_path in files_to_process:
                future = executor.submit(process_with_pooled_repo, file_path)
                futures[future] = file_path

            # Process completed futures as they finish
            start_time = time.monotonic()
            last_progress_time = start_time
            completed_count = 0
            stage_durations: dict[str, list[float]] = defaultdict(list)
            stage_total_faces: dict[str, int] = defaultdict(int)
            for future in as_completed(futures):
                file_path = futures.pop(future)
                try:
                    file_result = future.result()
                    result.processed += file_result.processed
                    result.skipped += file_result.skipped
                    result.failed += file_result.failed
                    result.failed_files.extend(file_result.failed_files)
                    for sname, (start, end) in file_result.stage_timings.items():
                        stage_durations[sname].append(end - start)
                    for sname, fcount in file_result.stage_face_counts.items():
                        stage_total_faces[sname] += fcount

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

                completed_count += 1
                now = time.monotonic()
                if now - last_progress_time >= self.progress_interval:
                    elapsed = now - start_time
                    pct = completed_count / result.total_files * 100
                    elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
                    per_photo = elapsed / completed_count
                    msg = (
                        f"Progress: {completed_count}/{result.total_files}"
                        f" ({pct:.1f}%) elapsed {elapsed_min}m{elapsed_sec:02d}s"
                        f" ({per_photo:.2f}s/photo)"
                    )
                    if self.force_progress:
                        logger.warning(msg)
                    else:
                        logger.info(msg)
                    last_progress_time = now

        # Log per-stage performance stats
        if stage_durations:
            total_wall = time.monotonic() - start_time
            total_wall_min, total_wall_sec = divmod(int(total_wall), 60)
            lines = [f"Performance stats ({total_wall_min}m{total_wall_sec:02d}s wall time):"]
            for sname in self._get_stages(stage):
                if sname in stage_durations:
                    durations = stage_durations[sname]
                    count = len(durations)
                    avg = sum(durations) / count
                    total_faces = stage_total_faces.get(sname)
                    if total_faces:
                        avg_face = sum(durations) / total_faces
                        lines.append(
                            f"  {sname}: {count} photos, {avg:.2f}s avg/photo"
                            f" | {total_faces} faces, {avg_face:.3f}s avg/face"
                        )
                    else:
                        lines.append(
                            f"  {sname}: {count} photos, {avg:.2f}s avg/photo"
                        )
            # Log batch coordinator stats
            if self._yolo_coordinator is not None:
                stats = self._yolo_coordinator.stats
                if stats["batches_processed"] > 0:
                    lines.append(
                        f"  YOLO batch coordinator: {stats['batches_processed']} batches,"
                        f" {stats['items_processed']} items,"
                        f" {stats['avg_batch_size']:.1f} avg batch size"
                    )
            if self._clip_image_coordinator is not None:
                stats = self._clip_image_coordinator.stats
                if stats["batches_processed"] > 0:
                    lines.append(
                        f"  CLIP image batch coordinator: {stats['batches_processed']} batches,"
                        f" {stats['items_processed']} items,"
                        f" {stats['avg_batch_size']:.1f} avg batch size"
                    )
            if self._clip_face_coordinator is not None:
                stats = self._clip_face_coordinator.stats
                if stats["batches_processed"] > 0:
                    lines.append(
                        f"  CLIP face batch coordinator: {stats['batches_processed']} batches,"
                        f" {stats['items_processed']} items,"
                        f" {stats['avg_batch_size']:.1f} avg batch size"
                    )

            logger.warning("\n".join(lines))

        result.success = result.failed == 0
        return result
