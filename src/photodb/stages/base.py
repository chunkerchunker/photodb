from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

from photodb.database.repository import PhotoRepository

from ..database.models import Photo, ProcessingStatus

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """Base class for all processing stages."""

    def __init__(self, repository: PhotoRepository, config: dict):
        self.repository = repository
        self.config = config
        self.stage_name = self.__class__.__name__.replace("Stage", "").lower()

    @abstractmethod
    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process a single photo. Must be implemented by subclasses.

        Returns:
            bool: True if processing was successful, False otherwise
        """
        pass

    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """Check if a file should be processed."""
        if force:
            return True

        photo = self.repository.get_photo_by_filename(str(file_path))
        if not photo:
            return True

        return not self.repository.has_been_processed(photo.id, self.stage_name)

    def process(self, file_path: Path) -> None:
        """Process a file through this stage."""
        photo = None
        try:
            # Quick DB operation: get or create photo
            photo = self._get_or_create_photo(file_path)

            # Quick DB write: mark as processing
            self._update_status(photo.id, "processing")

            # Do the heavy lifting OUTSIDE of any transaction
            # This is where image processing happens - no DB locks held!
            success = self.process_photo(photo, file_path)

            # Quick DB write: update final status
            if success:
                self._update_status(photo.id, "completed")
                logger.info(f"Successfully processed {file_path} through {self.stage_name}")
            else:
                self._update_status(photo.id, "failed", "Processing failed")
                logger.error(f"Processing failed for {file_path} through {self.stage_name}")

        except Exception as e:
            logger.error(f"Failed to process {file_path} through {self.stage_name}: {e}")
            if photo:
                self._update_status(photo.id, "failed", str(e))
            raise

    def _get_or_create_photo(self, file_path: Path) -> Photo:
        """Get existing photo or create new one."""
        filename = str(file_path.resolve())
        photo = self.repository.get_photo_by_filename(filename)

        if not photo:
            photo = Photo(
                id=self._generate_photo_id(file_path),
                filename=filename,
                normalized_path="",  # Will be set by normalize stage
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self.repository.create_photo(photo)
            logger.debug(f"Created new photo record: {photo.id}")

        return photo

    def _generate_photo_id(self, file_path: Path) -> str:
        """Generate a unique photo ID."""
        import hashlib

        return hashlib.sha256(str(file_path.resolve()).encode()).hexdigest()[:16]

    def _update_status(self, photo_id: str, status: str, error_message: Optional[str] = None):
        """Update processing status for this stage."""
        processing_status = ProcessingStatus(
            photo_id=photo_id,
            stage=self.stage_name,
            status=status,
            processed_at=datetime.now(),
            error_message=error_message,
        )
        self.repository.update_processing_status(processing_status)
