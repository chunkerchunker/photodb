"""
Stage 3: Face detection and embedding extraction.
"""

from pathlib import Path
import logging

from .base import BaseStage
from ..database.models import Photo, Face
from ..utils.face_extractor import FaceExtractor

logger = logging.getLogger(__name__)


class FacesStage(BaseStage):
    """Stage for detecting faces and extracting embeddings."""

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)

        # Check if we should force CPU mode due to MPS issues
        import os

        force_cpu = os.getenv("FACE_DETECTION_FORCE_CPU", "false").lower() == "true"

        self.face_extractor = FaceExtractor(force_cpu_fallback=force_cpu)
        logger.debug(f"FacesStage initialized with device: {self.face_extractor.device}")

    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """Check if face detection should be run for this file."""
        if force:
            return True

        # Get photo record
        photo = self.repository.get_photo_by_filename(str(file_path))
        if not photo:
            logger.debug(f"No photo record found for {file_path}")
            return True  # New photo, should process

        # Check if faces have been processed
        existing_faces = self.repository.get_faces_for_photo(photo.id)
        should_process = len(existing_faces) == 0

        if not should_process:
            logger.debug(f"Faces already processed for {file_path}, skipping")

        return should_process

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process face detection for a single photo."""
        try:
            # Check if normalized file exists
            if not photo.normalized_path:
                logger.warning(f"No normalized path for photo {photo.id}, skipping face detection")
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            logger.debug(f"Processing faces for {file_path} -> {normalized_path}")

            # Clear existing faces if reprocessing
            existing_faces = self.repository.get_faces_for_photo(photo.id)
            if existing_faces:
                logger.debug(f"Clearing {len(existing_faces)} existing faces")
                self.repository.delete_faces_for_photo(photo.id)

            # Extract faces using normalized image
            result = self.face_extractor.extract_from_image(str(normalized_path))

            if result["status"] == "no_faces_detected":
                logger.debug(f"No faces detected in {file_path}")
                # No faces found, but processing succeeded
                return True

            # Store detected faces
            faces_saved = 0
            for face_data in result["faces"]:
                bbox = face_data["bbox"]

                # Create Face record using existing schema
                face = Face.create(
                    photo_id=photo.id,
                    bbox_x=bbox["x1"],
                    bbox_y=bbox["y1"],
                    bbox_width=bbox["width"],
                    bbox_height=bbox["height"],
                    confidence=face_data["confidence"],
                )

                # Save face record
                self.repository.create_face(face)

                # Save embedding separately using pgvector
                self.repository.save_face_embedding(face.id, face_data["embedding"])
                faces_saved += 1

            logger.info(f"Saved {faces_saved} faces for {file_path}")
            return True

        except Exception as e:
            logger.error(f"Face detection failed for {file_path}: {e}")
            return False
