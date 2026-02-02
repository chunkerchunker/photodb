"""
Detection stage: Face and body detection using PersonDetector (YOLO + FaceNet).
"""

import os
from pathlib import Path
import logging

from .base import BaseStage
from ..database.models import Photo, PersonDetection
from ..utils.person_detector import PersonDetector

logger = logging.getLogger(__name__)


class DetectionStage(BaseStage):
    """Stage for detecting faces and bodies, extracting embeddings using PersonDetector."""

    stage_name = "detection"

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)

        # Check if we should force CPU mode
        force_cpu = os.getenv("DETECTION_FORCE_CPU", "false").lower() == "true"

        self.detector = PersonDetector(force_cpu=force_cpu)
        logger.debug(f"DetectionStage initialized with device: {self.detector.device}")

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process detection for a single photo."""
        try:
            # Check if normalized file exists
            if not photo.normalized_path:
                logger.warning(f"No normalized path for photo {photo.id}, skipping detection")
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            logger.debug(f"Processing detection for {file_path} -> {normalized_path}")

            # Clear existing detections if reprocessing
            existing_detections = self.repository.get_detections_for_photo(photo.id)
            if existing_detections:
                logger.debug(f"Clearing {len(existing_detections)} existing detections")
                self.repository.delete_detections_for_photo(photo.id)

            # Run detection using PersonDetector
            result = self.detector.detect(str(normalized_path))

            if result["status"] == "error":
                logger.error(
                    f"Detection failed for {file_path}: {result.get('error', 'Unknown error')}"
                )
                return False

            if result["status"] == "no_detections":
                logger.debug(f"No detections in {file_path}")
                # No detections found, but processing succeeded
                return True

            # Process each detection
            detections_saved = 0
            for detection_data in result["detections"]:
                face_data = detection_data.get("face")
                body_data = detection_data.get("body")

                # Create PersonDetection record
                detection = self._create_detection_record(photo.id, face_data, body_data)

                # Save detection to database
                self.repository.create_person_detection(detection)
                detections_saved += 1

                # Save embedding if face has one
                if (
                    face_data is not None
                    and detection.id is not None
                    and face_data.get("embedding") is not None
                ):
                    self.repository.save_detection_embedding(detection.id, face_data["embedding"])

            logger.info(f"Saved {detections_saved} detections for {file_path}")
            return True

        except Exception as e:
            logger.error(f"Detection failed for {file_path}: {e}")
            return False

    def _create_detection_record(
        self, photo_id: int, face_data: dict | None, body_data: dict | None
    ) -> PersonDetection:
        """Create a PersonDetection record from detector output."""
        # Extract face bounding box if present
        face_bbox_x = None
        face_bbox_y = None
        face_bbox_width = None
        face_bbox_height = None
        face_confidence = None

        if face_data is not None:
            bbox = face_data["bbox"]
            face_bbox_x = bbox["x1"]
            face_bbox_y = bbox["y1"]
            face_bbox_width = bbox["x2"] - bbox["x1"]
            face_bbox_height = bbox["y2"] - bbox["y1"]
            face_confidence = face_data["confidence"]

        # Extract body bounding box if present
        body_bbox_x = None
        body_bbox_y = None
        body_bbox_width = None
        body_bbox_height = None
        body_confidence = None

        if body_data is not None:
            bbox = body_data["bbox"]
            body_bbox_x = bbox["x1"]
            body_bbox_y = bbox["y1"]
            body_bbox_width = bbox["x2"] - bbox["x1"]
            body_bbox_height = bbox["y2"] - bbox["y1"]
            body_confidence = body_data["confidence"]

        return PersonDetection.create(
            photo_id=photo_id,
            face_bbox_x=face_bbox_x,
            face_bbox_y=face_bbox_y,
            face_bbox_width=face_bbox_width,
            face_bbox_height=face_bbox_height,
            face_confidence=face_confidence,
            body_bbox_x=body_bbox_x,
            body_bbox_y=body_bbox_y,
            body_bbox_width=body_bbox_width,
            body_bbox_height=body_bbox_height,
            body_confidence=body_confidence,
            detector_model="YOLO+FaceNet",
        )
