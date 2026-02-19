"""
Detection stage: Face and body detection using PersonDetector (YOLO + ArcFace).
"""

from pathlib import Path
import logging

from PIL import Image

from .base import BaseStage
from .. import config as defaults
from ..database.models import Photo, PersonDetection
from ..utils.person_detector import PersonDetector

logger = logging.getLogger(__name__)

# Subdirectory for face crops (matches pattern of full/ and med/ in normalize stage)
FACES_SUBDIR = "faces"

# WebP quality for face crops
FACE_WEBP_QUALITY = defaults.WEBP_QUALITY


class DetectionStage(BaseStage):
    """Stage for detecting faces and bodies, extracting embeddings using PersonDetector."""

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)

        force_cpu = defaults.DETECTION_FORCE_CPU

        self.detector = PersonDetector(force_cpu=force_cpu)
        self.detector.warmup()
        self.faces_output_dir = (
            Path(self.config.get("IMG_PATH", "./photos/processed")) / FACES_SUBDIR
        )
        self.yolo_batch_coordinator = None  # Set externally for cross-photo batching
        logger.debug(f"DetectionStage initialized with device: {self.detector.device}")

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process detection for a single photo."""
        if photo.id is None:
            logger.error(f"Photo {file_path} has no ID")
            return False

        photo_id = photo.id  # Capture for type narrowing

        image = None
        try:
            # Check if medium file exists
            if not photo.med_path:
                logger.warning(f"No medium path for photo {photo_id}, skipping detection")
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.med_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            logger.debug(f"Processing detection for {file_path} -> {normalized_path}")

            # Clear existing detections if reprocessing
            existing_detections = self.repository.get_detections_for_photo(photo_id)
            if existing_detections:
                logger.debug(f"Clearing {len(existing_detections)} existing detections")
                self.repository.delete_detections_for_photo(photo_id)

            # Run detection using PersonDetector
            if self.yolo_batch_coordinator is not None:
                # Batch mode: submit image to coordinator, parse result
                img_for_yolo = Image.open(normalized_path)
                if img_for_yolo.mode != "RGB":
                    img_for_yolo = img_for_yolo.convert("RGB")
                try:
                    yolo_result = self.yolo_batch_coordinator.submit(img_for_yolo).result()
                    result = self.detector.parse_yolo_result(img_for_yolo, yolo_result)
                finally:
                    img_for_yolo.close()
            else:
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

            # Ensure faces output directory exists
            self.faces_output_dir.mkdir(parents=True, exist_ok=True)

            # Open image for face cropping
            image = Image.open(normalized_path)

            # Process each detection
            detections_saved = 0
            face_index = 0
            for detection_data in result["detections"]:
                face_data = detection_data.get("face")
                body_data = detection_data.get("body")

                # Crop and save face if present
                face_path = None
                if face_data is not None:
                    face_path = self._crop_and_save_face(image, face_data, photo_id, face_index)
                    face_index += 1

                # Create PersonDetection record
                detection = self._create_detection_record(
                    photo_id, photo.collection_id, face_data, body_data, face_path
                )

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
        finally:
            if image:
                image.close()

    def _crop_and_save_face(
        self,
        image: Image.Image,
        face_data: dict,
        photo_id: int,
        face_index: int,
    ) -> str | None:
        """Crop face from image and save as WebP. Returns the face path or None on failure."""
        try:
            bbox = face_data["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            # Skip if crop is invalid
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid face bbox for photo {photo_id}: {bbox}")
                return None

            # Crop the face
            face_crop = image.crop((x1, y1, x2, y2))

            # Generate output filename: {photo_id}_{face_index}.webp
            output_filename = f"{photo_id}_{face_index}.webp"
            output_path = self.faces_output_dir / output_filename

            # Save as WebP
            face_crop.save(output_path, "WEBP", quality=FACE_WEBP_QUALITY)

            # Return the full path as stored in DB
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to crop face for photo {photo_id}: {e}")
            return None

    def _create_detection_record(
        self,
        photo_id: int,
        collection_id: int,
        face_data: dict | None,
        body_data: dict | None,
        face_path: str | None = None,
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
            collection_id=collection_id,
            face_bbox_x=face_bbox_x,
            face_bbox_y=face_bbox_y,
            face_bbox_width=face_bbox_width,
            face_bbox_height=face_bbox_height,
            face_confidence=face_confidence,
            face_path=face_path,
            body_bbox_x=body_bbox_x,
            body_bbox_y=body_bbox_y,
            body_bbox_width=body_bbox_width,
            body_bbox_height=body_bbox_height,
            body_confidence=body_confidence,
            detector_model="YOLO+ArcFace",
        )
