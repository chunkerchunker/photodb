"""
AgeGenderStage: Age and gender estimation using MiVOLO.

This stage processes existing person detections and estimates age and gender
using the MiVOLO model, which can use both face and body bounding boxes for
improved accuracy.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseStage
from ..database.models import Photo

logger = logging.getLogger(__name__)


class MiVOLOPredictor:
    """
    Wrapper for MiVOLO age/gender prediction.

    Handles import errors gracefully - MiVOLO may not be installed in all environments.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        """
        Initialize MiVOLO predictor.

        Args:
            checkpoint_path: Path to MiVOLO checkpoint file
            device: Device to use ('cuda', 'mps', 'cpu')
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._available = False
        self.predictor = None

        # Try to import and initialize MiVOLO
        try:
            from mivolo.predictor import Predictor

            self.predictor = Predictor(
                config=None,
                ckpt=checkpoint_path,
                device=device,
                with_persons=True,
            )
            self._available = True
            logger.info(f"MiVOLO predictor initialized on device: {device}")
        except ImportError as e:
            logger.warning(f"MiVOLO not available: {e}. Age/gender estimation disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize MiVOLO: {e}")

    def predict(
        self,
        image_path: str,
        face_bbox: Optional[tuple] = None,
        body_bbox: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Predict age and gender for a detection.

        Args:
            image_path: Path to image file
            face_bbox: (x, y, width, height) of face bounding box
            body_bbox: (x, y, width, height) of body bounding box

        Returns:
            Dict with keys: age, gender, gender_confidence
        """
        if not self._available:
            return {"age": None, "gender": "U", "gender_confidence": 0.0}

        try:
            import numpy as np
            from PIL import Image

            # Load image
            img = Image.open(image_path)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img_np = np.array(img)

            # Convert bboxes to MiVOLO format (x1, y1, x2, y2)
            faces = None
            bodies = None

            if face_bbox is not None and face_bbox[0] is not None:
                x, y, w, h = face_bbox
                faces = np.array([[x, y, x + w, y + h]])

            if body_bbox is not None and body_bbox[0] is not None:
                x, y, w, h = body_bbox
                bodies = np.array([[x, y, x + w, y + h]])

            # Run prediction
            result = self.predictor.recognize(
                img_np,
                detected_faces=faces,
                detected_bodies=bodies,
            )

            if result and len(result.ages) > 0:
                age = float(result.ages[0])
                gender = "M" if result.genders[0] == "male" else "F"
                # Get gender confidence if available
                gender_conf = (
                    float(result.gender_scores[0])
                    if hasattr(result, "gender_scores") and result.gender_scores is not None
                    else 0.9
                )
                return {
                    "age": age,
                    "gender": gender,
                    "gender_confidence": gender_conf,
                }

            return {"age": None, "gender": "U", "gender_confidence": 0.0}

        except Exception as e:
            logger.error(f"MiVOLO prediction failed: {e}")
            return {"age": None, "gender": "U", "gender_confidence": 0.0}


class AgeGenderStage(BaseStage):
    """
    Stage for estimating age and gender using MiVOLO.

    This stage:
    1. Gets existing person detections for the photo
    2. For each detection with a face or body bbox, runs MiVOLO prediction
    3. Updates the detection record with age_estimate, gender, gender_confidence
    """

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        # Override auto-generated stage_name
        self.stage_name = "age_gender"

        # Determine device
        force_cpu = os.getenv("MIVOLO_FORCE_CPU", "false").lower() == "true"
        if force_cpu:
            device = "cpu"
        else:
            # Check for CUDA availability
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        # Get checkpoint path from config or environment
        checkpoint_path = config.get(
            "MIVOLO_MODEL_PATH",
            os.getenv("MIVOLO_MODEL_PATH", "models/mivolo_imdb.pth.tar"),
        )

        self.predictor = MiVOLOPredictor(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.debug(f"AgeGenderStage initialized with device: {device}")

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """
        Process age/gender estimation for detections in a photo.

        Args:
            photo: Photo record with normalized_path
            file_path: Original file path (used for logging)

        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Check for normalized path
            if not photo.normalized_path:
                logger.warning(f"No normalized path for photo {photo.id}, skipping age/gender")
                return False

            # Build full path to normalized image
            normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            # Get existing detections for this photo
            detections = self.repository.get_detections_for_photo(photo.id)
            if not detections:
                logger.debug(f"No detections for photo {photo.id}, skipping age/gender")
                return True

            logger.debug(
                f"Processing age/gender for {len(detections)} detections in photo {photo.id}"
            )

            updated = 0
            for detection in detections:
                # Build bbox tuples from detection record
                face_bbox = None
                if detection.face_bbox_x is not None:
                    face_bbox = (
                        detection.face_bbox_x,
                        detection.face_bbox_y,
                        detection.face_bbox_width,
                        detection.face_bbox_height,
                    )

                body_bbox = None
                if detection.body_bbox_x is not None:
                    body_bbox = (
                        detection.body_bbox_x,
                        detection.body_bbox_y,
                        detection.body_bbox_width,
                        detection.body_bbox_height,
                    )

                # Skip if neither face nor body available
                if face_bbox is None and body_bbox is None:
                    logger.debug(f"Detection {detection.id} has no face or body bbox, skipping")
                    continue

                # Run MiVOLO prediction
                result = self.predictor.predict(
                    image_path=str(normalized_path),
                    face_bbox=face_bbox,
                    body_bbox=body_bbox,
                )

                # Update detection if we got meaningful results
                # (age is not None OR gender is not 'U')
                if result["age"] is not None or result["gender"] != "U":
                    self.repository.update_detection_age_gender(
                        detection_id=detection.id,
                        age_estimate=result["age"],
                        gender=result["gender"],
                        gender_confidence=result["gender_confidence"],
                        mivolo_output=result,
                    )
                    updated += 1
                    logger.debug(
                        f"Updated detection {detection.id}: age={result['age']}, "
                        f"gender={result['gender']} ({result['gender_confidence']:.2f})"
                    )

            logger.info(
                f"Updated age/gender for {updated}/{len(detections)} detections in photo {photo.id}"
            )
            return True

        except Exception as e:
            logger.error(f"Age/gender estimation failed for {file_path}: {e}")
            return False
