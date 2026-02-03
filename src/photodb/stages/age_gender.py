"""
AgeGenderStage: Age and gender estimation using MiVOLO.

This stage processes existing person detections and estimates age and gender
using the MiVOLO model, which can use both face and body bounding boxes for
improved accuracy.

Supports free-threaded Python 3.13t for true parallel inference on CPU.
"""

import os
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .base import BaseStage
from ..database.models import Photo

logger = logging.getLogger(__name__)


def _patch_torch_load():
    """
    Patch torch.load for PyTorch 2.6+ compatibility.

    PyTorch 2.6 changed weights_only default to True, which breaks loading
    older checkpoint files. We temporarily patch it during model loading.
    """
    original_torch_load = torch.load

    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    return original_torch_load, patched_load


class MiVOLOPredictor:
    """
    Wrapper for MiVOLO age/gender prediction.

    Handles import errors gracefully - MiVOLO may not be installed in all environments.
    Thread-safe for use with free-threaded Python 3.13t (PYTHON_GIL=0).

    Note: Lock IS REQUIRED - MiVOLO has lazy-initialized state (internal YOLO detector)
    that causes race conditions and inconsistent results without serialization.
    """

    def __init__(
        self,
        checkpoint_path: str,
        detector_weights_path: str,
        device: str = "cpu",
    ):
        """
        Initialize MiVOLO predictor.

        Args:
            checkpoint_path: Path to MiVOLO checkpoint file (.pth.tar format)
            detector_weights_path: Path to YOLO detector weights
            device: Device to use ('cuda', 'mps', 'cpu')
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._available = False
        self.predictor = None
        # Lock is REQUIRED for thread-safe inference - MiVOLO has internal state
        # that causes race conditions without serialization (tested empirically)
        self._lock = threading.Lock()

        # Try to import and initialize MiVOLO
        try:
            from types import SimpleNamespace

            # Apply timm compatibility shim before importing mivolo
            # (MiVOLO needs timm 0.8.x APIs that were changed in 0.9+)
            from ..utils import timm_compat  # noqa: F401
            from mivolo.predictor import Predictor

            # Patch torch.load for PyTorch 2.6+ compatibility
            original_load, patched_load = _patch_torch_load()
            torch.load = patched_load

            try:
                # MiVOLO expects an argparse-like config object
                config = SimpleNamespace(
                    checkpoint=checkpoint_path,
                    detector_weights=detector_weights_path,
                    device=device,
                    with_persons=True,
                    disable_faces=False,
                    draw=False,
                )

                self.predictor = Predictor(config, verbose=False)
                self._available = True
                logger.info(f"MiVOLO predictor initialized on device: {device} (thread-safe)")
            finally:
                # Restore original torch.load
                torch.load = original_load

        except ImportError as e:
            logger.warning(f"MiVOLO not available: {e}. Age/gender estimation disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize MiVOLO: {e}")

    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Predict age and gender for all detections in an image.

        MiVOLO's Predictor.recognize() runs its own detection and returns
        results with bounding boxes and age/gender estimates.

        Args:
            image_path: Path to image file

        Returns:
            List of dicts, each with keys: face_bbox, body_bbox, age, gender, gender_confidence
        """
        if not self._available:
            return []

        try:
            import cv2

            # Load image (MiVOLO expects BGR numpy array)
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return []

            # Run MiVOLO prediction (detection + age/gender in one pass)
            # Lock is REQUIRED - MiVOLO has lazy-initialized internal state
            with self._lock:
                result, _ = self.predictor.recognize(img)

            # Extract results from PersonAndFaceResult
            predictions = []

            # Process face-to-person mappings (faces with associated bodies)
            for face_ind, person_ind in result.face_to_person_map.items():
                age = result.ages[face_ind]
                gender = result.genders[face_ind]
                gender_score = result.gender_scores[face_ind]

                if age is None and gender is None:
                    continue

                # Get face bbox (x1, y1, x2, y2)
                face_bbox = result.get_bbox_by_ind(face_ind).cpu().numpy()

                # Get body bbox if associated
                body_bbox = None
                if person_ind is not None:
                    body_bbox = result.get_bbox_by_ind(person_ind).cpu().numpy()

                predictions.append(
                    {
                        "face_bbox": (
                            float(face_bbox[0]),  # x
                            float(face_bbox[1]),  # y
                            float(face_bbox[2] - face_bbox[0]),  # width
                            float(face_bbox[3] - face_bbox[1]),  # height
                        ),
                        "body_bbox": (
                            (
                                float(body_bbox[0]),
                                float(body_bbox[1]),
                                float(body_bbox[2] - body_bbox[0]),
                                float(body_bbox[3] - body_bbox[1]),
                            )
                            if body_bbox is not None
                            else None
                        ),
                        "age": float(age) if age is not None else None,
                        "gender": "M" if gender == "male" else ("F" if gender == "female" else "U"),
                        "gender_confidence": float(gender_score)
                        if gender_score is not None
                        else 0.0,
                    }
                )

            # Process unassigned persons (bodies without faces)
            for person_ind in result.unassigned_persons_inds:
                age = result.ages[person_ind]
                gender = result.genders[person_ind]
                gender_score = result.gender_scores[person_ind]

                if age is None and gender is None:
                    continue

                body_bbox = result.get_bbox_by_ind(person_ind).cpu().numpy()

                predictions.append(
                    {
                        "face_bbox": None,
                        "body_bbox": (
                            float(body_bbox[0]),
                            float(body_bbox[1]),
                            float(body_bbox[2] - body_bbox[0]),
                            float(body_bbox[3] - body_bbox[1]),
                        ),
                        "age": float(age) if age is not None else None,
                        "gender": "M" if gender == "male" else ("F" if gender == "female" else "U"),
                        "gender_confidence": float(gender_score)
                        if gender_score is not None
                        else 0.0,
                    }
                )

            return predictions

        except Exception as e:
            logger.error(f"MiVOLO prediction failed: {e}")
            return []


def _compute_iou(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection over Union between two bboxes.

    Args:
        bbox1: (x, y, width, height) - first bounding box
        bbox2: (x, y, width, height) - second bounding box

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to (x1, y1, x2, y2) format
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2

    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class AgeGenderStage(BaseStage):
    """
    Stage for estimating age and gender using MiVOLO.

    This stage:
    1. Runs MiVOLO on the normalized image (which does its own detection)
    2. Matches MiVOLO results to existing detections by bbox IoU
    3. Updates matched detection records with age_estimate, gender, gender_confidence
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
            # Check for CUDA/MPS availability
            try:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # MiVOLO may have issues with MPS, fall back to CPU
                    device = "cpu"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        # Get checkpoint path from config or environment
        checkpoint_path = config.get(
            "MIVOLO_MODEL_PATH",
            os.getenv("MIVOLO_MODEL_PATH", "models/mivolo_d1.pth.tar"),
        )

        # Get detector weights path (same YOLO model used by DetectionStage)
        detector_weights_path = config.get(
            "DETECTION_MODEL_PATH",
            os.getenv("DETECTION_MODEL_PATH", "models/yolov8x_person_face.pt"),
        )

        self.predictor = MiVOLOPredictor(
            checkpoint_path=checkpoint_path,
            detector_weights_path=detector_weights_path,
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

            # Run MiVOLO prediction
            predictions = self.predictor.predict(str(normalized_path))
            if not predictions:
                logger.debug(f"No MiVOLO predictions for photo {photo.id}")
                return True

            logger.debug(
                f"Got {len(predictions)} MiVOLO predictions for {len(detections)} detections"
            )

            # Match MiVOLO predictions to existing detections by bbox IoU
            updated = 0
            for prediction in predictions:
                best_match = None
                best_iou = 0.3  # Minimum IoU threshold

                for detection in detections:
                    # Try matching by face bbox first (all coordinates must be present)
                    if (
                        prediction["face_bbox"] is not None
                        and detection.face_bbox_x is not None
                        and detection.face_bbox_y is not None
                        and detection.face_bbox_width is not None
                        and detection.face_bbox_height is not None
                    ):
                        det_face_bbox = (
                            detection.face_bbox_x,
                            detection.face_bbox_y,
                            detection.face_bbox_width,
                            detection.face_bbox_height,
                        )
                        iou = _compute_iou(prediction["face_bbox"], det_face_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = detection

                    # Also try matching by body bbox (all coordinates must be present)
                    if (
                        prediction["body_bbox"] is not None
                        and detection.body_bbox_x is not None
                        and detection.body_bbox_y is not None
                        and detection.body_bbox_width is not None
                        and detection.body_bbox_height is not None
                    ):
                        det_body_bbox = (
                            detection.body_bbox_x,
                            detection.body_bbox_y,
                            detection.body_bbox_width,
                            detection.body_bbox_height,
                        )
                        iou = _compute_iou(prediction["body_bbox"], det_body_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = detection

                if best_match is not None:
                    # Update the matched detection
                    self.repository.update_detection_age_gender(
                        detection_id=best_match.id,
                        age_estimate=prediction["age"],
                        gender=prediction["gender"],
                        gender_confidence=prediction["gender_confidence"],
                        mivolo_output=prediction,
                    )
                    updated += 1
                    logger.debug(
                        f"Updated detection {best_match.id}: age={prediction['age']}, "
                        f"gender={prediction['gender']} ({prediction['gender_confidence']:.2f})"
                    )

            logger.info(
                f"Updated age/gender for {updated}/{len(detections)} detections in photo {photo.id}"
            )
            return True

        except Exception as e:
            logger.error(f"Age/gender estimation failed for {file_path}: {e}")
            return False
