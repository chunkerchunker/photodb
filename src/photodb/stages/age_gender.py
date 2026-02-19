"""
AgeGenderStage: Age and gender estimation using MiVOLO.

This stage processes existing person detections and estimates age and gender
using the MiVOLO model directly, passing pre-computed bounding boxes from the
detection stage to avoid redundant YOLO inference.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .base import BaseStage
from .. import config as defaults
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
    Wrapper for MiVOLO age/gender prediction using pre-computed bounding boxes.

    Unlike the original MiVOLO Predictor which runs its own YOLO detection,
    this wrapper loads only the MiVOLO age/gender model and accepts bounding
    boxes from the detection stage directly, eliminating redundant detection.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._available = False
        self.model = None

        try:
            # Apply timm compatibility shim before importing mivolo
            from ..utils import timm_compat  # noqa: F401
            from mivolo.model.mi_volo import MiVOLO

            # Patch torch.load for PyTorch 2.6+ compatibility
            original_load, patched_load = _patch_torch_load()
            torch.load = patched_load

            try:
                self.model = MiVOLO(
                    ckpt_path=checkpoint_path,
                    device=device,
                    half=True,
                    use_persons=True,
                    disable_faces=False,
                    verbose=False,
                )
                self._available = True
                logger.info(f"MiVOLO model initialized on device: {device}")
            finally:
                torch.load = original_load

        except ImportError as e:
            logger.warning(f"MiVOLO not available: {e}. Age/gender estimation disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize MiVOLO: {e}")

    @staticmethod
    def _build_synthetic_result(
        image: np.ndarray, detections: list
    ) -> Any:
        """
        Construct a PersonAndFaceResult from pre-computed detection bboxes.

        Each detection dict may have face and/or body bboxes in (x, y, w, h) format.
        These are converted to ultralytics Results format and wrapped in
        PersonAndFaceResult for MiVOLO consumption.
        """
        from mivolo.structures import PersonAndFaceResult
        from ultralytics.engine.results import Results

        boxes_list = []  # [x1, y1, x2, y2, conf, cls]
        # Track which detection indices map to which box indices
        # so we can map results back
        for det in detections:
            # Body bbox → class 0 ("person")
            if (
                det.body_bbox_x is not None
                and det.body_bbox_y is not None
                and det.body_bbox_width is not None
                and det.body_bbox_height is not None
            ):
                x1 = det.body_bbox_x
                y1 = det.body_bbox_y
                x2 = x1 + det.body_bbox_width
                y2 = y1 + det.body_bbox_height
                conf = det.body_confidence if det.body_confidence is not None else 0.9
                boxes_list.append([x1, y1, x2, y2, conf, 0])  # cls=0 → person

            # Face bbox → class 1 ("face")
            if (
                det.face_bbox_x is not None
                and det.face_bbox_y is not None
                and det.face_bbox_width is not None
                and det.face_bbox_height is not None
            ):
                x1 = det.face_bbox_x
                y1 = det.face_bbox_y
                x2 = x1 + det.face_bbox_width
                y2 = y1 + det.face_bbox_height
                conf = det.face_confidence if det.face_confidence is not None else 0.9
                boxes_list.append([x1, y1, x2, y2, conf, 1])  # cls=1 → face

        if not boxes_list:
            # Empty result — no valid bboxes
            boxes_tensor = torch.zeros((0, 6))
        else:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)

        results = Results(
            orig_img=image,
            path="synthetic",
            names={0: "person", 1: "face"},
            boxes=boxes_tensor,
        )

        return PersonAndFaceResult(results)

    def predict(self, image_path: str, detections: list) -> Dict[int, Dict[str, Any]]:
        """
        Predict age and gender using pre-computed bounding boxes.

        Args:
            image_path: Path to image file
            detections: List of PersonDetection objects from the database

        Returns:
            Dict mapping detection list index to prediction dict with keys:
            age, gender, gender_confidence, mivolo_output
        """
        if not self._available:
            return {}

        try:
            from PIL import Image

            pil_img = Image.open(image_path)
            pil_img.load()
            img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB → BGR
            pil_img.close()

            detected_objects = self._build_synthetic_result(img, detections)
            if detected_objects.n_objects == 0:
                return {}

            assert self.model is not None
            self.model.predict(img, detected_objects)

            # Map results back to detection indices.
            # We need to figure out which face/body indices correspond to which
            # original detection. We iterate detections in the same order we built
            # the boxes to reconstruct the mapping.
            #
            # Box layout (same order as _build_synthetic_result):
            #   For each detection: [body_box (if present), face_box (if present)]
            # PersonAndFaceResult.associate_faces_with_persons() (called inside
            # MiVOLO.predict) pairs faces with persons via IoU, which will
            # correctly pair our adjacent face+body boxes.

            results: Dict[int, Dict[str, Any]] = {}

            # Build a reverse map: box_index → (det_index, bbox_type)
            box_to_det: Dict[int, tuple] = {}
            box_idx = 0
            for det_idx, det in enumerate(detections):
                has_body = (
                    det.body_bbox_x is not None
                    and det.body_bbox_y is not None
                    and det.body_bbox_width is not None
                    and det.body_bbox_height is not None
                )
                has_face = (
                    det.face_bbox_x is not None
                    and det.face_bbox_y is not None
                    and det.face_bbox_width is not None
                    and det.face_bbox_height is not None
                )
                if has_body:
                    box_to_det[box_idx] = (det_idx, "body")
                    box_idx += 1
                if has_face:
                    box_to_det[box_idx] = (det_idx, "face")
                    box_idx += 1

            # Extract results from face_to_person_map (faces with associated bodies)
            for face_ind, person_ind in detected_objects.face_to_person_map.items():
                age = detected_objects.ages[face_ind]
                gender = detected_objects.genders[face_ind]
                gender_score = detected_objects.gender_scores[face_ind]

                if age is None and gender is None:
                    continue

                if face_ind in box_to_det:
                    det_idx = box_to_det[face_ind][0]
                elif person_ind is not None and person_ind in box_to_det:
                    det_idx = box_to_det[person_ind][0]
                else:
                    continue

                face_bbox = None
                body_bbox = None
                det = detections[det_idx]
                if (
                    det.face_bbox_x is not None
                    and det.face_bbox_y is not None
                    and det.face_bbox_width is not None
                    and det.face_bbox_height is not None
                ):
                    face_bbox = (
                        det.face_bbox_x,
                        det.face_bbox_y,
                        det.face_bbox_width,
                        det.face_bbox_height,
                    )
                if (
                    det.body_bbox_x is not None
                    and det.body_bbox_y is not None
                    and det.body_bbox_width is not None
                    and det.body_bbox_height is not None
                ):
                    body_bbox = (
                        det.body_bbox_x,
                        det.body_bbox_y,
                        det.body_bbox_width,
                        det.body_bbox_height,
                    )

                results[det_idx] = {
                    "face_bbox": face_bbox,
                    "body_bbox": body_bbox,
                    "age": float(age) if age is not None else None,
                    "gender": "M" if gender == "male" else ("F" if gender == "female" else "U"),
                    "gender_confidence": float(gender_score) if gender_score is not None else 0.0,
                }

            # Extract results from unassigned persons (bodies without faces)
            for person_ind in detected_objects.unassigned_persons_inds:
                age = detected_objects.ages[person_ind]
                gender = detected_objects.genders[person_ind]
                gender_score = detected_objects.gender_scores[person_ind]

                if age is None and gender is None:
                    continue

                if person_ind not in box_to_det:
                    continue

                det_idx = box_to_det[person_ind][0]
                det = detections[det_idx]
                body_bbox = None
                if (
                    det.body_bbox_x is not None
                    and det.body_bbox_y is not None
                    and det.body_bbox_width is not None
                    and det.body_bbox_height is not None
                ):
                    body_bbox = (
                        det.body_bbox_x,
                        det.body_bbox_y,
                        det.body_bbox_width,
                        det.body_bbox_height,
                    )

                results[det_idx] = {
                    "face_bbox": None,
                    "body_bbox": body_bbox,
                    "age": float(age) if age is not None else None,
                    "gender": "M" if gender == "male" else ("F" if gender == "female" else "U"),
                    "gender_confidence": float(gender_score) if gender_score is not None else 0.0,
                }

            return results

        except Exception as e:
            logger.error(f"MiVOLO prediction failed: {e}")
            return {}


class AgeGenderStage(BaseStage):
    """
    Stage for estimating age and gender using MiVOLO.

    This stage:
    1. Passes pre-computed detection bboxes to MiVOLO (no redundant YOLO detection)
    2. Results come back indexed by detection, so no IoU matching is needed
    3. Updates detection records with age_estimate, gender, gender_confidence
    """

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        self.stage_name = "age_gender"

        # Determine device
        force_cpu = defaults.MIVOLO_FORCE_CPU
        if force_cpu:
            device = "cpu"
        else:
            try:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "cpu"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        checkpoint_path = config.get(
            "MIVOLO_MODEL_PATH",
            defaults.MIVOLO_MODEL_PATH,
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
            photo: Photo record with med_path
            file_path: Original file path (used for logging)

        Returns:
            True if processing succeeded, False otherwise
        """
        if photo.id is None:
            logger.error(f"Photo {file_path} has no ID")
            return False

        photo_id = photo.id

        try:
            if not photo.med_path:
                logger.warning(f"No medium path for photo {photo_id}, skipping age/gender")
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.med_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            detections = self.repository.get_detections_for_photo(photo_id)
            if not detections:
                logger.debug(f"No detections for photo {photo_id}, skipping age/gender")
                return True

            # Pass detections directly — no redundant YOLO detection or IoU matching
            predictions = self.predictor.predict(str(normalized_path), detections)
            if not predictions:
                logger.debug(f"No MiVOLO predictions for photo {photo_id}")
                return True

            logger.debug(
                f"Got {len(predictions)} MiVOLO predictions for {len(detections)} detections"
            )

            updated = 0
            for det_idx, prediction in predictions.items():
                detection = detections[det_idx]
                if detection.id is None:
                    continue

                self.repository.update_detection_age_gender(
                    detection_id=detection.id,
                    age_estimate=prediction["age"],
                    gender=prediction["gender"],
                    gender_confidence=prediction["gender_confidence"],
                    mivolo_output=prediction,
                )
                updated += 1
                logger.debug(
                    f"Updated detection {detection.id}: age={prediction['age']}, "
                    f"gender={prediction['gender']} ({prediction['gender_confidence']:.2f})"
                )

            logger.info(
                f"Updated age/gender for {updated}/{len(detections)} detections in photo {photo_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Age/gender estimation failed for {file_path}: {e}")
            return False
