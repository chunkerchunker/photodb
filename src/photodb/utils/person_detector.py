"""
Person detection using YOLO for face+body detection and InsightFace for embeddings.

Supports CoreML on macOS for faster inference via Neural Engine.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

if sys.platform == "darwin":
    import objc

from ultralytics import YOLO

from .embedding_extractor import EmbeddingExtractor
from .. import config as defaults

logger = logging.getLogger(__name__)


def _load_yolo_model(model_path: str, task: Optional[str] = None) -> YOLO:
    """
    Load YOLO model with PyTorch 2.6+ compatibility.

    PyTorch 2.6 changed weights_only default to True, breaking older ultralytics.
    We temporarily patch torch.load for model loading since we trust models from
    official sources (HuggingFace/Ultralytics).

    Args:
        model_path: Path to the model file (.pt or .mlpackage)
        task: Task type for CoreML models (e.g., 'detect')
    """
    original_torch_load = torch.load

    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_load
        if task:
            return YOLO(model_path, task=task)
        return YOLO(model_path)
    finally:
        torch.load = original_torch_load


def _get_coreml_path(pt_model_path: str) -> Optional[str]:
    """
    Get the CoreML model path corresponding to a PyTorch model.

    Args:
        pt_model_path: Path to PyTorch model (.pt file)

    Returns:
        Path to CoreML model if it exists, None otherwise
    """
    pt_path = Path(pt_model_path)
    coreml_path = pt_path.with_suffix(".mlpackage")
    if coreml_path.exists():
        return str(coreml_path)
    return None


class PersonDetector:
    """Detect faces and bodies in images using YOLO, extract face embeddings with InsightFace.

    On macOS, automatically uses CoreML (.mlpackage) if available for 5x faster
    inference via the Neural Engine. CoreML is also thread-safe, unlike MPS.
    """

    # Class IDs from yolov8x_person_face model
    FACE_CLASS_ID = 1
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        force_cpu: bool = False,
        prefer_coreml: bool = defaults.DETECTION_PREFER_COREML,
    ):
        """
        Initialize person detection and embedding models.

        Args:
            model_path: Path to YOLO model. If None, uses DETECTION_MODEL_PATH env var.
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detects if not specified.
            force_cpu: Force CPU-only mode for PyTorch models.
            prefer_coreml: On macOS, prefer CoreML (.mlpackage) if available.
                          CoreML is faster (5x) and thread-safe. Default True.
        """

        # Get model path from env or parameter
        if model_path is None:
            model_path = defaults.DETECTION_MODEL_PATH

        # Check for CoreML model on macOS
        self.using_coreml = False
        coreml_path = None
        if prefer_coreml and sys.platform == "darwin":
            coreml_path = _get_coreml_path(model_path)
            if coreml_path:
                logger.info(f"Using CoreML model: {coreml_path}")
                self.using_coreml = True

        # Determine device for PyTorch operations (YOLO if not using CoreML)
        if force_cpu:
            self.device = "cpu"
        elif device is not None:
            self.device = device
        else:
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.min_confidence = defaults.DETECTION_MIN_CONFIDENCE

        # Load YOLO model
        if self.using_coreml:
            # CoreML model - task must be specified
            assert coreml_path is not None  # Guaranteed by using_coreml check above
            self.model = _load_yolo_model(coreml_path, task="detect")
            # CoreML handles device selection internally (ANE > GPU > CPU)
            self._yolo_device = None
            logger.info("PersonDetector using CoreML (Neural Engine)")
        else:
            # PyTorch model
            self.model = _load_yolo_model(model_path)
            self._yolo_device = self.device
            logger.info(f"PersonDetector using PyTorch on {self.device}")

        # Load InsightFace embedding model (ONNX-based)
        # Uses CoreML on macOS, CUDA on Linux, with CPU fallback
        self.embedding_extractor = EmbeddingExtractor()

    def warmup(self) -> None:
        """Run dummy inference to trigger lazy model compilation (CoreML/ONNX)."""
        dummy = Image.new("RGB", (640, 640), color=(128, 128, 128))
        try:
            if self.using_coreml:
                self.model(dummy, conf=self.min_confidence, verbose=False)
            else:
                self.model(dummy, conf=self.min_confidence, device=self._yolo_device, verbose=False)
            # Warmup embedding extractor with a small face-sized image
            dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
            self.embedding_extractor.model.get_feat([dummy_face])  # type: ignore[union-attr]
            logger.info("PersonDetector warmup complete")
        except Exception as e:
            logger.warning(f"PersonDetector warmup failed (non-fatal): {e}")
        finally:
            dummy.close()

    def run_yolo(self, images: list) -> list:
        """Run YOLO detection on a batch of PIL images.

        This is the inference function used by BatchCoordinator for batched detection.
        Runs on the BatchCoordinator's background thread, so we need an autorelease pool
        on macOS to drain CoreML ObjC objects (IOSurfaces, CIImages).

        Args:
            images: List of PIL.Image.Image objects.

        Returns:
            List of ultralytics Results objects, one per input image.
        """
        if sys.platform == "darwin":
            with objc.autorelease_pool():
                return self._run_yolo_impl(images)
        else:
            return self._run_yolo_impl(images)

    def _run_yolo_impl(self, images: list) -> list:
        """Internal YOLO batch inference."""
        if self.using_coreml:
            return self.model(images, conf=self.min_confidence, verbose=False)
        else:
            return self.model(
                images, conf=self.min_confidence, device=self._yolo_device, verbose=False
            )

    def parse_yolo_result(self, img: Image.Image, yolo_result) -> Dict[str, Any]:
        """Parse a YOLO detection result and extract embeddings.

        Takes a pre-computed YOLO Result (from run_yolo batch) and processes it
        the same way as _detect_impl: parse detections, match faces to bodies,
        extract face embeddings.

        Called from worker threads after batch YOLO inference. On macOS, wraps in
        autorelease pool for InsightFace ONNX/CoreML embedding extraction.

        Args:
            img: PIL Image (must be open and RGB).
            yolo_result: A single ultralytics Results object.

        Returns:
            Dict matching the format of detect(): {status, detections, image_dimensions}
        """
        if sys.platform == "darwin":
            with objc.autorelease_pool():
                return self._parse_yolo_result_impl(img, yolo_result)
        else:
            return self._parse_yolo_result_impl(img, yolo_result)

    def _parse_yolo_result_impl(self, img: Image.Image, yolo_result) -> Dict[str, Any]:
        """Internal parse implementation."""
        try:
            img_width, img_height = img.size

            faces = []
            bodies = []

            if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
                for box in yolo_result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()

                    detection = {
                        "bbox": {
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                            "width": float(xyxy[2] - xyxy[0]),
                            "height": float(xyxy[3] - xyxy[1]),
                        },
                        "confidence": conf,
                        "class_id": cls_id,
                    }

                    if cls_id == self.FACE_CLASS_ID:
                        faces.append(detection)
                    elif cls_id == self.PERSON_CLASS_ID:
                        bodies.append(detection)

            if not faces and not bodies:
                return {
                    "status": "no_detections",
                    "detections": [],
                    "image_dimensions": {"width": img_width, "height": img_height},
                }

            matched_detections = self._match_faces_to_bodies(faces, bodies)

            # Extract embeddings for faces
            for detection in matched_detections:
                if detection["face"] is not None:
                    try:
                        embedding = self.extract_embedding(img, detection["face"]["bbox"])
                        detection["face"]["embedding"] = embedding
                        detection["face"]["embedding_norm"] = float(np.linalg.norm(embedding))
                    except Exception:
                        detection["face"]["embedding"] = None
                        detection["face"]["embedding_norm"] = None

            return {
                "status": "success",
                "detections": matched_detections,
                "image_dimensions": {"width": img_width, "height": img_height},
            }

        except Exception as e:
            return {
                "status": "error",
                "detections": [],
                "image_dimensions": {"width": 0, "height": 0},
                "error": str(e),
            }

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces and bodies in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary with:
                - status: 'success', 'no_detections', or 'error'
                - detections: List of person detections with face/body info
                - image_dimensions: Dict with width and height
                - error: Error message (only if status is 'error')
        """
        # Wrap in autorelease pool on macOS to drain CoreML/ONNX Runtime ObjC objects
        # (IOSurfaces, CIImages) after each call instead of accumulating in thread pools.
        if sys.platform == "darwin":
            with objc.autorelease_pool():
                return self._detect_impl(image_path)
        else:
            return self._detect_impl(image_path)

    def _detect_impl(self, image_path: str) -> Dict[str, Any]:
        """Internal detect implementation."""
        img = None
        try:
            # Load image
            img = Image.open(image_path)

            # Convert RGBA to RGB if necessary
            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            img_width, img_height = img.size

            # Run YOLO detection
            if self.using_coreml:
                # CoreML - don't pass device parameter
                results = self.model(
                    img,
                    conf=self.min_confidence,
                    verbose=False,
                )
            else:
                # PyTorch - pass device parameter
                results = self.model(
                    img,
                    conf=self.min_confidence,
                    device=self._yolo_device,
                    verbose=False,
                )

            # Parse detections
            faces: List[Dict[str, Any]] = []
            bodies: List[Dict[str, Any]] = []

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy()

                        detection = {
                            "bbox": {
                                "x1": float(xyxy[0]),
                                "y1": float(xyxy[1]),
                                "x2": float(xyxy[2]),
                                "y2": float(xyxy[3]),
                                "width": float(xyxy[2] - xyxy[0]),
                                "height": float(xyxy[3] - xyxy[1]),
                            },
                            "confidence": conf,
                            "class_id": cls_id,
                        }

                        if cls_id == self.FACE_CLASS_ID:
                            faces.append(detection)
                        elif cls_id == self.PERSON_CLASS_ID:
                            bodies.append(detection)

            # Match faces to bodies
            if not faces and not bodies:
                return {
                    "status": "no_detections",
                    "detections": [],
                    "image_dimensions": {"width": img_width, "height": img_height},
                }

            matched_detections = self._match_faces_to_bodies(faces, bodies)

            # Extract embeddings for faces
            for detection in matched_detections:
                if detection["face"] is not None:
                    try:
                        embedding = self.extract_embedding(img, detection["face"]["bbox"])
                        detection["face"]["embedding"] = embedding
                        detection["face"]["embedding_norm"] = float(np.linalg.norm(embedding))
                    except Exception:
                        # If embedding extraction fails, continue without embedding
                        detection["face"]["embedding"] = None
                        detection["face"]["embedding_norm"] = None

            return {
                "status": "success",
                "detections": matched_detections,
                "image_dimensions": {"width": img_width, "height": img_height},
            }

        except FileNotFoundError:
            return {
                "status": "error",
                "detections": [],
                "image_dimensions": {"width": 0, "height": 0},
                "error": f"File not found: {image_path}",
            }
        except Exception as e:
            return {
                "status": "error",
                "detections": [],
                "image_dimensions": {"width": 0, "height": 0},
                "error": str(e),
            }
        finally:
            if img is not None:
                img.close()

    def _match_faces_to_bodies(
        self, faces: List[Dict[str, Any]], bodies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Match each face to its containing body based on spatial containment.

        Args:
            faces: List of face detections with bbox.
            bodies: List of body detections with bbox.

        Returns:
            List of matched detections with {'face': face_data|None, 'body': body_data|None}
        """
        matched: List[Dict[str, Any]] = []
        used_bodies: set = set()

        # Match each face to best containing body
        for face in faces:
            best_body = None
            best_containment = 0.0

            for i, body in enumerate(bodies):
                if i in used_bodies:
                    continue

                containment = self._compute_containment(face["bbox"], body["bbox"])
                if containment > best_containment:
                    best_containment = containment
                    best_body = (i, body)

            if best_body is not None and best_containment > 0.3:  # Minimum 30% overlap
                used_bodies.add(best_body[0])
                matched.append({"face": face, "body": best_body[1]})
            else:
                matched.append({"face": face, "body": None})

        # Add unmatched bodies
        for i, body in enumerate(bodies):
            if i not in used_bodies:
                matched.append({"face": None, "body": body})

        return matched

    def _compute_containment(self, face_bbox: Dict[str, Any], body_bbox: Dict[str, Any]) -> float:
        """
        Compute how much of the face bbox is contained within the body bbox.

        Args:
            face_bbox: Face bounding box with x1, y1, x2, y2.
            body_bbox: Body bounding box with x1, y1, x2, y2.

        Returns:
            Containment ratio (0.0 to 1.0), where 1.0 means fully contained.
        """
        # Calculate intersection
        x1 = max(face_bbox["x1"], body_bbox["x1"])
        y1 = max(face_bbox["y1"], body_bbox["y1"])
        x2 = min(face_bbox["x2"], body_bbox["x2"])
        y2 = min(face_bbox["y2"], body_bbox["y2"])

        # No intersection
        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)
        face_area = (face_bbox["x2"] - face_bbox["x1"]) * (face_bbox["y2"] - face_bbox["y1"])

        if face_area == 0:
            return 0.0

        return intersection_area / face_area

    def extract_embedding(self, image: Image.Image, bbox: Dict[str, Any]) -> List[float]:
        """
        Extract face embedding from a cropped face region.

        Args:
            image: PIL Image to extract face from.
            bbox: Bounding box with x1, y1, x2, y2.

        Returns:
            512-dimensional face embedding as list of floats.

        Raises:
            ValueError: If embedding extraction fails.
        """
        embedding = self.embedding_extractor.extract(image, bbox)
        if embedding is None:
            raise ValueError("Failed to extract face embedding")
        return embedding
