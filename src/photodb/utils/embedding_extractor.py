"""
Face embedding extraction using InsightFace's ArcFace (buffalo_l) model.

Provides 512-dimensional face embeddings for face clustering and recognition.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from insightface.model_zoo import get_model
from PIL import Image

logger = logging.getLogger(__name__)

# Default model directory (InsightFace convention)
DEFAULT_MODEL_ROOT = os.path.expanduser("~/.insightface/models")


def _get_providers() -> List[str]:
    """
    Get the best available ONNX Runtime execution providers in priority order.

    Priority: CoreML (macOS) > CUDA > CPU

    Returns:
        List of provider names in priority order.
    """
    available = ort.get_available_providers()
    providers = []

    # CoreML is preferred on macOS for Neural Engine acceleration
    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")

    # CUDA for GPU acceleration on Linux/Windows
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    # CPU as fallback (always available)
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    # Fallback if somehow none of the above
    if not providers:
        providers = available if available else ["CPUExecutionProvider"]

    return providers


def _ensure_model_available(model_name: str, model_root: str) -> str:
    """
    Ensure the recognition model is available, downloading if necessary.

    Args:
        model_name: InsightFace model name (e.g., "buffalo_l")
        model_root: Root directory for models

    Returns:
        Path to the recognition model ONNX file
    """
    model_dir = Path(model_root) / model_name

    # Check for common recognition model names
    recognition_models = [
        "w600k_r50.onnx",  # buffalo_l
        "w600k_mbf.onnx",  # buffalo_s
    ]

    for model_file in recognition_models:
        model_path = model_dir / model_file
        if model_path.exists():
            return str(model_path)

    # Model not found, try to download it
    # This will download the full model pack
    from insightface.utils.storage import ensure_available

    ensure_available("models", model_name, root=model_root)

    # Check again after download
    for model_file in recognition_models:
        model_path = model_dir / model_file
        if model_path.exists():
            return str(model_path)

    raise FileNotFoundError(f"Could not find recognition model for {model_name}")


class EmbeddingExtractor:
    """
    Extract 512-dimensional face embeddings using InsightFace's ArcFace model.

    This class uses InsightFace's ArcFace recognition model directly, since
    face detection is handled externally by YOLO. This avoids the FaceAnalysis
    requirement for a detection model.

    Example:
        extractor = EmbeddingExtractor()
        embedding = extractor.extract(pil_image, bbox)
        if embedding:
            # Use 512-dim embedding for clustering/recognition
            pass
    """

    # ArcFace standard input size
    INPUT_SIZE = (112, 112)

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        model_root: Optional[str] = None,
    ):
        """
        Initialize the embedding extractor with InsightFace ArcFace model.

        Args:
            model_name: InsightFace model name. Default "buffalo_l" uses ArcFace
                       with ResNet100 backbone for best accuracy.
            det_size: Detection size tuple (width, height). Default (640, 640).
                     This parameter is kept for API compatibility but not used
                     since we use external YOLO detection.
            model_root: Root directory for InsightFace models. Default uses
                       ~/.insightface/models
        """
        self.model_name = model_name
        self.det_size = det_size
        self.model_root = model_root or DEFAULT_MODEL_ROOT

        # Get optimal providers for current platform
        self.providers = _get_providers()
        logger.info(f"EmbeddingExtractor using ONNX providers: {self.providers}")

        # Find and load the recognition model
        model_path = _ensure_model_available(model_name, self.model_root)
        logger.info(f"Loading ArcFace model from: {model_path}")

        # Load the ArcFace model directly using model_zoo
        self.model = get_model(model_path, providers=self.providers)
        self.model.prepare(ctx_id=0)

    def extract(
        self,
        image: Image.Image,
        bbox: dict,
        padding: float = 0.2,
    ) -> Optional[List[float]]:
        """
        Extract face embedding from a bounding box region in an image.

        Adds padding around the bounding box before cropping to ensure the full
        face is captured for better embedding quality.

        Args:
            image: PIL Image in RGB format.
            bbox: Bounding box dict with x1, y1, x2, y2 keys.
            padding: Padding ratio to add around bbox. Default 0.2 (20%).

        Returns:
            512-dimensional embedding as list of floats, or None if extraction fails.
        """
        try:
            # Get bbox coordinates
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])

            # Calculate padding
            width = x2 - x1
            height = y2 - y1
            pad_x = width * padding
            pad_y = height * padding

            # Apply padding and clamp to image bounds
            img_width, img_height = image.size
            x1_padded = max(0, int(x1 - pad_x))
            y1_padded = max(0, int(y1 - pad_y))
            x2_padded = min(img_width, int(x2 + pad_x))
            y2_padded = min(img_height, int(y2 + pad_y))

            # Crop the padded region
            crop = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))

            # Convert PIL RGB to numpy BGR (InsightFace expects BGR)
            rgb_array = np.array(crop)
            bgr_array = rgb_array[:, :, ::-1].copy()

            # Resize to model input size (112x112)
            face_img = cv2.resize(bgr_array, self.INPUT_SIZE)

            # Extract embedding using get_feat (expects list of images)
            embeddings = self.model.get_feat([face_img])

            if embeddings is None or len(embeddings) == 0:
                logger.debug("Failed to extract embedding from crop")
                return None

            # Return embedding as list of floats
            embedding = embeddings[0]
            return [float(x) for x in embedding]

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None

    def extract_from_aligned(
        self,
        aligned_face: np.ndarray,
    ) -> Optional[List[float]]:
        """
        Extract face embedding from a pre-aligned face image.

        Args:
            aligned_face: Numpy array of aligned face in BGR format.
                         Should be 112x112 for ArcFace models.

        Returns:
            512-dimensional embedding as list of floats, or None if extraction fails.
        """
        try:
            # Ensure correct size
            if aligned_face.shape[:2] != self.INPUT_SIZE:
                aligned_face = cv2.resize(aligned_face, self.INPUT_SIZE)

            # Extract embedding using get_feat (expects list of images)
            embeddings = self.model.get_feat([aligned_face])

            if embeddings is None or len(embeddings) == 0:
                logger.debug("Failed to extract embedding from aligned face")
                return None

            # Return embedding as list of floats
            embedding = embeddings[0]
            return [float(x) for x in embedding]

        except Exception as e:
            logger.error(f"Failed to extract embedding from aligned face: {e}")
            return None
