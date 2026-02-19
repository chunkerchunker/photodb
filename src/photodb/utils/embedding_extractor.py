"""
Face embedding extraction using ArcFace (buffalo_l) via ONNX Runtime.

Provides 512-dimensional face embeddings for face clustering and recognition.
Uses direct ONNX Runtime inference instead of the InsightFace library.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from .. import config as defaults

logger = logging.getLogger(__name__)


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
    Ensure the recognition model ONNX file is available on disk.

    Args:
        model_name: Model pack name (e.g., "buffalo_l")
        model_root: Root directory for models

    Returns:
        Path to the recognition model ONNX file

    Raises:
        FileNotFoundError: If the model file is not found.
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

    raise FileNotFoundError(
        f"Could not find recognition model for {model_name} in {model_dir}. "
        f"Expected one of: {recognition_models}. "
        f"Run ./scripts/download_models.sh to download required models."
    )


class EmbeddingExtractor:
    """
    Extract 512-dimensional face embeddings using ArcFace via ONNX Runtime.

    This class loads the ArcFace recognition ONNX model directly and runs
    inference through ONNX Runtime, without depending on the InsightFace
    library. Face detection is handled externally by YOLO.

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
        model_name: Optional[str] = None,
        model_root: Optional[str] = None,
    ):
        """
        Initialize the embedding extractor with ArcFace ONNX model.

        Args:
            model_name: Model pack name. Default "buffalo_l" uses ArcFace
                       with ResNet50 backbone. Can be overridden with
                       EMBEDDING_MODEL_NAME environment variable.
            model_root: Root directory for models. Default uses
                       ~/.insightface/models. Can be overridden with
                       EMBEDDING_MODEL_ROOT environment variable.
        """
        self.model_name = model_name or defaults.EMBEDDING_MODEL_NAME
        self.model_root = model_root or defaults.EMBEDDING_MODEL_ROOT

        # Get optimal providers for current platform
        self.providers = _get_providers()
        logger.info(f"EmbeddingExtractor using ONNX providers: {self.providers}")

        # Find and load the recognition model
        model_path = _ensure_model_available(self.model_name, self.model_root)
        logger.info(f"Loading ArcFace model from: {model_path}")

        # Create ONNX Runtime session directly
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Normalization parameters for buffalo_l ArcFace (w600k_r50):
        # The ONNX graph uses Sub(0.0) and Div(1.0), making this a no-op.
        # Kept explicit for correctness and compatibility with other model packs.
        self.input_mean = 0.0
        self.input_std = 1.0

    def _preprocess(self, bgr_array: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR face image for ArcFace inference.

        Replicates the preprocessing from InsightFace's ArcFaceONNX.get_feat():
        BGR->RGB swap, HWC->CHW transpose, float32 cast, normalize.

        Args:
            bgr_array: (H, W, 3) uint8 BGR array, already 112x112.

        Returns:
            (1, 3, H, W) float32 array ready for ONNX inference.
        """
        # BGR -> RGB
        rgb = bgr_array[:, :, ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # float32 + normalize
        blob = (chw.astype(np.float32) - self.input_mean) / self.input_std
        # Add batch dimension
        return blob[np.newaxis]

    def get_feat(self, imgs: list) -> np.ndarray:
        """
        Extract embeddings from a list of pre-sized BGR face images.

        Replaces InsightFace model.get_feat() with direct ONNX Runtime inference.

        Args:
            imgs: List of (112, 112, 3) uint8 BGR numpy arrays.

        Returns:
            (N, 512) float32 numpy array of face embeddings.
        """
        blobs = np.concatenate([self._preprocess(img) for img in imgs], axis=0)
        return self.session.run(self.output_names, {self.input_name: blobs})[0]

    def extract(
        self,
        image: Image.Image,
        bbox: dict,
        padding: float = defaults.FACE_CROP_PADDING,
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

            # Crop the padded region and resize to model input size
            crop = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
            crop = crop.resize(self.INPUT_SIZE, Image.LANCZOS)

            # Convert PIL RGB to numpy BGR (ArcFace expects BGR input)
            rgb_array = np.array(crop)
            bgr_array = rgb_array[:, :, ::-1].copy()

            # Extract embedding
            embeddings = self.get_feat([bgr_array])

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
                # BGR -> RGB for PIL
                rgb = aligned_face[:, :, ::-1]
                pil_img = Image.fromarray(rgb)
                pil_img = pil_img.resize(self.INPUT_SIZE, Image.LANCZOS)
                rgb_resized = np.array(pil_img)
                aligned_face = rgb_resized[:, :, ::-1].copy()  # back to BGR

            # Extract embedding
            embeddings = self.get_feat([aligned_face])

            if embeddings is None or len(embeddings) == 0:
                logger.debug("Failed to extract embedding from aligned face")
                return None

            # Return embedding as list of floats
            embedding = embeddings[0]
            return [float(x) for x in embedding]

        except Exception as e:
            logger.error(f"Failed to extract embedding from aligned face: {e}")
            return None
