"""
CLIP analyzer for image and text encoding.

Provides efficient encoding for zero-shot classification using prompt embeddings.
Uses Apple's MobileCLIP-S2 model which produces 512-dimensional embeddings with
better efficiency than larger ViT models.

MobileCLIP-S2 uses FastViT backbone which requires timm>=0.9.16. We use a
compatibility shim (timm_compat.py) to make MiVOLO work with newer timm versions.
"""

import logging
import os
import threading
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Model configuration - can be changed via environment variables
DEFAULT_MODEL_NAME = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"

# Lazy-loaded model
_model = None
_preprocess = None
_tokenizer = None
_device = None
_model_name = None
_model_lock = threading.Lock()


def _load_model():
    """Lazy-load CLIP model."""
    global _model, _preprocess, _tokenizer, _device, _model_name

    # Fast path without lock
    if _model is not None:
        return _model, _preprocess, _tokenizer, _device

    with _model_lock:
        # Double-check inside lock
        if _model is not None:
            return _model, _preprocess, _tokenizer, _device

        import open_clip

        model_name = os.environ.get("CLIP_MODEL_NAME", DEFAULT_MODEL_NAME)
        pretrained = os.environ.get("CLIP_PRETRAINED", DEFAULT_PRETRAINED)

        logger.info(f"Loading CLIP model {model_name} (pretrained={pretrained})...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _tokenizer = open_clip.get_tokenizer(model_name)
        _model.eval()
        _model_name = model_name

        # Select device
        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"

        _model = _model.to(_device)
        logger.info(f"CLIP model {model_name} loaded on {_device}")

    return _model, _preprocess, _tokenizer, _device


class MobileCLIPAnalyzer:
    """CLIP image and text encoder for zero-shot classification.

    Uses OpenAI's ViT-B-32 CLIP model by default. Model can be configured
    via CLIP_MODEL_NAME and CLIP_PRETRAINED environment variables.

    Usage:
        analyzer = MobileCLIPAnalyzer()
        image_emb = analyzer.encode_image("photo.jpg")
        text_emb = analyzer.encode_text("a happy scene")
    """

    EMBEDDING_DIM = 512

    def __init__(self):
        """Initialize analyzer (model loaded lazily on first use)."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is None:
            self._model, self._preprocess, self._tokenizer, self._device = _load_model()

    @property
    def device(self) -> str:
        """Get the device being used."""
        self._ensure_loaded()
        return self._device

    @property
    def model_name(self) -> str:
        """Get the model name being used."""
        self._ensure_loaded()
        return _model_name

    def encode_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Encode an image file to embedding.

        Args:
            image_path: Path to image file.

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        self._ensure_loaded()

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding

    def encode_face(
        self,
        face_image: Image.Image,
    ) -> torch.Tensor:
        """
        Encode a face crop to embedding.

        Args:
            face_image: PIL Image of cropped face.

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        self._ensure_loaded()

        if face_image.mode != "RGB":
            face_image = face_image.convert("RGB")

        image_tensor = self._preprocess(face_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding

    def encode_face_from_bbox(
        self,
        image_path: Union[str, Path],
        bbox: dict,
    ) -> torch.Tensor:
        """
        Crop face from image and encode.

        Args:
            image_path: Path to full image.
            bbox: Bounding box with x1, y1, x2, y2.

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        image = Image.open(image_path).convert("RGB")

        x1 = max(0, int(bbox["x1"]))
        y1 = max(0, int(bbox["y1"]))
        x2 = min(image.width, int(bbox["x2"]))
        y2 = min(image.height, int(bbox["y2"]))

        face_crop = image.crop((x1, y1, x2, y2))
        return self.encode_face(face_crop)

    def encode_faces_batch(
        self,
        image_path: Union[str, Path],
        bboxes: List[dict],
    ) -> torch.Tensor:
        """
        Batch encode multiple face crops from one image.

        Args:
            image_path: Path to full image.
            bboxes: List of bounding boxes.

        Returns:
            Normalized embeddings tensor of shape (N, 512).
        """
        if not bboxes:
            return torch.empty(0, self.EMBEDDING_DIM)

        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")

        face_tensors = []
        for bbox in bboxes:
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(image.width, int(bbox["x2"]))
            y2 = min(image.height, int(bbox["y2"]))
            face_crop = image.crop((x1, y1, x2, y2))
            face_tensors.append(self._preprocess(face_crop))

        batch = torch.stack(face_tensors).to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to embedding.

        Args:
            text: Text string.

        Returns:
            Normalized embedding tensor of shape (1, 512).
        """
        self._ensure_loaded()

        tokens = self._tokenizer([text]).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Batch encode multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            Normalized embeddings tensor of shape (N, 512).
        """
        if not texts:
            return torch.empty(0, self.EMBEDDING_DIM)

        self._ensure_loaded()

        tokens = self._tokenizer(texts).to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings
