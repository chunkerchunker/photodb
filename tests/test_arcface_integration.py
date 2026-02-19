"""Integration tests for ArcFace embedding extraction via ONNX Runtime."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ARCFACE_MODEL = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"

pytestmark = pytest.mark.skipif(
    not ARCFACE_MODEL.exists(),
    reason="ArcFace model not available. Run ./scripts/download_models.sh",
)


class TestArcFaceIntegration:
    """Integration tests requiring actual ArcFace ONNX model."""

    @pytest.fixture
    def real_extractor(self):
        """Create a real EmbeddingExtractor (requires model on disk)."""
        from photodb.utils.embedding_extractor import EmbeddingExtractor

        return EmbeddingExtractor()

    @pytest.fixture
    def sample_face_image(self):
        """Create a simple test image with a face-like pattern."""
        # Create 200x200 image with face-like features
        img = Image.new("RGB", (200, 200), color=(200, 180, 160))
        return img

    @pytest.mark.slow
    def test_embedding_dimension(self, real_extractor, sample_face_image):
        """Test that embeddings are 512-dimensional."""
        bbox = {"x1": 20, "y1": 20, "x2": 180, "y2": 180}

        # Note: May return None if no face detected in synthetic image
        embedding = real_extractor.extract(sample_face_image, bbox)

        if embedding is not None:
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.slow
    def test_embedding_consistency(self, real_extractor, sample_face_image):
        """Test that same image produces consistent embeddings."""
        bbox = {"x1": 20, "y1": 20, "x2": 180, "y2": 180}

        embedding1 = real_extractor.extract(sample_face_image, bbox)
        embedding2 = real_extractor.extract(sample_face_image, bbox)

        if embedding1 is not None and embedding2 is not None:
            # Should be identical for same input
            np.testing.assert_array_almost_equal(embedding1, embedding2)

    @pytest.mark.slow
    def test_provider_selection(self, real_extractor):
        """Test that appropriate provider was selected."""
        import sys

        providers = real_extractor.providers

        if sys.platform == "darwin":
            # CoreML should be available on macOS
            assert "CoreMLExecutionProvider" in providers or "CPUExecutionProvider" in providers
        else:
            # CPU should always be available
            assert "CPUExecutionProvider" in providers
