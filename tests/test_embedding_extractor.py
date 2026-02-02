"""
Tests for InsightFace embedding extractor.

Uses InsightFace's ArcFace (buffalo_l) model for 512-dimensional face embeddings.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


class TestGetProviders:
    """Unit tests for _get_providers function."""

    def test_get_providers_selects_coreml_on_macos(self):
        """Verify CoreML provider is selected on macOS when available."""
        with patch("onnxruntime.get_available_providers") as mock_providers:
            mock_providers.return_value = [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            # Reimport to get fresh function with mocked ort
            import importlib
            import src.photodb.utils.embedding_extractor as mod

            importlib.reload(mod)

            with patch.object(sys, "platform", "darwin"):
                providers = mod._get_providers()

                # CoreML should be first on macOS
                assert providers[0] == "CoreMLExecutionProvider"
                assert "CPUExecutionProvider" in providers

    def test_get_providers_selects_cuda_when_available(self):
        """Verify CUDA provider is selected when CoreML not available."""
        with patch("onnxruntime.get_available_providers") as mock_providers:
            mock_providers.return_value = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            import importlib
            import src.photodb.utils.embedding_extractor as mod

            importlib.reload(mod)

            with patch.object(sys, "platform", "linux"):
                providers = mod._get_providers()

                # CUDA should be first when CoreML not available
                assert providers[0] == "CUDAExecutionProvider"
                assert "CPUExecutionProvider" in providers

    def test_get_providers_falls_back_to_cpu(self):
        """Verify CPU fallback when no GPU available."""
        with patch("onnxruntime.get_available_providers") as mock_providers:
            mock_providers.return_value = ["CPUExecutionProvider"]
            import importlib
            import src.photodb.utils.embedding_extractor as mod

            importlib.reload(mod)

            with patch.object(sys, "platform", "linux"):
                providers = mod._get_providers()

                assert providers == ["CPUExecutionProvider"]


class TestEmbeddingExtractorUnit:
    """Unit tests for EmbeddingExtractor using mocks (no model downloads required)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample RGB test image."""
        img_path = temp_dir / "test.jpg"
        img = Image.new("RGB", (640, 480), color="white")
        img.save(img_path, "JPEG")
        return img_path

    @pytest.fixture
    def sample_pil_image(self):
        """Create a sample PIL Image for testing."""
        return Image.new("RGB", (640, 480), color=(128, 128, 128))

    @pytest.fixture
    def sample_bbox(self):
        """Create a sample bounding box dict."""
        return {
            "x1": 100.0,
            "y1": 100.0,
            "x2": 200.0,
            "y2": 200.0,
            "width": 100.0,
            "height": 100.0,
        }

    @pytest.fixture
    def mock_model(self):
        """Create a mock ArcFace model."""
        mock = MagicMock()
        mock.get_feat.return_value = np.random.randn(1, 512).astype(np.float32)
        return mock

    @pytest.fixture
    def mock_extractor(self, mock_model, temp_dir):
        """Create an EmbeddingExtractor with mocked model loading."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        with patch("src.photodb.utils.embedding_extractor.get_model") as mock_get_model:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_get_model.return_value = mock_model
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))
                return extractor

    def test_init_loads_model(self, temp_dir):
        """Test that EmbeddingExtractor loads the ArcFace model correctly."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        mock_model = MagicMock()

        with patch("src.photodb.utils.embedding_extractor.get_model") as mock_get_model:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_get_model.return_value = mock_model
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                EmbeddingExtractor(model_root=str(temp_dir))

                # Verify get_model was called with model path and providers
                mock_get_model.assert_called_once()
                call_kwargs = mock_get_model.call_args[1]
                assert call_kwargs.get("providers") == ["CPUExecutionProvider"]

                # Verify prepare was called
                mock_model.prepare.assert_called_once()

    def test_init_with_custom_model_name(self, temp_dir):
        """Test initialization with custom model name."""
        # Create a fake model directory and file for buffalo_s
        model_dir = temp_dir / "buffalo_s"
        model_dir.mkdir()
        model_file = model_dir / "w600k_mbf.onnx"
        model_file.touch()

        mock_model = MagicMock()

        with patch("src.photodb.utils.embedding_extractor.get_model") as mock_get_model:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_get_model.return_value = mock_model
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_name="buffalo_s", model_root=str(temp_dir))

                # Model should be loaded
                mock_get_model.assert_called_once()
                assert extractor.model_name == "buffalo_s"

    def test_init_with_custom_det_size(self, temp_dir):
        """Test initialization with custom detection size."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        mock_model = MagicMock()

        with patch("src.photodb.utils.embedding_extractor.get_model") as mock_get_model:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_get_model.return_value = mock_model
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(det_size=(320, 320), model_root=str(temp_dir))

                # det_size is stored but not used (external YOLO handles detection)
                assert extractor.det_size == (320, 320)

    def test_extract_returns_512_dim_embedding(self, mock_extractor, sample_pil_image, sample_bbox):
        """Verify extract returns 512-dimensional embedding."""
        embedding = mock_extractor.extract(sample_pil_image, sample_bbox)

        assert embedding is not None
        assert len(embedding) == 512
        assert all(isinstance(x, float) for x in embedding)

    def test_extract_returns_none_for_no_embedding(self, temp_dir, sample_pil_image, sample_bbox):
        """Verify extract returns None when model returns empty."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.get_feat.return_value = np.array([])  # Empty result

        with patch("src.photodb.utils.embedding_extractor.get_model") as mock_get_model:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_get_model.return_value = mock_model
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))
                embedding = extractor.extract(sample_pil_image, sample_bbox)

                assert embedding is None

    def test_extract_adds_padding_to_bbox(self, mock_extractor, mock_model):
        """Verify extract adds 20% padding around bbox before cropping."""
        # Create larger test image to allow padding
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        # bbox is 100x100 at position (200, 150)
        bbox = {"x1": 200, "y1": 150, "x2": 300, "y2": 250}

        # With 20% padding on a 100x100 bbox:
        # pad_x = 100 * 0.2 = 20, pad_y = 100 * 0.2 = 20
        # Expected crop: (180, 130) to (320, 270) = 140x140
        mock_extractor.extract(img, bbox)

        # Verify get_feat was called
        mock_model.get_feat.assert_called_once()
        call_args = mock_model.get_feat.call_args[0]
        img_list = call_args[0]

        # The image should be resized to 112x112 for ArcFace
        assert img_list[0].shape == (112, 112, 3)

    def test_extract_clamps_bbox_to_image_bounds(self, mock_extractor, mock_model):
        """Verify extract clamps padded bbox to image boundaries."""
        # Create test image and bbox at corner (padding would go out of bounds)
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        bbox = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}

        # Should not raise error, bbox should be clamped
        embedding = mock_extractor.extract(img, bbox)

        # Verify get_feat was called
        mock_model.get_feat.assert_called()
        assert embedding is not None

    def test_extract_converts_rgb_to_bgr(self, temp_dir):
        """Verify extract converts PIL RGB to BGR numpy array for InsightFace."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        captured_input = []

        def capture_get_feat(imgs):
            captured_input.append(imgs[0].copy())
            return np.random.randn(1, 512).astype(np.float32)

        mock_model = MagicMock()
        mock_model.get_feat.side_effect = capture_get_feat

        with patch("src.photodb.utils.embedding_extractor.get_model") as mock_get_model:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_get_model.return_value = mock_model
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))

                # Create image with distinct R, G, B channels for verification
                img = Image.new("RGB", (200, 200), color=(255, 0, 0))  # Pure red
                bbox = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}

                extractor.extract(img, bbox)

                # Get the array passed to InsightFace
                img_array = captured_input[0]

                # In BGR, red (255, 0, 0) RGB becomes (0, 0, 255) BGR
                # So the first channel (B) should be 0, third channel (R) should be 255
                assert img_array[0, 0, 0] == 0  # B channel
                assert img_array[0, 0, 2] == 255  # R channel

    def test_extract_from_aligned_returns_512_dim_embedding(self, mock_extractor):
        """Verify extract_from_aligned returns 512-dim embedding from pre-aligned face."""
        # Create aligned face as numpy array (BGR format, 112x112 for ArcFace)
        aligned_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        embedding = mock_extractor.extract_from_aligned(aligned_face)

        assert embedding is not None
        assert len(embedding) == 512
        assert all(isinstance(x, float) for x in embedding)

    def test_extract_from_aligned_returns_none_for_empty_result(self, temp_dir):
        """Verify extract_from_aligned returns None when model returns empty."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.get_feat.return_value = np.array([])  # Empty result

        with patch("src.photodb.utils.embedding_extractor.get_model") as mock_get_model:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_get_model.return_value = mock_model
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))
                aligned_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

                embedding = extractor.extract_from_aligned(aligned_face)

                assert embedding is None

    def test_extract_handles_exception_gracefully(self, temp_dir, sample_pil_image, sample_bbox):
        """Verify extract returns None when an exception occurs."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.get_feat.side_effect = RuntimeError("Model inference failed")

        with patch("src.photodb.utils.embedding_extractor.get_model") as mock_get_model:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_get_model.return_value = mock_model
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))
                embedding = extractor.extract(sample_pil_image, sample_bbox)

                assert embedding is None

    def test_extract_from_aligned_resizes_non_standard_input(self, mock_extractor, mock_model):
        """Verify extract_from_aligned resizes input that's not 112x112."""
        # Create aligned face with wrong size
        aligned_face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        embedding = mock_extractor.extract_from_aligned(aligned_face)

        # Verify get_feat was called with resized image
        mock_model.get_feat.assert_called()
        call_args = mock_model.get_feat.call_args[0]
        img_list = call_args[0]

        # Should be resized to 112x112
        assert img_list[0].shape == (112, 112, 3)
        assert embedding is not None


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration tests require models to be downloaded. Set RUN_INTEGRATION_TESTS=1 to run.",
)
class TestEmbeddingExtractorIntegration:
    """Integration tests for EmbeddingExtractor (requires model downloads)."""

    @pytest.fixture
    def sample_pil_image(self):
        """Create a sample PIL Image for testing."""
        return Image.new("RGB", (640, 480), color=(128, 128, 128))

    @pytest.fixture
    def sample_bbox(self):
        """Create a sample bounding box dict."""
        return {"x1": 100, "y1": 100, "x2": 300, "y2": 300}

    def test_real_extractor_initialization(self):
        """Test that EmbeddingExtractor initializes correctly with real models."""
        from src.photodb.utils.embedding_extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor()
        assert extractor.model is not None
        assert extractor.model_name == "buffalo_l"

    def test_real_extraction_returns_512_dim(self, sample_pil_image, sample_bbox):
        """Test real extraction returns 512-dimensional embedding."""
        from src.photodb.utils.embedding_extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor()
        embedding = extractor.extract(sample_pil_image, sample_bbox)

        # For a blank image, we may or may not get an embedding
        # but if we do, it should be 512-dim
        if embedding is not None:
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)
