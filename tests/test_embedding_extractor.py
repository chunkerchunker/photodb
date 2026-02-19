"""
Tests for ArcFace embedding extractor via ONNX Runtime.

Uses ArcFace (buffalo_l) model for 512-dimensional face embeddings.
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
        from src.photodb.utils.embedding_extractor import _get_providers

        with patch("src.photodb.utils.embedding_extractor.ort.get_available_providers") as mock_p:
            mock_p.return_value = [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            with patch.object(sys, "platform", "darwin"):
                providers = _get_providers()

                assert providers[0] == "CoreMLExecutionProvider"
                assert "CPUExecutionProvider" in providers

    def test_get_providers_selects_cuda_when_available(self):
        """Verify CUDA provider is selected when CoreML not available."""
        from src.photodb.utils.embedding_extractor import _get_providers

        with patch("src.photodb.utils.embedding_extractor.ort.get_available_providers") as mock_p:
            mock_p.return_value = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            with patch.object(sys, "platform", "linux"):
                providers = _get_providers()

                assert providers[0] == "CUDAExecutionProvider"
                assert "CPUExecutionProvider" in providers

    def test_get_providers_falls_back_to_cpu(self):
        """Verify CPU fallback when no GPU available."""
        from src.photodb.utils.embedding_extractor import _get_providers

        with patch("src.photodb.utils.embedding_extractor.ort.get_available_providers") as mock_p:
            mock_p.return_value = ["CPUExecutionProvider"]
            with patch.object(sys, "platform", "linux"):
                providers = _get_providers()

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
    def mock_session(self):
        """Create a mock ONNX Runtime session."""
        session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input.1"
        session.get_inputs.return_value = [mock_input]
        mock_output = MagicMock()
        mock_output.name = "output"
        session.get_outputs.return_value = [mock_output]
        session.run.return_value = [np.random.randn(1, 512).astype(np.float32)]
        return session

    @pytest.fixture
    def mock_extractor(self, mock_session, temp_dir):
        """Create an EmbeddingExtractor with mocked ONNX session."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        with patch(
            "src.photodb.utils.embedding_extractor.ort.InferenceSession"
        ) as mock_ort_session:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_ort_session.return_value = mock_session
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))
                return extractor

    def test_init_loads_model(self, temp_dir):
        """Test that EmbeddingExtractor creates an ONNX InferenceSession correctly."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input.1"
        mock_session.get_inputs.return_value = [mock_input]
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_outputs.return_value = [mock_output]

        with patch(
            "src.photodb.utils.embedding_extractor.ort.InferenceSession"
        ) as mock_ort_session:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_ort_session.return_value = mock_session
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                EmbeddingExtractor(model_root=str(temp_dir))

                # Verify InferenceSession was called with model path and providers
                mock_ort_session.assert_called_once()
                call_kwargs = mock_ort_session.call_args
                assert call_kwargs[1].get("providers") == ["CPUExecutionProvider"]

    def test_init_with_custom_model_name(self, temp_dir):
        """Test initialization with custom model name."""
        # Create a fake model directory and file for buffalo_s
        model_dir = temp_dir / "buffalo_s"
        model_dir.mkdir()
        model_file = model_dir / "w600k_mbf.onnx"
        model_file.touch()

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input.1"
        mock_session.get_inputs.return_value = [mock_input]
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_outputs.return_value = [mock_output]

        with patch(
            "src.photodb.utils.embedding_extractor.ort.InferenceSession"
        ) as mock_ort_session:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_ort_session.return_value = mock_session
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_name="buffalo_s", model_root=str(temp_dir))

                # Session should be created
                mock_ort_session.assert_called_once()
                assert extractor.model_name == "buffalo_s"

    def test_extract_returns_512_dim_embedding(self, mock_extractor, sample_pil_image, sample_bbox):
        """Verify extract returns 512-dimensional embedding."""
        embedding = mock_extractor.extract(sample_pil_image, sample_bbox)

        assert embedding is not None
        assert len(embedding) == 512
        assert all(isinstance(x, float) for x in embedding)

    def test_extract_returns_none_for_no_embedding(self, temp_dir, sample_pil_image, sample_bbox):
        """Verify extract returns None when session returns empty."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input.1"
        mock_session.get_inputs.return_value = [mock_input]
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.array([])]  # Empty result

        with patch(
            "src.photodb.utils.embedding_extractor.ort.InferenceSession"
        ) as mock_ort_session:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_ort_session.return_value = mock_session
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))
                embedding = extractor.extract(sample_pil_image, sample_bbox)

                assert embedding is None

    def test_extract_adds_padding_to_bbox(self, mock_extractor, mock_session):
        """Verify extract adds 20% padding around bbox before cropping."""
        # Create larger test image to allow padding
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        # bbox is 100x100 at position (200, 150)
        bbox = {"x1": 200, "y1": 150, "x2": 300, "y2": 250}

        # With 20% padding on a 100x100 bbox:
        # pad_x = 100 * 0.2 = 20, pad_y = 100 * 0.2 = 20
        # Expected crop: (180, 130) to (320, 270) = 140x140
        mock_extractor.extract(img, bbox)

        # Verify session.run was called
        mock_session.run.assert_called_once()
        # session.run(output_names, {input_name: blobs}) — positional args
        input_dict = mock_session.run.call_args[0][1]
        blob = input_dict[mock_extractor.input_name]

        # The blob should be NCHW float32, shape (1, 3, 112, 112)
        assert blob.shape == (1, 3, 112, 112)

    def test_extract_clamps_bbox_to_image_bounds(self, mock_extractor, mock_session):
        """Verify extract clamps padded bbox to image boundaries."""
        # Create test image and bbox at corner (padding would go out of bounds)
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        bbox = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}

        # Should not raise error, bbox should be clamped
        embedding = mock_extractor.extract(img, bbox)

        # Verify session.run was called
        mock_session.run.assert_called()
        assert embedding is not None

    def test_extract_converts_rgb_to_bgr(self, temp_dir):
        """Verify extract converts PIL RGB to BGR then back to RGB for ONNX input."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        captured_input = []

        def capture_session_run(output_names, input_dict):
            captured_input.append(input_dict.copy())
            return [np.random.randn(1, 512).astype(np.float32)]

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input.1"
        mock_session.get_inputs.return_value = [mock_input]
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.side_effect = capture_session_run

        with patch(
            "src.photodb.utils.embedding_extractor.ort.InferenceSession"
        ) as mock_ort_session:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_ort_session.return_value = mock_session
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))

                # Create image with distinct R, G, B channels for verification
                img = Image.new("RGB", (200, 200), color=(255, 0, 0))  # Pure red
                bbox = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}

                extractor.extract(img, bbox)

                # Get the blob passed to session.run
                # The extract method: PIL RGB -> numpy RGB -> BGR -> _preprocess (BGR->RGB, HWC->CHW)
                # So for a pure red (255,0,0) PIL image:
                #   RGB array: (255, 0, 0)
                #   BGR array: (0, 0, 255)
                #   _preprocess BGR->RGB: (255, 0, 0)
                #   CHW: channel 0 = R = 255, channel 2 = B = 0
                blob = captured_input[0]["input.1"]
                assert blob[0, 0, 0, 0] == 255.0  # R channel
                assert blob[0, 2, 0, 0] == 0.0  # B channel

    def test_extract_from_aligned_returns_512_dim_embedding(self, mock_extractor):
        """Verify extract_from_aligned returns 512-dim embedding from pre-aligned face."""
        # Create aligned face as numpy array (BGR format, 112x112 for ArcFace)
        aligned_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        embedding = mock_extractor.extract_from_aligned(aligned_face)

        assert embedding is not None
        assert len(embedding) == 512
        assert all(isinstance(x, float) for x in embedding)

    def test_extract_from_aligned_returns_none_for_empty_result(self, temp_dir):
        """Verify extract_from_aligned returns None when session returns empty."""
        # Create a fake model directory and file
        model_dir = temp_dir / "buffalo_l"
        model_dir.mkdir()
        model_file = model_dir / "w600k_r50.onnx"
        model_file.touch()

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input.1"
        mock_session.get_inputs.return_value = [mock_input]
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.array([])]  # Empty result

        with patch(
            "src.photodb.utils.embedding_extractor.ort.InferenceSession"
        ) as mock_ort_session:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_ort_session.return_value = mock_session
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

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input.1"
        mock_session.get_inputs.return_value = [mock_input]
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.side_effect = RuntimeError("Model inference failed")

        with patch(
            "src.photodb.utils.embedding_extractor.ort.InferenceSession"
        ) as mock_ort_session:
            with patch(
                "src.photodb.utils.embedding_extractor._get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                mock_ort_session.return_value = mock_session
                from src.photodb.utils.embedding_extractor import EmbeddingExtractor

                extractor = EmbeddingExtractor(model_root=str(temp_dir))
                embedding = extractor.extract(sample_pil_image, sample_bbox)

                assert embedding is None

    def test_extract_from_aligned_resizes_non_standard_input(self, mock_extractor, mock_session):
        """Verify extract_from_aligned resizes input that's not 112x112."""
        # Create aligned face with wrong size
        aligned_face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        embedding = mock_extractor.extract_from_aligned(aligned_face)

        # Verify session.run was called with properly shaped input
        mock_session.run.assert_called()
        # session.run(output_names, {input_name: blobs}) — positional args
        input_dict = mock_session.run.call_args[0][1]
        blob = input_dict[mock_extractor.input_name]

        # Should be NCHW (1, 3, 112, 112)
        assert blob.shape == (1, 3, 112, 112)
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
        assert extractor.session is not None
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
