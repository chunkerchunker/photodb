"""
Tests for PersonDetector utility.

Uses YOLO for face+body detection and ArcFace (ONNX Runtime) for embeddings.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


class TestPersonDetectorUnit:
    """Unit tests for PersonDetector using mocks (no model downloads required)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample white test image (no faces expected)."""
        img_path = temp_dir / "test.jpg"
        img = Image.new("RGB", (640, 480), color="white")
        img.save(img_path, "JPEG")
        return img_path

    @pytest.fixture
    def sample_rgba_image(self, temp_dir):
        """Create a sample RGBA image."""
        img_path = temp_dir / "test_rgba.png"
        img = Image.new("RGBA", (640, 480), color=(255, 255, 255, 255))
        img.save(img_path, "PNG")
        return img_path

    @pytest.fixture
    def mock_yolo(self):
        """Create a mock YOLO model."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        return mock_model

    @pytest.fixture
    def mock_embedding_extractor(self):
        """Create a mock EmbeddingExtractor."""
        mock_extractor = MagicMock()
        # Return a 512-dim embedding list
        mock_extractor.extract.return_value = [0.1] * 512
        mock_extractor.get_feat.return_value = np.random.randn(1, 512).astype(np.float32)
        return mock_extractor

    @pytest.fixture
    def mock_detector(self, mock_yolo, mock_embedding_extractor):
        """Create a PersonDetector with mocked models."""
        with patch("src.photodb.utils.person_detector.YOLO", return_value=mock_yolo):
            with patch(
                "src.photodb.utils.person_detector.EmbeddingExtractor",
                return_value=mock_embedding_extractor,
            ):
                from src.photodb.utils.person_detector import PersonDetector

                detector = PersonDetector(force_cpu=True)
                return detector

    def test_class_constants(self):
        """Test that class constants are correctly defined."""
        from src.photodb.utils.person_detector import PersonDetector

        assert PersonDetector.FACE_CLASS_ID == 1
        assert PersonDetector.PERSON_CLASS_ID == 0

    def test_detector_initialization(self, mock_yolo, mock_embedding_extractor):
        """Test that PersonDetector initializes correctly with force_cpu."""
        with patch("src.photodb.utils.person_detector.YOLO", return_value=mock_yolo):
            with patch(
                "src.photodb.utils.person_detector.EmbeddingExtractor",
                return_value=mock_embedding_extractor,
            ):
                from src.photodb.utils.person_detector import PersonDetector

                detector = PersonDetector(force_cpu=True)
                assert detector.device == "cpu"
                assert detector.model is not None
                assert detector.embedding_extractor is not None

    def test_detector_initialization_with_custom_confidence(
        self, mock_yolo, mock_embedding_extractor
    ):
        """Test detector respects DETECTION_MIN_CONFIDENCE config value."""
        with patch("src.photodb.utils.person_detector.YOLO", return_value=mock_yolo):
            with patch(
                "src.photodb.utils.person_detector.EmbeddingExtractor",
                return_value=mock_embedding_extractor,
            ):
                with patch("src.photodb.utils.person_detector.defaults") as mock_defaults:
                    mock_defaults.DETECTION_MIN_CONFIDENCE = 0.7
                    mock_defaults.DETECTION_MODEL_PATH = "models/yolov8x_person_face.pt"
                    mock_defaults.DETECTION_FORCE_CPU = False
                    mock_defaults.DETECTION_PREFER_COREML = False
                    from src.photodb.utils.person_detector import PersonDetector

                    detector = PersonDetector(force_cpu=True)
                    assert detector.min_confidence == 0.7

    def test_compute_containment_full_overlap(self, mock_detector):
        """Test containment computation when face is fully inside body."""
        face_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        body_bbox = {"x1": 50, "y1": 50, "x2": 300, "y2": 400}

        containment = mock_detector._compute_containment(face_bbox, body_bbox)
        assert containment == 1.0  # Face is fully contained

    def test_compute_containment_partial_overlap(self, mock_detector):
        """Test containment computation with partial overlap."""
        face_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        body_bbox = {"x1": 150, "y1": 100, "x2": 300, "y2": 400}

        containment = mock_detector._compute_containment(face_bbox, body_bbox)
        # Face is 100x100, overlap is 50x100 = 5000, face area = 10000
        assert 0 < containment < 1.0
        assert containment == pytest.approx(0.5, rel=0.01)

    def test_compute_containment_no_overlap(self, mock_detector):
        """Test containment computation with no overlap."""
        face_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        body_bbox = {"x1": 300, "y1": 300, "x2": 400, "y2": 500}

        containment = mock_detector._compute_containment(face_bbox, body_bbox)
        assert containment == 0.0

    def test_compute_containment_zero_area_face(self, mock_detector):
        """Test containment computation with zero-area face."""
        face_bbox = {"x1": 100, "y1": 100, "x2": 100, "y2": 100}  # Zero area
        body_bbox = {"x1": 50, "y1": 50, "x2": 300, "y2": 400}

        containment = mock_detector._compute_containment(face_bbox, body_bbox)
        assert containment == 0.0

    def test_match_faces_to_bodies_single_match(self, mock_detector):
        """Test matching a single face to a single body."""
        faces = [{"bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150}, "confidence": 0.9}]
        bodies = [{"bbox": {"x1": 50, "y1": 20, "x2": 300, "y2": 400}, "confidence": 0.95}]

        matches = mock_detector._match_faces_to_bodies(faces, bodies)

        assert len(matches) == 1
        assert matches[0]["face"] is not None
        assert matches[0]["body"] is not None

    def test_match_faces_to_bodies_no_bodies(self, mock_detector):
        """Test matching faces when no bodies are detected."""
        faces = [{"bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150}, "confidence": 0.9}]
        bodies = []

        matches = mock_detector._match_faces_to_bodies(faces, bodies)

        assert len(matches) == 1
        assert matches[0]["face"] is not None
        assert matches[0]["body"] is None

    def test_match_faces_to_bodies_no_faces(self, mock_detector):
        """Test matching when no faces but bodies are detected."""
        faces = []
        bodies = [{"bbox": {"x1": 50, "y1": 20, "x2": 300, "y2": 400}, "confidence": 0.95}]

        matches = mock_detector._match_faces_to_bodies(faces, bodies)

        assert len(matches) == 1
        assert matches[0]["face"] is None
        assert matches[0]["body"] is not None

    def test_match_faces_to_bodies_multiple(self, mock_detector):
        """Test matching multiple faces to multiple bodies."""
        faces = [
            {"bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150}, "confidence": 0.9},
            {"bbox": {"x1": 400, "y1": 50, "x2": 500, "y2": 150}, "confidence": 0.85},
        ]
        bodies = [
            {"bbox": {"x1": 50, "y1": 20, "x2": 300, "y2": 400}, "confidence": 0.95},
            {"bbox": {"x1": 350, "y1": 20, "x2": 600, "y2": 400}, "confidence": 0.92},
        ]

        matches = mock_detector._match_faces_to_bodies(faces, bodies)

        assert len(matches) == 2
        # Both faces should be matched to their respective bodies
        assert all(m["face"] is not None for m in matches)
        assert all(m["body"] is not None for m in matches)

    def test_detect_returns_dict_structure(self, mock_detector, sample_image):
        """Test detect() returns expected dictionary structure."""
        result = mock_detector.detect(str(sample_image))

        assert isinstance(result, dict)
        assert "status" in result
        assert "detections" in result
        assert "image_dimensions" in result
        assert result["status"] in ["success", "no_detections"]

    def test_detect_empty_image(self, mock_detector, sample_image):
        """Test that a plain white image returns no detections."""
        result = mock_detector.detect(str(sample_image))

        # Empty white image with mocked model should have no detections
        assert result["status"] in ["success", "no_detections"]
        if result["status"] == "no_detections":
            assert len(result["detections"]) == 0

    def test_detect_image_dimensions(self, mock_detector, sample_image):
        """Test that image dimensions are correctly reported."""
        result = mock_detector.detect(str(sample_image))

        assert result["image_dimensions"]["width"] == 640
        assert result["image_dimensions"]["height"] == 480

    def test_detect_rgba_image(self, mock_detector, sample_rgba_image):
        """Test detection on RGBA image (should convert to RGB)."""
        result = mock_detector.detect(str(sample_rgba_image))

        assert "status" in result
        assert result["status"] in ["success", "no_detections", "error"]

    def test_detect_nonexistent_file(self, mock_detector, temp_dir):
        """Test detection on non-existent file returns error."""
        result = mock_detector.detect(str(temp_dir / "nonexistent.jpg"))

        assert result["status"] == "error"
        assert "error" in result

    def test_extract_embedding_returns_list(self, mock_detector, sample_image):
        """Test that extract_embedding returns a list of floats."""
        img = Image.open(sample_image)
        bbox = {"x1": 0, "y1": 0, "x2": 160, "y2": 160}

        embedding = mock_detector.extract_embedding(img, bbox)

        assert isinstance(embedding, list)
        assert len(embedding) == 512  # InsightFace ArcFace embedding dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_device_auto_detection_cpu_fallback(self, mock_yolo, mock_embedding_extractor):
        """Test device auto-detection falls back to CPU when forced."""
        with patch("src.photodb.utils.person_detector.YOLO", return_value=mock_yolo):
            with patch(
                "src.photodb.utils.person_detector.EmbeddingExtractor",
                return_value=mock_embedding_extractor,
            ):
                from src.photodb.utils.person_detector import PersonDetector

                detector = PersonDetector(force_cpu=True)
                assert detector.device == "cpu"

    def test_detection_result_format(self, mock_detector, sample_image):
        """Test the format of detection results."""
        result = mock_detector.detect(str(sample_image))

        # Verify top-level structure
        assert "status" in result
        assert "detections" in result
        assert "image_dimensions" in result
        assert isinstance(result["detections"], list)
        assert isinstance(result["image_dimensions"], dict)
        assert "width" in result["image_dimensions"]
        assert "height" in result["image_dimensions"]

    def test_detect_with_face_detections(self, mock_yolo, mock_embedding_extractor, sample_image):
        """Test detection when YOLO returns face detections."""
        # Create mock boxes with face detection
        mock_box = MagicMock()
        mock_box.cls = torch.tensor([1])  # FACE_CLASS_ID
        mock_box.conf = torch.tensor([0.95])
        mock_box.xyxy = torch.tensor([[100, 100, 200, 200]])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_yolo.return_value = [mock_result]

        with patch("src.photodb.utils.person_detector.YOLO", return_value=mock_yolo):
            with patch(
                "src.photodb.utils.person_detector.EmbeddingExtractor",
                return_value=mock_embedding_extractor,
            ):
                from src.photodb.utils.person_detector import PersonDetector

                detector = PersonDetector(force_cpu=True)
                result = detector.detect(str(sample_image))

                assert result["status"] == "success"
                assert len(result["detections"]) == 1
                assert result["detections"][0]["face"] is not None

    def test_detect_with_person_detections(self, mock_yolo, mock_embedding_extractor, sample_image):
        """Test detection when YOLO returns person/body detections."""
        # Create mock boxes with person detection
        mock_box = MagicMock()
        mock_box.cls = torch.tensor([0])  # PERSON_CLASS_ID
        mock_box.conf = torch.tensor([0.9])
        mock_box.xyxy = torch.tensor([[50, 50, 300, 400]])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_yolo.return_value = [mock_result]

        with patch("src.photodb.utils.person_detector.YOLO", return_value=mock_yolo):
            with patch(
                "src.photodb.utils.person_detector.EmbeddingExtractor",
                return_value=mock_embedding_extractor,
            ):
                from src.photodb.utils.person_detector import PersonDetector

                detector = PersonDetector(force_cpu=True)
                result = detector.detect(str(sample_image))

                assert result["status"] == "success"
                assert len(result["detections"]) == 1
                assert result["detections"][0]["body"] is not None
                assert result["detections"][0]["face"] is None

    def test_detect_with_matched_face_and_body(
        self, mock_yolo, mock_embedding_extractor, sample_image
    ):
        """Test detection when YOLO returns both face and body that match."""
        # Create mock boxes with face inside body
        mock_face_box = MagicMock()
        mock_face_box.cls = torch.tensor([1])  # FACE_CLASS_ID
        mock_face_box.conf = torch.tensor([0.95])
        mock_face_box.xyxy = torch.tensor([[100, 50, 200, 150]])

        mock_body_box = MagicMock()
        mock_body_box.cls = torch.tensor([0])  # PERSON_CLASS_ID
        mock_body_box.conf = torch.tensor([0.9])
        mock_body_box.xyxy = torch.tensor([[50, 20, 300, 400]])

        mock_result = MagicMock()
        mock_result.boxes = [mock_face_box, mock_body_box]

        mock_yolo.return_value = [mock_result]

        with patch("src.photodb.utils.person_detector.YOLO", return_value=mock_yolo):
            with patch(
                "src.photodb.utils.person_detector.EmbeddingExtractor",
                return_value=mock_embedding_extractor,
            ):
                from src.photodb.utils.person_detector import PersonDetector

                detector = PersonDetector(force_cpu=True)
                result = detector.detect(str(sample_image))

                assert result["status"] == "success"
                assert len(result["detections"]) == 1
                # Face and body should be matched together
                assert result["detections"][0]["face"] is not None
                assert result["detections"][0]["body"] is not None


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration tests require models to be downloaded. Set RUN_INTEGRATION_TESTS=1 to run.",
)
class TestPersonDetectorIntegration:
    """Integration tests for PersonDetector (requires model downloads)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample white test image (no faces expected)."""
        img_path = temp_dir / "test.jpg"
        img = Image.new("RGB", (640, 480), color="white")
        img.save(img_path, "JPEG")
        return img_path

    def test_real_detector_initialization(self):
        """Test that PersonDetector initializes correctly with real models."""
        from src.photodb.utils.person_detector import PersonDetector

        detector = PersonDetector(force_cpu=True)
        assert detector.device == "cpu"
        assert detector.model is not None
        assert detector.embedding_extractor is not None

    def test_real_detection_on_blank_image(self, sample_image):
        """Test real detection on a blank image."""
        from src.photodb.utils.person_detector import PersonDetector

        detector = PersonDetector(force_cpu=True)
        result = detector.detect(str(sample_image))

        assert result["status"] in ["success", "no_detections"]
        assert result["image_dimensions"]["width"] == 640
        assert result["image_dimensions"]["height"] == 480
