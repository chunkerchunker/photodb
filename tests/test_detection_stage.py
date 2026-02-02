"""
Tests for DetectionStage.

Tests the detection stage that uses PersonDetector to detect faces/bodies
and save PersonDetection records.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


class TestDetectionStageUnit:
    """Unit tests for DetectionStage using mocks."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample test image."""
        img_path = temp_dir / "test.jpg"
        img = Image.new("RGB", (640, 480), color="white")
        img.save(img_path, "JPEG")
        return img_path

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository."""
        repo = MagicMock()
        repo.get_detections_for_photo.return_value = []
        repo.delete_detections_for_photo.return_value = 0
        repo.create_person_detection.return_value = None
        repo.save_detection_embedding.return_value = None
        return repo

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return {
            "IMG_PATH": str(temp_dir),
        }

    @pytest.fixture
    def mock_person_detector(self):
        """Create a mock PersonDetector."""
        detector = MagicMock()
        detector.detect.return_value = {
            "status": "no_detections",
            "detections": [],
            "image_dimensions": {"width": 640, "height": 480},
        }
        return detector

    def test_stage_name_is_detection(self, mock_repository, config, mock_person_detector):
        """Test that stage_name is 'detection'."""
        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage

            stage = DetectionStage(mock_repository, config)
            assert stage.stage_name == "detection"

    def test_inherits_from_base_stage(self, mock_repository, config, mock_person_detector):
        """Test that DetectionStage inherits from BaseStage."""
        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.stages.base import BaseStage

            stage = DetectionStage(mock_repository, config)
            assert isinstance(stage, BaseStage)

    def test_process_photo_skips_without_normalized_path(
        self, mock_repository, config, mock_person_detector
    ):
        """Test that process_photo returns False if photo has no normalized_path."""
        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=None,  # No normalized path
                width=640,
                height=480,
                normalized_width=None,
                normalized_height=None,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is False
            mock_person_detector.detect.assert_not_called()

    def test_process_photo_skips_nonexistent_normalized_file(
        self, mock_repository, config, mock_person_detector, temp_dir
    ):
        """Test that process_photo returns False if normalized file doesn't exist."""
        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path="nonexistent.jpg",  # File doesn't exist
                width=640,
                height=480,
                normalized_width=None,
                normalized_height=None,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is False
            mock_person_detector.detect.assert_not_called()

    def test_process_photo_clears_existing_detections_on_reprocess(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that existing detections are cleared when reprocessing."""
        from src.photodb.database.models import PersonDetection

        existing_detection = PersonDetection(
            id=100,
            photo_id=1,
            face_bbox_x=0.1,
            face_bbox_y=0.1,
            face_bbox_width=0.2,
            face_bbox_height=0.2,
            face_confidence=0.9,
        )
        mock_repository.get_detections_for_photo.return_value = [existing_detection]

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            stage.process_photo(photo, Path("/path/to/photo.jpg"))
            mock_repository.delete_detections_for_photo.assert_called_once_with(1)

    def test_process_photo_returns_true_on_no_detections(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that process_photo returns True when no detections found."""
        mock_person_detector.detect.return_value = {
            "status": "no_detections",
            "detections": [],
            "image_dimensions": {"width": 640, "height": 480},
        }

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is True

    def test_process_photo_creates_detection_with_face_only(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that face-only detections are saved correctly."""
        mock_person_detector.detect.return_value = {
            "status": "success",
            "detections": [
                {
                    "face": {
                        "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
                        "confidence": 0.95,
                        "embedding": [0.1] * 512,
                        "embedding_norm": 1.0,
                    },
                    "body": None,
                }
            ],
            "image_dimensions": {"width": 640, "height": 480},
        }

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            mock_repository.create_person_detection.assert_called_once()

            # Verify the PersonDetection object passed to create_person_detection
            call_args = mock_repository.create_person_detection.call_args
            detection = call_args[0][0]
            assert detection.photo_id == 1
            assert detection.face_bbox_x == 100
            assert detection.face_bbox_y == 50
            assert detection.face_bbox_width == 100  # 200 - 100
            assert detection.face_bbox_height == 100  # 150 - 50
            assert detection.face_confidence == 0.95
            assert detection.body_bbox_x is None

    def test_process_photo_creates_detection_with_body_only(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that body-only detections are saved correctly."""
        mock_person_detector.detect.return_value = {
            "status": "success",
            "detections": [
                {
                    "face": None,
                    "body": {
                        "bbox": {"x1": 50, "y1": 20, "x2": 300, "y2": 400},
                        "confidence": 0.9,
                    },
                }
            ],
            "image_dimensions": {"width": 640, "height": 480},
        }

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            mock_repository.create_person_detection.assert_called_once()

            # Verify the PersonDetection object
            call_args = mock_repository.create_person_detection.call_args
            detection = call_args[0][0]
            assert detection.photo_id == 1
            assert detection.face_bbox_x is None
            assert detection.body_bbox_x == 50
            assert detection.body_bbox_y == 20
            assert detection.body_bbox_width == 250  # 300 - 50
            assert detection.body_bbox_height == 380  # 400 - 20
            assert detection.body_confidence == 0.9

    def test_process_photo_creates_detection_with_face_and_body(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that matched face+body detections are saved correctly."""
        mock_person_detector.detect.return_value = {
            "status": "success",
            "detections": [
                {
                    "face": {
                        "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
                        "confidence": 0.95,
                        "embedding": [0.1] * 512,
                        "embedding_norm": 1.0,
                    },
                    "body": {
                        "bbox": {"x1": 50, "y1": 20, "x2": 300, "y2": 400},
                        "confidence": 0.9,
                    },
                }
            ],
            "image_dimensions": {"width": 640, "height": 480},
        }

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            mock_repository.create_person_detection.assert_called_once()

            # Verify both face and body data are present
            call_args = mock_repository.create_person_detection.call_args
            detection = call_args[0][0]
            assert detection.face_bbox_x == 100
            assert detection.face_confidence == 0.95
            assert detection.body_bbox_x == 50
            assert detection.body_confidence == 0.9

    def test_process_photo_saves_embedding_for_face(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that face embeddings are saved to the database."""
        embedding = [0.1] * 512
        mock_person_detector.detect.return_value = {
            "status": "success",
            "detections": [
                {
                    "face": {
                        "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
                        "confidence": 0.95,
                        "embedding": embedding,
                        "embedding_norm": 1.0,
                    },
                    "body": None,
                }
            ],
            "image_dimensions": {"width": 640, "height": 480},
        }

        # Need to set the detection.id when create_person_detection is called
        def set_detection_id(detection):
            detection.id = 42

        mock_repository.create_person_detection.side_effect = set_detection_id

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            mock_repository.save_detection_embedding.assert_called_once_with(42, embedding)

    def test_process_photo_does_not_save_embedding_without_face(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that embeddings are not saved for body-only detections."""
        mock_person_detector.detect.return_value = {
            "status": "success",
            "detections": [
                {
                    "face": None,
                    "body": {
                        "bbox": {"x1": 50, "y1": 20, "x2": 300, "y2": 400},
                        "confidence": 0.9,
                    },
                }
            ],
            "image_dimensions": {"width": 640, "height": 480},
        }

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            mock_repository.save_detection_embedding.assert_not_called()

    def test_process_photo_handles_multiple_detections(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that multiple detections in one photo are all saved."""
        mock_person_detector.detect.return_value = {
            "status": "success",
            "detections": [
                {
                    "face": {
                        "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
                        "confidence": 0.95,
                        "embedding": [0.1] * 512,
                        "embedding_norm": 1.0,
                    },
                    "body": {
                        "bbox": {"x1": 50, "y1": 20, "x2": 300, "y2": 400},
                        "confidence": 0.9,
                    },
                },
                {
                    "face": {
                        "bbox": {"x1": 400, "y1": 50, "x2": 500, "y2": 150},
                        "confidence": 0.85,
                        "embedding": [0.2] * 512,
                        "embedding_norm": 1.0,
                    },
                    "body": {
                        "bbox": {"x1": 350, "y1": 20, "x2": 550, "y2": 400},
                        "confidence": 0.88,
                    },
                },
            ],
            "image_dimensions": {"width": 640, "height": 480},
        }

        detection_id_counter = [0]

        def set_detection_id(detection):
            detection_id_counter[0] += 1
            detection.id = detection_id_counter[0]

        mock_repository.create_person_detection.side_effect = set_detection_id

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            assert mock_repository.create_person_detection.call_count == 2
            assert mock_repository.save_detection_embedding.call_count == 2

    def test_process_photo_returns_false_on_detector_error(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that process_photo returns False when detector returns error."""
        mock_person_detector.detect.return_value = {
            "status": "error",
            "detections": [],
            "image_dimensions": {"width": 0, "height": 0},
            "error": "Some error occurred",
        }

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is False

    def test_process_photo_handles_exception(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that exceptions during detection are handled gracefully."""
        mock_person_detector.detect.side_effect = Exception("Detection failed")

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is False

    def test_process_photo_stores_detector_model_info(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that detector model info is stored in detection record."""
        mock_person_detector.detect.return_value = {
            "status": "success",
            "detections": [
                {
                    "face": {
                        "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
                        "confidence": 0.95,
                        "embedding": [0.1] * 512,
                        "embedding_norm": 1.0,
                    },
                    "body": None,
                }
            ],
            "image_dimensions": {"width": 640, "height": 480},
        }

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            call_args = mock_repository.create_person_detection.call_args
            detection = call_args[0][0]
            # Should have detector model info
            assert detection.detector_model is not None

    def test_does_not_clear_detections_when_no_existing(
        self, mock_repository, config, mock_person_detector, sample_image, temp_dir
    ):
        """Test that delete is not called when there are no existing detections."""
        mock_repository.get_detections_for_photo.return_value = []

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_person_detector,
        ):
            from src.photodb.stages.detection import DetectionStage
            from src.photodb.database.models import Photo

            stage = DetectionStage(mock_repository, config)
            photo = Photo(
                id=1,
                filename="/path/to/photo.jpg",
                normalized_path=sample_image.name,
                width=640,
                height=480,
                normalized_width=640,
                normalized_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            stage.process_photo(photo, Path("/path/to/photo.jpg"))
            mock_repository.delete_detections_for_photo.assert_not_called()

    def test_respects_force_cpu_env_var(self, mock_repository, config, monkeypatch):
        """Test that DETECTION_FORCE_CPU env var is respected."""
        monkeypatch.setenv("DETECTION_FORCE_CPU", "true")

        mock_detector = MagicMock()
        mock_detector.detect.return_value = {
            "status": "no_detections",
            "detections": [],
            "image_dimensions": {"width": 640, "height": 480},
        }

        with patch(
            "src.photodb.stages.detection.PersonDetector",
            return_value=mock_detector,
        ) as mock_class:
            from src.photodb.stages.detection import DetectionStage

            DetectionStage(mock_repository, config)
            # PersonDetector should have been called with force_cpu=True
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs.get("force_cpu") is True
