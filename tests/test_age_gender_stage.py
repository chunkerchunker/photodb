"""
Tests for AgeGenderStage.

Tests the age/gender estimation stage that uses MiVOLO to estimate age and gender
for existing person detections.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


class TestAgeGenderStageUnit:
    """Unit tests for AgeGenderStage using mocks."""

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
        repo.update_detection_age_gender.return_value = None
        return repo

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return {
            "IMG_PATH": str(temp_dir),
        }

    @pytest.fixture
    def mock_mivolo_predictor(self):
        """Create a mock MiVOLOPredictor."""
        predictor = MagicMock()
        predictor.predict.return_value = {
            "age": 25.5,
            "gender": "M",
            "gender_confidence": 0.95,
        }
        predictor._available = True
        return predictor

    def test_stage_name_is_age_gender(self, mock_repository, config, mock_mivolo_predictor):
        """Test that stage_name is 'age_gender'."""
        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage

            stage = AgeGenderStage(mock_repository, config)
            assert stage.stage_name == "age_gender"

    def test_inherits_from_base_stage(self, mock_repository, config, mock_mivolo_predictor):
        """Test that AgeGenderStage inherits from BaseStage."""
        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.stages.base import BaseStage

            stage = AgeGenderStage(mock_repository, config)
            assert isinstance(stage, BaseStage)

    def test_process_photo_skips_without_normalized_path(
        self, mock_repository, config, mock_mivolo_predictor
    ):
        """Test that process_photo returns False if photo has no normalized_path."""
        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            mock_mivolo_predictor.predict.assert_not_called()

    def test_process_photo_skips_nonexistent_normalized_file(
        self, mock_repository, config, mock_mivolo_predictor, temp_dir
    ):
        """Test that process_photo returns False if normalized file doesn't exist."""
        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            mock_mivolo_predictor.predict.assert_not_called()

    def test_process_photo_returns_true_when_no_detections(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that process_photo returns True when there are no detections."""
        mock_repository.get_detections_for_photo.return_value = []

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            mock_mivolo_predictor.predict.assert_not_called()

    def test_process_photo_updates_detection_with_age_gender(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that age/gender data is updated for detections with faces."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=100,
            face_bbox_height=100,
            face_confidence=0.95,
            body_bbox_x=50,
            body_bbox_y=20,
            body_bbox_width=250,
            body_bbox_height=380,
            body_confidence=0.9,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        mock_mivolo_predictor.predict.return_value = {
            "age": 32.5,
            "gender": "F",
            "gender_confidence": 0.92,
        }

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            mock_repository.update_detection_age_gender.assert_called_once()

            # Verify the update call arguments
            call_args = mock_repository.update_detection_age_gender.call_args
            assert call_args[1]["detection_id"] == 10
            assert call_args[1]["age_estimate"] == 32.5
            assert call_args[1]["gender"] == "F"
            assert call_args[1]["gender_confidence"] == 0.92

    def test_process_photo_passes_face_bbox_to_predictor(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that face bounding box is passed to MiVOLO predictor."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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

            # Check that predict was called with face bbox
            call_kwargs = mock_mivolo_predictor.predict.call_args[1]
            assert call_kwargs["face_bbox"] == (100, 50, 80, 100)

    def test_process_photo_passes_body_bbox_to_predictor(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that body bounding box is passed to MiVOLO predictor."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            body_bbox_x=50,
            body_bbox_y=20,
            body_bbox_width=250,
            body_bbox_height=380,
            body_confidence=0.9,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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

            # Check that predict was called with body bbox
            call_kwargs = mock_mivolo_predictor.predict.call_args[1]
            assert call_kwargs["body_bbox"] == (50, 20, 250, 380)

    def test_process_photo_handles_multiple_detections(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that multiple detections are all processed."""
        from src.photodb.database.models import PersonDetection

        detections = [
            PersonDetection(
                id=10,
                photo_id=1,
                face_bbox_x=100,
                face_bbox_y=50,
                face_bbox_width=80,
                face_bbox_height=100,
                face_confidence=0.95,
            ),
            PersonDetection(
                id=11,
                photo_id=1,
                face_bbox_x=300,
                face_bbox_y=50,
                face_bbox_width=90,
                face_bbox_height=110,
                face_confidence=0.85,
            ),
        ]
        mock_repository.get_detections_for_photo.return_value = detections

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            assert mock_mivolo_predictor.predict.call_count == 2
            assert mock_repository.update_detection_age_gender.call_count == 2

    def test_process_photo_skips_detection_without_face_or_body(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that detections without face or body bboxes are skipped."""
        from src.photodb.database.models import PersonDetection

        # Detection with no face or body bbox (shouldn't happen but handle it)
        detection = PersonDetection(
            id=10,
            photo_id=1,
            # No face_bbox_* or body_bbox_* fields set
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            mock_mivolo_predictor.predict.assert_not_called()
            mock_repository.update_detection_age_gender.assert_not_called()

    def test_process_photo_does_not_update_when_no_prediction(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that update is not called when predictor returns no results."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        # Predictor returns no useful data
        mock_mivolo_predictor.predict.return_value = {
            "age": None,
            "gender": "U",
            "gender_confidence": 0.0,
        }

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            mock_repository.update_detection_age_gender.assert_not_called()

    def test_process_photo_updates_when_age_only(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that update is called when only age is available."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        mock_mivolo_predictor.predict.return_value = {
            "age": 45.0,
            "gender": "U",
            "gender_confidence": 0.0,
        }

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            mock_repository.update_detection_age_gender.assert_called_once()

    def test_process_photo_updates_when_gender_only(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that update is called when only gender is available."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        mock_mivolo_predictor.predict.return_value = {
            "age": None,
            "gender": "F",
            "gender_confidence": 0.88,
        }

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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
            mock_repository.update_detection_age_gender.assert_called_once()

    def test_process_photo_handles_exception(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that exceptions during prediction are handled gracefully."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]
        mock_mivolo_predictor.predict.side_effect = Exception("Prediction failed")

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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

    def test_respects_force_cpu_env_var(self, mock_repository, config, monkeypatch):
        """Test that MIVOLO_FORCE_CPU env var is respected."""
        monkeypatch.setenv("MIVOLO_FORCE_CPU", "true")

        mock_predictor = MagicMock()
        mock_predictor._available = True

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_predictor,
        ) as mock_class:
            from src.photodb.stages.age_gender import AgeGenderStage

            AgeGenderStage(mock_repository, config)
            # MiVOLOPredictor should have been called with device="cpu"
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs.get("device") == "cpu"

    def test_stores_mivolo_output_in_update(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that full MiVOLO output is stored."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        mivolo_result = {
            "age": 32.5,
            "gender": "M",
            "gender_confidence": 0.95,
        }
        mock_mivolo_predictor.predict.return_value = mivolo_result

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
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

            call_args = mock_repository.update_detection_age_gender.call_args
            assert call_args[1]["mivolo_output"] == mivolo_result


class TestMiVOLOPredictorUnit:
    """Unit tests for MiVOLOPredictor wrapper class."""

    def test_predictor_gracefully_handles_import_error(self):
        """Test that MiVOLOPredictor works when MiVOLO is not installed."""
        with patch.dict("sys.modules", {"mivolo": None, "mivolo.predictor": None}):
            # This should not raise an error
            from src.photodb.stages.age_gender import MiVOLOPredictor

            predictor = MiVOLOPredictor(
                checkpoint_path="models/mivolo.pth",
                device="cpu",
            )

            # Should return default values when MiVOLO is unavailable
            result = predictor.predict(
                image_path="/path/to/image.jpg",
                face_bbox=(100, 50, 80, 100),
            )

            assert result["age"] is None
            assert result["gender"] == "U"
            assert result["gender_confidence"] == 0.0

    def test_predictor_available_flag(self):
        """Test that _available flag is set correctly."""
        with patch.dict("sys.modules", {"mivolo": None, "mivolo.predictor": None}):
            from src.photodb.stages.age_gender import MiVOLOPredictor

            predictor = MiVOLOPredictor(
                checkpoint_path="models/mivolo.pth",
                device="cpu",
            )

            assert predictor._available is False
