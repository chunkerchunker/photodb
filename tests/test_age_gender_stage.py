"""
Tests for AgeGenderStage.

Tests the age/gender estimation stage that uses MiVOLO to estimate age and gender
for existing person detections. MiVOLO runs its own detection and results are
matched to existing detections via IoU.
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
        """Create a mock MiVOLOPredictor that returns list of predictions."""
        predictor = MagicMock()
        # Default: return empty list (no predictions)
        predictor.predict.return_value = []
        predictor._available = True
        predictor._lock = MagicMock()
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
        """Test that age/gender data is updated for detections matched by IoU."""
        from src.photodb.database.models import PersonDetection

        # Existing detection in database
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

        # MiVOLO prediction with overlapping face bbox (high IoU)
        mock_mivolo_predictor.predict.return_value = [
            {
                "face_bbox": (100, 50, 100, 100),  # Same as detection
                "body_bbox": (50, 20, 250, 380),
                "age": 32.5,
                "gender": "F",
                "gender_confidence": 0.92,
            }
        ]

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

    def test_process_photo_calls_predict_with_image_path(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that predict is called with the normalized image path."""
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
        mock_mivolo_predictor.predict.return_value = []

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

            # Check that predict was called with the image path string
            mock_mivolo_predictor.predict.assert_called_once()
            call_args = mock_mivolo_predictor.predict.call_args[0]
            assert sample_image.name in call_args[0]

    def test_process_photo_matches_by_body_bbox_when_no_face(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that matching works via body bbox when face bbox is not available."""
        from src.photodb.database.models import PersonDetection

        # Detection with only body bbox
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

        # MiVOLO prediction with matching body bbox
        mock_mivolo_predictor.predict.return_value = [
            {
                "face_bbox": None,
                "body_bbox": (50, 20, 250, 380),  # Same as detection
                "age": 28.0,
                "gender": "M",
                "gender_confidence": 0.88,
            }
        ]

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
            call_args = mock_repository.update_detection_age_gender.call_args
            assert call_args[1]["detection_id"] == 10

    def test_process_photo_handles_multiple_predictions(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that multiple MiVOLO predictions are matched to multiple detections."""
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

        # Two MiVOLO predictions matching each detection
        mock_mivolo_predictor.predict.return_value = [
            {
                "face_bbox": (100, 50, 80, 100),
                "body_bbox": None,
                "age": 25.0,
                "gender": "M",
                "gender_confidence": 0.9,
            },
            {
                "face_bbox": (300, 50, 90, 110),
                "body_bbox": None,
                "age": 30.0,
                "gender": "F",
                "gender_confidence": 0.85,
            },
        ]

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
            # predict is called once for the whole image
            assert mock_mivolo_predictor.predict.call_count == 1
            # update is called for each matched detection
            assert mock_repository.update_detection_age_gender.call_count == 2

    def test_process_photo_skips_low_iou_predictions(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that predictions with low IoU are not matched."""
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

        # MiVOLO prediction with non-overlapping bbox (low IoU)
        mock_mivolo_predictor.predict.return_value = [
            {
                "face_bbox": (500, 400, 80, 100),  # Far from detection
                "body_bbox": None,
                "age": 25.0,
                "gender": "M",
                "gender_confidence": 0.9,
            }
        ]

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
            # No match due to low IoU
            mock_repository.update_detection_age_gender.assert_not_called()

    def test_process_photo_returns_true_when_no_predictions(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that process_photo returns True when MiVOLO returns no predictions."""
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
        mock_mivolo_predictor.predict.return_value = []

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
            "face_bbox": (100, 50, 80, 100),
            "body_bbox": None,
            "age": 32.5,
            "gender": "M",
            "gender_confidence": 0.95,
        }
        mock_mivolo_predictor.predict.return_value = [mivolo_result]

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

    def test_handles_detection_with_partial_bbox(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that detections with partial bbox data (some None) are handled."""
        from src.photodb.database.models import PersonDetection

        # Detection with only x coordinate set (invalid state, but should not crash)
        detection = PersonDetection(
            id=10,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=None,  # Missing
            face_bbox_width=None,  # Missing
            face_bbox_height=None,  # Missing
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        mock_mivolo_predictor.predict.return_value = [
            {
                "face_bbox": (100, 50, 80, 100),
                "body_bbox": None,
                "age": 25.0,
                "gender": "M",
                "gender_confidence": 0.9,
            }
        ]

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

            # Should not crash
            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is True
            # No match because detection bbox is incomplete
            mock_repository.update_detection_age_gender.assert_not_called()


class TestMiVOLOPredictorUnit:
    """Unit tests for MiVOLOPredictor wrapper class."""

    def test_predictor_gracefully_handles_import_error(self):
        """Test that MiVOLOPredictor works when MiVOLO is not installed."""
        with patch.dict("sys.modules", {"mivolo": None, "mivolo.predictor": None}):
            # This should not raise an error
            from src.photodb.stages.age_gender import MiVOLOPredictor

            predictor = MiVOLOPredictor(
                checkpoint_path="models/mivolo.pth",
                detector_weights_path="models/yolo.pt",
                device="cpu",
            )

            # Should return empty list when MiVOLO is unavailable
            result = predictor.predict("/path/to/image.jpg")
            assert result == []

    def test_predictor_available_flag(self):
        """Test that _available flag is set correctly."""
        with patch.dict("sys.modules", {"mivolo": None, "mivolo.predictor": None}):
            from src.photodb.stages.age_gender import MiVOLOPredictor

            predictor = MiVOLOPredictor(
                checkpoint_path="models/mivolo.pth",
                detector_weights_path="models/yolo.pt",
                device="cpu",
            )

            assert predictor._available is False

    def test_predictor_has_thread_lock(self):
        """Test that predictor has a threading lock for thread safety."""
        with patch.dict("sys.modules", {"mivolo": None, "mivolo.predictor": None}):
            from src.photodb.stages.age_gender import MiVOLOPredictor
            import threading

            predictor = MiVOLOPredictor(
                checkpoint_path="models/mivolo.pth",
                detector_weights_path="models/yolo.pt",
                device="cpu",
            )

            assert hasattr(predictor, "_lock")
            assert isinstance(predictor._lock, type(threading.Lock()))


class TestComputeIoU:
    """Tests for the _compute_iou helper function."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes is 1.0."""
        from src.photodb.stages.age_gender import _compute_iou

        bbox = (100, 100, 50, 50)
        assert _compute_iou(bbox, bbox) == 1.0

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        from src.photodb.stages.age_gender import _compute_iou

        bbox1 = (0, 0, 50, 50)
        bbox2 = (100, 100, 50, 50)
        assert _compute_iou(bbox1, bbox2) == 0.0

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        from src.photodb.stages.age_gender import _compute_iou

        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 100, 100)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 ≈ 0.143
        iou = _compute_iou(bbox1, bbox2)
        assert 0.14 < iou < 0.15

    def test_contained_box(self):
        """Test IoU when one box is fully contained in another."""
        from src.photodb.stages.age_gender import _compute_iou

        bbox1 = (0, 0, 100, 100)
        bbox2 = (25, 25, 50, 50)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 2500 - 2500 = 10000
        # IoU: 2500/10000 = 0.25
        iou = _compute_iou(bbox1, bbox2)
        assert iou == 0.25

    def test_zero_area_box(self):
        """Test IoU with zero area box returns 0.0."""
        from src.photodb.stages.age_gender import _compute_iou

        bbox1 = (100, 100, 0, 0)  # Zero area
        bbox2 = (100, 100, 50, 50)
        assert _compute_iou(bbox1, bbox2) == 0.0
