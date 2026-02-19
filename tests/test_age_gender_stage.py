"""
Tests for AgeGenderStage.

Tests the age/gender estimation stage that uses MiVOLO to estimate age and gender
for existing person detections. Pre-computed bounding boxes from the detection stage
are passed directly to MiVOLO, eliminating redundant YOLO detection and IoU matching.
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
        """Create a mock MiVOLOPredictor that returns dict of predictions."""
        predictor = MagicMock()
        # Default: return empty dict (no predictions)
        predictor.predict.return_value = {}
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

    def test_process_photo_skips_without_med_path(
        self, mock_repository, config, mock_mivolo_predictor
    ):
        """Test that process_photo returns False if photo has no med_path."""
        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
            photo = Photo(
                id=1,
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=None,
                width=640,
                height=480,
                med_width=None,
                med_height=None,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is False
            mock_mivolo_predictor.predict.assert_not_called()

    def test_process_photo_skips_nonexistent_med_file(
        self, mock_repository, config, mock_mivolo_predictor, temp_dir
    ):
        """Test that process_photo returns False if medium file doesn't exist."""
        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
            photo = Photo(
                id=1,
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path="nonexistent.jpg",
                width=640,
                height=480,
                med_width=None,
                med_height=None,
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
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is True
            mock_mivolo_predictor.predict.assert_not_called()

    def test_process_photo_updates_detection_with_age_gender(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that age/gender data is updated for matched detections."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            collection_id=1,
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

        # Prediction indexed by detection position
        mock_mivolo_predictor.predict.return_value = {
            0: {
                "face_bbox": (100, 50, 100, 100),
                "body_bbox": (50, 20, 250, 380),
                "age": 32.5,
                "gender": "F",
                "gender_confidence": 0.92,
            }
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
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            mock_repository.update_detection_age_gender.assert_called_once()

            call_args = mock_repository.update_detection_age_gender.call_args
            assert call_args[1]["detection_id"] == 10
            assert call_args[1]["age_estimate"] == 32.5
            assert call_args[1]["gender"] == "F"
            assert call_args[1]["gender_confidence"] == 0.92

    def test_process_photo_calls_predict_with_detections(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that predict is called with image path and detections list."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            collection_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]
        mock_mivolo_predictor.predict.return_value = {}

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
            photo = Photo(
                id=1,
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            stage.process_photo(photo, Path("/path/to/photo.jpg"))

            mock_mivolo_predictor.predict.assert_called_once()
            call_args = mock_mivolo_predictor.predict.call_args
            # First arg: image path string
            assert sample_image.name in call_args[0][0]
            # Second arg: detections list
            assert call_args[0][1] == [detection]

    def test_process_photo_handles_body_only_detection(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that detections with only body bbox are handled correctly."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            collection_id=1,
            body_bbox_x=50,
            body_bbox_y=20,
            body_bbox_width=250,
            body_bbox_height=380,
            body_confidence=0.9,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        mock_mivolo_predictor.predict.return_value = {
            0: {
                "face_bbox": None,
                "body_bbox": (50, 20, 250, 380),
                "age": 28.0,
                "gender": "M",
                "gender_confidence": 0.88,
            }
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
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
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
        """Test that multiple predictions are mapped to multiple detections."""
        from src.photodb.database.models import PersonDetection

        detections = [
            PersonDetection(
                id=10,
                photo_id=1,
                collection_id=1,
                face_bbox_x=100,
                face_bbox_y=50,
                face_bbox_width=80,
                face_bbox_height=100,
                face_confidence=0.95,
            ),
            PersonDetection(
                id=11,
                photo_id=1,
                collection_id=1,
                face_bbox_x=300,
                face_bbox_y=50,
                face_bbox_width=90,
                face_bbox_height=110,
                face_confidence=0.85,
            ),
        ]
        mock_repository.get_detections_for_photo.return_value = detections

        mock_mivolo_predictor.predict.return_value = {
            0: {
                "face_bbox": (100, 50, 80, 100),
                "body_bbox": None,
                "age": 25.0,
                "gender": "M",
                "gender_confidence": 0.9,
            },
            1: {
                "face_bbox": (300, 50, 90, 110),
                "body_bbox": None,
                "age": 30.0,
                "gender": "F",
                "gender_confidence": 0.85,
            },
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
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))

            assert result is True
            assert mock_mivolo_predictor.predict.call_count == 1
            assert mock_repository.update_detection_age_gender.call_count == 2

    def test_process_photo_returns_true_when_no_predictions(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that process_photo returns True when MiVOLO returns no predictions."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=10,
            photo_id=1,
            collection_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]
        mock_mivolo_predictor.predict.return_value = {}

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
            photo = Photo(
                id=1,
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
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
            collection_id=1,
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
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
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
            collection_id=1,
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
        mock_mivolo_predictor.predict.return_value = {0: mivolo_result}

        with patch(
            "src.photodb.stages.age_gender.MiVOLOPredictor",
            return_value=mock_mivolo_predictor,
        ):
            from src.photodb.stages.age_gender import AgeGenderStage
            from src.photodb.database.models import Photo

            stage = AgeGenderStage(mock_repository, config)
            photo = Photo(
                id=1,
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            stage.process_photo(photo, Path("/path/to/photo.jpg"))

            call_args = mock_repository.update_detection_age_gender.call_args
            assert call_args[1]["mivolo_output"] == mivolo_result

    def test_process_photo_skips_detection_with_no_id(
        self, mock_repository, config, mock_mivolo_predictor, sample_image
    ):
        """Test that detections without an ID are skipped."""
        from src.photodb.database.models import PersonDetection

        detection = PersonDetection(
            id=None,
            photo_id=1,
            collection_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=80,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [detection]

        mock_mivolo_predictor.predict.return_value = {
            0: {
                "face_bbox": (100, 50, 80, 100),
                "body_bbox": None,
                "age": 25.0,
                "gender": "M",
                "gender_confidence": 0.9,
            }
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
                collection_id=1,
                orig_path="/path/to/photo.jpg",
                full_path=None,
                med_path=sample_image.name,
                width=640,
                height=480,
                med_width=640,
                med_height=480,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            result = stage.process_photo(photo, Path("/path/to/photo.jpg"))
            assert result is True
            mock_repository.update_detection_age_gender.assert_not_called()


class TestMiVOLOPredictorUnit:
    """Unit tests for MiVOLOPredictor wrapper class."""

    def test_predictor_gracefully_handles_import_error(self):
        """Test that MiVOLOPredictor works when MiVOLO is not installed."""
        with patch.dict("sys.modules", {"mivolo": None, "mivolo.model": None, "mivolo.model.mi_volo": None}):
            from src.photodb.stages.age_gender import MiVOLOPredictor

            predictor = MiVOLOPredictor(
                checkpoint_path="models/mivolo.pth",
                device="cpu",
            )

            result = predictor.predict("/path/to/image.jpg", [])
            assert result == {}

    def test_predictor_available_flag(self):
        """Test that _available flag is set correctly."""
        with patch.dict("sys.modules", {"mivolo": None, "mivolo.model": None, "mivolo.model.mi_volo": None}):
            from src.photodb.stages.age_gender import MiVOLOPredictor

            predictor = MiVOLOPredictor(
                checkpoint_path="models/mivolo.pth",
                device="cpu",
            )

            assert predictor._available is False

    def test_predictor_has_no_thread_lock(self):
        """Test that predictor does not have a threading lock (YOLO removed)."""
        with patch.dict("sys.modules", {"mivolo": None, "mivolo.model": None, "mivolo.model.mi_volo": None}):
            from src.photodb.stages.age_gender import MiVOLOPredictor

            predictor = MiVOLOPredictor(
                checkpoint_path="models/mivolo.pth",
                device="cpu",
            )

            assert not hasattr(predictor, "_lock")

    def test_predictor_no_detector_weights_param(self):
        """Test that MiVOLOPredictor does not accept detector_weights_path."""
        import inspect
        from src.photodb.stages.age_gender import MiVOLOPredictor

        sig = inspect.signature(MiVOLOPredictor.__init__)
        assert "detector_weights_path" not in sig.parameters


class TestMiVOLOThreadSafety:
    """Thread safety test for MiVOLO.predict() without serialization lock."""

    MIVOLO_CHECKPOINT = "models/mivolo_d1.pth.tar"

    @pytest.fixture
    def mivolo_model(self):
        """Create a real MiVOLO model instance, skip if model files unavailable."""
        import os

        if not os.path.exists(self.MIVOLO_CHECKPOINT):
            pytest.skip(f"MiVOLO checkpoint not found: {self.MIVOLO_CHECKPOINT}")

        try:
            from src.photodb.utils import timm_compat  # noqa: F401
            from src.photodb.stages.age_gender import _patch_torch_load
            import torch

            original_load, patched_load = _patch_torch_load()
            torch.load = patched_load
            try:
                from mivolo.model.mi_volo import MiVOLO

                model = MiVOLO(
                    ckpt_path=self.MIVOLO_CHECKPOINT,
                    device="cpu",
                    half=False,
                    use_persons=True,
                    disable_faces=False,
                    verbose=False,
                )
            finally:
                torch.load = original_load

            return model
        except Exception as e:
            pytest.skip(f"Could not initialize MiVOLO model: {e}")

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create multiple sample test images with synthetic detections."""
        import cv2
        import numpy as np

        images = []
        for i in range(4):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            path = tmp_path / f"test_{i}.jpg"
            cv2.imwrite(str(path), img)
            images.append(str(path))
        return images

    def test_mivolo_predict_thread_safety(self, mivolo_model, sample_images):
        """
        Run MiVOLO.predict() concurrently from 4 threads and verify consistent results.

        If results are inconsistent across runs, the lock needs to be re-added.
        """
        import cv2
        import concurrent.futures
        from src.photodb.stages.age_gender import MiVOLOPredictor

        def run_prediction(image_path):
            """Run prediction and return results for comparison."""
            img = cv2.imread(image_path)
            from src.photodb.database.models import PersonDetection

            # Create a synthetic detection covering the whole image
            det = PersonDetection(
                id=1,
                photo_id=1,
                collection_id=1,
                body_bbox_x=0,
                body_bbox_y=0,
                body_bbox_width=640,
                body_bbox_height=480,
                body_confidence=0.9,
                face_bbox_x=200,
                face_bbox_y=100,
                face_bbox_width=100,
                face_bbox_height=120,
                face_confidence=0.9,
            )

            detected_objects = MiVOLOPredictor._build_synthetic_result(img, [det])
            mivolo_model.predict(img, detected_objects)

            # Collect results
            ages = [a for a in detected_objects.ages if a is not None]
            genders = [g for g in detected_objects.genders if g is not None]
            return (tuple(ages), tuple(genders))

        # Run each image 3 times serially to establish baseline results
        baseline = {}
        for image_path in sample_images:
            baseline[image_path] = run_prediction(image_path)

        # Run concurrently from 4 threads, repeat 5 times
        inconsistent = 0
        for _ in range(5):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(run_prediction, path): path for path in sample_images
                }
                for future in concurrent.futures.as_completed(futures):
                    path = futures[future]
                    result = future.result()
                    if result != baseline[path]:
                        inconsistent += 1

        if inconsistent > 0:
            pytest.fail(
                f"MiVOLO.predict() produced inconsistent results in {inconsistent} "
                f"out of {5 * len(sample_images)} concurrent runs. "
                f"Thread lock should be re-added to MiVOLOPredictor."
            )
