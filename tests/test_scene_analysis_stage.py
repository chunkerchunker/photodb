"""Tests for scene analysis stage."""

import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image


class TestSceneAnalysisStage:
    """Unit tests for SceneAnalysisStage using mocks."""

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
        repo.get_prompt_categories.return_value = []
        repo.create_analysis_output.return_value = None
        repo.bulk_upsert_photo_tags.return_value = None
        repo.bulk_upsert_detection_tags.return_value = None
        repo.upsert_scene_analysis.return_value = None
        return repo

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return {
            "IMG_PATH": str(temp_dir),
        }

    @pytest.fixture
    def mock_photo(self, sample_image):
        """Create a mock Photo object."""
        from photodb.database.models import Photo

        return Photo(
            id=1,
            filename="/path/to/test.jpg",
            normalized_path=sample_image.name,
            width=640,
            height=480,
            normalized_width=640,
            normalized_height=480,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def mock_mobileclip_analyzer(self):
        """Create a mock MobileCLIPAnalyzer."""
        analyzer = MagicMock()
        analyzer.encode_image.return_value = torch.randn(1, 512)
        analyzer.encode_faces_batch.return_value = torch.randn(0, 512)
        analyzer.device = "cpu"
        return analyzer

    @pytest.fixture
    def mock_prompt_cache(self):
        """Create a mock PromptCache."""
        from photodb.database.models import PromptCategory

        cache = MagicMock()
        cache.classify.return_value = {"joyful": 0.8, "peaceful": 0.1}
        cache.classify_multi.return_value = [("beach", 0.7, 1)]
        cache.get_prompt_ids.return_value = [1, 2]
        cache.get_category.return_value = (
            ["joyful", "peaceful"],
            torch.randn(2, 512),
            PromptCategory(
                id=1,
                name="scene_mood",
                target="scene",
                selection_mode="single",
                min_confidence=0.1,
                max_results=5,
                description=None,
                display_order=0,
                is_active=True,
                created_at=None,
                updated_at=None,
            ),
        )
        return cache

    def test_stage_name_is_scene_analysis(
        self, mock_repository, config, mock_mobileclip_analyzer, mock_prompt_cache
    ):
        """Test that stage_name is 'scene_analysis'."""
        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                from photodb.stages.scene_analysis import SceneAnalysisStage

                stage = SceneAnalysisStage(mock_repository, config)
                assert stage.stage_name == "scene_analysis"

    def test_inherits_from_base_stage(
        self, mock_repository, config, mock_mobileclip_analyzer, mock_prompt_cache
    ):
        """Test that SceneAnalysisStage inherits from BaseStage."""
        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                from photodb.stages.scene_analysis import SceneAnalysisStage
                from photodb.stages.base import BaseStage

                stage = SceneAnalysisStage(mock_repository, config)
                assert isinstance(stage, BaseStage)

    def test_process_photo_skips_without_normalized_path(
        self, mock_repository, config, mock_mobileclip_analyzer, mock_prompt_cache
    ):
        """Test that process_photo returns False if photo has no normalized_path."""
        from photodb.database.models import Photo

        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                from photodb.stages.scene_analysis import SceneAnalysisStage

                stage = SceneAnalysisStage(mock_repository, config)
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

    def test_process_photo_skips_nonexistent_normalized_file(
        self, mock_repository, config, mock_mobileclip_analyzer, mock_prompt_cache, temp_dir
    ):
        """Test that process_photo returns False if normalized file doesn't exist."""
        from photodb.database.models import Photo

        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                from photodb.stages.scene_analysis import SceneAnalysisStage

                stage = SceneAnalysisStage(mock_repository, config)
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

    def test_process_creates_photo_tags(
        self,
        mock_repository,
        config,
        mock_mobileclip_analyzer,
        mock_prompt_cache,
        mock_photo,
    ):
        """Test that processing creates photo tags."""
        from photodb.database.models import PromptCategory

        # Set up a scene category
        scene_category = PromptCategory(
            id=1,
            name="scene_mood",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repository.get_prompt_categories.return_value = [scene_category]

        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                # Also mock Apple Vision to not run on non-macOS
                with patch("photodb.stages.scene_analysis._apple_vision_available", False):
                    from photodb.stages.scene_analysis import SceneAnalysisStage

                    stage = SceneAnalysisStage(mock_repository, config)
                    result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                    assert result is True
                    assert mock_repository.bulk_upsert_photo_tags.called

    def test_process_stores_scene_analysis(
        self,
        mock_repository,
        config,
        mock_mobileclip_analyzer,
        mock_prompt_cache,
        mock_photo,
    ):
        """Test that processing stores SceneAnalysis record."""
        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                with patch("photodb.stages.scene_analysis._apple_vision_available", False):
                    from photodb.stages.scene_analysis import SceneAnalysisStage

                    stage = SceneAnalysisStage(mock_repository, config)
                    result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                    assert result is True
                    assert mock_repository.upsert_scene_analysis.called

    def test_process_stores_analysis_output(
        self,
        mock_repository,
        config,
        mock_mobileclip_analyzer,
        mock_prompt_cache,
        mock_photo,
    ):
        """Test that processing stores AnalysisOutput record."""
        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                with patch("photodb.stages.scene_analysis._apple_vision_available", False):
                    from photodb.stages.scene_analysis import SceneAnalysisStage

                    stage = SceneAnalysisStage(mock_repository, config)
                    result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                    assert result is True
                    assert mock_repository.create_analysis_output.called

    def test_process_handles_multi_select_categories(
        self,
        mock_repository,
        config,
        mock_mobileclip_analyzer,
        mock_prompt_cache,
        mock_photo,
    ):
        """Test that multi-select categories are handled correctly."""
        from photodb.database.models import PromptCategory

        # Set up a multi-select scene category
        multi_category = PromptCategory(
            id=2,
            name="scene_objects",
            target="scene",
            selection_mode="multi",
            min_confidence=0.3,
            max_results=10,
            description=None,
            display_order=1,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repository.get_prompt_categories.return_value = [multi_category]

        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                with patch("photodb.stages.scene_analysis._apple_vision_available", False):
                    from photodb.stages.scene_analysis import SceneAnalysisStage

                    stage = SceneAnalysisStage(mock_repository, config)
                    result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                    assert result is True
                    # Should have called classify_multi for multi category
                    assert mock_prompt_cache.classify_multi.called

    def test_process_handles_face_detections(
        self,
        mock_repository,
        config,
        mock_mobileclip_analyzer,
        mock_prompt_cache,
        mock_photo,
    ):
        """Test that face detections are processed for face categories."""
        from photodb.database.models import PersonDetection, PromptCategory

        # Set up face detection
        face_detection = PersonDetection(
            id=1,
            photo_id=1,
            face_bbox_x=100,
            face_bbox_y=50,
            face_bbox_width=100,
            face_bbox_height=100,
            face_confidence=0.95,
        )
        mock_repository.get_detections_for_photo.return_value = [face_detection]

        # Set up face category
        face_category = PromptCategory(
            id=3,
            name="face_emotion",
            target="face",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repository.get_prompt_categories.side_effect = lambda target=None: (
            [face_category] if target == "face" else []
        )

        # Mock face embedding
        mock_mobileclip_analyzer.encode_faces_batch.return_value = torch.randn(1, 512)

        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                with patch("photodb.stages.scene_analysis._apple_vision_available", False):
                    from photodb.stages.scene_analysis import SceneAnalysisStage

                    stage = SceneAnalysisStage(mock_repository, config)
                    result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                    assert result is True
                    # Should have encoded faces
                    assert mock_mobileclip_analyzer.encode_faces_batch.called

    def test_process_handles_exception(
        self,
        mock_repository,
        config,
        mock_mobileclip_analyzer,
        mock_prompt_cache,
        mock_photo,
    ):
        """Test that exceptions during processing are handled gracefully."""
        mock_mobileclip_analyzer.encode_image.side_effect = Exception("Encoding failed")

        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                with patch("photodb.stages.scene_analysis._apple_vision_available", False):
                    from photodb.stages.scene_analysis import SceneAnalysisStage

                    stage = SceneAnalysisStage(mock_repository, config)
                    result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))
                    assert result is False

    def test_process_returns_true_with_no_categories(
        self,
        mock_repository,
        config,
        mock_mobileclip_analyzer,
        mock_prompt_cache,
        mock_photo,
    ):
        """Test that processing succeeds even without categories."""
        mock_repository.get_prompt_categories.return_value = []

        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                with patch("photodb.stages.scene_analysis._apple_vision_available", False):
                    from photodb.stages.scene_analysis import SceneAnalysisStage

                    stage = SceneAnalysisStage(mock_repository, config)
                    result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                    assert result is True

    def test_category_classification_error_continues(
        self,
        mock_repository,
        config,
        mock_mobileclip_analyzer,
        mock_prompt_cache,
        mock_photo,
    ):
        """Test that category classification errors don't stop processing."""
        from photodb.database.models import PromptCategory

        # Set up a scene category
        scene_category = PromptCategory(
            id=1,
            name="scene_mood",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repository.get_prompt_categories.return_value = [scene_category]

        # Make classify raise an exception
        mock_prompt_cache.classify.side_effect = ValueError("Category not found")

        with patch(
            "photodb.stages.scene_analysis.MobileCLIPAnalyzer",
            return_value=mock_mobileclip_analyzer,
        ):
            with patch(
                "photodb.stages.scene_analysis.PromptCache",
                return_value=mock_prompt_cache,
            ):
                with patch("photodb.stages.scene_analysis._apple_vision_available", False):
                    from photodb.stages.scene_analysis import SceneAnalysisStage

                    stage = SceneAnalysisStage(mock_repository, config)
                    # Should still succeed even if category classification fails
                    result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                    assert result is True
