"""Tests for Apple Vision scene classifier."""

import sys
import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Apple Vision only available on macOS"
)


class TestAppleVisionClassifier:
    """Tests for AppleVisionClassifier using real Apple Vision framework."""

    @pytest.fixture
    def test_image_path(self):
        """Return path to test image."""
        return Path(__file__).parent.parent / "test_photos" / "test.jpg"

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance."""
        from photodb.utils.apple_vision_classifier import AppleVisionClassifier

        return AppleVisionClassifier()

    def test_classify_returns_success_status(self, classifier, test_image_path):
        """Test that classify returns success status for valid image."""
        result = classifier.classify(str(test_image_path))

        assert result["status"] == "success"
        assert "classifications" in result
        assert "processing_time_ms" in result

    def test_classify_returns_labels(self, classifier, test_image_path):
        """Test that classify returns classification labels with confidence."""
        result = classifier.classify(str(test_image_path))

        assert result["status"] == "success"
        assert len(result["classifications"]) > 0

        # Check structure of first classification
        first = result["classifications"][0]
        assert "identifier" in first
        assert "confidence" in first
        assert isinstance(first["identifier"], str)
        assert isinstance(first["confidence"], float)
        assert 0.0 <= first["confidence"] <= 1.0

    def test_classify_returns_top_k(self, classifier, test_image_path):
        """Test that classify respects top_k parameter."""
        result = classifier.classify(str(test_image_path), top_k=5)

        assert result["status"] == "success"
        assert len(result["classifications"]) <= 5

    def test_classify_returns_sorted_by_confidence(self, classifier, test_image_path):
        """Test that classifications are sorted by confidence descending."""
        result = classifier.classify(str(test_image_path), top_k=10)

        assert result["status"] == "success"
        classifications = result["classifications"]

        if len(classifications) > 1:
            for i in range(len(classifications) - 1):
                assert classifications[i]["confidence"] >= classifications[i + 1]["confidence"]

    def test_classify_respects_min_confidence(self, classifier, test_image_path):
        """Test that classify filters by min_confidence."""
        result = classifier.classify(str(test_image_path), min_confidence=0.5)

        assert result["status"] == "success"
        for item in result["classifications"]:
            assert item["confidence"] >= 0.5

    def test_classify_nonexistent_file_returns_error(self, classifier):
        """Test that classify returns error for nonexistent file."""
        result = classifier.classify("/nonexistent/path/to/image.jpg")

        assert result["status"] == "error"
        assert result["classifications"] == []
        assert "error" in result
        assert "processing_time_ms" in result

    def test_classify_invalid_file_returns_error(self, classifier, tmp_path):
        """Test that classify returns error for invalid image file."""
        # Create an invalid image file (just text content)
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_text("this is not an image")

        result = classifier.classify(str(invalid_file))

        assert result["status"] == "error"
        assert result["classifications"] == []
        assert "error" in result

    def test_processing_time_is_positive(self, classifier, test_image_path):
        """Test that processing_time_ms is a positive integer."""
        result = classifier.classify(str(test_image_path))

        assert result["processing_time_ms"] >= 0
        assert isinstance(result["processing_time_ms"], int)


class TestAppleVisionClassifierInit:
    """Tests for AppleVisionClassifier initialization."""

    def test_init_creates_instance(self):
        """Test that classifier can be initialized."""
        from photodb.utils.apple_vision_classifier import AppleVisionClassifier

        classifier = AppleVisionClassifier()
        assert classifier is not None
