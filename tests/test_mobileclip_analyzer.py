"""Tests for MobileCLIP analyzer."""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np


class TestMobileCLIPAnalyzer:
    def test_encode_image(self, test_image_path):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        embedding = analyzer.encode_image(str(test_image_path))

        assert embedding.shape == (1, 512)

    def test_encode_text(self):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        embedding = analyzer.encode_text("a happy scene")

        assert embedding.shape == (1, 512)

    def test_encode_texts_batch(self):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        texts = ["happy", "sad", "neutral"]
        embeddings = analyzer.encode_texts(texts)

        assert embeddings.shape == (3, 512)

    def test_encode_face_crop(self, face_crop):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        embedding = analyzer.encode_face(face_crop)

        assert embedding.shape == (1, 512)

    def test_encode_face_from_bbox(self, test_image_path):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        bbox = {"x1": 10, "y1": 10, "x2": 100, "y2": 100}
        embedding = analyzer.encode_face_from_bbox(test_image_path, bbox)

        assert embedding.shape == (1, 512)

    def test_encode_faces_batch(self, test_image_path):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        bboxes = [
            {"x1": 10, "y1": 10, "x2": 100, "y2": 100},
            {"x1": 50, "y1": 50, "x2": 150, "y2": 150},
        ]
        embeddings = analyzer.encode_faces_batch(test_image_path, bboxes)

        assert embeddings.shape == (2, 512)

    def test_encode_faces_batch_empty(self, test_image_path):
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        embeddings = analyzer.encode_faces_batch(test_image_path, [])

        assert embeddings.shape == (0, 512)


@pytest.fixture
def test_image_path():
    return Path(__file__).parent.parent / "test_photos" / "test.jpg"


@pytest.fixture
def face_crop():
    return Image.fromarray(np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8))
