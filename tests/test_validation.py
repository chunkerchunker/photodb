import pytest
from pathlib import Path
from PIL import Image
import tempfile
import hashlib
from src.photodb.utils.validation import ImageValidator


class TestImageValidator:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def valid_image(self, temp_dir):
        """Create a valid test image."""
        img_path = temp_dir / "valid.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path, "JPEG")
        return img_path

    @pytest.fixture
    def small_file(self, temp_dir):
        """Create a file that's too small."""
        file_path = temp_dir / "tiny.jpg"
        file_path.write_bytes(b"x" * 30)  # 30 bytes < 50 bytes
        return file_path

    @pytest.fixture
    def large_file(self, temp_dir):
        """Create a file that's too large."""
        file_path = temp_dir / "huge.jpg"
        # Create a 501MB file (exceeds 500MB limit)
        # Note: For testing, we'll just check the logic without actually creating 501MB
        # Instead, we'll mock this in the test
        img = Image.new("RGB", (100, 100))
        img.save(file_path, "JPEG")
        return file_path

    @pytest.fixture
    def corrupt_image(self, temp_dir):
        """Create a corrupt image file."""
        file_path = temp_dir / "corrupt.jpg"
        # Write invalid JPEG data
        file_path.write_bytes(b"Not a valid JPEG" + b"\xff\xd8\xff")
        return file_path

    def test_validate_file_valid(self, valid_image):
        """Test validating a valid image file."""
        assert ImageValidator.validate_file(valid_image) is True

    def test_validate_file_not_exists(self, temp_dir):
        """Test validating non-existent file."""
        non_existent = temp_dir / "does_not_exist.jpg"
        assert ImageValidator.validate_file(non_existent) is False

    def test_validate_file_too_small(self, small_file):
        """Test validating file that's too small."""
        assert ImageValidator.validate_file(small_file) is False

    def test_validate_file_too_large(self, temp_dir, monkeypatch):
        """Test validating file that's too large."""
        # Create a normal file but mock its size
        file_path = temp_dir / "large.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(file_path, "JPEG")

        # Mock the Path.stat method at the class level
        import pathlib

        original_stat = pathlib.Path.stat

        class MockStat:
            st_size = 501 * 1024 * 1024  # 501MB

        def mock_stat(self, **kwargs):
            if self == file_path:
                return MockStat()
            return original_stat(self, **kwargs)

        monkeypatch.setattr(pathlib.Path, "stat", mock_stat)

        assert ImageValidator.validate_file(file_path) is False

    def test_validate_file_corrupt(self, corrupt_image):
        """Test validating corrupt image file."""
        assert ImageValidator.validate_file(corrupt_image) is False

    def test_validate_file_text(self, temp_dir):
        """Test validating text file pretending to be image."""
        text_file = temp_dir / "text.jpg"
        text_file.write_text("This is just text, not an image")
        assert ImageValidator.validate_file(text_file) is False

    def test_calculate_checksum(self, valid_image):
        """Test checksum calculation."""
        checksum = ImageValidator.calculate_checksum(valid_image)

        # Verify it's a valid SHA-256 hash (64 hex characters)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

        # Verify same file gives same checksum
        checksum2 = ImageValidator.calculate_checksum(valid_image)
        assert checksum == checksum2

    def test_calculate_checksum_different_files(self, temp_dir):
        """Test that different files have different checksums."""
        # Create two different images
        img1_path = temp_dir / "img1.jpg"
        img1 = Image.new("RGB", (100, 100), color="red")
        img1.save(img1_path, "JPEG")

        img2_path = temp_dir / "img2.jpg"
        img2 = Image.new("RGB", (100, 100), color="blue")
        img2.save(img2_path, "JPEG")

        checksum1 = ImageValidator.calculate_checksum(img1_path)
        checksum2 = ImageValidator.calculate_checksum(img2_path)

        assert checksum1 != checksum2

    def test_calculate_checksum_manual_verification(self, temp_dir):
        """Test checksum calculation against known value."""
        # Create a file with known content
        test_file = temp_dir / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        checksum = ImageValidator.calculate_checksum(test_file)

        # Calculate expected checksum
        expected = hashlib.sha256(test_content).hexdigest()

        assert checksum == expected

    def test_validate_various_formats(self, temp_dir):
        """Test validation of various image formats."""
        formats = [
            ("RGB", "JPEG", ".jpg"),
            ("RGB", "PNG", ".png"),
            ("RGB", "BMP", ".bmp"),
            ("RGB", "TIFF", ".tiff"),
            ("RGB", "WEBP", ".webp"),
            ("P", "GIF", ".gif"),
        ]

        for mode, format_name, extension in formats:
            file_path = temp_dir / f"test{extension}"
            img = Image.new(mode, (100, 100))
            if mode == "P":
                # For palette mode, create a simple palette
                img.putpalette([i for i in range(256)] * 3)
            img.save(file_path, format_name)

            assert ImageValidator.validate_file(file_path) is True, f"Failed for {format_name}"

    def test_validate_file_edge_cases(self, temp_dir):
        """Test edge cases for file validation."""
        # Exactly 50 bytes file (should be valid)
        edge_file = temp_dir / "edge.jpg"
        with open(edge_file, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0")  # JPEG header
            f.write(b"X" * (50 - 4))  # Pad to 50 bytes

        # This will fail image verification but tests size boundary
        result = ImageValidator.validate_file(edge_file)
        # Should fail due to invalid image, not size
        assert result is False

        # Test with valid small image
        valid_edge = temp_dir / "valid_edge.jpg"
        img = Image.new("RGB", (50, 50))
        img.save(valid_edge, "JPEG", quality=95)

        # Should be valid as it's over 50 bytes
        assert ImageValidator.validate_file(valid_edge) is True
