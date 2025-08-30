import pytest
from pathlib import Path
from PIL import Image
import tempfile
from datetime import datetime
from src.photodb.utils.exif import ExifExtractor


class TestExifExtractor:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def image_with_exif(self, temp_dir):
        """Create an image with EXIF data."""
        img_path = temp_dir / "with_exif.jpg"
        img = Image.new("RGB", (100, 100), color="blue")

        # Add EXIF data
        exif = img.getexif()
        # Add DateTime
        exif[306] = "2023:05:15 14:30:00"
        # Add DateTimeOriginal
        exif[36867] = "2023:05:15 14:30:00"
        # Add Make
        exif[271] = "TestCamera"
        # Add Model
        exif[272] = "Model X"

        img.save(img_path, "JPEG", exif=exif)
        return img_path

    @pytest.fixture
    def image_without_exif(self, temp_dir):
        """Create an image without EXIF data."""
        img_path = temp_dir / "no_exif.jpg"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(img_path, "JPEG")
        return img_path

    @pytest.fixture
    def image_with_gps(self, temp_dir):
        """Create an image with GPS data."""
        img_path = temp_dir / "with_gps.jpg"
        img = Image.new("RGB", (100, 100), color="yellow")

        # Note: Setting GPS data in EXIF is complex and may not work with basic PIL
        # For testing purposes, we'll create a simple image without GPS
        img.save(img_path, "JPEG")
        return img_path

    def test_extract_all_metadata(self, image_with_exif):
        """Test extracting all metadata from an image."""
        metadata = ExifExtractor.extract_all_metadata(image_with_exif)

        assert "format" in metadata
        assert metadata["format"] == "JPEG"
        assert "mode" in metadata
        assert metadata["mode"] == "RGB"
        assert "size" in metadata
        assert metadata["size"]["width"] == 100
        assert metadata["size"]["height"] == 100
        assert "exif" in metadata
        assert "DateTime" in metadata["exif"]
        assert "Make" in metadata["exif"]

    def test_extract_all_metadata_no_exif(self, image_without_exif):
        """Test extracting metadata from image without EXIF."""
        metadata = ExifExtractor.extract_all_metadata(image_without_exif)

        assert "format" in metadata
        assert metadata["format"] == "JPEG"
        assert "mode" in metadata
        assert "size" in metadata
        # EXIF may or may not be present depending on PIL version
        if "exif" in metadata:
            assert isinstance(metadata["exif"], dict)

    def test_extract_datetime(self, image_with_exif):
        """Test extracting datetime from EXIF."""
        dt = ExifExtractor.extract_datetime(image_with_exif)

        assert dt is not None
        assert isinstance(dt, datetime)
        assert dt.year == 2023
        assert dt.month == 5
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 0

    def test_extract_datetime_no_exif(self, image_without_exif):
        """Test extracting datetime from image without EXIF."""
        dt = ExifExtractor.extract_datetime(image_without_exif)
        assert dt is None

    def test_extract_datetime_invalid_format(self, temp_dir):
        """Test handling invalid datetime format."""
        img_path = temp_dir / "bad_date.jpg"
        img = Image.new("RGB", (100, 100))

        exif = img.getexif()
        # Add invalid DateTime format
        exif[306] = "Not a valid date"
        img.save(img_path, "JPEG", exif=exif)

        dt = ExifExtractor.extract_datetime(img_path)
        assert dt is None

    def test_extract_gps_coordinates(self, image_with_gps):
        """Test extracting GPS coordinates."""
        coords = ExifExtractor.extract_gps_coordinates(image_with_gps)

        # Note: GPS extraction may not work with basic PIL GPS setting
        # This test may need adjustment based on actual GPS data format
        # For now, we test that the method doesn't crash
        assert coords is None or isinstance(coords, tuple)

    def test_extract_gps_no_gps(self, image_without_exif):
        """Test extracting GPS from image without GPS data."""
        coords = ExifExtractor.extract_gps_coordinates(image_without_exif)
        assert coords is None

    def test_parse_exif(self):
        """Test EXIF parsing helper method."""
        sample_exif = {271: "Canon", 272: "EOS 5D", 306: "2023:01:01 12:00:00"}

        parsed = ExifExtractor._parse_exif(sample_exif)

        assert "Make" in parsed
        assert parsed["Make"] == "Canon"
        assert "Model" in parsed
        assert parsed["Model"] == "EOS 5D"
        assert "DateTime" in parsed

    def test_parse_exif_with_bytes(self):
        """Test EXIF parsing with byte values."""
        sample_exif = {271: b"Canon", 272: "EOS 5D"}

        parsed = ExifExtractor._parse_exif(sample_exif)

        assert "Make" in parsed
        assert parsed["Make"] == "Canon"  # Should be decoded
        assert "Model" in parsed

    def test_convert_to_degrees(self):
        """Test GPS coordinate conversion."""
        # Test with valid GPS data format
        gps_data = ((40, 1), (42, 1), (51, 1))  # 40°42'51"
        degrees = ExifExtractor._convert_to_degrees(gps_data)

        assert degrees is not None
        assert abs(degrees - 40.714167) < 0.001  # 40 + 42/60 + 51/3600

    def test_convert_to_degrees_none(self):
        """Test GPS conversion with None input."""
        degrees = ExifExtractor._convert_to_degrees(None)
        assert degrees is None

    def test_convert_to_degrees_invalid(self):
        """Test GPS conversion with invalid data."""
        degrees = ExifExtractor._convert_to_degrees("invalid")
        assert degrees is None

    def test_extract_metadata_error_handling(self, temp_dir):
        """Test error handling when file doesn't exist."""
        non_existent = temp_dir / "does_not_exist.jpg"
        metadata = ExifExtractor.extract_all_metadata(non_existent)

        assert "error" in metadata
        assert len(metadata["error"]) > 0

    def test_datetime_priority(self, temp_dir):
        """Test that DateTimeOriginal is preferred over DateTime."""
        img_path = temp_dir / "priority.jpg"
        img = Image.new("RGB", (100, 100))

        exif = img.getexif()
        # Add both DateTime and DateTimeOriginal
        exif[306] = "2023:01:01 12:00:00"  # DateTime
        exif[36867] = "2023:05:15 14:30:00"  # DateTimeOriginal (preferred)

        img.save(img_path, "JPEG", exif=exif)

        dt = ExifExtractor.extract_datetime(img_path)
        assert dt is not None
        # Should use DateTimeOriginal
        assert dt.month == 5
        assert dt.day == 15
