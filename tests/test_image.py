import pytest
from pathlib import Path
from PIL import Image
import tempfile
from src.photodb.utils.image import ImageHandler


class TestImageHandler:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample test image."""
        img_path = temp_dir / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path, "JPEG")
        return img_path

    @pytest.fixture
    def sample_rgba_image(self, temp_dir):
        """Create a sample RGBA image."""
        img_path = temp_dir / "test_rgba.png"
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(img_path, "PNG")
        return img_path

    def test_is_supported(self):
        """Test format support detection."""
        assert ImageHandler.is_supported(Path("test.jpg"))
        assert ImageHandler.is_supported(Path("test.JPEG"))
        assert ImageHandler.is_supported(Path("test.png"))
        assert ImageHandler.is_supported(Path("test.heic"))
        assert ImageHandler.is_supported(Path("test.heif"))
        assert ImageHandler.is_supported(Path("test.bmp"))
        assert ImageHandler.is_supported(Path("test.tiff"))
        assert ImageHandler.is_supported(Path("test.webp"))
        assert ImageHandler.is_supported(Path("test.gif"))
        assert not ImageHandler.is_supported(Path("test.txt"))
        assert not ImageHandler.is_supported(Path("test.pdf"))

    def test_open_image(self, sample_image):
        """Test opening an image file."""
        img = ImageHandler.open_image(sample_image)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_open_image_unsupported(self, temp_dir):
        """Test opening unsupported file format."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("not an image")

        with pytest.raises(ValueError, match="Unsupported format"):
            ImageHandler.open_image(txt_file)

    def test_open_image_rgba_conversion(self, sample_rgba_image):
        """Test RGBA to RGB conversion."""
        img = ImageHandler.open_image(sample_rgba_image)
        assert img.mode == "RGB"
        # Should have white background where alpha was transparent
        assert img.size == (100, 100)

    def test_open_image_decompression_bomb(self, temp_dir):
        """Test protection against decompression bombs."""
        # Create a fake large image
        img_path = temp_dir / "large.jpg"
        # Create image just over the limit
        width = 15000
        height = 12000  # 180 megapixels > 179 megapixel limit
        img = Image.new("RGB", (width, height))
        img.save(img_path, "JPEG")

        with pytest.raises(ValueError, match="Image too large"):
            ImageHandler.open_image(img_path)

    def test_get_image_info(self, sample_image):
        """Test getting image information."""
        info = ImageHandler.get_image_info(sample_image)
        assert info["width"] == 100
        assert info["height"] == 100
        assert info["format"] == "JPEG"
        assert info["mode"] == "RGB"
        assert "size_bytes" in info
        assert info["size_bytes"] > 0

    def test_calculate_resize_dimensions_no_resize(self):
        """Test resize calculation when no resize is needed."""
        current_size = (500, 500)
        max_dims = {"1:1": (1092, 1092)}
        result = ImageHandler.calculate_resize_dimensions(current_size, max_dims)
        assert result is None  # No resize needed

    def test_calculate_resize_dimensions_square(self):
        """Test resize calculation for square image."""
        current_size = (2000, 2000)
        max_dims = {"1:1": (1092, 1092)}
        result = ImageHandler.calculate_resize_dimensions(current_size, max_dims)
        assert result == (1092, 1092)

    def test_calculate_resize_dimensions_landscape(self):
        """Test resize calculation for landscape image."""
        current_size = (3000, 2000)  # 3:2 aspect ratio
        max_dims = {"3:2": (1344, 896)}
        result = ImageHandler.calculate_resize_dimensions(current_size, max_dims)
        assert result == (1344, 896)

    def test_calculate_resize_dimensions_portrait(self):
        """Test resize calculation for portrait image."""
        current_size = (2000, 3000)  # 2:3 aspect ratio
        max_dims = {"2:3": (896, 1344)}
        result = ImageHandler.calculate_resize_dimensions(current_size, max_dims)
        assert result == (896, 1344)

    def test_resize_image(self, sample_image):
        """Test image resizing."""
        img = ImageHandler.open_image(sample_image)
        resized = ImageHandler.resize_image(img, (50, 50))
        assert resized.size == (50, 50)
        assert resized.mode == img.mode

    def test_save_as_png(self, sample_image, temp_dir):
        """Test saving image as PNG."""
        img = ImageHandler.open_image(sample_image)
        output_path = temp_dir / "output.png"

        ImageHandler.save_as_png(img, output_path, optimize=True)

        assert output_path.exists()
        # Verify it's a PNG
        with Image.open(output_path) as saved_img:
            assert saved_img.format == "PNG"
            assert saved_img.size == img.size

    def test_save_as_png_creates_directory(self, sample_image, temp_dir):
        """Test that save_as_png creates parent directories."""
        img = ImageHandler.open_image(sample_image)
        output_path = temp_dir / "nested" / "dir" / "output.png"

        ImageHandler.save_as_png(img, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_mode_conversions(self, temp_dir):
        """Test various image mode conversions."""
        # Test palette mode
        img_path = temp_dir / "palette.gif"
        img = Image.new("P", (100, 100))
        img.save(img_path, "GIF")
        converted = ImageHandler.open_image(img_path)
        assert converted.mode == "RGB"

        # Test LA mode (grayscale with alpha)
        img_path = temp_dir / "la.png"
        img = Image.new("LA", (100, 100))
        img.save(img_path, "PNG")
        converted = ImageHandler.open_image(img_path)
        assert converted.mode == "RGB"

        # Test L mode (grayscale)
        img_path = temp_dir / "gray.jpg"
        img = Image.new("L", (100, 100))
        img.save(img_path, "JPEG")
        converted = ImageHandler.open_image(img_path)
        assert converted.mode == "L"  # L mode is kept as-is
