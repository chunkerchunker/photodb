# pyvips Normalize Stage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Pillow + pillow_heif with pyvips in the normalize stage for 3-5x faster image processing on Apple Silicon.

**Architecture:** Rewrite `ImageHandler` to use pyvips for all pixel operations (decode, resize, rotate, encode). The normalize stage simplifies dramatically because pyvips `thumbnail()` does shrink-on-load + auto-rotate in one call, and `autorot()` handles EXIF orientation without manual code. Pillow remains in the codebase for other stages (detection crops, CLIP preprocessing, EXIF metadata) that are unaffected.

**Tech Stack:** pyvips (Python bindings for libvips), installed via `brew install vips` + `uv add pyvips`

---

### Task 1: Add dependencies

**Files:**
- Modify: `Brewfile`
- Modify: `pyproject.toml`
- Modify: `Install.md`

**Step 1: Add vips to Brewfile**

```ruby
# Image processing (libvips for fast HEIF/JPEG → WebP conversion)
brew "vips"
```

Add after the `# Utilities` section.

**Step 2: Add pyvips to pyproject.toml**

Add `"pyvips>=2.2.0"` to the `dependencies` list.

**Step 3: Update Install.md**

Change line 11 from:
```
This installs from the `Brewfile`: git, just, node, pnpm, uv, PostgreSQL 18, pgvector, jq.
```
to:
```
This installs from the `Brewfile`: git, just, node, pnpm, uv, PostgreSQL 18, pgvector, jq, vips.
```

**Step 4: Install**

Run:
```bash
brew bundle
uv sync
```

**Step 5: Verify pyvips can load**

Run: `uv run python -c "import pyvips; print(pyvips.version(0), pyvips.version(1), pyvips.version(2))"`
Expected: Prints the libvips version numbers (e.g., `8 16 1`)

**Step 6: Commit**

```bash
git add Brewfile pyproject.toml Install.md uv.lock
git commit -m "add vips/pyvips dependency for fast image processing"
```

---

### Task 2: Rewrite ImageHandler with pyvips

**Files:**
- Modify: `src/photodb/utils/image.py`

This is the core change. Replace all Pillow pixel operations with pyvips equivalents. Keep `is_supported()` and `calculate_resize_dimensions()` unchanged (pure logic, no Pillow dependency).

**Step 1: Rewrite `image.py`**

Replace the full file contents with:

```python
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import os

import pyvips

from .. import config as defaults

logger = logging.getLogger(__name__)


class ImageHandler:
    """Handles reading, converting, and basic operations on various image formats."""

    SUPPORTED_FORMATS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
    }

    # Maximum pixel count (to prevent memory issues)
    MAX_PIXELS = defaults.MAX_IMAGE_PIXELS

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in cls.SUPPORTED_FORMATS

    @classmethod
    def open_image(cls, file_path: Path) -> pyvips.Image:
        """
        Open an image file with proper format handling.

        Args:
            file_path: Path to the image file

        Returns:
            pyvips.Image object

        Raises:
            ValueError: If format is not supported or image is too large
            IOError: If file cannot be read
        """
        if not cls.is_supported(file_path):
            raise ValueError(f"Unsupported format: {file_path.suffix}")

        try:
            image = pyvips.Image.new_from_file(str(file_path), access="sequential")

            # Check for decompression bomb
            if image.width * image.height > cls.MAX_PIXELS:
                raise ValueError(f"Image too large: {image.width}x{image.height}")

            # Flatten alpha to white background if present
            if image.hasalpha():
                image = image.flatten(background=[255, 255, 255])

            # Convert to sRGB 8-bit if needed (handles CMYK, 16-bit, etc.)
            if image.interpretation != "srgb" and image.bands >= 3:
                image = image.colourspace("srgb")
            if image.format != "uchar":
                image = image.cast("uchar")

            return image

        except pyvips.Error as e:
            logger.error(f"Failed to open image {file_path}: {e}")
            raise IOError(f"Failed to open image {file_path}: {e}")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to open image {file_path}: {e}")
            raise

    @classmethod
    def get_image_info(cls, file_path: Path) -> Dict[str, Any]:
        """
        Get basic information about an image without fully loading it.

        Returns:
            Dictionary with width, height, format, mode
        """
        image = pyvips.Image.new_from_file(str(file_path), access="sequential")

        # Map pyvips interpretation to PIL-like mode names
        mode_map = {"srgb": "RGB", "b-w": "L", "rgb16": "RGB", "grey16": "L"}
        mode = mode_map.get(image.interpretation, "RGB")
        if image.hasalpha():
            mode += "A"

        # Detect format from loader
        loader = image.get("vips-loader") if image.get_typeof("vips-loader") else ""
        format_map = {
            "jpegload": "JPEG",
            "pngload": "PNG",
            "webpload": "WEBP",
            "heifload": "HEIF",
            "tiffload": "TIFF",
            "gifload": "GIF",
        }
        fmt = next((v for k, v in format_map.items() if k in loader), None)

        return {
            "width": image.width,
            "height": image.height,
            "format": fmt,
            "mode": mode,
            "size_bytes": file_path.stat().st_size,
        }

    @classmethod
    def calculate_resize_dimensions(
        cls, current_size: Tuple[int, int], max_dimensions: Dict[str, Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Calculate resize dimensions based on aspect ratio and max dimensions.

        Args:
            current_size: Current (width, height)
            max_dimensions: Dict of aspect ratio strings to max dimensions

        Returns:
            New (width, height) or None if no resize needed
        """
        width, height = current_size
        aspect_ratio = width / height

        # Find closest matching aspect ratio
        closest_ratio = None
        closest_diff = float("inf")

        # Get resize scale from environment variable
        resize_scale = float(os.getenv("RESIZE_SCALE", "1.0"))

        aspect_ratios = {
            "1:1": (1.0, (1092, 1092)),
            "3:4": (0.75, (951, 1268)),
            "4:3": (1.33, (1268, 951)),
            "2:3": (0.67, (896, 1344)),
            "3:2": (1.5, (1344, 896)),
            "9:16": (0.56, (819, 1456)),
            "16:9": (1.78, (1456, 819)),
            "1:2": (0.5, (784, 1568)),
            "2:1": (2.0, (1568, 784)),
        }

        for name, (ratio, max_dims) in aspect_ratios.items():
            diff = abs(aspect_ratio - ratio)
            if diff < closest_diff:
                closest_diff = diff
                closest_ratio = max_dims

        if closest_ratio:
            max_width, max_height = closest_ratio
            max_width = int(max_width * resize_scale)
            max_height = int(max_height * resize_scale)

            # Check if resize is needed
            if width <= max_width and height <= max_height:
                return None

            # Calculate scale factor
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            return (new_width, new_height)

        return None

    @classmethod
    def resize_image(cls, image: pyvips.Image, target_size: Tuple[int, int]) -> pyvips.Image:
        """
        Resize image to target dimensions.

        Args:
            image: pyvips Image object
            target_size: Target (width, height)

        Returns:
            Resized pyvips Image
        """
        target_width, target_height = target_size
        h_scale = target_width / image.width
        v_scale = target_height / image.height
        return image.resize(h_scale, vscale=v_scale / h_scale, kernel="lanczos3")

    @classmethod
    def autorotate(cls, image: pyvips.Image) -> pyvips.Image:
        """Apply EXIF orientation and strip the orientation tag."""
        return image.autorot()

    @classmethod
    def save_as_webp(
        cls,
        image: pyvips.Image,
        output_path: Path,
        quality: int = 95,
        original_path: Optional[Path] = None,
    ) -> None:
        """
        Save image as WebP with lossy compression and EXIF orientation correction.

        Args:
            image: pyvips Image object
            output_path: Path to save WebP
            quality: WebP quality (0-100, default 95 for high quality lossy)
            original_path: Path to original file for EXIF data (ignored, rotation
                applied before save via autorotate)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip alpha for lossy WebP
        if image.hasalpha():
            image = image.flatten(background=[255, 255, 255])

        image.webpsave(str(output_path), Q=quality, effort=defaults.WEBP_METHOD)
        logger.debug(f"Saved WebP image to {output_path}")

    @classmethod
    def save_as_png(
        cls,
        image: pyvips.Image,
        output_path: Path,
        optimize: bool = True,
        original_path: Optional[Path] = None,
    ) -> None:
        """
        Save image as PNG with compression.

        Args:
            image: pyvips Image object
            output_path: Path to save PNG
            optimize: Whether to optimize file size
            original_path: Path to original file for EXIF data (ignored, rotation
                applied before save via autorotate)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.pngsave(str(output_path), compression=defaults.PNG_COMPRESS_LEVEL)
        logger.debug(f"Saved image to {output_path}")
```

Key changes:
- `open_image` returns `pyvips.Image` instead of `PIL.Image.Image`
- No `pillow_heif` import — pyvips reads HEIC natively via libheif
- No `_apply_exif_orientation` — replaced by `autorotate()` (callers apply before save)
- `save_as_webp` no longer reads the original file to apply EXIF — the `original_path` parameter is kept for API compat but ignored (rotation must be applied before calling save)
- `resize_image` uses pyvips `resize()` with Lanczos3 kernel
- No explicit `close()` calls — pyvips uses reference counting
- `get_image_info` maps pyvips metadata to the same dict format

**Step 2: Run existing tests to see what fails**

Run: `uv run pytest tests/test_image.py -v`
Expected: Several failures due to PIL-specific assertions (e.g., `isinstance(img, Image.Image)`)

**Step 3: Commit**

```bash
git add src/photodb/utils/image.py
git commit -m "rewrite ImageHandler to use pyvips instead of Pillow"
```

---

### Task 3: Rewrite test_image.py for pyvips

**Files:**
- Modify: `tests/test_image.py`

Tests use Pillow to create fixture images and verify outputs. Update to use pyvips for fixtures where possible, and Pillow only for creating input fixture files (since pyvips can't create JPEG/GIF test images as easily as `Image.new().save()`).

**Step 1: Rewrite `test_image.py`**

```python
import pytest
from pathlib import Path
from PIL import Image
import pyvips
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
        assert isinstance(img, pyvips.Image)
        assert img.bands >= 3
        assert img.width == 100
        assert img.height == 100

    def test_open_image_unsupported(self, temp_dir):
        """Test opening unsupported file format."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("not an image")

        with pytest.raises(ValueError, match="Unsupported format"):
            ImageHandler.open_image(txt_file)

    def test_open_image_rgba_conversion(self, sample_rgba_image):
        """Test RGBA to RGB conversion (alpha flattened to white)."""
        img = ImageHandler.open_image(sample_rgba_image)
        assert not img.hasalpha()
        assert img.width == 100
        assert img.height == 100

    def test_open_image_decompression_bomb(self, temp_dir):
        """Test protection against decompression bombs."""
        img_path = temp_dir / "large.jpg"
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
        assert resized.width == 50
        assert resized.height == 50
        assert resized.bands == img.bands

    def test_save_as_webp(self, sample_image, temp_dir):
        """Test saving image as WebP."""
        img = ImageHandler.open_image(sample_image)
        output_path = temp_dir / "output.webp"

        ImageHandler.save_as_webp(img, output_path)

        assert output_path.exists()
        # Verify it's a valid WebP by re-reading
        saved = pyvips.Image.new_from_file(str(output_path))
        assert saved.width == 100
        assert saved.height == 100

    def test_save_as_png(self, sample_image, temp_dir):
        """Test saving image as PNG."""
        img = ImageHandler.open_image(sample_image)
        output_path = temp_dir / "output.png"

        ImageHandler.save_as_png(img, output_path, optimize=True)

        assert output_path.exists()
        saved = pyvips.Image.new_from_file(str(output_path))
        assert saved.width == 100
        assert saved.height == 100

    def test_save_creates_directory(self, sample_image, temp_dir):
        """Test that save methods create parent directories."""
        img = ImageHandler.open_image(sample_image)
        output_path = temp_dir / "nested" / "dir" / "output.webp"

        ImageHandler.save_as_webp(img, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_autorotate(self, sample_image):
        """Test EXIF autorotation returns a valid image."""
        img = ImageHandler.open_image(sample_image)
        rotated = ImageHandler.autorotate(img)
        assert isinstance(rotated, pyvips.Image)
        assert rotated.width == img.width
        assert rotated.height == img.height

    def test_mode_conversions(self, temp_dir):
        """Test various image mode conversions."""
        # Test palette mode (GIF → RGB)
        img_path = temp_dir / "palette.gif"
        img = Image.new("P", (100, 100))
        img.save(img_path, "GIF")
        converted = ImageHandler.open_image(img_path)
        assert converted.bands >= 3

        # Test LA mode (grayscale with alpha → flattened)
        img_path = temp_dir / "la.png"
        img = Image.new("LA", (100, 100))
        img.save(img_path, "PNG")
        converted = ImageHandler.open_image(img_path)
        assert not converted.hasalpha()

        # Test L mode (grayscale)
        img_path = temp_dir / "gray.jpg"
        img = Image.new("L", (100, 100))
        img.save(img_path, "JPEG")
        converted = ImageHandler.open_image(img_path)
        assert converted.width == 100
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_image.py -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/test_image.py
git commit -m "update image handler tests for pyvips"
```

---

### Task 4: Simplify normalize.py

**Files:**
- Modify: `src/photodb/stages/normalize.py`

The normalize stage currently has complex manual EXIF handling because Pillow requires it. With pyvips, `autorot()` handles all EXIF orientations automatically, and the dimensions after autorot are the final dimensions. This eliminates the separate EXIF orientation check, the `ORIENTATION_SWAPS_DIMENSIONS` constant, and the conditional dimension swapping.

**Step 1: Rewrite `normalize.py`**

```python
from pathlib import Path
import logging

from .base import BaseStage
from .. import config as defaults
from ..database.models import Photo
from ..utils.image import ImageHandler

logger = logging.getLogger(__name__)


class NormalizeStage(BaseStage):
    """Stage 1: Normalize photos to WebP format with standard sizes."""

    # WebP quality for lossy compression (0-100)
    WEBP_QUALITY = defaults.WEBP_QUALITY

    # Subdirectory for medium-sized images (preparation for multiple sizes)
    MED_SUBDIR = "med"

    # Subdirectory for full-sized images
    FULL_SUBDIR = "full"

    def __init__(self, repository, config):
        super().__init__(repository, config)
        self.base_output_dir = Path(config.get("IMG_PATH", "./photos/processed"))
        self.output_dir = self.base_output_dir / self.MED_SUBDIR
        self.full_output_dir = self.base_output_dir / self.FULL_SUBDIR

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Normalize a photo to WebP format (both medium and full-size versions).

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.full_output_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{self._generate_photo_id(Path(photo.orig_path))}.webp"
            med_output_path = self.output_dir / output_filename
            full_output_path = self.full_output_dir / output_filename

            # Open and auto-rotate (EXIF orientation applied, tag stripped)
            image = ImageHandler.open_image(file_path)
            image = ImageHandler.autorotate(image)

            # Store original dimensions (post-rotation = final orientation)
            photo.width = image.width
            photo.height = image.height
            logger.debug(f"Original size (post-rotation): {image.width}x{image.height}")

            # Save full-size WebP (format conversion only, no resize)
            ImageHandler.save_as_webp(image, full_output_path, quality=self.WEBP_QUALITY)
            logger.debug(f"Full-size photo saved to {full_output_path}")
            photo.full_path = str(full_output_path)

            # Calculate and apply resize for medium version
            original_size = (image.width, image.height)
            new_size = ImageHandler.calculate_resize_dimensions(original_size, {})

            if new_size and new_size != original_size:
                logger.debug(f"Resizing to: {new_size[0]}x{new_size[1]}")
                med_image = ImageHandler.resize_image(image, new_size)
                photo.med_width = new_size[0]
                photo.med_height = new_size[1]
            else:
                med_image = image
                photo.med_width = image.width
                photo.med_height = image.height

            # Save medium WebP
            ImageHandler.save_as_webp(med_image, med_output_path, quality=self.WEBP_QUALITY)
            logger.debug(f"Medium photo saved to {med_output_path}")
            logger.debug(f"Final dimensions: {photo.med_width}x{photo.med_height}")

            photo.med_path = str(med_output_path)
            self.repository.update_photo(photo)
            return True

        except Exception as e:
            logger.error(f"Failed to normalize photo {file_path}: {e}")
            return False
```

Key simplifications:
- **Removed**: `pillow_heif` import and `register_heif_opener()` — pyvips reads HEIC natively
- **Removed**: `from PIL import Image` and `from PIL.ExifTags import Base as ExifBase`
- **Removed**: `ORIENTATION_SWAPS_DIMENSIONS` constant and all manual EXIF orientation handling
- **Removed**: Second `Image.open()` for EXIF probing — `autorotate()` does it in one call
- **Removed**: `try/finally` with `image.close()`/`full_image.close()` — pyvips uses ref counting
- **Single decode**: The image is decoded once and reused for both full and medium (previously decoded twice via two `ImageHandler.open_image()` calls)
- **Dimensions are trivial**: After `autorotate()`, `image.width`/`image.height` are the final dimensions — no conditional swapping needed

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/photodb/stages/normalize.py
git commit -m "simplify normalize stage using pyvips autorotate and single decode"
```

---

### Task 5: Remove unused Pillow dependencies from normalize path

**Files:**
- Modify: `pyproject.toml` (check, but do NOT remove `pillow` or `pillow-heif`)

`pillow` and `pillow-heif` are still used by:
- `src/photodb/utils/exif.py` (EXIF metadata extraction, `pillow_heif.open_heif()`)
- `src/photodb/stages/detection.py` (face crop from normalized WebP)
- `src/photodb/utils/person_detector.py` (YOLO inference)
- `src/photodb/utils/embedding_extractor.py` (face embedding)
- `src/photodb/utils/mobileclip_analyzer.py` (CLIP preprocessing)
- `src/photodb/utils/apple_vision_classifier.py` (warmup)
- Multiple test files (fixture creation)

**Step 1: Verify Pillow is still needed**

Do NOT remove `pillow` or `pillow-heif` from `pyproject.toml`. They are required by other stages.

**Step 2: Update backfill_full_size.py**

This script mirrors normalize logic and should use pyvips too.

Modify: `scripts/backfill_full_size.py`

Find the section that does:
```python
import pillow_heif
pillow_heif.register_heif_opener()
from PIL import Image, ImageOps
```

And the processing section that does:
```python
img = Image.open(source_path)
img = ImageOps.exif_transpose(img)
img.save(full_path, format="WEBP", quality=quality)
```

Replace with:
```python
import pyvips
```

And:
```python
img = pyvips.Image.new_from_file(str(source_path), access="sequential")
img = img.autorot()
if img.hasalpha():
    img = img.flatten(background=[255, 255, 255])
img.webpsave(str(full_path), Q=quality)
```

**Step 3: Commit**

```bash
git add scripts/backfill_full_size.py
git commit -m "update backfill_full_size.py to use pyvips"
```

---

### Task 6: Performance benchmark

**Files:**
- Modify: `tmp/perf testing.md` (append results)

**Step 1: Run benchmark with the 83-photo test set**

Run single-threaded:
```bash
just local "/Volumes/media/Pictures/capture/Order\#04236/Album\#23070/" --skip-directory-scan --force --collection-id 3 --show-progress --stage normalize
```

Record the avg/photo time. Expected: ~0.4-0.8s (down from ~2.3s).

**Step 2: Run 4-thread benchmark**

```bash
just local "/Volumes/media/Pictures/capture/Order\#04236/Album\#23070/" --skip-directory-scan --force --collection-id 3 --show-progress --stage normalize --parallel 4
```

**Step 3: Run full pipeline benchmark**

```bash
just local "/Volumes/media/Pictures/capture/Order\#04236/Album\#23070/" --skip-directory-scan --force --collection-id 3 --show-progress --parallel 4
```

Compare overall wall time against the previous 1m19s (4-thread) result.

**Step 4: Spot-check output quality**

Visually compare a few output WebP files (both med/ and full/) against the previous Pillow-generated outputs. Check:
- EXIF rotation is correct (portrait photos display correctly)
- Colors look the same
- No artifacts or corruption

**Step 5: Append results to perf testing.md and commit**

```bash
git add tmp/perf\ testing.md
git commit -m "add pyvips normalize benchmark results"
```

---

## Files Changed Summary

| File | Action | Reason |
|---|---|---|
| `Brewfile` | Add `vips` | System dependency |
| `pyproject.toml` | Add `pyvips>=2.2.0` | Python dependency |
| `Install.md` | Mention `vips` | Documentation |
| `src/photodb/utils/image.py` | Rewrite | Pillow → pyvips |
| `src/photodb/stages/normalize.py` | Simplify | Use pyvips autorotate, single decode |
| `tests/test_image.py` | Rewrite | PIL assertions → pyvips assertions |
| `scripts/backfill_full_size.py` | Update | Match normalize changes |

## Not Changed (Pillow remains)

| File | Reason |
|---|---|
| `src/photodb/utils/exif.py` | `pillow_heif.open_heif()` for raw EXIF bytes — no pyvips equivalent |
| `src/photodb/stages/detection.py` | Face crop from normalized WebP — downstream, not a bottleneck |
| `src/photodb/utils/person_detector.py` | YOLO inference expects PIL Image |
| `src/photodb/utils/embedding_extractor.py` | InsightFace expects PIL Image |
| `src/photodb/utils/mobileclip_analyzer.py` | open_clip transform expects PIL Image |
