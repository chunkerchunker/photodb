import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# macOS: ensure Homebrew's libvips is discoverable by cffi's dlopen
if sys.platform == "darwin":
    _fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    _brew_paths = "/opt/homebrew/lib:/usr/local/lib"
    if _brew_paths not in _fallback:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
            f"{_brew_paths}:{_fallback}" if _fallback else _brew_paths
        )

import pyvips  # noqa: E402  (must come after library-path fix)

from .. import config as defaults

logger = logging.getLogger(__name__)

# Map vips-loader names to canonical format strings
_LOADER_FORMAT_MAP: Dict[str, str] = {
    "jpegload": "JPEG",
    "jpegload_buffer": "JPEG",
    "pngload": "PNG",
    "pngload_buffer": "PNG",
    "webpload": "WEBP",
    "webpload_buffer": "WEBP",
    "heifload": "HEIF",
    "heifload_buffer": "HEIF",
    "tiffload": "TIFF",
    "tiffload_buffer": "TIFF",
    "gifload": "GIF",
    "gifload_buffer": "GIF",
    "ppmload": "PPM",
    "ppmload_buffer": "PPM",
    "magickload": "MAGICK",
    "magickload_buffer": "MAGICK",
}


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

        pyvips uses reference counting -- no explicit close() needed.

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
            image = pyvips.Image.new_from_file(str(file_path))

            # Check for decompression bomb
            if image.width * image.height > cls.MAX_PIXELS:
                raise ValueError(f"Image too large: {image.width}x{image.height}")

            # Flatten alpha channel onto white background
            if image.hasalpha():
                image = image.flatten(background=[255, 255, 255])

            # Convert to sRGB if needed (e.g. CMYK, Lab)
            interpretation = image.interpretation
            if interpretation not in ("srgb", "b-w", "rgb", "multiband"):
                image = image.colourspace("srgb")

            # Cast to 8-bit unsigned if needed
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
        Get basic information about an image without fully decoding it.

        Returns:
            Dictionary with width, height, format, mode, size_bytes
        """
        image = pyvips.Image.new_from_file(str(file_path), access="sequential")

        # Derive format from the vips loader name
        loader = image.get("vips-loader") if image.get_typeof("vips-loader") else None
        fmt = _LOADER_FORMAT_MAP.get(loader, loader) if loader else None

        # Derive a Pillow-compatible mode string
        bands = image.bands
        has_alpha = image.hasalpha()
        interpretation = image.interpretation
        if interpretation == "b-w":
            mode = "LA" if has_alpha else "L"
        elif bands == 4 and has_alpha:
            mode = "RGBA"
        elif bands == 3:
            mode = "RGB"
        else:
            mode = "RGB"

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
        Resize image to target dimensions using Lanczos3.

        Args:
            image: pyvips.Image object
            target_size: Target (width, height)

        Returns:
            Resized pyvips.Image
        """
        target_width, target_height = target_size
        h_scale = target_width / image.width
        v_scale = target_height / image.height
        return image.resize(h_scale, vscale=v_scale, kernel="lanczos3")

    @classmethod
    def autorotate(cls, image: pyvips.Image) -> pyvips.Image:
        """
        Apply EXIF orientation and strip the orientation tag.

        Args:
            image: pyvips.Image object

        Returns:
            Rotated pyvips.Image (may be the same object if no rotation needed)
        """
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
        Save image as WebP with lossy compression.

        The original_path parameter is kept for API compatibility but ignored.
        Callers should apply rotation via autorotate() before calling this.

        Args:
            image: pyvips.Image object
            output_path: Path to save WebP
            quality: WebP quality (0-100, default 95)
            original_path: Ignored (kept for API compat)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip alpha if present (lossy WebP should be opaque)
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
        Save image as PNG.

        The original_path parameter is kept for API compatibility but ignored.
        Callers should apply rotation via autorotate() before calling this.

        Args:
            image: pyvips.Image object
            output_path: Path to save PNG
            optimize: Ignored (pyvips always optimizes)
            original_path: Ignored (kept for API compat)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image.pngsave(str(output_path), compression=defaults.PNG_COMPRESS_LEVEL)
        logger.debug(f"Saved image to {output_path}")
