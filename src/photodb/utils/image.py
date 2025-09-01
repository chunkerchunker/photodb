from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import logging
import os

# Register HEIF opener with Pillow
register_heif_opener()

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
    MAX_PIXELS = 178_956_970  # ~179 megapixels

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in cls.SUPPORTED_FORMATS

    @classmethod
    def open_image(cls, file_path: Path) -> Image.Image:
        """
        Open an image file with proper format handling.

        IMPORTANT: The returned image MUST be closed after use to prevent file handle leaks!
        Use image.close() or a context manager.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image object (caller must close it!)

        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be read
        """
        if not cls.is_supported(file_path):
            raise ValueError(f"Unsupported format: {file_path.suffix}")

        try:
            # Set PIL's max image pixels to our limit
            Image.MAX_IMAGE_PIXELS = cls.MAX_PIXELS

            image = Image.open(file_path)

            # Load image data immediately to allow closing the file
            image.load()

            # Check for decompression bomb (redundant but explicit)
            if image.width * image.height > cls.MAX_PIXELS:
                image.close()
                raise ValueError(f"Image too large: {image.width}x{image.height}")

            # Convert RGBA to RGB if needed (for JPEG compatibility)
            if image.mode in ("RGBA", "LA", "P"):
                # Create white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    converted = image.convert("RGBA")
                    image.close()
                    image = converted
                background.paste(image, mask=image.split()[-1] if "A" in image.mode else None)
                if image != background:
                    image.close()
                image = background
            elif image.mode not in ("RGB", "L"):
                converted = image.convert("RGB")
                image.close()
                image = converted

            return image

        except Image.DecompressionBombError as e:
            logger.error(f"Decompression bomb detected in {file_path}: {e}")
            raise ValueError("Image too large: potential decompression bomb")
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
        with Image.open(file_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
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
    def resize_image(cls, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image to target dimensions maintaining aspect ratio.

        Args:
            image: PIL Image object
            target_size: Target (width, height)

        Returns:
            Resized PIL Image
        """
        # Use high-quality Lanczos resampling
        resized = image.resize(target_size, Image.Resampling.LANCZOS)
        return resized

    @classmethod
    def save_as_png(
        cls,
        image: Image.Image,
        output_path: Path,
        optimize: bool = True,
        original_path: Optional[Path] = None,
    ) -> None:
        """
        Save image as PNG with optimization and EXIF orientation correction.

        Args:
            image: PIL Image object
            output_path: Path to save PNG
            optimize: Whether to optimize file size
            original_path: Path to original file for EXIF data (optional)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply EXIF orientation if we have the original path
        # Use the exact same approach as the working test script
        if original_path:
            try:
                with Image.open(original_path) as original:
                    # Apply EXIF orientation using PIL's built-in method (same as test script)
                    corrected = ImageOps.exif_transpose(original)

                    # If orientation was applied (different object returned), we need to apply the same to our image
                    if corrected is not original:
                        # The test script worked by applying exif_transpose directly
                        # We need to determine what transformation to apply to our processed image

                        # Check orientation tag to apply the same transformation
                        exif = original.getexif()
                        orientation = exif.get(0x0112) if exif else None

                        if orientation and orientation != 1:
                            original_image = image

                            # Apply the same transformation that exif_transpose would do
                            if orientation == 6:  # 90° CCW
                                image = image.rotate(-90, expand=True)
                            elif orientation == 8:  # 90° CW
                                image = image.rotate(90, expand=True)
                            elif orientation == 3:  # 180°
                                image = image.rotate(180, expand=True)
                            elif orientation == 2:  # Flip horizontal
                                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                            # Add other orientations as needed

                            if image is not original_image:
                                original_image.close()
                                logger.debug(f"Applied EXIF orientation {orientation}")

                    corrected.close()

            except Exception as e:
                logger.debug(f"Could not apply EXIF orientation at save time: {e}")

        save_kwargs = {"format": "PNG", "optimize": optimize}

        # Add compression for RGB images
        if image.mode == "RGB":
            save_kwargs["compress_level"] = 9

        image.save(output_path, **save_kwargs)
        logger.debug(f"Saved image to {output_path}")
