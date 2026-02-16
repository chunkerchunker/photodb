from pathlib import Path
import logging
import pillow_heif
from PIL import Image
from PIL.ExifTags import Base as ExifBase

from .base import BaseStage
from .. import config as defaults
from ..database.models import Photo
from ..utils.image import ImageHandler

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()

# EXIF orientations that swap width/height (involve 90° or 270° rotation)
ORIENTATION_SWAPS_DIMENSIONS = {5, 6, 7, 8}


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
        image = None
        full_image = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.full_output_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{self._generate_photo_id(Path(photo.orig_path))}.webp"
            med_output_path = self.output_dir / output_filename
            full_output_path = self.full_output_dir / output_filename

            # Open image using ImageHandler (orientation applied at save time)
            image = ImageHandler.open_image(file_path)
            original_size = (image.width, image.height)

            # Store original dimensions
            photo.width = original_size[0]
            photo.height = original_size[1]

            logger.debug(f"Original size: {original_size[0]}x{original_size[1]}")

            # Calculate resize dimensions using ImageHandler
            new_size = ImageHandler.calculate_resize_dimensions(original_size, {})

            # Check EXIF orientation to determine if dimensions will be swapped
            exif_swaps_dimensions = False
            try:
                with Image.open(file_path) as orig:
                    exif = orig.getexif()
                    orientation = exif.get(ExifBase.Orientation) if exif else None
                    if orientation in ORIENTATION_SWAPS_DIMENSIONS:
                        exif_swaps_dimensions = True
                        logger.debug(f"EXIF orientation {orientation} will swap dimensions")
            except Exception as e:
                logger.debug(f"Could not read EXIF orientation: {e}")

            # Create full-size WebP (no resize, just format conversion with rotation baking)
            full_image = ImageHandler.open_image(file_path)
            ImageHandler.save_as_webp(
                full_image, full_output_path, quality=self.WEBP_QUALITY, original_path=file_path
            )
            logger.debug(f"Full-size photo saved to {full_output_path}")
            photo.full_path = str(full_output_path)

            # Resize for medium version if needed
            if new_size and new_size != original_size:
                logger.debug(f"Resizing to: {new_size[0]}x{new_size[1]}")
                image = ImageHandler.resize_image(image, new_size)
                pre_rotation_width, pre_rotation_height = new_size
            else:
                pre_rotation_width, pre_rotation_height = original_size

            # Save medium as WebP using ImageHandler
            # Note: save_as_webp applies EXIF rotation
            ImageHandler.save_as_webp(
                image, med_output_path, quality=self.WEBP_QUALITY, original_path=file_path
            )
            logger.debug(f"Medium photo saved to {med_output_path}")

            # Store final dimensions (accounting for EXIF rotation)
            if exif_swaps_dimensions:
                photo.med_width = pre_rotation_height
                photo.med_height = pre_rotation_width
            else:
                photo.med_width = pre_rotation_width
                photo.med_height = pre_rotation_height
            logger.debug(f"Final dimensions: {photo.med_width}x{photo.med_height}")

            # Only update DB after image processing is complete
            photo.med_path = str(med_output_path)
            self.repository.update_photo(photo)
            return True

        except Exception as e:
            logger.error(f"Failed to normalize photo {file_path}: {e}")
            return False
        finally:
            # CRITICAL: Always close images to free file handles
            if image:
                image.close()
            if full_image:
                full_image.close()
