from pathlib import Path
import logging

from .base import BaseStage
from .. import config as defaults
from ..database.models import Photo
from ..utils.image import ImageHandler

logger = logging.getLogger(__name__)


class NormalizeStage(BaseStage):
    """Stage 1: Normalize photos to WebP format with standard sizes."""

    WEBP_QUALITY = defaults.WEBP_QUALITY
    MED_SUBDIR = "med"
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
