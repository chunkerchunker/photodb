from pathlib import Path
import logging
from typing import Tuple
from PIL import Image
import pillow_heif

from .base import BaseStage
from ..database.models import Photo
from ..utils.image import ImageHandler

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()


class NormalizeStage(BaseStage):
    """Stage 1: Normalize photos to PNG format with standard sizes."""
    
    def __init__(self, repository, config):
        super().__init__(repository, config)
        self.output_dir = Path(config.get('img_path', './photos/processed'))
    
    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Normalize a photo to PNG format.
        
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{photo.id}.png"
            output_path = self.output_dir / output_filename
            
            # Open image using ImageHandler
            image = ImageHandler.open_image(file_path)
            original_size = (image.width, image.height)
            
            logger.debug(f"Original size: {original_size[0]}x{original_size[1]}")
            
            # Calculate resize dimensions using ImageHandler
            new_size = ImageHandler.calculate_resize_dimensions(original_size, {})
            
            # Resize if needed
            if new_size and new_size != original_size:
                logger.debug(f"Resizing to: {new_size[0]}x{new_size[1]}")
                image = ImageHandler.resize_image(image, new_size)
            else:
                logger.debug("No resize needed")
            
            # Save as PNG using ImageHandler
            ImageHandler.save_as_png(image, output_path, optimize=True)
            
            photo.normalized_path = str(output_path)
            self.repository.update_photo(photo)
            
            logger.debug(f"Normalized photo saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to normalize photo {file_path}: {e}")
            return False