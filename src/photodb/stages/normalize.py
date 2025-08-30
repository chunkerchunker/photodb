from pathlib import Path
import logging
from typing import Tuple
from PIL import Image
import pillow_heif

from .base import BaseStage
from ..database.models import Photo

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()


class NormalizeStage(BaseStage):
    """Stage 1: Normalize photos to JPEG format with standard sizes."""
    
    def __init__(self, repository, config):
        super().__init__(repository, config)
        self.output_dir = Path(config.get('img_path', './photos/processed'))
        self.max_size = (2048, 2048)
        self.quality = 85
    
    def process_photo(self, photo: Photo, file_path: Path) -> None:
        """Normalize a photo to JPEG format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{photo.id}.jpg"
        output_path = self.output_dir / output_filename
        
        with Image.open(file_path) as img:
            if img.format != 'JPEG' or self._needs_resize(img):
                logger.debug(f"Converting/resizing {file_path.name}")
                img = self._process_image(img)
                img.save(output_path, 'JPEG', quality=self.quality, optimize=True)
            else:
                logger.debug(f"Copying {file_path.name} (already normalized)")
                img.save(output_path, 'JPEG', quality=self.quality, optimize=True)
        
        photo.normalized_path = str(output_path)
        self.repository.update_photo(photo)
        
        logger.debug(f"Normalized photo saved to {output_path}")
    
    def _needs_resize(self, img: Image.Image) -> bool:
        """Check if image needs resizing."""
        return img.width > self.max_size[0] or img.height > self.max_size[1]
    
    def _process_image(self, img: Image.Image) -> Image.Image:
        """Process image (resize and convert)."""
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        
        if self._needs_resize(img):
            img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        return img