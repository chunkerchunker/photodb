from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from .base import BaseStage
from ..database.models import Photo, Metadata

logger = logging.getLogger(__name__)


class MetadataStage(BaseStage):
    """Stage 2: Extract and store photo metadata."""
    
    def process_photo(self, photo: Photo, file_path: Path) -> None:
        """Extract metadata from a photo."""
        existing_metadata = self.repository.get_metadata(photo.id)
        if existing_metadata:
            logger.debug(f"Metadata already exists for {photo.id}, updating")
        
        metadata = self._extract_metadata(file_path, photo.id)
        
        if existing_metadata:
            self._update_metadata(existing_metadata, metadata)
        else:
            self.repository.create_metadata(metadata)
        
        logger.debug(f"Metadata extracted for {photo.id}")
    
    def _extract_metadata(self, file_path: Path, photo_id: str) -> Metadata:
        """Extract metadata from image file."""
        metadata = Metadata(
            photo_id=photo_id,
            captured_at=None,
            latitude=None,
            longitude=None,
            created_at=datetime.now(),
            extra={}
        )
        
        try:
            with Image.open(file_path) as img:
                exif_data = img._getexif()
                
                if exif_data:
                    metadata = self._parse_exif(exif_data, metadata)
                
                metadata.extra['width'] = img.width
                metadata.extra['height'] = img.height
                metadata.extra['format'] = img.format
                metadata.extra['mode'] = img.mode
                
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
        
        if not metadata.captured_at:
            stat = file_path.stat()
            metadata.captured_at = datetime.fromtimestamp(stat.st_mtime)
        
        return metadata
    
    def _parse_exif(self, exif_data: dict, metadata: Metadata) -> Metadata:
        """Parse EXIF data into metadata."""
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            
            if tag == 'DateTime':
                try:
                    metadata.captured_at = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    logger.debug(f"Could not parse DateTime: {value}")
            
            elif tag == 'DateTimeOriginal':
                try:
                    metadata.captured_at = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    logger.debug(f"Could not parse DateTimeOriginal: {value}")
            
            elif tag == 'GPSInfo':
                gps_data = self._parse_gps(value)
                if 'latitude' in gps_data:
                    metadata.latitude = gps_data['latitude']
                if 'longitude' in gps_data:
                    metadata.longitude = gps_data['longitude']
                metadata.extra['gps'] = gps_data
            
            elif tag in ['Make', 'Model', 'LensModel', 'Software']:
                metadata.extra[tag.lower()] = str(value)
            
            elif tag in ['ISO', 'ISOSpeedRatings']:
                metadata.extra['iso'] = value
            
            elif tag in ['FNumber', 'ExposureTime', 'FocalLength']:
                metadata.extra[tag.lower()] = self._format_rational(value)
        
        return metadata
    
    def _parse_gps(self, gps_info: dict) -> Dict[str, Any]:
        """Parse GPS information from EXIF."""
        gps_data = {}
        
        for key, value in gps_info.items():
            decode = GPSTAGS.get(key, key)
            gps_data[decode] = value
        
        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
            lat = self._convert_to_degrees(gps_data['GPSLatitude'])
            lon = self._convert_to_degrees(gps_data['GPSLongitude'])
            
            if 'GPSLatitudeRef' in gps_data and gps_data['GPSLatitudeRef'] == 'S':
                lat = -lat
            if 'GPSLongitudeRef' in gps_data and gps_data['GPSLongitudeRef'] == 'W':
                lon = -lon
            
            gps_data['latitude'] = lat
            gps_data['longitude'] = lon
        
        return gps_data
    
    def _convert_to_degrees(self, value) -> float:
        """Convert GPS coordinates to decimal degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    
    def _format_rational(self, value) -> float:
        """Format EXIF rational values."""
        if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
            if value.denominator != 0:
                return float(value.numerator) / float(value.denominator)
        return float(value)
    
    def _update_metadata(self, existing: Metadata, new: Metadata) -> None:
        """Update existing metadata with new values."""
        if new.captured_at:
            existing.captured_at = new.captured_at
        if new.latitude is not None:
            existing.latitude = new.latitude
        if new.longitude is not None:
            existing.longitude = new.longitude
        
        existing.extra.update(new.extra)