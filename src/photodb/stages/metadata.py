from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json

from .base import BaseStage
from ..database.models import Photo, Metadata, ProcessingStatus
from ..utils.exif import ExifExtractor

logger = logging.getLogger(__name__)


class MetadataStage(BaseStage):
    """Stage 2: Extract and store photo metadata."""
    
    STAGE_NAME = 'metadata'
    
    def process_photo(self, photo: Photo, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from a photo."""
        logger.info(f"Extracting metadata from: {file_path}")
        
        # Update processing status
        self.repository.update_processing_status(
            ProcessingStatus(
                photo_id=photo.id,
                stage=self.STAGE_NAME,
                status='processing',
                processed_at=datetime.now(),
                error_message=None
            )
        )
        
        try:
            # Extract all metadata using ExifExtractor
            all_metadata = ExifExtractor.extract_all_metadata(file_path)
            
            # Extract specific fields
            captured_at = ExifExtractor.extract_datetime(file_path)
            gps_coords = ExifExtractor.extract_gps_coordinates(file_path)
            
            # Parse additional metadata
            parsed_metadata = self._parse_metadata(all_metadata)
            
            # Create metadata record
            metadata = Metadata(
                photo_id=photo.id,
                captured_at=captured_at or datetime.fromtimestamp(file_path.stat().st_mtime),
                latitude=gps_coords[0] if gps_coords else None,
                longitude=gps_coords[1] if gps_coords else None,
                created_at=datetime.now(),
                extra=parsed_metadata
            )
            
            # Check if metadata exists (for updates)
            existing_metadata = self.repository.get_metadata(photo.id)
            if existing_metadata:
                logger.debug(f"Metadata already exists for {photo.id}, updating")
                metadata.created_at = existing_metadata.created_at
                self._update_metadata(metadata)
            else:
                self.repository.create_metadata(metadata)
            
            # Update processing status
            self.repository.update_processing_status(
                ProcessingStatus(
                    photo_id=photo.id,
                    stage=self.STAGE_NAME,
                    status='completed',
                    processed_at=datetime.now(),
                    error_message=None
                )
            )
            
            result = {
                'success': True,
                'photo_id': photo.id,
                'captured_at': captured_at.isoformat() if captured_at else None,
                'has_location': gps_coords is not None,
                'metadata_fields': len(parsed_metadata),
                'camera_info': self._extract_camera_info(parsed_metadata)
            }
            
            logger.info(f"Successfully extracted metadata for {file_path}")
            logger.debug(f"Metadata extracted for {photo.id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            
            # Update processing status
            self.repository.update_processing_status(
                ProcessingStatus(
                    photo_id=photo.id,
                    stage=self.STAGE_NAME,
                    status='failed',
                    processed_at=datetime.now(),
                    error_message=str(e)
                )
            )
            
            return {
                'success': False,
                'photo_id': photo.id,
                'error': str(e)
            }
    
    def _parse_metadata(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and structure metadata for storage.
        
        Args:
            raw_metadata: Raw metadata from extractor
            
        Returns:
            Structured metadata dict
        """
        parsed = {}
        
        # Basic image properties
        if 'size' in raw_metadata:
            parsed['dimensions'] = raw_metadata['size']
            parsed['width'] = raw_metadata['size'].get('width')
            parsed['height'] = raw_metadata['size'].get('height')
        
        if 'format' in raw_metadata:
            parsed['format'] = raw_metadata['format']
        
        if 'mode' in raw_metadata:
            parsed['color_mode'] = raw_metadata['mode']
        
        # EXIF data
        if 'exif' in raw_metadata:
            exif = raw_metadata['exif']
            
            # Camera information
            camera_fields = {
                'Make': 'camera_make',
                'Model': 'camera_model',
                'LensModel': 'lens_model',
                'Software': 'software'
            }
            
            for exif_key, parsed_key in camera_fields.items():
                if exif_key in exif:
                    parsed[parsed_key] = str(exif[exif_key])
            
            # Shooting parameters
            shooting_fields = {
                'ExposureTime': 'exposure_time',
                'FNumber': 'f_number',
                'ISO': 'iso',
                'ISOSpeedRatings': 'iso',  # Alternative ISO field
                'FocalLength': 'focal_length',
                'Flash': 'flash',
                'WhiteBalance': 'white_balance',
                'ExposureMode': 'exposure_mode',
                'ExposureProgram': 'exposure_program',
                'MeteringMode': 'metering_mode'
            }
            
            for exif_key, parsed_key in shooting_fields.items():
                if exif_key in exif:
                    value = exif[exif_key]
                    # Convert tuples to values
                    if isinstance(value, tuple) and len(value) == 2:
                        value = value[0] / value[1] if value[1] != 0 else value[0]
                    parsed[parsed_key] = value
            
            # Image orientation
            if 'Orientation' in exif:
                parsed['orientation'] = exif['Orientation']
            
            # Copyright and artist
            if 'Copyright' in exif:
                parsed['copyright'] = exif['Copyright']
            if 'Artist' in exif:
                parsed['artist'] = exif['Artist']
            
            # Store complete EXIF for reference
            parsed['exif_full'] = exif
        
        # Additional info
        if 'info' in raw_metadata:
            parsed['additional_info'] = raw_metadata['info']
        
        return parsed
    
    def _extract_camera_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract camera information from metadata."""
        camera_info = {}
        
        if 'camera_make' in metadata:
            camera_info['make'] = metadata['camera_make']
        
        if 'camera_model' in metadata:
            camera_info['model'] = metadata['camera_model']
        
        if 'lens_model' in metadata:
            camera_info['lens'] = metadata['lens_model']
        
        if 'iso' in metadata:
            camera_info['iso'] = metadata['iso']
        
        if 'f_number' in metadata:
            camera_info['aperture'] = f"f/{metadata['f_number']}"
        
        if 'exposure_time' in metadata:
            exp = metadata['exposure_time']
            if isinstance(exp, (int, float)):
                if exp < 1:
                    camera_info['shutter'] = f"1/{int(1/exp)}s"
                else:
                    camera_info['shutter'] = f"{exp}s"
        
        return camera_info
    
    def _update_metadata(self, metadata: Metadata) -> None:
        """Update existing metadata record."""
        # This would need to be implemented in the repository
        # For now, delete and recreate
        with self.repository.db.transaction() as conn:
            conn.execute(
                "DELETE FROM metadata WHERE photo_id = ?",
                (metadata.photo_id,)
            )
        self.repository.create_metadata(metadata)
    
    # Keeping old methods for backwards compatibility
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