# Task 07: Stage 2 - Metadata Extraction

## Objective
Implement Stage 2 of the processing pipeline: extracting comprehensive metadata from photos including EXIF data, timestamps, GPS coordinates, and storing all information in the database.

## Dependencies
- Task 02: Database Setup (for metadata storage)
- Task 04: Image Format Handling (for EXIF extraction utilities)
- Task 06: Photo Normalization (Stage 1 must complete first)

## Deliverables

### 1. Metadata Extraction Stage (src/photodb/stages/metadata.py)
```python
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import json

from ..database.repository import PhotoRepository
from ..database.models import Metadata, ProcessingStatus
from ..utils.exif import ExifExtractor

logger = logging.getLogger(__name__)

class MetadataStage:
    """Stage 2: Extract and store photo metadata."""
    
    STAGE_NAME = 'metadata'
    
    def __init__(self, repository: PhotoRepository, config: dict):
        self.repository = repository
        self.config = config
        self.ingest_path = Path(config.get('ingest_path', './photos/raw'))
    
    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """
        Check if metadata extraction is needed.
        
        Args:
            file_path: Path to check
            force: Force reprocessing
            
        Returns:
            True if processing needed
        """
        if force:
            return True
        
        # Get photo record
        rel_path = self._get_relative_path(file_path)
        photo = self.repository.get_photo_by_filename(str(rel_path))
        
        if not photo:
            logger.warning(f"No photo record found for {file_path}")
            return False
        
        # Check if Stage 1 is complete
        stage1_status = self.repository.get_processing_status(photo.id, 'normalize')
        if not stage1_status or stage1_status.status != 'completed':
            logger.debug(f"Stage 1 not complete for {file_path}")
            return False
        
        # Check if metadata already extracted
        metadata = self.repository.get_metadata(photo.id)
        if metadata and not force:
            return False
        
        # Check processing status
        status = self.repository.get_processing_status(photo.id, self.STAGE_NAME)
        if status and status.status == 'completed' and not force:
            return False
        
        return True
    
    def process(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from photo.
        
        Args:
            file_path: Path to photo
            
        Returns:
            Dict with extraction results
        """
        logger.info(f"Extracting metadata from: {file_path}")
        
        # Get photo record
        rel_path = self._get_relative_path(file_path)
        photo = self.repository.get_photo_by_filename(str(rel_path))
        
        if not photo:
            return {
                'success': False,
                'error': f"No photo record found for {file_path}"
            }
        
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
            # Extract all metadata
            all_metadata = ExifExtractor.extract_all_metadata(file_path)
            
            # Extract specific fields
            captured_at = ExifExtractor.extract_datetime(file_path)
            gps_coords = ExifExtractor.extract_gps_coordinates(file_path)
            
            # Parse additional metadata
            parsed_metadata = self._parse_metadata(all_metadata)
            
            # Create metadata record
            metadata = Metadata.create(
                photo_id=photo.id,
                captured_at=captured_at,
                latitude=gps_coords[0] if gps_coords else None,
                longitude=gps_coords[1] if gps_coords else None,
                extra=parsed_metadata
            )
            
            # Check if metadata exists (for updates)
            existing = self.repository.get_metadata(photo.id)
            if existing:
                # Update existing metadata
                metadata.created_at = existing.created_at
                self._update_metadata(metadata)
            else:
                # Create new metadata
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
                    parsed[parsed_key] = exif[exif_key]
            
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
    
    def _update_metadata(self, metadata: Metadata):
        """Update existing metadata record."""
        # This would need to be implemented in the repository
        # For now, delete and recreate
        with self.repository.db.transaction() as conn:
            conn.execute(
                "DELETE FROM metadata WHERE photo_id = ?",
                (metadata.photo_id,)
            )
        self.repository.create_metadata(metadata)
    
    def _get_relative_path(self, file_path: Path) -> Path:
        """Get path relative to ingest path."""
        try:
            return file_path.relative_to(self.ingest_path)
        except ValueError:
            return file_path
```

### 2. Metadata Enrichment (src/photodb/stages/enrich.py)
```python
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetadataEnricher:
    """Enrich metadata with derived information."""
    
    @staticmethod
    def enrich_location(latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Enrich location data with reverse geocoding.
        
        Args:
            latitude: GPS latitude
            longitude: GPS longitude
            
        Returns:
            Dict with location details
        """
        # This would integrate with a geocoding service
        # For now, return basic info
        return {
            'coordinates': {
                'latitude': latitude,
                'longitude': longitude
            },
            'map_url': f"https://maps.google.com/?q={latitude},{longitude}"
        }
    
    @staticmethod
    def enrich_datetime(captured_at: datetime) -> Dict[str, Any]:
        """
        Enrich datetime with additional temporal information.
        
        Args:
            captured_at: Photo capture datetime
            
        Returns:
            Dict with temporal details
        """
        return {
            'datetime': captured_at.isoformat(),
            'date': captured_at.date().isoformat(),
            'time': captured_at.time().isoformat(),
            'year': captured_at.year,
            'month': captured_at.month,
            'day': captured_at.day,
            'weekday': captured_at.strftime('%A'),
            'hour': captured_at.hour,
            'season': get_season(captured_at),
            'time_of_day': get_time_of_day(captured_at.hour)
        }
    
    @staticmethod
    def analyze_shooting_conditions(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze shooting conditions from metadata.
        
        Args:
            metadata: Extracted metadata
            
        Returns:
            Dict with shooting analysis
        """
        conditions = {}
        
        # Analyze ISO
        if 'iso' in metadata:
            iso = metadata['iso']
            if iso <= 200:
                conditions['light_conditions'] = 'bright'
            elif iso <= 800:
                conditions['light_conditions'] = 'normal'
            elif iso <= 3200:
                conditions['light_conditions'] = 'dim'
            else:
                conditions['light_conditions'] = 'dark'
        
        # Analyze aperture
        if 'f_number' in metadata:
            f = metadata['f_number']
            if f <= 2.8:
                conditions['depth_of_field'] = 'shallow'
            elif f <= 5.6:
                conditions['depth_of_field'] = 'moderate'
            else:
                conditions['depth_of_field'] = 'deep'
        
        # Analyze shutter speed
        if 'exposure_time' in metadata:
            exp = metadata['exposure_time']
            if isinstance(exp, (int, float)):
                if exp < 1/500:
                    conditions['motion'] = 'frozen'
                elif exp < 1/60:
                    conditions['motion'] = 'normal'
                else:
                    conditions['motion'] = 'motion_blur'
        
        # Flash usage
        if 'flash' in metadata:
            conditions['flash_used'] = metadata['flash'] != 0
        
        return conditions

def get_season(dt: datetime) -> str:
    """Get season from datetime (Northern Hemisphere)."""
    month = dt.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

def get_time_of_day(hour: int) -> str:
    """Get time of day from hour."""
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'
```

### 3. Metadata Statistics (src/photodb/stages/stats.py)
```python
from typing import Dict, List, Any
from collections import Counter
import logging

from ..database.repository import PhotoRepository

logger = logging.getLogger(__name__)

class MetadataStatistics:
    """Generate statistics from metadata."""
    
    def __init__(self, repository: PhotoRepository):
        self.repository = repository
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate overall statistics from all metadata."""
        stats = {
            'total_photos': 0,
            'with_location': 0,
            'with_datetime': 0,
            'cameras': Counter(),
            'years': Counter(),
            'locations': [],
            'date_range': None
        }
        
        # Query all metadata
        # This would need a repository method to get all metadata
        # For now, this is a placeholder
        
        return stats
    
    def analyze_camera_usage(self) -> Dict[str, int]:
        """Analyze camera usage statistics."""
        cameras = Counter()
        
        # Query metadata and count cameras
        # Placeholder implementation
        
        return dict(cameras)
    
    def analyze_temporal_distribution(self) -> Dict[str, Any]:
        """Analyze temporal distribution of photos."""
        distribution = {
            'by_year': Counter(),
            'by_month': Counter(),
            'by_hour': Counter(),
            'by_weekday': Counter()
        }
        
        # Query metadata and analyze dates
        # Placeholder implementation
        
        return distribution
```

## Implementation Steps

1. **Implement MetadataStage**
   - EXIF extraction integration
   - GPS coordinate parsing
   - Datetime extraction
   - Metadata structuring
   - Database storage

2. **Add enrichment capabilities**
   - Location enrichment (geocoding ready)
   - Temporal analysis
   - Shooting condition analysis
   - Derived metadata generation

3. **Create statistics module**
   - Camera usage analysis
   - Temporal distribution
   - Location clustering
   - Metadata completeness metrics

4. **Handle edge cases**
   - Missing EXIF data
   - Corrupted metadata
   - Non-standard formats
   - Timezone handling

5. **Write tests**
   - Test EXIF extraction
   - Test GPS parsing
   - Test datetime parsing
   - Test metadata storage
   - Test enrichment functions

## Testing Checklist

- [ ] EXIF data extracts correctly
- [ ] GPS coordinates parse with proper hemisphere
- [ ] Datetime extracts from various EXIF fields
- [ ] Metadata stores in database
- [ ] Stage 2 only runs after Stage 1
- [ ] Force flag triggers re-extraction
- [ ] Missing metadata handled gracefully
- [ ] Camera info extracts correctly
- [ ] Shooting parameters parse properly
- [ ] JSON serialization works

## Notes

- Consider adding face detection metadata
- Implement color analysis (dominant colors, histogram)
- Add image quality metrics
- Consider extracting keywords/tags
- Implement timezone detection from GPS coordinates