# Task 04: Image Format Handling

## Objective
Implement comprehensive image format handling utilities to support reading, converting, and processing various image formats including JPEG, PNG, HEIC, and others.

## Dependencies
- Task 01: Project Setup (dependencies like Pillow and pillow-heif)

## Deliverables

### 1. Image Format Handler (src/photodb/utils/image.py)
```python
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image
from pillow_heif import register_heif_opener
import logging

# Register HEIF opener with Pillow
register_heif_opener()

logger = logging.getLogger(__name__)

class ImageHandler:
    """Handles reading, converting, and basic operations on various image formats."""
    
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.heic', '.heif', 
        '.bmp', '.tiff', '.tif', '.webp', '.gif'
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
        
        Args:
            file_path: Path to the image file
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be read
        """
        if not cls.is_supported(file_path):
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        try:
            image = Image.open(file_path)
            
            # Check for decompression bomb
            if image.width * image.height > cls.MAX_PIXELS:
                raise ValueError(f"Image too large: {image.width}x{image.height}")
            
            # Convert RGBA to RGB if needed (for JPEG compatibility)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if 'A' in image.mode else None)
                image = background
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            return image
            
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
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'size_bytes': file_path.stat().st_size
            }
    
    @classmethod
    def calculate_resize_dimensions(
        cls, 
        current_size: Tuple[int, int],
        max_dimensions: Dict[str, Tuple[int, int]]
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
        closest_diff = float('inf')
        
        aspect_ratios = {
            '1:1': (1.0, (1092, 1092)),
            '3:4': (0.75, (951, 1268)),
            '4:3': (1.33, (1268, 951)),
            '2:3': (0.67, (896, 1344)),
            '3:2': (1.5, (1344, 896)),
            '9:16': (0.56, (819, 1456)),
            '16:9': (1.78, (1456, 819)),
            '1:2': (0.5, (784, 1568)),
            '2:1': (2.0, (1568, 784))
        }
        
        for name, (ratio, max_dims) in aspect_ratios.items():
            diff = abs(aspect_ratio - ratio)
            if diff < closest_diff:
                closest_diff = diff
                closest_ratio = max_dims
        
        if closest_ratio:
            max_width, max_height = closest_ratio
            
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
    def resize_image(
        cls,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Image.Image:
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
        optimize: bool = True
    ) -> None:
        """
        Save image as PNG with optimization.
        
        Args:
            image: PIL Image object
            output_path: Path to save PNG
            optimize: Whether to optimize file size
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_kwargs = {
            'format': 'PNG',
            'optimize': optimize
        }
        
        # Add compression for RGB images
        if image.mode == 'RGB':
            save_kwargs['compress_level'] = 9
        
        image.save(output_path, **save_kwargs)
        logger.debug(f"Saved image to {output_path}")
```

### 2. EXIF Data Extractor (src/photodb/utils/exif.py)
```python
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import logging

logger = logging.getLogger(__name__)

class ExifExtractor:
    """Extract and parse EXIF metadata from images."""
    
    @classmethod
    def extract_all_metadata(cls, file_path: Path) -> Dict[str, Any]:
        """
        Extract all available EXIF/metadata from an image.
        
        Returns:
            Dictionary containing all metadata
        """
        metadata = {}
        
        try:
            with Image.open(file_path) as img:
                # Get basic info
                metadata['format'] = img.format
                metadata['mode'] = img.mode
                metadata['size'] = {'width': img.width, 'height': img.height}
                
                # Extract EXIF data
                exifdata = img.getexif()
                if exifdata:
                    metadata['exif'] = cls._parse_exif(exifdata)
                
                # Extract other metadata
                if hasattr(img, 'info'):
                    # Filter out non-serializable data
                    info = {}
                    for key, value in img.info.items():
                        try:
                            # Test if serializable
                            import json
                            json.dumps(value)
                            info[key] = value
                        except (TypeError, ValueError):
                            info[key] = str(value)
                    metadata['info'] = info
                
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            metadata['error'] = str(e)
        
        return metadata
    
    @classmethod
    def extract_datetime(cls, file_path: Path) -> Optional[datetime]:
        """
        Extract capture datetime from EXIF data.
        
        Returns:
            Datetime when photo was taken, or None
        """
        try:
            with Image.open(file_path) as img:
                exifdata = img.getexif()
                if not exifdata:
                    return None
                
                # Try different datetime tags in order of preference
                datetime_tags = [
                    36867,  # DateTimeOriginal
                    36868,  # DateTimeDigitized
                    306,    # DateTime
                ]
                
                for tag in datetime_tags:
                    if tag in exifdata:
                        date_str = exifdata[tag]
                        try:
                            # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                            return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            logger.warning(f"Invalid datetime format: {date_str}")
                
        except Exception as e:
            logger.error(f"Failed to extract datetime from {file_path}: {e}")
        
        return None
    
    @classmethod
    def extract_gps_coordinates(cls, file_path: Path) -> Optional[Tuple[float, float]]:
        """
        Extract GPS coordinates from EXIF data.
        
        Returns:
            Tuple of (latitude, longitude) or None
        """
        try:
            with Image.open(file_path) as img:
                exifdata = img.getexif()
                if not exifdata:
                    return None
                
                # Get GPS IFD
                gps_ifd = exifdata.get_ifd(0x8825)  # GPS IFD pointer
                if not gps_ifd:
                    return None
                
                gps_data = {}
                for tag, value in gps_ifd.items():
                    decoded = GPSTAGS.get(tag, tag)
                    gps_data[decoded] = value
                
                # Extract coordinates
                lat = cls._convert_to_degrees(gps_data.get('GPSLatitude'))
                lon = cls._convert_to_degrees(gps_data.get('GPSLongitude'))
                
                if lat is None or lon is None:
                    return None
                
                # Apply hemisphere
                if gps_data.get('GPSLatitudeRef') == 'S':
                    lat = -lat
                if gps_data.get('GPSLongitudeRef') == 'W':
                    lon = -lon
                
                return (lat, lon)
                
        except Exception as e:
            logger.error(f"Failed to extract GPS from {file_path}: {e}")
        
        return None
    
    @classmethod
    def _parse_exif(cls, exifdata) -> Dict[str, Any]:
        """Parse EXIF data into readable format."""
        parsed = {}
        
        for tag_id, value in exifdata.items():
            tag = TAGS.get(tag_id, tag_id)
            
            # Handle IFD (Image File Directory) data
            if isinstance(value, dict):
                parsed[tag] = cls._parse_exif(value)
            else:
                # Convert bytes to string
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = str(value)
                        
                parsed[tag] = value
        
        return parsed
    
    @classmethod
    def _convert_to_degrees(cls, value) -> Optional[float]:
        """Convert GPS coordinates to degrees."""
        if not value:
            return None
        
        try:
            # GPS coordinates are stored as ((degrees, 1), (minutes, 1), (seconds, 100))
            d, m, s = value
            degrees = d[0] / d[1] if d[1] != 0 else 0
            minutes = m[0] / m[1] if m[1] != 0 else 0
            seconds = s[0] / s[1] if s[1] != 0 else 0
            
            return degrees + (minutes / 60.0) + (seconds / 3600.0)
        except:
            return None
```

### 3. Image Validation Utilities (src/photodb/utils/validation.py)
```python
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)

class ImageValidator:
    """Validate image files for processing."""
    
    @classmethod
    def validate_file(cls, file_path: Path) -> bool:
        """
        Validate that a file is a processable image.
        
        Returns:
            True if valid, False otherwise
        """
        # Check file exists
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        # Check file size (skip if too small or too large)
        size = file_path.stat().st_size
        if size < 1024:  # Less than 1KB
            logger.warning(f"File too small: {file_path} ({size} bytes)")
            return False
        
        if size > 500 * 1024 * 1024:  # More than 500MB
            logger.warning(f"File too large: {file_path} ({size} bytes)")
            return False
        
        # Try to open as image
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"Invalid image file {file_path}: {e}")
            return False
    
    @classmethod
    def calculate_checksum(cls, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
```

## Implementation Steps

1. **Implement ImageHandler class**
   - Support all required formats
   - Handle HEIC with pillow-heif
   - Implement resize logic
   - Add PNG conversion

2. **Create EXIF extractor**
   - Extract all metadata
   - Parse datetime fields
   - Extract GPS coordinates
   - Handle edge cases

3. **Add validation utilities**
   - File existence checks
   - Size validation
   - Format verification
   - Checksum calculation

4. **Write comprehensive tests**
   - Test each image format
   - Test resize calculations
   - Test EXIF extraction
   - Test error handling

## Testing Checklist

- [ ] All supported formats open correctly
- [ ] HEIC/HEIF files are handled
- [ ] Resize dimensions calculate correctly
- [ ] Images resize maintaining aspect ratio
- [ ] PNG conversion preserves quality
- [ ] EXIF data extracts properly
- [ ] GPS coordinates parse correctly
- [ ] Invalid files are rejected
- [ ] Large files are handled safely

## Notes

- HEIC support requires pillow-heif library
- Be careful with memory usage for large images
- Some EXIF data may be in different formats
- GPS coordinates need hemisphere correction
- Consider implementing image rotation based on EXIF orientation