from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import piexif
from PIL import Image
import logging
import json

logger = logging.getLogger(__name__)


class ExifExtractor:
    """Extract and parse EXIF metadata from images using piexif."""

    @classmethod
    def extract_all_metadata(cls, file_path: Path) -> Dict[str, Any]:
        """
        Extract all available EXIF/metadata from an image.

        Returns:
            Dictionary containing all metadata
        """
        metadata = {}

        try:
            # Get basic image info
            with Image.open(file_path) as img:
                metadata["format"] = img.format
                metadata["mode"] = img.mode
                metadata["size"] = {"width": img.width, "height": img.height}

            # Extract EXIF data using piexif
            try:
                exif_dict = piexif.load(str(file_path))
                
                # Convert EXIF data to readable format
                metadata["exif"] = cls._parse_exif_dict(exif_dict)
                
                # Store raw EXIF for reference (serializable version)
                metadata["exif_raw"] = cls._make_serializable(exif_dict)
                
            except Exception as e:
                logger.debug(f"Could not load EXIF data with piexif: {e}")
                # Fall back to PIL's basic EXIF extraction
                with Image.open(file_path) as img:
                    exifdata = img.getexif()
                    if exifdata:
                        metadata["exif"] = cls._parse_pil_exif(exifdata)

        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            metadata["error"] = str(e)

        return metadata

    @classmethod
    def extract_datetime(cls, file_path: Path) -> Optional[datetime]:
        """
        Extract capture datetime from EXIF data.

        Returns:
            Datetime when photo was taken, or None
        """
        try:
            exif_dict = piexif.load(str(file_path))
            
            # Try different datetime tags in order of preference
            datetime_fields = [
                ("Exif", piexif.ExifIFD.DateTimeOriginal),
                ("Exif", piexif.ExifIFD.DateTimeDigitized),
                ("0th", piexif.ImageIFD.DateTime),
            ]
            
            for ifd, tag in datetime_fields:
                if ifd in exif_dict and tag in exif_dict[ifd]:
                    date_str = exif_dict[ifd][tag]
                    if isinstance(date_str, bytes):
                        date_str = date_str.decode('utf-8')
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
        Extract GPS coordinates from EXIF data using piexif.

        Returns:
            Tuple of (latitude, longitude) or None
        """
        try:
            exif_dict = piexif.load(str(file_path))
            
            if "GPS" not in exif_dict or not exif_dict["GPS"]:
                return None
            
            gps_data = exif_dict["GPS"]
            
            # Check for required GPS fields
            if (piexif.GPSIFD.GPSLatitude not in gps_data or 
                piexif.GPSIFD.GPSLongitude not in gps_data):
                return None
            
            # Extract and convert coordinates
            lat = cls._convert_to_degrees(gps_data[piexif.GPSIFD.GPSLatitude])
            lon = cls._convert_to_degrees(gps_data[piexif.GPSIFD.GPSLongitude])
            
            if lat is None or lon is None:
                return None
            
            # Apply hemisphere
            lat_ref = gps_data.get(piexif.GPSIFD.GPSLatitudeRef, b'N')
            if isinstance(lat_ref, bytes):
                lat_ref = lat_ref.decode('utf-8')
            if lat_ref == 'S':
                lat = -lat
                
            lon_ref = gps_data.get(piexif.GPSIFD.GPSLongitudeRef, b'E')
            if isinstance(lon_ref, bytes):
                lon_ref = lon_ref.decode('utf-8')
            if lon_ref == 'W':
                lon = -lon
            
            return (lat, lon)

        except Exception as e:
            logger.error(f"Failed to extract GPS from {file_path}: {e}")

        return None

    @classmethod
    def _parse_exif_dict(cls, exif_dict: dict) -> Dict[str, Any]:
        """Parse piexif dictionary into readable format."""
        parsed = {}
        
        # Parse each IFD (Image File Directory)
        ifd_names = {
            "0th": "Image",
            "Exif": "Exif",
            "GPS": "GPS",
            "1st": "Thumbnail",
            "Interop": "Interoperability"
        }
        
        for ifd_key, ifd_name in ifd_names.items():
            if ifd_key in exif_dict and exif_dict[ifd_key]:
                ifd_data = {}
                for tag, value in exif_dict[ifd_key].items():
                    # Get human-readable tag name
                    tag_name = cls._get_tag_name(ifd_key, tag)
                    
                    # Convert value to readable format
                    readable_value = cls._convert_value(value)
                    
                    if tag_name and readable_value is not None:
                        ifd_data[tag_name] = readable_value
                
                if ifd_data:
                    parsed[ifd_name] = ifd_data
        
        return parsed

    @classmethod
    def _get_tag_name(cls, ifd: str, tag: int) -> Optional[str]:
        """Get human-readable tag name from piexif."""
        try:
            if ifd == "0th":
                return piexif.TAGS["Image"].get(tag, {}).get("name")
            elif ifd == "Exif":
                return piexif.TAGS["Exif"].get(tag, {}).get("name")
            elif ifd == "GPS":
                return piexif.TAGS["GPS"].get(tag, {}).get("name")
            elif ifd == "1st":
                return piexif.TAGS["Image"].get(tag, {}).get("name")
            elif ifd == "Interop":
                return piexif.TAGS["Interop"].get(tag, {}).get("name")
        except:
            pass
        return f"Tag_{tag}"

    @classmethod
    def _convert_value(cls, value: Any) -> Any:
        """Convert EXIF value to readable format."""
        if isinstance(value, bytes):
            try:
                # Try to decode as UTF-8
                return value.decode('utf-8').rstrip('\x00')
            except:
                # Return as hex string if not decodable
                return value.hex()
        elif isinstance(value, tuple) and len(value) == 2:
            # Rational number (numerator, denominator)
            if value[1] != 0:
                result = value[0] / value[1]
                # Return as int if it's a whole number
                if result == int(result):
                    return int(result)
                return result
            return value[0]
        elif isinstance(value, (list, tuple)):
            # Convert each element
            return [cls._convert_value(v) for v in value]
        else:
            return value

    @classmethod
    def _convert_to_degrees(cls, value) -> Optional[float]:
        """Convert GPS coordinates to degrees."""
        if not value or len(value) != 3:
            return None
        
        try:
            # GPS coordinates in EXIF are stored as:
            # ((degrees_num, degrees_den), (minutes_num, minutes_den), (seconds_num, seconds_den))
            degrees = value[0][0] / value[0][1] if value[0][1] != 0 else 0
            minutes = value[1][0] / value[1][1] if value[1][1] != 0 else 0
            seconds = value[2][0] / value[2][1] if value[2][1] != 0 else 0
            
            return degrees + (minutes / 60.0) + (seconds / 3600.0)
        except (TypeError, IndexError, ZeroDivisionError) as e:
            logger.error(f"Error converting GPS coordinates: {e}")
            return None

    @classmethod
    def _parse_pil_exif(cls, exifdata) -> Dict[str, Any]:
        """Parse PIL EXIF data as fallback."""
        from PIL.ExifTags import TAGS
        
        parsed = {}
        for tag_id, value in exifdata.items():
            tag = TAGS.get(tag_id, tag_id)
            parsed[str(tag)] = cls._convert_value(value)
        return parsed

    @classmethod
    def _make_serializable(cls, obj: Any) -> Any:
        """Convert piexif data to JSON-serializable format."""
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {str(k): cls._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [cls._make_serializable(item) for item in obj]
        elif isinstance(obj, bytes):
            try:
                return obj.decode('utf-8').rstrip('\x00')
            except:
                return obj.hex()
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        else:
            return str(obj)