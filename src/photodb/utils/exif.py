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
                metadata["format"] = img.format
                metadata["mode"] = img.mode
                metadata["size"] = {"width": img.width, "height": img.height}

                # Extract EXIF data
                exifdata = img.getexif()
                if exifdata:
                    metadata["exif"] = cls._parse_exif(exifdata)

                # Extract other metadata
                if hasattr(img, "info"):
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
                    metadata["info"] = info

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
            with Image.open(file_path) as img:
                exifdata = img.getexif()
                if not exifdata:
                    return None

                # Try different datetime tags in order of preference
                datetime_tags = [
                    36867,  # DateTimeOriginal
                    36868,  # DateTimeDigitized
                    306,  # DateTime
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
                lat = cls._convert_to_degrees(gps_data.get("GPSLatitude"))
                lon = cls._convert_to_degrees(gps_data.get("GPSLongitude"))

                if lat is None or lon is None:
                    return None

                # Apply hemisphere
                if gps_data.get("GPSLatitudeRef") == "S":
                    lat = -lat
                if gps_data.get("GPSLongitudeRef") == "W":
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
                        value = value.decode("utf-8", errors="ignore")
                    except (UnicodeDecodeError, AttributeError):
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
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            return None
