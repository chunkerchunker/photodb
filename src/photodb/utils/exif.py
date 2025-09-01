from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import piexif
from PIL import Image
import logging
from pillow_heif import register_heif_opener
import pillow_heif

register_heif_opener()

logger = logging.getLogger(__name__)


class ExifExtractor:
    """Extract and parse EXIF metadata from images using piexif."""

    def __init__(self, file_path: Path):
        """Initialize with image path and load EXIF data once."""
        self.file_path = file_path
        self.img_info = {}
        self.exif_dict = None
        self.error = None

        # Load all metadata once
        self._load_metadata()

    def _load_metadata(self):
        """Load image info and EXIF data once during initialization."""
        try:
            # Get basic image info and EXIF data
            with Image.open(self.file_path) as img:
                self.img_info = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": {"width": img.width, "height": img.height},
                }

                # Get EXIF data for piexif
                exif_data = self._get_exif_data_for_piexif(img, self.file_path)

            # Load EXIF data using piexif when available
            if exif_data:
                try:
                    self.exif_dict = piexif.load(exif_data)
                except Exception as e:
                    logger.debug(f"Could not load EXIF data with piexif: {e}")
                    # Fall back to PIL's basic EXIF extraction
                    with Image.open(self.file_path) as img:
                        exifdata = img.getexif()
                        if exifdata:
                            self.exif_dict = self._convert_pil_exif_to_piexif_format(exifdata)
            else:
                # Fall back to PIL's EXIF extraction for unsupported formats
                with Image.open(self.file_path) as img:
                    exifdata = img.getexif()
                    if exifdata:
                        self.exif_dict = self._convert_pil_exif_to_piexif_format(exifdata)

        except Exception as e:
            logger.error(f"Failed to load metadata from {self.file_path}: {e}")
            self.error = str(e)

    def extract_all_metadata(self) -> Dict[str, Any]:
        """
        Extract all available EXIF/metadata from an image.

        Returns:
            Dictionary containing all metadata
        """
        metadata = self.img_info.copy()

        if self.error:
            metadata["error"] = self.error
            return metadata

        if self.exif_dict:
            # Convert EXIF data to readable format
            metadata["exif"] = self._parse_exif_dict(self.exif_dict)

            # Store raw EXIF for reference (serializable version)
            metadata["exif_raw"] = self._make_serializable(self.exif_dict)

        return metadata

    def extract_datetime(self) -> Optional[datetime]:
        """
        Extract capture datetime from EXIF data.

        Returns:
            Datetime when photo was taken, or None
        """
        if not self.exif_dict:
            return None

        try:
            # Try different datetime tags in order of preference
            datetime_fields = [
                ("Exif", piexif.ExifIFD.DateTimeOriginal),
                ("Exif", piexif.ExifIFD.DateTimeDigitized),
                ("0th", piexif.ImageIFD.DateTime),
            ]

            for ifd, tag in datetime_fields:
                if ifd in self.exif_dict and tag in self.exif_dict[ifd]:
                    date_str = self.exif_dict[ifd][tag]
                    if isinstance(date_str, bytes):
                        date_str = date_str.decode("utf-8")
                    try:
                        # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        logger.warning(f"Invalid datetime format: {date_str}")

        except Exception as e:
            logger.error(f"Failed to extract datetime from {self.file_path}: {e}")

        return None

    def extract_gps_coordinates(self) -> Optional[Tuple[float, float]]:
        """
        Extract GPS coordinates from EXIF data using piexif.

        Returns:
            Tuple of (latitude, longitude) or None
        """
        if not self.exif_dict:
            return None

        try:
            if "GPS" not in self.exif_dict or not self.exif_dict["GPS"]:
                return None

            gps_data = self.exif_dict["GPS"]

            # Check for required GPS fields
            if (
                piexif.GPSIFD.GPSLatitude not in gps_data
                or piexif.GPSIFD.GPSLongitude not in gps_data
            ):
                return None

            # Extract and convert coordinates
            lat = self._convert_to_degrees(gps_data[piexif.GPSIFD.GPSLatitude])
            lon = self._convert_to_degrees(gps_data[piexif.GPSIFD.GPSLongitude])

            if lat is None or lon is None:
                return None

            # Apply hemisphere
            lat_ref = gps_data.get(piexif.GPSIFD.GPSLatitudeRef, b"N")
            if isinstance(lat_ref, bytes):
                lat_ref = lat_ref.decode("utf-8")
            if lat_ref == "S":
                lat = -lat

            lon_ref = gps_data.get(piexif.GPSIFD.GPSLongitudeRef, b"E")
            if isinstance(lon_ref, bytes):
                lon_ref = lon_ref.decode("utf-8")
            if lon_ref == "W":
                lon = -lon

            return (lat, lon)

        except Exception as e:
            logger.error(f"Failed to extract GPS from {self.file_path}: {e}")

        return None

    def _convert_pil_exif_to_piexif_format(self, pil_exif) -> Optional[dict]:
        """Convert PIL EXIF data to a format similar to piexif structure."""
        try:
            # Create a basic structure similar to piexif
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "Interop": {}}

            from PIL.ExifTags import TAGS, GPS

            for tag_id, value in pil_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)

                # Map common tags to piexif structure
                if tag_name == "DateTime":
                    exif_dict["0th"][piexif.ImageIFD.DateTime] = value
                elif tag_name == "DateTimeOriginal":
                    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = value
                elif tag_name == "DateTimeDigitized":
                    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = value
                elif tag_name == "GPSInfo" and isinstance(value, dict):
                    # Handle GPS data
                    for gps_tag_id, gps_value in value.items():
                        if gps_tag_id in GPS:
                            gps_key = GPS[gps_tag_id]
                            # Map common GPS tags
                            if gps_key == "GPSLatitude":
                                exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = gps_value
                            elif gps_key == "GPSLongitude":
                                exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = gps_value
                            elif gps_key == "GPSLatitudeRef":
                                exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = gps_value
                            elif gps_key == "GPSLongitudeRef":
                                exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = gps_value

                # Store in Exif section for most other tags
                if isinstance(tag_id, int) and tag_id not in [34853]:  # Skip GPSInfo
                    exif_dict["Exif"][tag_id] = value

            return exif_dict

        except Exception as e:
            logger.error(f"Failed to convert PIL EXIF to piexif format: {e}")
            return None

    def _get_exif_data_for_piexif(self, img: Image.Image, file_path: Path) -> Optional[bytes]:
        """
        Get EXIF data suitable for piexif.load().
        For HEIF files, extract raw EXIF data using pyheif.
        """
        try:
            # Check if format is supported by piexif
            if img.format in ["JPEG", "TIFF"]:
                # Read original file bytes for supported formats
                with open(file_path, "rb") as f:
                    return f.read()
            elif img.format in ["HEIF", "HEIC"]:
                # Extract raw EXIF data from HEIF using pillow-heif
                logger.debug(f"Extracting EXIF from {img.format} using pillow-heif")
                heif_file = pillow_heif.open_heif(file_path)
                # Get EXIF data from the first image
                exif_data = heif_file.info.get("exif")
                if exif_data:
                    return exif_data
                return None
            else:
                # Unsupported format for piexif or pillow-heif not available
                if img.format in ["HEIF", "HEIC"]:
                    logger.debug("pillow-heif not available for HEIF EXIF extraction")
                else:
                    logger.debug(f"Format {img.format} not supported by piexif")
                return None
        except Exception as e:
            logger.error(f"Failed to extract EXIF data for piexif: {e}")
            return None

    def _parse_exif_dict(self, exif_dict: dict) -> Dict[str, Any]:
        """Parse piexif dictionary into readable format."""
        parsed = {}

        # Parse each IFD (Image File Directory)
        ifd_names = {
            "0th": "Image",
            "Exif": "Exif",
            "GPS": "GPS",
            "1st": "Thumbnail",
            "Interop": "Interoperability",
        }

        for ifd_key, ifd_name in ifd_names.items():
            if ifd_key in exif_dict and exif_dict[ifd_key]:
                ifd_data = {}
                for tag, value in exif_dict[ifd_key].items():
                    # Get human-readable tag name
                    tag_name = self._get_tag_name(ifd_key, tag)

                    # Convert value to readable format
                    readable_value = self._convert_value(value)

                    if tag_name and readable_value is not None:
                        ifd_data[tag_name] = readable_value

                if ifd_data:
                    parsed[ifd_name] = ifd_data

        return parsed

    def _get_tag_name(self, ifd: str, tag: int) -> Optional[str]:
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
        except (KeyError, AttributeError):
            pass
        return f"Tag_{tag}"

    def _convert_value(self, value: Any) -> Any:
        """Convert EXIF value to readable format."""
        if isinstance(value, bytes):
            try:
                # Try to decode as UTF-8
                return value.decode("utf-8").rstrip("\x00")
            except UnicodeDecodeError:
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
            return [self._convert_value(v) for v in value]
        else:
            return value

    def _convert_to_degrees(self, value) -> Optional[float]:
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

    def _parse_pil_exif(self, exifdata) -> Dict[str, Any]:
        """Parse PIL EXIF data as fallback."""
        from PIL.ExifTags import TAGS

        parsed = {}
        for tag_id, value in exifdata.items():
            tag = TAGS.get(tag_id, tag_id)
            parsed[str(tag)] = self._convert_value(value)
        return parsed

    def _make_serializable(self, obj: Any) -> Any:
        """Convert piexif data to JSON-serializable format."""
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, bytes):
            try:
                return obj.decode("utf-8").rstrip("\x00")
            except UnicodeDecodeError:
                return obj.hex()
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        else:
            return str(obj)

    # Class methods for backward compatibility
    @classmethod
    def extract_all_metadata_from_path(cls, file_path: Path) -> Dict[str, Any]:
        """Extract all metadata from a file path (backward compatibility)."""
        extractor = cls(file_path)
        return extractor.extract_all_metadata()

    @classmethod
    def extract_datetime_from_path(cls, file_path: Path) -> Optional[datetime]:
        """Extract datetime from a file path (backward compatibility)."""
        extractor = cls(file_path)
        return extractor.extract_datetime()

    @classmethod
    def extract_gps_coordinates_from_path(cls, file_path: Path) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from a file path (backward compatibility)."""
        extractor = cls(file_path)
        return extractor.extract_gps_coordinates()
