from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from PIL.ExifTags import TAGS, GPSTAGS

from .base import BaseStage
from ..database.models import Photo, Metadata
from ..utils.exif import ExifExtractor

logger = logging.getLogger(__name__)


class MetadataStage(BaseStage):
    """Stage 2: Extract and store photo metadata."""

    STAGE_NAME = "metadata"

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Extract metadata from a photo.

        Returns:
            bool: True if processing was successful, False otherwise
        """
        logger.info(f"Extracting metadata from: {file_path}")

        try:
            # Create ExifExtractor instance - loads EXIF data once for efficiency
            extractor = ExifExtractor(file_path)

            # Extract all metadata using the instance (already serializable with piexif)
            all_metadata = extractor.extract_all_metadata()

            # Extract specific fields using the same instance
            captured_at = extractor.extract_datetime()
            gps_coords = extractor.extract_gps_coordinates()

            # Parse additional metadata from the already-serializable data
            parsed_metadata = self._parse_metadata(all_metadata)

            # Ensure everything is JSON serializable (safety check)
            serializable_metadata = self._make_json_serializable(parsed_metadata)

            # Create metadata record
            metadata = Metadata(
                photo_id=photo.id,
                captured_at=captured_at or self._infer_date_from_filename(photo.filename),
                latitude=gps_coords[0] if gps_coords else None,
                longitude=gps_coords[1] if gps_coords else None,
                created_at=datetime.now(),
                extra=serializable_metadata,
            )

            # Check if metadata exists (for updates)
            existing_metadata = self.repository.get_metadata(photo.id)
            if existing_metadata:
                logger.debug(f"Metadata already exists for {photo.id}, updating")
                self._update_metadata(metadata)
            else:
                self.repository.create_metadata(metadata)

            logger.info(f"Successfully extracted metadata for {file_path}")
            logger.debug(f"Metadata extracted for {photo.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return False

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
        if "size" in raw_metadata:
            parsed["dimensions"] = raw_metadata["size"]
            parsed["width"] = raw_metadata["size"].get("width")
            parsed["height"] = raw_metadata["size"].get("height")

        if "format" in raw_metadata:
            parsed["format"] = raw_metadata["format"]

        if "mode" in raw_metadata:
            parsed["color_mode"] = raw_metadata["mode"]

        # EXIF data
        if "exif" in raw_metadata:
            exif = raw_metadata["exif"]

            # Camera information
            camera_fields = {
                "Make": "camera_make",
                "Model": "camera_model",
                "LensModel": "lens_model",
                "Software": "software",
            }

            for exif_key, parsed_key in camera_fields.items():
                if exif_key in exif:
                    parsed[parsed_key] = str(exif[exif_key])

            # Shooting parameters
            shooting_fields = {
                "ExposureTime": "exposure_time",
                "FNumber": "f_number",
                "ISO": "iso",
                "ISOSpeedRatings": "iso",  # Alternative ISO field
                "FocalLength": "focal_length",
                "Flash": "flash",
                "WhiteBalance": "white_balance",
                "ExposureMode": "exposure_mode",
                "ExposureProgram": "exposure_program",
                "MeteringMode": "metering_mode",
            }

            for exif_key, parsed_key in shooting_fields.items():
                if exif_key in exif:
                    value = exif[exif_key]
                    # Values should already be serializable from _make_json_serializable
                    parsed[parsed_key] = value

            # Image orientation
            if "Orientation" in exif:
                parsed["orientation"] = exif["Orientation"]

            # Copyright and artist
            if "Copyright" in exif:
                parsed["copyright"] = exif["Copyright"]
            if "Artist" in exif:
                parsed["artist"] = exif["Artist"]

            # Store complete EXIF for reference
            parsed["exif_full"] = exif

        # Additional info
        if "info" in raw_metadata:
            parsed["additional_info"] = raw_metadata["info"]

        return parsed

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON serializable types.
        Handles PIL IFDRational, tuples, and other non-serializable types.
        """
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "numerator") and hasattr(obj, "denominator"):
            # Handle PIL IFDRational objects
            if obj.denominator != 0:
                return float(obj.numerator) / float(obj.denominator)
            else:
                return float(obj.numerator)
        elif isinstance(obj, str):
            # Remove null bytes that PostgreSQL can't handle
            return obj.replace("\x00", "")
        elif isinstance(obj, (int, float, bool)):
            return obj
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            # Convert unknown types to string and remove null bytes
            return str(obj).replace("\x00", "")

    def _extract_camera_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract camera information from metadata."""
        camera_info = {}

        if "camera_make" in metadata:
            camera_info["make"] = metadata["camera_make"]

        if "camera_model" in metadata:
            camera_info["model"] = metadata["camera_model"]

        if "lens_model" in metadata:
            camera_info["lens"] = metadata["lens_model"]

        if "iso" in metadata:
            camera_info["iso"] = metadata["iso"]

        if "f_number" in metadata:
            camera_info["aperture"] = f"f/{metadata['f_number']}"

        if "exposure_time" in metadata:
            exp = metadata["exposure_time"]
            if isinstance(exp, (int, float)):
                if exp < 1:
                    camera_info["shutter"] = f"1/{int(1 / exp)}s"
                else:
                    camera_info["shutter"] = f"{exp}s"

        return camera_info

    def _update_metadata(self, metadata: Metadata) -> None:
        """Update existing metadata record."""
        # PostgreSQL repository handles updates automatically with ON CONFLICT
        # Just create/update the metadata - the repository will handle it
        self.repository.create_metadata(metadata)

    # Keeping old methods for backwards compatibility
    def _parse_exif(self, exif_data: dict, metadata: Metadata) -> Metadata:
        """Parse EXIF data into metadata."""
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)

            if tag == "DateTime":
                try:
                    metadata.captured_at = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    logger.debug(f"Could not parse DateTime: {value}")

            elif tag == "DateTimeOriginal":
                try:
                    metadata.captured_at = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    logger.debug(f"Could not parse DateTimeOriginal: {value}")

            elif tag == "GPSInfo":
                gps_data = self._parse_gps(value)
                if "latitude" in gps_data:
                    metadata.latitude = gps_data["latitude"]
                if "longitude" in gps_data:
                    metadata.longitude = gps_data["longitude"]
                metadata.extra["gps"] = gps_data

            elif tag in ["Make", "Model", "LensModel", "Software"]:
                metadata.extra[tag.lower()] = str(value)

            elif tag in ["ISO", "ISOSpeedRatings"]:
                metadata.extra["iso"] = value

            elif tag in ["FNumber", "ExposureTime", "FocalLength"]:
                metadata.extra[tag.lower()] = self._format_rational(value)

        return metadata

    def _parse_gps(self, gps_info: dict) -> Dict[str, Any]:
        """Parse GPS information from EXIF."""
        gps_data = {}

        for key, value in gps_info.items():
            decode = GPSTAGS.get(key, key)
            gps_data[decode] = value

        if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
            lat = self._convert_to_degrees(gps_data["GPSLatitude"])
            lon = self._convert_to_degrees(gps_data["GPSLongitude"])

            if "GPSLatitudeRef" in gps_data and gps_data["GPSLatitudeRef"] == "S":
                lat = -lat
            if "GPSLongitudeRef" in gps_data and gps_data["GPSLongitudeRef"] == "W":
                lon = -lon

            gps_data["latitude"] = lat
            gps_data["longitude"] = lon

        return gps_data

    def _convert_to_degrees(self, value) -> float:
        """Convert GPS coordinates to decimal degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    def _format_rational(self, value) -> float:
        """Format EXIF rational values."""
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            if value.denominator != 0:
                return float(value.numerator) / float(value.denominator)
        return float(value)

    def _infer_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Infer date from filename if possible."""
        import re

        # Example pattern: 2020/01/10/IMG_7529.JPG (YYYY/MM/DD/xxxx.ext)
        match = re.search(r"(\d{4})/(\d{2})/(\d{2})/[^/]+$", filename)
        if match:
            try:
                return datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
            except ValueError:
                logger.debug(f"Could not parse date from filename: {filename}")
        return None
