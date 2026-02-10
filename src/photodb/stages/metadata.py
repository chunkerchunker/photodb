from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional

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
        if photo.id is None:
            logger.error(f"Photo {file_path} has no ID")
            return False

        photo_id = photo.id  # Capture for type narrowing

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
                photo_id=photo_id,
                collection_id=photo.collection_id,
                captured_at=captured_at or self._infer_date_from_filename(photo.orig_path),
                latitude=gps_coords[0] if gps_coords else None,
                longitude=gps_coords[1] if gps_coords else None,
                created_at=datetime.now(),
                extra=serializable_metadata,
            )

            # Check if metadata exists (for updates)
            existing_metadata = self.repository.get_metadata(photo_id)
            if existing_metadata:
                logger.debug(f"Metadata already exists for {photo_id}, updating")
                self._update_metadata(metadata)
            else:
                self.repository.create_metadata(metadata)

            logger.info(f"Successfully extracted metadata for {file_path}")
            logger.debug(f"Metadata extracted for {photo_id}")
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

    def _update_metadata(self, metadata: Metadata) -> None:
        """Update existing metadata record."""
        # PostgreSQL repository handles updates automatically with ON CONFLICT
        # Just create/update the metadata - the repository will handle it
        self.repository.create_metadata(metadata)

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
