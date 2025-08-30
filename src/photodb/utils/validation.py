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
        if size < 50:  # Less than 50 bytes (extremely small, likely not a real image)
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
