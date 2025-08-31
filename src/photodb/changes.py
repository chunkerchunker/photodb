from pathlib import Path
from typing import List, Dict, Optional
import logging
import hashlib
from datetime import datetime

from .database.pg_repository import PostgresPhotoRepository

logger = logging.getLogger(__name__)


class ChangeDetector:
    """Detects changes in photos that may require reprocessing."""

    def __init__(self, repository):
        self.repository = repository

    def detect_modified_files(
        self, directory: Path, since: Optional[datetime] = None
    ) -> List[Path]:
        """
        Find files modified since a given time.

        Args:
            directory: Directory to check
            since: Check files modified after this time

        Returns:
            List of modified file paths
        """
        if since is None:
            # Default to last 24 hours
            from datetime import timedelta

            since = datetime.now() - timedelta(days=1)

        modified_files = []

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            # Check modification time
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime > since:
                modified_files.append(file_path)

        return modified_files

    def detect_moved_files(self, directory: Path) -> Dict[Path, Path]:
        """
        Detect files that have been moved/renamed.

        Returns:
            Dict mapping old path to new path
        """
        moved_files = {}

        # Get all current files with checksums
        current_files = {}
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                checksum = self._calculate_checksum(file_path)
                current_files[checksum] = file_path

        # Compare with database records
        # (This would need checksum storage in database)
        # For now, this is a placeholder implementation

        return moved_files

    def detect_corrupted_files(self, files: List[Path]) -> List[Path]:
        """
        Detect potentially corrupted image files.

        Args:
            files: List of files to check

        Returns:
            List of corrupted file paths
        """
        corrupted = []

        for file_path in files:
            try:
                # Try to open and verify image
                from PIL import Image

                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                logger.warning(f"Potentially corrupted file {file_path}: {e}")
                corrupted.append(file_path)

        return corrupted

    def _calculate_checksum(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate MD5 checksum of file."""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
