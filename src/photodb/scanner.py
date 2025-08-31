from pathlib import Path
from typing import List, Set, Optional, Generator, TYPE_CHECKING
from dataclasses import dataclass
import logging
import os

from .utils.image import ImageHandler
from .utils.validation import ImageValidator

if TYPE_CHECKING:
    from .database.repository import PhotoRepository as PhotoRepository

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Results from a directory scan."""

    total_files: int
    new_files: List[Path]
    existing_files: List[Path]
    skipped_files: List[Path]

    @property
    def has_new_files(self) -> bool:
        return len(self.new_files) > 0


class FileScanner:
    """Scans directories and identifies new photos to process."""

    def __init__(self, repository: "PhotoRepository", base_path: Optional[str] = None):
        self.repository = repository
        self.base_path = Path(base_path or os.getenv("INGEST_PATH", "./photos/raw"))

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        pattern: str = "*",
        check_existing: bool = True,
    ) -> ScanResult:
        """
        Scan directory for image files.

        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            pattern: File pattern to match
            check_existing: Whether to check if files are already in database

        Returns:
            ScanResult with categorized files
        """
        logger.info(f"Scanning directory: {directory}")

        # Find all files matching pattern
        all_files = self._find_files(directory, recursive, pattern)

        # Filter to supported image files
        image_files = [f for f in all_files if ImageHandler.is_supported(f)]

        # Validate files
        valid_files = []
        skipped_files = []

        for file_path in image_files:
            if ImageValidator.validate_file(file_path):
                valid_files.append(file_path)
            else:
                skipped_files.append(file_path)
                logger.debug(f"Skipping invalid file: {file_path}")

        # Check which files are already processed
        new_files = []
        existing_files = []

        if check_existing:
            for file_path in valid_files:
                rel_path = self._get_relative_path(file_path)
                if self.repository.get_photo_by_filename(str(rel_path)):
                    existing_files.append(file_path)
                else:
                    new_files.append(file_path)
        else:
            new_files = valid_files

        result = ScanResult(
            total_files=len(image_files),
            new_files=new_files,
            existing_files=existing_files,
            skipped_files=skipped_files,
        )

        logger.info(
            f"Scan complete: {len(new_files)} new, "
            f"{len(existing_files)} existing, "
            f"{len(skipped_files)} skipped"
        )

        return result

    def scan_file(self, file_path: Path) -> ScanResult:
        """
        Check a single file for processing.

        Args:
            file_path: Path to check

        Returns:
            ScanResult with single file categorized
        """
        result = ScanResult(total_files=1, new_files=[], existing_files=[], skipped_files=[])

        # Check if supported format
        if not ImageHandler.is_supported(file_path):
            result.skipped_files.append(file_path)
            return result

        # Validate file
        if not ImageValidator.validate_file(file_path):
            result.skipped_files.append(file_path)
            return result

        # Check if already processed
        rel_path = self._get_relative_path(file_path)
        if self.repository.get_photo_by_filename(str(rel_path)):
            result.existing_files.append(file_path)
        else:
            result.new_files.append(file_path)

        return result

    def watch_directory(
        self, directory: Path, recursive: bool = True, pattern: str = "*", poll_interval: int = 60
    ) -> Generator[ScanResult, None, None]:
        """
        Watch directory for new files (generator that yields scan results).

        Args:
            directory: Directory to watch
            recursive: Whether to watch subdirectories
            pattern: File pattern to match
            poll_interval: Seconds between scans

        Yields:
            ScanResult when new files are found
        """
        import time

        logger.info(f"Watching directory: {directory}")
        processed_files: Set[Path] = set()

        while True:
            try:
                # Scan for all files
                result = self.scan_directory(
                    directory, recursive=recursive, pattern=pattern, check_existing=False
                )

                # Find truly new files (not seen in this session)
                new_in_session = []
                for file_path in result.new_files + result.existing_files:
                    if file_path not in processed_files:
                        new_in_session.append(file_path)
                        processed_files.add(file_path)

                if new_in_session:
                    yield ScanResult(
                        total_files=result.total_files,
                        new_files=new_in_session,
                        existing_files=[],
                        skipped_files=result.skipped_files,
                    )

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Watch interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during watch: {e}")
                time.sleep(poll_interval)

    def _find_files(self, directory: Path, recursive: bool, pattern: str) -> List[Path]:
        """Find all files matching pattern in directory."""
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Filter to actual files (not directories)
        return [f for f in files if f.is_file()]

    def _get_relative_path(self, file_path: Path) -> Path:
        """Get path relative to base ingest path."""
        try:
            return file_path.relative_to(self.base_path)
        except ValueError:
            # File is outside base path, use absolute path
            return file_path
