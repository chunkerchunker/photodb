from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    total_files: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    failed_files: List[tuple] = field(default_factory=list)
    success: bool = True


class BaseProcessor:
    """Base class for photo processors with common functionality."""

    def __init__(
        self,
        repository,
        config: dict,
        force: bool = False,
        dry_run: bool = False,
        max_photos: Optional[int] = None,
    ):
        self.repository = repository
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.max_photos = max_photos

    def _find_files_generator(self, directory: Path, recursive: bool, pattern: str):
        """Generate matching image files in directory."""
        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".heic",
            ".heif",
            ".bmp",
            ".tiff",
            ".webp",
        }

        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        # Yield files as they are discovered
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                yield file_path

    def _find_files(self, directory: Path, recursive: bool, pattern: str) -> List[Path]:
        """Find all matching image files in directory."""
        return list(self._find_files_generator(directory, recursive, pattern))

    def _get_stages(self, stage: str) -> List[str]:
        """Get list of stages to run."""
        if stage == "all":
            return ["normalize", "metadata", "enrich"]
        return [stage]

    def process_file(self, file_path: Path, stage: str = "all") -> ProcessingResult:
        """Process a single file through specified stages."""
        raise NotImplementedError("Subclasses must implement process_file")

    def process_directory(
        self, directory: Path, stage: str = "all", recursive: bool = True, pattern: str = "*"
    ) -> ProcessingResult:
        """Process all matching files in a directory."""
        raise NotImplementedError("Subclasses must implement process_directory")