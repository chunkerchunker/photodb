# Task 05: File Discovery and Directory Scanning

## Objective
Implement the file discovery system that scans directories for new photos, tracks processed files, and manages the queue of images to be processed.

## Dependencies
- Task 02: Database Setup (for tracking processed files)
- Task 04: Image Format Handling (for file validation)

## Deliverables

### 1. File Scanner Module (src/photodb/scanner.py)
```python
from pathlib import Path
from typing import List, Set, Optional, Generator
from dataclasses import dataclass
import logging
import os

from .database.repository import PhotoRepository
from .utils.image import ImageHandler
from .utils.validation import ImageValidator

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
    
    def __init__(self, repository: PhotoRepository, base_path: Optional[str] = None):
        self.repository = repository
        self.base_path = Path(base_path or os.getenv('INGEST_PATH', './photos/raw'))
        
    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        pattern: str = '*',
        check_existing: bool = True
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
            skipped_files=skipped_files
        )
        
        logger.info(f"Scan complete: {len(new_files)} new, "
                   f"{len(existing_files)} existing, "
                   f"{len(skipped_files)} skipped")
        
        return result
    
    def scan_file(self, file_path: Path) -> ScanResult:
        """
        Check a single file for processing.
        
        Args:
            file_path: Path to check
            
        Returns:
            ScanResult with single file categorized
        """
        result = ScanResult(
            total_files=1,
            new_files=[],
            existing_files=[],
            skipped_files=[]
        )
        
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
        self,
        directory: Path,
        recursive: bool = True,
        pattern: str = '*',
        poll_interval: int = 60
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
                    directory,
                    recursive=recursive,
                    pattern=pattern,
                    check_existing=False
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
                        skipped_files=result.skipped_files
                    )
                
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Watch interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during watch: {e}")
                time.sleep(poll_interval)
    
    def _find_files(
        self,
        directory: Path,
        recursive: bool,
        pattern: str
    ) -> List[Path]:
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
```

### 2. Batch Processing Queue (src/photodb/queue.py)
```python
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from queue import Queue, PriorityQueue
from threading import Lock
import logging

logger = logging.getLogger(__name__)

@dataclass(order=True)
class QueueItem:
    """Item in processing queue with priority."""
    priority: int
    file_path: Path = field(compare=False)
    retry_count: int = field(default=0, compare=False)
    
class ProcessingQueue:
    """Manages queue of files to be processed."""
    
    def __init__(self, max_retries: int = 3):
        self.queue: PriorityQueue = PriorityQueue()
        self.processing: Set[Path] = set()
        self.failed: List[Tuple[Path, str]] = []
        self.completed: List[Path] = []
        self.max_retries = max_retries
        self.lock = Lock()
    
    def add_files(self, files: List[Path], priority: int = 5):
        """
        Add files to processing queue.
        
        Args:
            files: List of file paths to add
            priority: Priority (lower = higher priority)
        """
        for file_path in files:
            self.queue.put(QueueItem(priority, file_path))
        
        logger.debug(f"Added {len(files)} files to queue")
    
    def get_next(self) -> Optional[Path]:
        """
        Get next file to process.
        
        Returns:
            Path to next file or None if queue empty
        """
        try:
            item = self.queue.get_nowait()
            with self.lock:
                self.processing.add(item.file_path)
            return item.file_path
        except:
            return None
    
    def mark_completed(self, file_path: Path):
        """Mark file as successfully processed."""
        with self.lock:
            if file_path in self.processing:
                self.processing.remove(file_path)
            self.completed.append(file_path)
    
    def mark_failed(self, file_path: Path, error: str):
        """
        Mark file as failed and potentially retry.
        
        Args:
            file_path: Path to failed file
            error: Error message
        """
        with self.lock:
            if file_path in self.processing:
                self.processing.remove(file_path)
        
        # Check if we should retry
        # (In real implementation, track retry count per file)
        self.failed.append((file_path, error))
        logger.error(f"Failed to process {file_path}: {error}")
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty() and len(self.processing) == 0
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            'queued': self.queue.qsize(),
            'processing': len(self.processing),
            'completed': len(self.completed),
            'failed': len(self.failed)
        }
```

### 3. Change Detection (src/photodb/changes.py)
```python
from pathlib import Path
from typing import List, Dict, Set
import logging
import hashlib
from datetime import datetime

from .database.repository import PhotoRepository

logger = logging.getLogger(__name__)

class ChangeDetector:
    """Detects changes in photos that may require reprocessing."""
    
    def __init__(self, repository: PhotoRepository):
        self.repository = repository
    
    def detect_modified_files(
        self,
        directory: Path,
        since: Optional[datetime] = None
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
        
        for file_path in directory.rglob('*'):
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
        for file_path in directory.rglob('*'):
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
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
```

## Implementation Steps

1. **Implement FileScanner**
   - Directory traversal logic
   - File filtering and validation
   - Database lookup for existing files
   - Relative path handling

2. **Create ProcessingQueue**
   - Priority queue implementation
   - Thread-safe operations
   - Retry logic
   - Statistics tracking

3. **Build ChangeDetector**
   - Modification time checking
   - File movement detection
   - Corruption detection
   - Checksum calculation

4. **Add watch mode**
   - Periodic scanning
   - New file detection
   - Event generation

5. **Write tests**
   - Test directory scanning
   - Test file filtering
   - Test queue operations
   - Test change detection

## Testing Checklist

- [ ] Scanner finds all image files
- [ ] Recursive scanning works
- [ ] Pattern matching filters correctly
- [ ] Existing files are identified
- [ ] Invalid files are skipped
- [ ] Queue maintains priority order
- [ ] Thread safety in queue operations
- [ ] Modified files are detected
- [ ] Corrupted files are identified
- [ ] Watch mode detects new files

## Notes

- Consider using watchdog library for real-time file system monitoring
- Implement incremental scanning for large directories
- Add support for exclusion patterns
- Consider caching directory listings for performance
- Implement parallel scanning for large directory trees