from pathlib import Path
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, field
from queue import PriorityQueue, Empty
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
        except Empty:
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
            "queued": self.queue.qsize(),
            "processing": len(self.processing),
            "completed": len(self.completed),
            "failed": len(self.failed),
        }
