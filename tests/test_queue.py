import pytest
from pathlib import Path
import threading
import time

from photodb.queue import ProcessingQueue, QueueItem


@pytest.fixture
def queue():
    """Create a ProcessingQueue instance."""
    return ProcessingQueue(max_retries=3)


class TestQueueItem:
    def test_queue_item_ordering(self):
        """Test that QueueItems are ordered by priority."""
        item1 = QueueItem(priority=1, file_path=Path("file1.jpg"))
        item2 = QueueItem(priority=5, file_path=Path("file2.jpg"))
        item3 = QueueItem(priority=3, file_path=Path("file3.jpg"))

        # Lower priority number should come first
        assert item1 < item2
        assert item1 < item3
        assert item3 < item2

    def test_queue_item_comparison_ignores_path(self):
        """Test that comparison only uses priority, not path."""
        item1 = QueueItem(priority=1, file_path=Path("z_file.jpg"))
        item2 = QueueItem(priority=2, file_path=Path("a_file.jpg"))

        # Even though 'a_file' comes before 'z_file' alphabetically,
        # item1 should still be less due to priority
        assert item1 < item2


class TestProcessingQueue:
    def test_add_files(self, queue):
        """Test adding files to the queue."""
        files = [Path("file1.jpg"), Path("file2.jpg"), Path("file3.jpg")]

        queue.add_files(files, priority=5)

        stats = queue.get_stats()
        assert stats["queued"] == 3
        assert stats["processing"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0

    def test_add_files_with_priority(self, queue):
        """Test adding files with different priorities."""
        high_priority = [Path("urgent.jpg")]
        low_priority = [Path("normal.jpg")]

        queue.add_files(low_priority, priority=10)
        queue.add_files(high_priority, priority=1)

        # High priority file should come first
        next_file = queue.get_next()
        assert next_file == Path("urgent.jpg")

    def test_get_next(self, queue):
        """Test getting next file from queue."""
        files = [Path("file1.jpg"), Path("file2.jpg")]
        queue.add_files(files)

        # Get first file
        file1 = queue.get_next()
        assert file1 in files
        assert file1 in queue.processing

        # Get second file
        file2 = queue.get_next()
        assert file2 in files
        assert file2 in queue.processing
        assert len(queue.processing) == 2

        # Queue should be empty now
        file3 = queue.get_next()
        assert file3 is None

    def test_get_next_empty_queue(self, queue):
        """Test getting from empty queue returns None."""
        result = queue.get_next()
        assert result is None

    def test_mark_completed(self, queue):
        """Test marking files as completed."""
        files = [Path("file1.jpg")]
        queue.add_files(files)

        file_path = queue.get_next()
        assert file_path in queue.processing

        queue.mark_completed(file_path)
        assert file_path not in queue.processing
        assert file_path in queue.completed

        stats = queue.get_stats()
        assert stats["processing"] == 0
        assert stats["completed"] == 1

    def test_mark_failed(self, queue):
        """Test marking files as failed."""
        files = [Path("file1.jpg")]
        queue.add_files(files)

        file_path = queue.get_next()
        error_msg = "Failed to process"

        queue.mark_failed(file_path, error_msg)
        assert file_path not in queue.processing
        assert (file_path, error_msg) in queue.failed

        stats = queue.get_stats()
        assert stats["processing"] == 0
        assert stats["failed"] == 1

    def test_is_empty(self, queue):
        """Test checking if queue is empty."""
        assert queue.is_empty()

        # Add files
        queue.add_files([Path("file1.jpg")])
        assert not queue.is_empty()

        # Get file (now processing)
        file_path = queue.get_next()
        assert not queue.is_empty()  # Still processing

        # Mark completed
        queue.mark_completed(file_path)
        assert queue.is_empty()

    def test_get_stats(self, queue):
        """Test getting queue statistics."""
        # Initial state
        stats = queue.get_stats()
        assert stats == {"queued": 0, "processing": 0, "completed": 0, "failed": 0}

        # Add files
        queue.add_files([Path("file1.jpg"), Path("file2.jpg")])
        stats = queue.get_stats()
        assert stats["queued"] == 2

        # Process one
        file1 = queue.get_next()
        stats = queue.get_stats()
        assert stats["queued"] == 1
        assert stats["processing"] == 1

        # Complete one
        queue.mark_completed(file1)
        stats = queue.get_stats()
        assert stats["processing"] == 0
        assert stats["completed"] == 1

        # Fail one
        file2 = queue.get_next()
        queue.mark_failed(file2, "error")
        stats = queue.get_stats()
        assert stats["failed"] == 1

    def test_thread_safety(self, queue):
        """Test thread-safe operations."""
        num_files = 100
        files = [Path(f"file{i}.jpg") for i in range(num_files)]
        queue.add_files(files)

        processed = []
        lock = threading.Lock()

        def worker():
            while True:
                file_path = queue.get_next()
                if file_path is None:
                    break

                # Simulate processing
                time.sleep(0.001)

                with lock:
                    processed.append(file_path)

                queue.mark_completed(file_path)

        # Start multiple worker threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify all files were processed exactly once
        assert len(processed) == num_files
        assert set(processed) == set(files)
        assert queue.is_empty()
        assert len(queue.completed) == num_files

    def test_priority_order(self, queue):
        """Test that files are processed in priority order."""
        # Add files with different priorities
        queue.add_files([Path("low1.jpg")], priority=10)
        queue.add_files([Path("high.jpg")], priority=1)
        queue.add_files([Path("medium.jpg")], priority=5)
        queue.add_files([Path("low2.jpg")], priority=10)

        # Files should come out in priority order
        assert queue.get_next() == Path("high.jpg")
        assert queue.get_next() == Path("medium.jpg")

        # The two priority-10 files can come in any order
        low_files = {queue.get_next(), queue.get_next()}
        assert low_files == {Path("low1.jpg"), Path("low2.jpg")}

    def test_retry_tracking(self, queue):
        """Test that retry count is tracked in QueueItem."""
        item = QueueItem(priority=5, file_path=Path("file.jpg"))
        assert item.retry_count == 0

        item.retry_count += 1
        assert item.retry_count == 1
