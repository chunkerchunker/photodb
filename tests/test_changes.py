import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from PIL import Image

from photodb.changes import ChangeDetector
from photodb.database.repository import PhotoRepository


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    return Mock(spec=PhotoRepository)


@pytest.fixture
def detector(mock_repository):
    """Create a ChangeDetector instance."""
    return ChangeDetector(mock_repository)


def create_test_image(path: Path, size=(100, 100)):
    """Create a test image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', size, color='blue')
    img.save(path, 'JPEG')
    return path


def create_test_file(path: Path, content="test"):
    """Create a test file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestChangeDetector:
    
    def test_detect_modified_files_default_timeframe(self, detector, temp_dir):
        """Test detecting modified files with default 24-hour timeframe."""
        # Create files with different modification times
        old_file = create_test_file(temp_dir / "old.txt")
        new_file = create_test_file(temp_dir / "new.txt")
        
        # Set old file's modification time to 2 days ago
        old_time = time.time() - (2 * 24 * 60 * 60)
        os.utime(old_file, (old_time, old_time))
        
        # Detect modified files (should only find new.txt)
        modified = detector.detect_modified_files(temp_dir)
        
        assert len(modified) == 1
        assert new_file in modified
        assert old_file not in modified
    
    def test_detect_modified_files_custom_timeframe(self, detector, temp_dir):
        """Test detecting modified files with custom timeframe."""
        import os
        
        # Create test files
        file1 = create_test_file(temp_dir / "file1.txt")
        file2 = create_test_file(temp_dir / "file2.txt")
        file3 = create_test_file(temp_dir / "file3.txt")
        
        # Set different modification times
        now = time.time()
        os.utime(file1, (now - 7200, now - 7200))  # 2 hours ago
        os.utime(file2, (now - 3600, now - 3600))  # 1 hour ago
        os.utime(file3, (now - 300, now - 300))    # 5 minutes ago
        
        # Check files modified in last 30 minutes
        since = datetime.now() - timedelta(minutes=30)
        modified = detector.detect_modified_files(temp_dir, since=since)
        
        assert len(modified) == 1
        assert file3 in modified
    
    def test_detect_modified_files_recursive(self, detector, temp_dir):
        """Test detecting modified files in subdirectories."""
        # Create nested structure
        create_test_file(temp_dir / "root.txt")
        create_test_file(temp_dir / "subdir" / "nested.txt")
        create_test_file(temp_dir / "subdir" / "deep" / "file.txt")
        
        modified = detector.detect_modified_files(temp_dir)
        
        assert len(modified) == 3
    
    def test_detect_modified_files_ignores_directories(self, detector, temp_dir):
        """Test that directories themselves are not included."""
        # Create directory and file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        file_path = create_test_file(subdir / "file.txt")
        
        modified = detector.detect_modified_files(temp_dir)
        
        # Should only find the file, not the directory
        assert len(modified) == 1
        assert file_path in modified
        assert subdir not in modified
    
    def test_detect_corrupted_files(self, detector, temp_dir):
        """Test detecting corrupted image files."""
        # Create valid image
        valid_img = create_test_image(temp_dir / "valid.jpg")
        
        # Create corrupted image (invalid JPEG data)
        corrupt_img = temp_dir / "corrupt.jpg"
        corrupt_img.write_bytes(b"not a valid jpeg")
        
        # Create empty image file
        empty_img = temp_dir / "empty.jpg"
        empty_img.touch()
        
        files = [valid_img, corrupt_img, empty_img]
        corrupted = detector.detect_corrupted_files(files)
        
        assert valid_img not in corrupted
        assert corrupt_img in corrupted
        assert empty_img in corrupted
    
    def test_detect_corrupted_files_empty_list(self, detector):
        """Test detecting corrupted files with empty input."""
        corrupted = detector.detect_corrupted_files([])
        assert corrupted == []
    
    def test_detect_corrupted_files_non_image(self, detector, temp_dir):
        """Test that non-image files are detected as corrupted."""
        text_file = create_test_file(temp_dir / "document.txt")
        
        corrupted = detector.detect_corrupted_files([text_file])
        
        assert text_file in corrupted
    
    def test_calculate_checksum(self, detector, temp_dir):
        """Test MD5 checksum calculation."""
        # Create file with known content
        file_path = temp_dir / "test.txt"
        file_path.write_text("Hello, World!")
        
        checksum = detector._calculate_checksum(file_path)
        
        # Known MD5 of "Hello, World!"
        expected = "65a8e27d8879283831b664bd8b7f0ad4"
        assert checksum == expected
    
    def test_calculate_checksum_identical_files(self, detector, temp_dir):
        """Test that identical files have same checksum."""
        # Create two files with same content
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        
        content = "Same content in both files"
        file1.write_text(content)
        file2.write_text(content)
        
        checksum1 = detector._calculate_checksum(file1)
        checksum2 = detector._calculate_checksum(file2)
        
        assert checksum1 == checksum2
    
    def test_calculate_checksum_different_files(self, detector, temp_dir):
        """Test that different files have different checksums."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        checksum1 = detector._calculate_checksum(file1)
        checksum2 = detector._calculate_checksum(file2)
        
        assert checksum1 != checksum2
    
    def test_calculate_checksum_large_file(self, detector, temp_dir):
        """Test checksum calculation for large file."""
        # Create a large file (10MB)
        large_file = temp_dir / "large.bin"
        data = b"x" * (10 * 1024 * 1024)
        large_file.write_bytes(data)
        
        # Should handle large file without issues
        checksum = detector._calculate_checksum(large_file)
        assert len(checksum) == 32  # MD5 hash is 32 hex characters
    
    def test_detect_moved_files(self, detector, temp_dir):
        """Test detection of moved files (placeholder test)."""
        # Create some files
        create_test_file(temp_dir / "file1.txt", "content1")
        create_test_file(temp_dir / "file2.txt", "content2")
        
        # This is a placeholder implementation in the actual code
        moved = detector.detect_moved_files(temp_dir)
        
        # Should return empty dict for now
        assert isinstance(moved, dict)
        assert len(moved) == 0
    
    def test_detect_moved_files_with_checksums(self, detector, temp_dir):
        """Test that checksum calculation works for move detection."""
        # Create files
        file1 = create_test_file(temp_dir / "original.txt", "unique content")
        file2 = create_test_file(temp_dir / "another.txt", "different content")
        
        # Calculate checksums for all files
        checksums = {}
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                checksum = detector._calculate_checksum(file_path)
                checksums[checksum] = file_path
        
        assert len(checksums) == 2
        
        # Each file should have unique checksum
        assert len(set(checksums.keys())) == 2
    
    @patch('photodb.changes.datetime')
    def test_detect_modified_with_mocked_time(self, mock_datetime, detector, temp_dir):
        """Test modified file detection with mocked datetime."""
        # Set up mock
        mock_now = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        mock_datetime.fromtimestamp = datetime.fromtimestamp
        
        # Create file
        test_file = create_test_file(temp_dir / "test.txt")
        
        # Set file modification time to 12 hours ago
        file_time = time.mktime((mock_now - timedelta(hours=12)).timetuple())
        os.utime(test_file, (file_time, file_time))
        
        # Should be found when checking last 24 hours
        modified = detector.detect_modified_files(temp_dir)
        assert test_file in modified


import os  # Add this import at the top of the file for the os.utime calls