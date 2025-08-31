import pytest
from pathlib import Path
from unittest.mock import Mock
import tempfile
import shutil
from PIL import Image

from photodb.scanner import FileScanner
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
def scanner(mock_repository, temp_dir):
    """Create a FileScanner instance."""
    return FileScanner(mock_repository, base_path=str(temp_dir))


def create_test_image(path: Path, size=(100, 100), format='JPEG'):
    """Create a test image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', size, color='red')
    img.save(path, format=format)
    return path


def create_test_file(path: Path, content="test"):
    """Create a test non-image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestFileScanner:
    
    def test_scan_empty_directory(self, scanner, temp_dir):
        """Test scanning an empty directory."""
        result = scanner.scan_directory(temp_dir)
        
        assert result.total_files == 0
        assert len(result.new_files) == 0
        assert len(result.existing_files) == 0
        assert len(result.skipped_files) == 0
        assert not result.has_new_files
    
    def test_scan_directory_with_images(self, scanner, temp_dir, mock_repository):
        """Test scanning directory with image files."""
        # Create test images
        img1 = create_test_image(temp_dir / "photo1.jpg")
        img2 = create_test_image(temp_dir / "photo2.png")
        
        # Mock repository to return None (new files)
        mock_repository.get_photo_by_filename.return_value = None
        
        result = scanner.scan_directory(temp_dir)
        
        assert result.total_files == 2
        assert len(result.new_files) == 2
        assert len(result.existing_files) == 0
        assert result.has_new_files
    
    def test_scan_directory_recursive(self, scanner, temp_dir, mock_repository):
        """Test recursive directory scanning."""
        # Create nested structure
        create_test_image(temp_dir / "photo1.jpg")
        create_test_image(temp_dir / "subdir" / "photo2.jpg")
        create_test_image(temp_dir / "subdir" / "nested" / "photo3.jpg")
        
        mock_repository.get_photo_by_filename.return_value = None
        
        # Recursive scan
        result = scanner.scan_directory(temp_dir, recursive=True)
        assert result.total_files == 3
        assert len(result.new_files) == 3
        
        # Non-recursive scan
        result = scanner.scan_directory(temp_dir, recursive=False)
        assert result.total_files == 1
        assert len(result.new_files) == 1
    
    def test_scan_directory_with_pattern(self, scanner, temp_dir, mock_repository):
        """Test scanning with file pattern."""
        create_test_image(temp_dir / "photo1.jpg")
        create_test_image(temp_dir / "photo2.png")
        create_test_image(temp_dir / "image.bmp")
        
        mock_repository.get_photo_by_filename.return_value = None
        
        # Match only JPG files
        result = scanner.scan_directory(temp_dir, pattern="*.jpg")
        assert result.total_files == 1
        assert len(result.new_files) == 1
    
    def test_scan_existing_files(self, scanner, temp_dir, mock_repository):
        """Test detection of existing files in database."""
        img1 = create_test_image(temp_dir / "photo1.jpg")
        img2 = create_test_image(temp_dir / "photo2.jpg")
        
        # Mock repository to return existing photo for photo1
        def mock_get_photo(filename):
            if "photo1" in filename:
                return Mock()  # Existing photo
            return None
        
        mock_repository.get_photo_by_filename.side_effect = mock_get_photo
        
        result = scanner.scan_directory(temp_dir)
        
        assert result.total_files == 2
        assert len(result.new_files) == 1
        assert len(result.existing_files) == 1
    
    def test_scan_skip_invalid_files(self, scanner, temp_dir, mock_repository):
        """Test skipping of invalid/corrupted files."""
        # Create valid image
        create_test_image(temp_dir / "valid.jpg")
        
        # Create invalid image (empty file)
        invalid_file = temp_dir / "invalid.jpg"
        invalid_file.touch()
        
        # Create non-image file with image extension
        corrupt_file = temp_dir / "corrupt.jpg"
        corrupt_file.write_text("not an image")
        
        mock_repository.get_photo_by_filename.return_value = None
        
        result = scanner.scan_directory(temp_dir)
        
        assert result.total_files == 3
        assert len(result.new_files) == 1
        assert len(result.skipped_files) == 2
    
    def test_scan_mixed_file_types(self, scanner, temp_dir, mock_repository):
        """Test scanning directory with mixed file types."""
        create_test_image(temp_dir / "photo.jpg")
        create_test_file(temp_dir / "document.txt")
        create_test_file(temp_dir / "script.py")
        create_test_image(temp_dir / "image.png")
        
        mock_repository.get_photo_by_filename.return_value = None
        
        result = scanner.scan_directory(temp_dir)
        
        # Only image files should be counted
        assert result.total_files == 2
        assert len(result.new_files) == 2
    
    def test_scan_single_file(self, scanner, temp_dir, mock_repository):
        """Test scanning a single file."""
        img_path = create_test_image(temp_dir / "photo.jpg")
        
        mock_repository.get_photo_by_filename.return_value = None
        
        result = scanner.scan_file(img_path)
        
        assert result.total_files == 1
        assert len(result.new_files) == 1
        assert result.new_files[0] == img_path
    
    def test_scan_single_file_unsupported(self, scanner, temp_dir):
        """Test scanning unsupported file type."""
        txt_path = create_test_file(temp_dir / "document.txt")
        
        result = scanner.scan_file(txt_path)
        
        assert result.total_files == 1
        assert len(result.skipped_files) == 1
        assert len(result.new_files) == 0
    
    def test_relative_path_handling(self, scanner, temp_dir, mock_repository):
        """Test relative path calculation."""
        # Create image in subdirectory
        img_path = create_test_image(temp_dir / "subdir" / "photo.jpg")
        
        mock_repository.get_photo_by_filename.return_value = None
        
        result = scanner.scan_directory(temp_dir, recursive=True)
        
        # Verify relative path was used in repository check
        mock_repository.get_photo_by_filename.assert_called_with("subdir/photo.jpg")
    
    def test_watch_directory(self, scanner, temp_dir, mock_repository):
        """Test watch directory generator."""
        # Create initial file
        create_test_image(temp_dir / "photo1.jpg")
        
        mock_repository.get_photo_by_filename.return_value = None
        
        # Use a short poll interval for testing
        watcher = scanner.watch_directory(temp_dir, poll_interval=0.1)
        
        # Get first result
        result = next(watcher)
        assert len(result.new_files) == 1
        
        # Create another file
        create_test_image(temp_dir / "photo2.jpg")
        
        # Get second result
        result = next(watcher)
        assert len(result.new_files) == 1
        
        # Stop the generator
        watcher.close()
    
    def test_supported_formats(self, scanner, temp_dir, mock_repository):
        """Test all supported image formats."""
        formats = [
            ("photo.jpg", "JPEG"),
            ("photo.png", "PNG"),
            ("photo.bmp", "BMP"),
            ("photo.gif", "GIF"),
            ("photo.webp", "WEBP"),
            ("photo.tiff", "TIFF"),
        ]
        
        for filename, format in formats:
            create_test_image(temp_dir / filename, format=format)
        
        mock_repository.get_photo_by_filename.return_value = None
        
        result = scanner.scan_directory(temp_dir)
        
        assert result.total_files == len(formats)
        assert len(result.new_files) == len(formats)