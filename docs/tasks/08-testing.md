# Task 08: Testing Infrastructure

## Objective
Establish comprehensive testing infrastructure including unit tests, integration tests, fixtures, and test utilities for all components of the PhotoDB pipeline.

## Dependencies
- All previous tasks (testing covers all components)

## Deliverables

### 1. Test Configuration (tests/conftest.py)
```python
import pytest
import tempfile
import shutil
from pathlib import Path
import os
from typing import Generator
import sqlite3

from photodb.database.connection import DatabaseConnection
from photodb.database.repository import PhotoRepository

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def test_db(temp_dir) -> Generator[DatabaseConnection, None, None]:
    """Create a test database."""
    db_path = temp_dir / "test.db"
    db = DatabaseConnection(str(db_path))
    yield db
    # Cleanup handled by temp_dir fixture

@pytest.fixture
def repository(test_db) -> PhotoRepository:
    """Create a test repository."""
    return PhotoRepository(test_db)

@pytest.fixture
def sample_images(temp_dir) -> dict:
    """Create sample test images."""
    from PIL import Image
    
    images = {}
    
    # Create various test images
    sizes_and_formats = [
        ('small_jpg', (800, 600), 'JPEG'),
        ('large_jpg', (4000, 3000), 'JPEG'),
        ('square_png', (1000, 1000), 'PNG'),
        ('portrait', (768, 1024), 'PNG'),
        ('landscape', (1920, 1080), 'JPEG'),
    ]
    
    for name, size, format_type in sizes_and_formats:
        img = Image.new('RGB', size, color='red')
        
        # Add some EXIF data for JPEG images
        if format_type == 'JPEG':
            from PIL import ExifTags
            exif = img.getexif()
            exif[0x9003] = '2024:01:15 14:30:00'  # DateTimeOriginal
            exif[0x010F] = 'TestCamera'  # Make
            exif[0x0110] = 'Model X'  # Model
        
        path = temp_dir / f"{name}.{format_type.lower()}"
        img.save(path, format_type)
        images[name] = path
    
    return images

@pytest.fixture
def mock_config(temp_dir) -> dict:
    """Create mock configuration."""
    return {
        'db_path': str(temp_dir / 'test.db'),
        'ingest_path': str(temp_dir / 'ingest'),
        'img_path': str(temp_dir / 'processed'),
        'log_level': 'DEBUG',
        'log_file': str(temp_dir / 'test.log')
    }

@pytest.fixture
def setup_directories(mock_config):
    """Set up test directory structure."""
    for key in ['ingest_path', 'img_path']:
        Path(mock_config[key]).mkdir(parents=True, exist_ok=True)
    return mock_config
```

### 2. Database Tests (tests/test_database.py)
```python
import pytest
from datetime import datetime
import json

from photodb.database.models import Photo, Metadata, ProcessingStatus

class TestDatabaseConnection:
    def test_connection_creation(self, test_db):
        """Test database connection is created."""
        assert test_db.db_path.endswith('test.db')
        
        # Check tables exist
        with test_db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            assert 'photos' in tables
            assert 'metadata' in tables
            assert 'processing_status' in tables
    
    def test_transaction_rollback(self, test_db):
        """Test transaction rollback on error."""
        with pytest.raises(Exception):
            with test_db.transaction() as conn:
                conn.execute(
                    "INSERT INTO photos (id, filename, normalized_path) VALUES (?, ?, ?)",
                    ('test-id', 'test.jpg', 'test.png')
                )
                # Force an error
                raise Exception("Test error")
        
        # Check that insert was rolled back
        with test_db.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
            assert count == 0

class TestPhotoRepository:
    def test_create_and_retrieve_photo(self, repository):
        """Test creating and retrieving a photo."""
        photo = Photo.create('test.jpg', 'processed/test.png')
        repository.create_photo(photo)
        
        # Retrieve by filename
        retrieved = repository.get_photo_by_filename('test.jpg')
        assert retrieved is not None
        assert retrieved.filename == 'test.jpg'
        assert retrieved.normalized_path == 'processed/test.png'
        
        # Retrieve by ID
        retrieved_by_id = repository.get_photo_by_id(photo.id)
        assert retrieved_by_id is not None
        assert retrieved_by_id.id == photo.id
    
    def test_create_metadata(self, repository):
        """Test creating metadata."""
        # First create a photo
        photo = Photo.create('test.jpg', 'processed/test.png')
        repository.create_photo(photo)
        
        # Create metadata
        metadata = Metadata.create(
            photo_id=photo.id,
            captured_at=datetime(2024, 1, 15, 14, 30),
            latitude=37.7749,
            longitude=-122.4194,
            extra={'camera': 'TestCamera'}
        )
        repository.create_metadata(metadata)
        
        # Retrieve metadata
        retrieved = repository.get_metadata(photo.id)
        assert retrieved is not None
        assert retrieved.latitude == 37.7749
        assert retrieved.longitude == -122.4194
        assert retrieved.extra['camera'] == 'TestCamera'
    
    def test_processing_status(self, repository):
        """Test processing status tracking."""
        # Create a photo
        photo = Photo.create('test.jpg', 'processed/test.png')
        repository.create_photo(photo)
        
        # Update status
        status = ProcessingStatus(
            photo_id=photo.id,
            stage='normalize',
            status='completed',
            processed_at=datetime.now(),
            error_message=None
        )
        repository.update_processing_status(status)
        
        # Check status
        retrieved = repository.get_processing_status(photo.id, 'normalize')
        assert retrieved is not None
        assert retrieved.status == 'completed'
        
        # Check has_been_processed
        assert repository.has_been_processed(photo.id, 'normalize')
        assert not repository.has_been_processed(photo.id, 'metadata')
```

### 3. Image Handler Tests (tests/test_image_handler.py)
```python
import pytest
from PIL import Image

from photodb.utils.image import ImageHandler
from photodb.utils.validation import ImageValidator

class TestImageHandler:
    def test_supported_formats(self):
        """Test format support detection."""
        from pathlib import Path
        
        assert ImageHandler.is_supported(Path('test.jpg'))
        assert ImageHandler.is_supported(Path('test.JPEG'))
        assert ImageHandler.is_supported(Path('test.png'))
        assert ImageHandler.is_supported(Path('test.heic'))
        assert not ImageHandler.is_supported(Path('test.txt'))
        assert not ImageHandler.is_supported(Path('test.pdf'))
    
    def test_open_image(self, sample_images):
        """Test opening various image formats."""
        for name, path in sample_images.items():
            img = ImageHandler.open_image(path)
            assert isinstance(img, Image.Image)
            assert img.mode in ('RGB', 'L')
    
    def test_resize_calculation(self):
        """Test resize dimension calculations."""
        # Test image that needs resizing
        large_size = (4000, 3000)
        new_size = ImageHandler.calculate_resize_dimensions(
            large_size,
            ImageHandler.MAX_DIMENSIONS
        )
        assert new_size is not None
        assert new_size[0] <= 1344
        assert new_size[1] <= 896
        
        # Test image that doesn't need resizing
        small_size = (800, 600)
        new_size = ImageHandler.calculate_resize_dimensions(
            small_size,
            ImageHandler.MAX_DIMENSIONS
        )
        assert new_size is None
    
    def test_resize_maintains_aspect_ratio(self, sample_images):
        """Test that resizing maintains aspect ratio."""
        img_path = sample_images['large_jpg']
        img = ImageHandler.open_image(img_path)
        
        original_ratio = img.width / img.height
        
        # Resize
        new_size = (800, 600)
        resized = ImageHandler.resize_image(img, new_size)
        
        resized_ratio = resized.width / resized.height
        assert abs(original_ratio - resized_ratio) < 0.01
    
    def test_save_as_png(self, temp_dir, sample_images):
        """Test PNG saving."""
        img_path = sample_images['small_jpg']
        img = ImageHandler.open_image(img_path)
        
        output_path = temp_dir / 'output.png'
        ImageHandler.save_as_png(img, output_path)
        
        assert output_path.exists()
        
        # Verify it's a PNG
        with Image.open(output_path) as saved:
            assert saved.format == 'PNG'

class TestImageValidator:
    def test_validate_valid_file(self, sample_images):
        """Test validation of valid image files."""
        for name, path in sample_images.items():
            assert ImageValidator.validate_file(path)
    
    def test_validate_invalid_file(self, temp_dir):
        """Test validation of invalid files."""
        # Non-existent file
        from pathlib import Path
        assert not ImageValidator.validate_file(Path('nonexistent.jpg'))
        
        # Empty file
        empty_file = temp_dir / 'empty.jpg'
        empty_file.write_bytes(b'')
        assert not ImageValidator.validate_file(empty_file)
        
        # Text file pretending to be image
        fake_image = temp_dir / 'fake.jpg'
        fake_image.write_text('This is not an image')
        assert not ImageValidator.validate_file(fake_image)
```

### 4. Stage Tests (tests/test_stages.py)
```python
import pytest
from pathlib import Path
import uuid

from photodb.stages.normalize import NormalizeStage
from photodb.stages.metadata import MetadataStage

class TestNormalizeStage:
    def test_should_process(self, repository, mock_config, sample_images):
        """Test processing detection."""
        stage = NormalizeStage(repository, mock_config)
        
        img_path = sample_images['small_jpg']
        
        # Should process new file
        assert stage.should_process(img_path)
        
        # Process the file
        result = stage.process(img_path)
        assert result['success']
        
        # Should not process again
        assert not stage.should_process(img_path)
        
        # Should process with force
        assert stage.should_process(img_path, force=True)
    
    def test_process_creates_uuid(self, repository, mock_config, sample_images):
        """Test that processing assigns UUID."""
        stage = NormalizeStage(repository, mock_config)
        
        img_path = sample_images['small_jpg']
        result = stage.process(img_path)
        
        assert result['success']
        assert 'photo_id' in result
        
        # Verify UUID format
        try:
            uuid.UUID(result['photo_id'])
        except ValueError:
            pytest.fail("Invalid UUID format")
    
    def test_resize_large_image(self, repository, mock_config, sample_images):
        """Test resizing of large images."""
        stage = NormalizeStage(repository, mock_config)
        
        img_path = sample_images['large_jpg']
        result = stage.process(img_path)
        
        assert result['success']
        assert result['was_resized']
        assert result['new_size'][0] <= 1344
        assert result['new_size'][1] <= 1344

class TestMetadataStage:
    def test_requires_stage1(self, repository, mock_config, sample_images):
        """Test that Stage 2 requires Stage 1 completion."""
        metadata_stage = MetadataStage(repository, mock_config)
        
        img_path = sample_images['small_jpg']
        
        # Should not process without Stage 1
        assert not metadata_stage.should_process(img_path)
        
        # Run Stage 1
        normalize_stage = NormalizeStage(repository, mock_config)
        normalize_result = normalize_stage.process(img_path)
        assert normalize_result['success']
        
        # Now should process
        assert metadata_stage.should_process(img_path)
    
    def test_extract_metadata(self, repository, mock_config, sample_images):
        """Test metadata extraction."""
        # First run normalization
        normalize_stage = NormalizeStage(repository, mock_config)
        img_path = sample_images['small_jpg']
        normalize_stage.process(img_path)
        
        # Extract metadata
        metadata_stage = MetadataStage(repository, mock_config)
        result = metadata_stage.process(img_path)
        
        assert result['success']
        assert 'photo_id' in result
        assert 'metadata_fields' in result
```

### 5. CLI Tests (tests/test_cli.py)
```python
import pytest
from click.testing import CliRunner
from pathlib import Path

from photodb.cli import main

class TestCLI:
    def test_help(self):
        """Test CLI help."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Process photos' in result.output
    
    def test_process_single_file(self, sample_images, setup_directories):
        """Test processing single file."""
        runner = CliRunner()
        img_path = sample_images['small_jpg']
        
        result = runner.invoke(main, [str(img_path)])
        assert result.exit_code == 0
    
    def test_process_directory(self, sample_images, setup_directories):
        """Test processing directory."""
        runner = CliRunner()
        dir_path = Path(sample_images['small_jpg']).parent
        
        result = runner.invoke(main, [str(dir_path)])
        assert result.exit_code == 0
    
    def test_dry_run(self, sample_images, setup_directories):
        """Test dry run mode."""
        runner = CliRunner()
        img_path = sample_images['small_jpg']
        
        result = runner.invoke(main, [str(img_path), '--dry-run'])
        assert result.exit_code == 0
        assert 'DRY RUN' in result.output or 'Would process' in result.output
```

### 6. Test Utilities (tests/utils.py)
```python
from pathlib import Path
from PIL import Image
import random
from datetime import datetime, timedelta

def create_test_image_with_exif(
    path: Path,
    size: tuple = (1024, 768),
    format: str = 'JPEG',
    exif_data: dict = None
):
    """Create a test image with EXIF data."""
    img = Image.new('RGB', size, color=(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ))
    
    if exif_data and format == 'JPEG':
        exif = img.getexif()
        for tag, value in exif_data.items():
            exif[tag] = value
        img.save(path, format, exif=exif)
    else:
        img.save(path, format)
    
    return path

def create_test_directory_structure(base_path: Path) -> dict:
    """Create a test directory structure with images."""
    structure = {
        'root': base_path,
        'subdirs': [],
        'images': []
    }
    
    # Create subdirectories
    for i in range(3):
        subdir = base_path / f'folder_{i}'
        subdir.mkdir(parents=True, exist_ok=True)
        structure['subdirs'].append(subdir)
        
        # Add images to each subdirectory
        for j in range(5):
            img_path = subdir / f'image_{i}_{j}.jpg'
            create_test_image_with_exif(img_path)
            structure['images'].append(img_path)
    
    return structure

def generate_random_datetime(
    start: datetime = None,
    end: datetime = None
) -> datetime:
    """Generate random datetime between start and end."""
    if not start:
        start = datetime(2020, 1, 1)
    if not end:
        end = datetime.now()
    
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86400)
    
    return start + timedelta(days=random_days, seconds=random_seconds)
```

## Implementation Steps

1. **Set up pytest configuration**
   - Configure pytest.ini
   - Set up coverage reporting
   - Configure test discovery

2. **Create fixtures**
   - Temporary directories
   - Test database
   - Sample images
   - Mock configurations

3. **Write unit tests**
   - Database operations
   - Image handling
   - Metadata extraction
   - File discovery

4. **Write integration tests**
   - Full pipeline processing
   - CLI commands
   - Error scenarios

5. **Add performance tests**
   - Large directory processing
   - Parallel processing
   - Memory usage

## Testing Checklist

- [ ] All modules have test coverage
- [ ] Fixtures work correctly
- [ ] Database tests pass
- [ ] Image processing tests pass
- [ ] Stage tests verify dependencies
- [ ] CLI tests cover all options
- [ ] Error cases are tested
- [ ] Performance tests pass
- [ ] Coverage > 80%

## Notes

- Use pytest-mock for complex mocking scenarios
- Add benchmark tests for performance-critical paths
- Consider property-based testing with hypothesis
- Add stress tests for large datasets
- Implement continuous integration testing