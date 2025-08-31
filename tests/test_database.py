import pytest
import tempfile
import os
from datetime import datetime
import sqlite3

from photodb.database.connection import DatabaseConnection
from photodb.database.models import Photo, Metadata, ProcessingStatus
from photodb.database.repository import PhotoRepository


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def db_connection(temp_db):
    """Create a database connection for testing."""
    return DatabaseConnection(db_path=temp_db)


@pytest.fixture
def repository(db_connection):
    """Create a repository instance for testing."""
    return PhotoRepository(db_connection)


class TestDatabaseConnection:
    def test_database_initialization(self, db_connection, temp_db):
        """Test that database is properly initialized with schema."""
        # Check that database file was created
        assert os.path.exists(temp_db)
        
        # Check that tables were created
        with db_connection.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            
            assert 'photos' in tables
            assert 'metadata' in tables
            assert 'processing_status' in tables
    
    def test_transaction_rollback(self, db_connection):
        """Test that transactions are rolled back on error."""
        with pytest.raises(sqlite3.IntegrityError):
            with db_connection.transaction() as conn:
                # This should succeed
                conn.execute(
                    "INSERT INTO photos (id, filename, normalized_path) VALUES (?, ?, ?)",
                    ('test-id', 'test.jpg', 'normalized/test.jpg')
                )
                # This should fail due to duplicate primary key
                conn.execute(
                    "INSERT INTO photos (id, filename, normalized_path) VALUES (?, ?, ?)",
                    ('test-id', 'test2.jpg', 'normalized/test2.jpg')
                )
        
        # Check that no records were inserted due to rollback
        with db_connection.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
            assert count == 0
    
    def test_foreign_keys_enabled(self, db_connection):
        """Test that foreign key constraints are enforced."""
        with pytest.raises(sqlite3.IntegrityError):
            with db_connection.transaction() as conn:
                # Try to insert metadata for non-existent photo
                conn.execute(
                    "INSERT INTO metadata (photo_id, created_at) VALUES (?, ?)",
                    ('non-existent-id', datetime.now())
                )


class TestModels:
    def test_photo_create(self):
        """Test Photo.create factory method."""
        photo = Photo.create('test.jpg', 'normalized/test.jpg')
        
        assert photo.id is not None
        assert photo.filename == 'test.jpg'
        assert photo.normalized_path == 'normalized/test.jpg'
        assert isinstance(photo.created_at, datetime)
        assert isinstance(photo.updated_at, datetime)
        assert photo.created_at == photo.updated_at
    
    def test_metadata_create(self):
        """Test Metadata.create factory method."""
        metadata = Metadata.create(
            'photo-id',
            captured_at=datetime(2024, 1, 1, 12, 0, 0),
            latitude=37.7749,
            longitude=-122.4194,
            extra={'camera': 'iPhone 15'}
        )
        
        assert metadata.photo_id == 'photo-id'
        assert metadata.captured_at == datetime(2024, 1, 1, 12, 0, 0)
        assert metadata.latitude == 37.7749
        assert metadata.longitude == -122.4194
        assert metadata.extra == {'camera': 'iPhone 15'}
        assert isinstance(metadata.created_at, datetime)
    
    def test_processing_status_to_dict(self):
        """Test ProcessingStatus.to_dict method."""
        status = ProcessingStatus(
            photo_id='photo-id',
            stage='normalize',
            status='completed',
            processed_at=datetime(2024, 1, 1, 12, 0, 0),
            error_message=None
        )
        
        result = status.to_dict()
        assert result['photo_id'] == 'photo-id'
        assert result['stage'] == 'normalize'
        assert result['status'] == 'completed'
        assert result['processed_at'] == '2024-01-01T12:00:00'
        assert result['error_message'] is None


class TestPhotoRepository:
    def test_create_and_get_photo(self, repository):
        """Test creating and retrieving a photo."""
        photo = Photo.create('test.jpg', 'normalized/test.jpg')
        repository.create_photo(photo)
        
        # Get by ID
        retrieved = repository.get_photo_by_id(photo.id)
        assert retrieved is not None
        assert retrieved.id == photo.id
        assert retrieved.filename == 'test.jpg'
        
        # Get by filename
        retrieved = repository.get_photo_by_filename('test.jpg')
        assert retrieved is not None
        assert retrieved.id == photo.id
    
    def test_update_photo(self, repository):
        """Test updating a photo record."""
        photo = Photo.create('test.jpg', 'normalized/test.jpg')
        repository.create_photo(photo)
        
        # Update the photo
        photo.normalized_path = 'updated/test.jpg'
        repository.update_photo(photo)
        
        # Retrieve and verify
        retrieved = repository.get_photo_by_id(photo.id)
        assert retrieved.normalized_path == 'updated/test.jpg'
        assert retrieved.updated_at > retrieved.created_at
    
    def test_create_and_get_metadata(self, repository):
        """Test creating and retrieving metadata."""
        # First create a photo
        photo = Photo.create('test.jpg', 'normalized/test.jpg')
        repository.create_photo(photo)
        
        # Create metadata
        metadata = Metadata.create(
            photo.id,
            captured_at=datetime(2024, 1, 1),
            latitude=37.7749,
            longitude=-122.4194,
            extra={'camera': 'iPhone'}
        )
        repository.create_metadata(metadata)
        
        # Retrieve and verify
        retrieved = repository.get_metadata(photo.id)
        assert retrieved is not None
        assert retrieved.photo_id == photo.id
        assert retrieved.latitude == 37.7749
        assert retrieved.extra == {'camera': 'iPhone'}
    
    def test_processing_status(self, repository):
        """Test processing status tracking."""
        # Create a photo
        photo = Photo.create('test.jpg', 'normalized/test.jpg')
        repository.create_photo(photo)
        
        # Update processing status
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
    
    def test_get_unprocessed_photos(self, repository):
        """Test getting unprocessed photos for a stage."""
        # Create photos with different processing states
        photo1 = Photo.create('photo1.jpg', 'normalized/photo1.jpg')
        photo2 = Photo.create('photo2.jpg', 'normalized/photo2.jpg')
        photo3 = Photo.create('photo3.jpg', 'normalized/photo3.jpg')
        
        repository.create_photo(photo1)
        repository.create_photo(photo2)
        repository.create_photo(photo3)
        
        # Mark photo1 as completed for 'normalize' stage
        repository.update_processing_status(ProcessingStatus(
            photo_id=photo1.id,
            stage='normalize',
            status='completed',
            processed_at=datetime.now(),
            error_message=None
        ))
        
        # Mark photo2 as failed
        repository.update_processing_status(ProcessingStatus(
            photo_id=photo2.id,
            stage='normalize',
            status='failed',
            processed_at=datetime.now(),
            error_message='Test error'
        ))
        
        # Get unprocessed photos
        unprocessed = repository.get_unprocessed_photos('normalize')
        unprocessed_ids = {p.id for p in unprocessed}
        
        assert photo1.id not in unprocessed_ids  # completed
        assert photo2.id in unprocessed_ids      # failed
        assert photo3.id in unprocessed_ids      # not processed
    
    def test_get_failed_photos(self, repository):
        """Test getting failed photos for a stage."""
        photo = Photo.create('test.jpg', 'normalized/test.jpg')
        repository.create_photo(photo)
        
        # Mark as failed
        repository.update_processing_status(ProcessingStatus(
            photo_id=photo.id,
            stage='normalize',
            status='failed',
            processed_at=datetime.now(),
            error_message='Test error message'
        ))
        
        # Get failed photos
        failed = repository.get_failed_photos('normalize')
        assert len(failed) == 1
        assert failed[0]['id'] == photo.id
        assert failed[0]['error_message'] == 'Test error message'
    
    def test_get_photo_count_by_status(self, repository):
        """Test getting photo counts by status."""
        # Create photos and set different statuses
        for i in range(5):
            photo = Photo.create(f'photo{i}.jpg', f'normalized/photo{i}.jpg')
            repository.create_photo(photo)
            
            if i < 2:
                status = 'completed'
            elif i < 4:
                status = 'processing'
            else:
                status = 'failed'
            
            repository.update_processing_status(ProcessingStatus(
                photo_id=photo.id,
                stage='normalize',
                status=status,
                processed_at=datetime.now() if status != 'processing' else None,
                error_message='Error' if status == 'failed' else None
            ))
        
        # Get counts
        counts = repository.get_photo_count_by_status('normalize')
        assert counts['completed'] == 2
        assert counts['processing'] == 2
        assert counts['failed'] == 1
    
    def test_unique_filename_constraint(self, repository):
        """Test that duplicate filenames are rejected."""
        photo1 = Photo.create('test.jpg', 'normalized/test1.jpg')
        photo2 = Photo.create('test.jpg', 'normalized/test2.jpg')
        
        repository.create_photo(photo1)
        
        with pytest.raises(sqlite3.IntegrityError):
            repository.create_photo(photo2)