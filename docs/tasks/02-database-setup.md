# Task 02: Database Setup and Schema Implementation

## Objective
Implement the SQLite database layer including schema creation, connection management, and data models for the PhotoDB pipeline.

## Dependencies
- Task 01: Project Setup (pyproject.toml, project structure)

## Deliverables

### 1. Database Schema (schema.sql)
Based on the requirements, implement the following schema:

```sql
-- Photos table: Core photo records
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,  -- UUID
    filename TEXT NOT NULL UNIQUE,  -- Relative path from INGEST_PATH
    normalized_path TEXT NOT NULL,  -- Path to normalized image in IMG_PATH
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metadata table: Extracted photo metadata
CREATE TABLE IF NOT EXISTS metadata (
    photo_id TEXT PRIMARY KEY,
    captured_at TIMESTAMP,  -- When photo was taken
    latitude REAL,
    longitude REAL,
    extra JSON,  -- All EXIF/TIFF/IFD metadata as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

-- Processing status table: Track processing stages
CREATE TABLE IF NOT EXISTS processing_status (
    photo_id TEXT NOT NULL,
    stage TEXT NOT NULL,  -- 'normalize', 'metadata', etc.
    status TEXT NOT NULL,  -- 'pending', 'processing', 'completed', 'failed'
    processed_at TIMESTAMP,
    error_message TEXT,
    PRIMARY KEY (photo_id, stage),
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_photos_filename ON photos(filename);
CREATE INDEX IF NOT EXISTS idx_metadata_captured_at ON metadata(captured_at);
CREATE INDEX IF NOT EXISTS idx_metadata_location ON metadata(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_status(status, stage);
```

### 2. Database Connection Manager (src/photodb/database/connection.py)
```python
import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv('DB_PATH', './data/photos.db')
        self._ensure_directory()
        self._init_database()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent.parent.parent.parent / 'schema.sql'
        
        with self.get_connection() as conn:
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    conn.executescript(f.read())
            else:
                # Embedded schema fallback
                conn.executescript(self._get_embedded_schema())
            conn.commit()
            logger.info(f"Database initialized at: {self.db_path}")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Execute operations within a transaction."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise
    
    def _get_embedded_schema(self) -> str:
        """Return embedded schema as fallback."""
        return '''
        -- Embedded schema (same as schema.sql)
        CREATE TABLE IF NOT EXISTS photos (...);
        CREATE TABLE IF NOT EXISTS metadata (...);
        CREATE TABLE IF NOT EXISTS processing_status (...);
        '''
```

### 3. Data Models (src/photodb/database/models.py)
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import json
import uuid

@dataclass
class Photo:
    id: str
    filename: str
    normalized_path: str
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def create(cls, filename: str, normalized_path: str) -> 'Photo':
        """Create a new photo record."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            filename=filename,
            normalized_path=normalized_path,
            created_at=now,
            updated_at=now
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'filename': self.filename,
            'normalized_path': self.normalized_path,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

@dataclass
class Metadata:
    photo_id: str
    captured_at: Optional[datetime]
    latitude: Optional[float]
    longitude: Optional[float]
    extra: Dict[str, Any]
    created_at: datetime
    
    @classmethod
    def create(cls, photo_id: str, **kwargs) -> 'Metadata':
        """Create metadata record from extracted data."""
        return cls(
            photo_id=photo_id,
            captured_at=kwargs.get('captured_at'),
            latitude=kwargs.get('latitude'),
            longitude=kwargs.get('longitude'),
            extra=kwargs.get('extra', {}),
            created_at=datetime.now()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'photo_id': self.photo_id,
            'captured_at': self.captured_at.isoformat() if self.captured_at else None,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'extra': json.dumps(self.extra),
            'created_at': self.created_at.isoformat()
        }

@dataclass
class ProcessingStatus:
    photo_id: str
    stage: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    processed_at: Optional[datetime]
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'photo_id': self.photo_id,
            'stage': self.stage,
            'status': self.status,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'error_message': self.error_message
        }
```

### 4. Database Repository (src/photodb/database/repository.py)
```python
import sqlite3
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import logging

from .connection import DatabaseConnection
from .models import Photo, Metadata, ProcessingStatus

logger = logging.getLogger(__name__)

class PhotoRepository:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def create_photo(self, photo: Photo) -> None:
        """Insert a new photo record."""
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO photos (id, filename, normalized_path, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (photo.id, photo.filename, photo.normalized_path, 
                 photo.created_at, photo.updated_at)
            )
    
    def get_photo_by_filename(self, filename: str) -> Optional[Photo]:
        """Get photo by filename."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM photos WHERE filename = ?", (filename,)
            ).fetchone()
            
            if row:
                return Photo(**dict(row))
            return None
    
    def get_photo_by_id(self, photo_id: str) -> Optional[Photo]:
        """Get photo by ID."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM photos WHERE id = ?", (photo_id,)
            ).fetchone()
            
            if row:
                return Photo(**dict(row))
            return None
    
    def update_photo(self, photo: Photo) -> None:
        """Update existing photo record."""
        photo.updated_at = datetime.now()
        with self.db.transaction() as conn:
            conn.execute(
                """UPDATE photos 
                   SET normalized_path = ?, updated_at = ?
                   WHERE id = ?""",
                (photo.normalized_path, photo.updated_at, photo.id)
            )
    
    def create_metadata(self, metadata: Metadata) -> None:
        """Insert metadata record."""
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO metadata 
                   (photo_id, captured_at, latitude, longitude, extra, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (metadata.photo_id, metadata.captured_at, metadata.latitude,
                 metadata.longitude, json.dumps(metadata.extra), metadata.created_at)
            )
    
    def get_metadata(self, photo_id: str) -> Optional[Metadata]:
        """Get metadata for a photo."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM metadata WHERE photo_id = ?", (photo_id,)
            ).fetchone()
            
            if row:
                data = dict(row)
                data['extra'] = json.loads(data['extra']) if data['extra'] else {}
                return Metadata(**data)
            return None
    
    def update_processing_status(self, status: ProcessingStatus) -> None:
        """Update processing status for a photo stage."""
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO processing_status 
                   (photo_id, stage, status, processed_at, error_message)
                   VALUES (?, ?, ?, ?, ?)""",
                (status.photo_id, status.stage, status.status,
                 status.processed_at, status.error_message)
            )
    
    def get_processing_status(self, photo_id: str, stage: str) -> Optional[ProcessingStatus]:
        """Get processing status for a specific stage."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """SELECT * FROM processing_status 
                   WHERE photo_id = ? AND stage = ?""",
                (photo_id, stage)
            ).fetchone()
            
            if row:
                return ProcessingStatus(**dict(row))
            return None
    
    def has_been_processed(self, photo_id: str, stage: str) -> bool:
        """Check if a photo has been processed for a specific stage."""
        status = self.get_processing_status(photo_id, stage)
        return status is not None and status.status == 'completed'
```

## Implementation Steps

1. **Create schema.sql file**
   - Place in project root
   - Include all tables and indexes

2. **Implement database connection**
   - Create connection.py with context managers
   - Handle transactions properly
   - Ensure thread safety

3. **Create data models**
   - Use dataclasses for clean structure
   - Include factory methods
   - Add serialization methods

4. **Build repository layer**
   - Implement CRUD operations
   - Add processing status tracking
   - Include helper methods

5. **Write unit tests**
   - Test connection management
   - Test all CRUD operations
   - Test transaction rollback

## Testing Checklist

- [ ] Database creates successfully
- [ ] Schema applies without errors
- [ ] Can insert and retrieve photos
- [ ] Can update processing status
- [ ] Transactions roll back on error
- [ ] Foreign key constraints work
- [ ] Indexes improve query performance
- [ ] JSON metadata storage works

## Notes

- Use WAL mode for better concurrency if needed
- Consider adding database migrations for future updates
- Implement proper connection pooling for multi-threaded use
- Add vacuum/analyze commands for maintenance