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
        '''