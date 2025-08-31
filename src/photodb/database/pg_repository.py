from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import logging
from psycopg2.extras import RealDictCursor

from .pg_connection import PostgresConnectionPool
from .models import Photo, Metadata, ProcessingStatus

logger = logging.getLogger(__name__)


class PostgresPhotoRepository:
    def __init__(self, connection_pool: PostgresConnectionPool):
        self.pool = connection_pool
    
    def create_photo(self, photo: Photo) -> None:
        """Insert a new photo record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO photos (id, filename, normalized_path, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (photo.id, photo.filename, photo.normalized_path, 
                     photo.created_at, photo.updated_at)
                )
    
    def get_photo_by_filename(self, filename: str) -> Optional[Photo]:
        """Get photo by filename."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM photos WHERE filename = %s", (filename,)
                )
                row = cursor.fetchone()
                
                if row:
                    return Photo(**row)
                return None
    
    def get_photo_by_id(self, photo_id: str) -> Optional[Photo]:
        """Get photo by ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM photos WHERE id = %s", (photo_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return Photo(**row)
                return None
    
    def update_photo(self, photo: Photo) -> None:
        """Update existing photo record."""
        photo.updated_at = datetime.now()
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE photos 
                       SET normalized_path = %s, updated_at = %s
                       WHERE id = %s""",
                    (photo.normalized_path, photo.updated_at, photo.id)
                )
    
    def create_metadata(self, metadata: Metadata) -> None:
        """Insert or update metadata record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO metadata 
                       (photo_id, captured_at, latitude, longitude, extra, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (photo_id) 
                       DO UPDATE SET 
                           captured_at = EXCLUDED.captured_at,
                           latitude = EXCLUDED.latitude,
                           longitude = EXCLUDED.longitude,
                           extra = EXCLUDED.extra""",
                    (metadata.photo_id, metadata.captured_at, metadata.latitude,
                     metadata.longitude, json.dumps(metadata.extra), metadata.created_at)
                )
    
    def get_metadata(self, photo_id: str) -> Optional[Metadata]:
        """Get metadata for a photo."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM metadata WHERE photo_id = %s", (photo_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    # PostgreSQL returns JSONB as dict already
                    if row['extra'] is None:
                        row['extra'] = {}
                    return Metadata(**row)
                return None
    
    def update_processing_status(self, status: ProcessingStatus) -> None:
        """Update processing status for a photo stage."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO processing_status 
                       (photo_id, stage, status, processed_at, error_message)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (photo_id, stage) 
                       DO UPDATE SET 
                           status = EXCLUDED.status,
                           processed_at = EXCLUDED.processed_at,
                           error_message = EXCLUDED.error_message""",
                    (status.photo_id, status.stage, status.status,
                     status.processed_at, status.error_message)
                )
    
    def get_processing_status(self, photo_id: str, stage: str) -> Optional[ProcessingStatus]:
        """Get processing status for a specific stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT * FROM processing_status 
                       WHERE photo_id = %s AND stage = %s""",
                    (photo_id, stage)
                )
                row = cursor.fetchone()
                
                if row:
                    return ProcessingStatus(**row)
                return None
    
    def has_been_processed(self, photo_id: str, stage: str) -> bool:
        """Check if a photo has been processed for a specific stage."""
        status = self.get_processing_status(photo_id, stage)
        return status is not None and status.status == 'completed'
    
    def get_unprocessed_photos(self, stage: str, limit: int = 100) -> List[Photo]:
        """Get photos that haven't been processed for a specific stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT p.* FROM photos p
                       LEFT JOIN processing_status ps 
                       ON p.id = ps.photo_id AND ps.stage = %s
                       WHERE ps.status IS NULL OR ps.status != 'completed'
                       LIMIT %s""",
                    (stage, limit)
                )
                rows = cursor.fetchall()
                
                return [Photo(**row) for row in rows]
    
    def get_failed_photos(self, stage: str) -> List[Dict[str, Any]]:
        """Get photos that failed processing for a specific stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT p.*, ps.error_message, ps.processed_at 
                       FROM photos p
                       JOIN processing_status ps ON p.id = ps.photo_id
                       WHERE ps.stage = %s AND ps.status = 'failed'""",
                    (stage,)
                )
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
    
    def get_photo_count_by_status(self, stage: str) -> Dict[str, int]:
        """Get count of photos by processing status for a stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT status, COUNT(*) as count 
                       FROM processing_status
                       WHERE stage = %s
                       GROUP BY status""",
                    (stage,)
                )
                rows = cursor.fetchall()
                
                return {row['status']: row['count'] for row in rows}