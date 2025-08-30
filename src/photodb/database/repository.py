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
    
    def get_unprocessed_photos(self, stage: str, limit: int = 100) -> List[Photo]:
        """Get photos that haven't been processed for a specific stage."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT p.* FROM photos p
                   LEFT JOIN processing_status ps 
                   ON p.id = ps.photo_id AND ps.stage = ?
                   WHERE ps.status IS NULL OR ps.status != 'completed'
                   LIMIT ?""",
                (stage, limit)
            ).fetchall()
            
            return [Photo(**dict(row)) for row in rows]
    
    def get_failed_photos(self, stage: str) -> List[Dict[str, Any]]:
        """Get photos that failed processing for a specific stage."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT p.*, ps.error_message, ps.processed_at 
                   FROM photos p
                   JOIN processing_status ps ON p.id = ps.photo_id
                   WHERE ps.stage = ? AND ps.status = 'failed'""",
                (stage,)
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_photo_count_by_status(self, stage: str) -> Dict[str, int]:
        """Get count of photos by processing status for a stage."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT status, COUNT(*) as count 
                   FROM processing_status
                   WHERE stage = ?
                   GROUP BY status""",
                (stage,)
            ).fetchall()
            
            return {row['status']: row['count'] for row in rows}