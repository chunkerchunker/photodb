from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import logging
from psycopg2.extras import RealDictCursor

from .connection import ConnectionPool
from .models import Photo, Metadata, ProcessingStatus, LLMAnalysis, BatchJob, Person, Face

logger = logging.getLogger(__name__)


class PhotoRepository:
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool

    def create_photo(self, photo: Photo) -> None:
        """Insert a new photo record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO photos (id, filename, normalized_path, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (
                        photo.id,
                        photo.filename,
                        photo.normalized_path,
                        photo.created_at,
                        photo.updated_at,
                    ),
                )

    def get_photo_by_filename(self, filename: str) -> Optional[Photo]:
        """Get photo by filename."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM photos WHERE filename = %s", (filename,))
                row = cursor.fetchone()

                if row:
                    return Photo(**dict(row))  # type: ignore[arg-type]
                return None

    def get_photo_by_id(self, photo_id: str) -> Optional[Photo]:
        """Get photo by ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM photos WHERE id = %s", (photo_id,))
                row = cursor.fetchone()

                if row:
                    return Photo(**dict(row))  # type: ignore[arg-type]
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
                    (photo.normalized_path, photo.updated_at, photo.id),
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
                           created_at = EXCLUDED.created_at,
                           captured_at = EXCLUDED.captured_at,
                           latitude = EXCLUDED.latitude,
                           longitude = EXCLUDED.longitude,
                           extra = EXCLUDED.extra""",
                    (
                        metadata.photo_id,
                        metadata.captured_at,
                        metadata.latitude,
                        metadata.longitude,
                        json.dumps(metadata.extra),
                        metadata.created_at,
                    ),
                )

    def get_metadata(self, photo_id: str) -> Optional[Metadata]:
        """Get metadata for a photo."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM metadata WHERE photo_id = %s", (photo_id,))
                row = cursor.fetchone()

                if row:
                    # PostgreSQL returns JSONB as dict already
                    if row["extra"] is None:
                        row["extra"] = {}
                    return Metadata(**dict(row))  # type: ignore[arg-type]
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
                    (
                        status.photo_id,
                        status.stage,
                        status.status,
                        status.processed_at,
                        status.error_message,
                    ),
                )

    def get_processing_status(self, photo_id: str, stage: str) -> Optional[ProcessingStatus]:
        """Get processing status for a specific stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT * FROM processing_status 
                       WHERE photo_id = %s AND stage = %s""",
                    (photo_id, stage),
                )
                row = cursor.fetchone()

                if row:
                    return ProcessingStatus(**dict(row))  # type: ignore[arg-type]
                return None

    def has_been_processed(self, photo_id: str, stage: str) -> bool:
        """Check if a photo has been processed for a specific stage."""
        status = self.get_processing_status(photo_id, stage)
        return status is not None and status.status == "completed"

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
                    (stage, limit),
                )
                rows = cursor.fetchall()

                return [Photo(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def get_failed_photos(self, stage: str) -> List[Dict[str, Any]]:
        """Get photos that failed processing for a specific stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT p.*, ps.error_message, ps.processed_at 
                       FROM photos p
                       JOIN processing_status ps ON p.id = ps.photo_id
                       WHERE ps.stage = %s AND ps.status = 'failed'""",
                    (stage,),
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
                    (stage,),
                )
                rows = cursor.fetchall()

                return {row["status"]: row["count"] for row in rows}

    # LLM Analysis methods

    def create_llm_analysis(self, analysis: LLMAnalysis) -> None:
        """Insert or update LLM analysis record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO llm_analysis 
                       (id, photo_id, model_name, model_version, processed_at, batch_id,
                        analysis, description, objects, people_count, location_description,
                        emotional_tone, confidence_score, processing_duration_ms,
                        input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
                        error_message)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (photo_id) 
                       DO UPDATE SET 
                           model_name = EXCLUDED.model_name,
                           model_version = EXCLUDED.model_version,
                           processed_at = EXCLUDED.processed_at,
                           batch_id = EXCLUDED.batch_id,
                           analysis = EXCLUDED.analysis,
                           description = EXCLUDED.description,
                           objects = EXCLUDED.objects,
                           people_count = EXCLUDED.people_count,
                           location_description = EXCLUDED.location_description,
                           emotional_tone = EXCLUDED.emotional_tone,
                           confidence_score = EXCLUDED.confidence_score,
                           processing_duration_ms = EXCLUDED.processing_duration_ms,
                           input_tokens = EXCLUDED.input_tokens,
                           output_tokens = EXCLUDED.output_tokens,
                           cache_creation_tokens = EXCLUDED.cache_creation_tokens,
                           cache_read_tokens = EXCLUDED.cache_read_tokens,
                           error_message = EXCLUDED.error_message""",
                    (
                        analysis.id,
                        analysis.photo_id,
                        analysis.model_name,
                        analysis.model_version,
                        analysis.processed_at,
                        analysis.batch_id,
                        json.dumps(analysis.analysis),
                        analysis.description,
                        analysis.objects,
                        analysis.people_count,
                        analysis.location_description,
                        analysis.emotional_tone,
                        analysis.confidence_score,
                        analysis.processing_duration_ms,
                        analysis.input_tokens,
                        analysis.output_tokens,
                        analysis.cache_creation_tokens,
                        analysis.cache_read_tokens,
                        analysis.error_message,
                    ),
                )

    def get_llm_analysis(self, photo_id: str) -> Optional[LLMAnalysis]:
        """Get LLM analysis for a photo."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM llm_analysis WHERE photo_id = %s", (photo_id,))
                row = cursor.fetchone()

                if row:
                    # Convert objects array back to Python list if needed
                    if row.get("objects") is None:
                        row["objects"] = []
                    return LLMAnalysis(**dict(row))  # type: ignore[arg-type]
                return None

    def has_llm_analysis(self, photo_id: str) -> bool:
        """Check if a photo has LLM analysis."""
        analysis = self.get_llm_analysis(photo_id)
        return analysis is not None and analysis.error_message is None

    def get_photos_for_llm_analysis(self, limit: int = 100) -> List[Photo]:
        """Get photos that need LLM analysis (have normalized image but no analysis)."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT p.* FROM photos p
                       WHERE p.normalized_path != ''
                       AND NOT EXISTS (
                           SELECT 1 FROM llm_analysis la 
                           WHERE la.photo_id = p.id AND la.error_message IS NULL
                       )
                       LIMIT %s""",
                    (limit,),
                )
                rows = cursor.fetchall()

                return [Photo(**dict(row)) for row in rows]  # type: ignore[arg-type]

    # Batch Job methods

    def create_batch_job(self, batch_job: BatchJob) -> None:
        """Create a new batch job record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO batch_jobs 
                       (id, provider_batch_id, status, submitted_at, completed_at,
                        photo_count, processed_count, failed_count, photo_ids,
                        total_input_tokens, total_output_tokens, total_cache_creation_tokens, total_cache_read_tokens,
                        estimated_cost_cents, actual_cost_cents, model_name, batch_discount_applied,
                        error_message)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        batch_job.id,
                        batch_job.provider_batch_id,
                        batch_job.status,
                        batch_job.submitted_at,
                        batch_job.completed_at,
                        batch_job.photo_count,
                        batch_job.processed_count,
                        batch_job.failed_count,
                        batch_job.photo_ids,
                        batch_job.total_input_tokens,
                        batch_job.total_output_tokens,
                        batch_job.total_cache_creation_tokens,
                        batch_job.total_cache_read_tokens,
                        batch_job.estimated_cost_cents,
                        batch_job.actual_cost_cents,
                        batch_job.model_name,
                        batch_job.batch_discount_applied,
                        batch_job.error_message,
                    ),
                )

    def get_batch_job_by_provider_id(self, provider_batch_id: str) -> Optional[BatchJob]:
        """Get batch job by provider batch ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM batch_jobs WHERE provider_batch_id = %s", (provider_batch_id,)
                )
                row = cursor.fetchone()

                if row:
                    # PostgreSQL returns arrays as lists already
                    row_dict = dict(row)
                    if row_dict.get("photo_ids") is None:
                        row_dict["photo_ids"] = []
                    return BatchJob(**row_dict)  # type: ignore[arg-type]
                return None

    def update_batch_job(self, batch_job: BatchJob) -> None:
        """Update batch job status and counts."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE batch_jobs 
                       SET status = %s, completed_at = %s, processed_count = %s,
                           failed_count = %s, 
                           total_input_tokens = %s, total_output_tokens = %s,
                           total_cache_creation_tokens = %s, total_cache_read_tokens = %s,
                           estimated_cost_cents = %s, actual_cost_cents = %s,
                           error_message = %s
                       WHERE id = %s""",
                    (
                        batch_job.status,
                        batch_job.completed_at,
                        batch_job.processed_count,
                        batch_job.failed_count,
                        batch_job.total_input_tokens,
                        batch_job.total_output_tokens,
                        batch_job.total_cache_creation_tokens,
                        batch_job.total_cache_read_tokens,
                        batch_job.estimated_cost_cents,
                        batch_job.actual_cost_cents,
                        batch_job.error_message,
                        batch_job.id,
                    ),
                )

    def get_active_batch_jobs(self) -> List[BatchJob]:
        """Get all batch jobs that are still processing."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT * FROM batch_jobs 
                       WHERE status IN ('submitted', 'processing')
                       ORDER BY submitted_at"""
                )
                rows = cursor.fetchall()

                batch_jobs = []
                for row in rows:
                    row_dict = dict(row)
                    # PostgreSQL returns arrays as lists already
                    if row_dict.get("photo_ids") is None:
                        row_dict["photo_ids"] = []
                    batch_jobs.append(BatchJob(**row_dict))  # type: ignore[arg-type]
                return batch_jobs

    # Person methods

    def create_person(self, person: Person) -> None:
        """Create a new person record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO person (id, name, created_at, updated_at)
                       VALUES (%s, %s, %s, %s)""",
                    (
                        person.id,
                        person.name,
                        person.created_at,
                        person.updated_at,
                    ),
                )

    def get_person_by_id(self, person_id: str) -> Optional[Person]:
        """Get person by ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM person WHERE id = %s", (person_id,))
                row = cursor.fetchone()

                if row:
                    return Person(**dict(row))  # type: ignore[arg-type]
                return None

    def get_person_by_name(self, name: str) -> Optional[Person]:
        """Get person by name."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM person WHERE name = %s", (name,))
                row = cursor.fetchone()

                if row:
                    return Person(**dict(row))  # type: ignore[arg-type]
                return None

    def update_person(self, person: Person) -> None:
        """Update person record."""
        person.updated_at = datetime.now()
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person 
                       SET name = %s, updated_at = %s
                       WHERE id = %s""",
                    (person.name, person.updated_at, person.id),
                )

    def list_people(self) -> List[Person]:
        """List all people."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM person ORDER BY name")
                rows = cursor.fetchall()

                return [Person(**dict(row)) for row in rows]  # type: ignore[arg-type]

    # Face methods

    def create_face(self, face: Face) -> None:
        """Create a new face detection record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO face 
                       (id, photo_id, bbox_x, bbox_y, bbox_width, bbox_height, 
                        person_id, confidence)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        face.id,
                        face.photo_id,
                        face.bbox_x,
                        face.bbox_y,
                        face.bbox_width,
                        face.bbox_height,
                        face.person_id,
                        face.confidence,
                    ),
                )

    def get_face_by_id(self, face_id: str) -> Optional[Face]:
        """Get face by ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM face WHERE id = %s", (face_id,))
                row = cursor.fetchone()

                if row:
                    return Face(**dict(row))  # type: ignore[arg-type]
                return None

    def get_faces_for_photo(self, photo_id: str) -> List[Face]:
        """Get all faces detected in a photo."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT * FROM face 
                       WHERE photo_id = %s 
                       ORDER BY confidence DESC""",
                    (photo_id,),
                )
                rows = cursor.fetchall()

                return [Face(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def get_faces_for_person(self, person_id: str) -> List[Face]:
        """Get all faces identified as a specific person."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT * FROM face 
                       WHERE person_id = %s 
                       ORDER BY confidence DESC""",
                    (person_id,),
                )
                rows = cursor.fetchall()

                return [Face(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def update_face_person(self, face_id: str, person_id: Optional[str]) -> None:
        """Update the person association for a face."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE face 
                       SET person_id = %s
                       WHERE id = %s""",
                    (person_id, face_id),
                )

    def get_photos_with_person(self, person_id: str) -> List[Photo]:
        """Get all photos containing a specific person."""
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT DISTINCT p.* FROM photos p
                       JOIN face f ON p.id = f.photo_id
                       WHERE f.person_id = %s""",
                    (person_id,),
                )
                rows = cursor.fetchall()

                return [Photo(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def delete_faces_for_photo(self, photo_id: str) -> None:
        """Delete all face detections for a photo."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM face WHERE photo_id = %s", (photo_id,))
