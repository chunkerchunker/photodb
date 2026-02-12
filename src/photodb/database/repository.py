from typing import Optional, List, Dict, Any
from datetime import date, datetime
import json
import logging
import os
from psycopg.rows import dict_row

from .connection import ConnectionPool
from .models import (
    AnalysisOutput,
    BatchJob,
    Cluster,
    DetectionTag,
    Face,
    FamilyMember,
    LLMAnalysis,
    Metadata,
    Person,
    PersonBirthOrder,
    PersonDetection,
    PersonNotRelated,
    PersonParent,
    PersonPartnership,
    Photo,
    PhotoTag,
    ProcessingStatus,
    PromptCategory,
    PromptEmbedding,
    SceneAnalysis,
    Sibling,
)

logger = logging.getLogger(__name__)

# Minimum face size for clustering (in pixels)
# Faces smaller than this (in either dimension) will be excluded from clustering
MIN_FACE_SIZE_PX = int(os.environ.get("MIN_FACE_SIZE_PX", 50))  # Default 50 pixels

# Minimum face detection confidence for clustering
# Faces with lower confidence will be excluded from clustering
MIN_FACE_CONFIDENCE = float(os.environ.get("MIN_FACE_CONFIDENCE", 0.9))  # Default 90%


class PhotoRepository:
    def __init__(self, connection_pool: ConnectionPool, collection_id: Optional[int] = None):
        self.pool = connection_pool
        env_collection_id = os.environ.get("COLLECTION_ID")
        self.collection_id = collection_id or int(env_collection_id) if env_collection_id else 1

    def _resolve_collection_id(self, collection_id: Optional[int]) -> int:
        return collection_id if collection_id is not None else self.collection_id

    def create_photo(self, photo: Photo) -> None:
        """Insert a new photo record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO photo (collection_id, orig_path, full_path, med_path, width, height,
                                         med_width, med_height, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (
                        photo.collection_id,
                        photo.orig_path,
                        photo.full_path,
                        photo.med_path,
                        photo.width,
                        photo.height,
                        photo.med_width,
                        photo.med_height,
                        photo.created_at,
                        photo.updated_at,
                    ),
                )
                photo.id = cursor.fetchone()[0]

    def get_photo_by_orig_path(
        self, orig_path: str, collection_id: Optional[int] = None
    ) -> Optional[Photo]:
        """Get photo by original path."""
        collection_id = self._resolve_collection_id(collection_id)
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM photo WHERE collection_id = %s AND orig_path = %s",
                    (collection_id, orig_path),
                )
                row = cursor.fetchone()

                if row:
                    return Photo(**dict(row))  # type: ignore[arg-type]
                return None

    def get_photo_by_id(
        self, photo_id: int, collection_id: Optional[int] = None
    ) -> Optional[Photo]:
        """Get photo by ID."""
        collection_id = self._resolve_collection_id(collection_id)
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM photo WHERE id = %s AND collection_id = %s",
                    (photo_id, collection_id),
                )
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
                    """UPDATE photo
                       SET full_path = %s, med_path = %s, width = %s, height = %s,
                           med_width = %s, med_height = %s, updated_at = %s
                       WHERE id = %s""",
                    (
                        photo.full_path,
                        photo.med_path,
                        photo.width,
                        photo.height,
                        photo.med_width,
                        photo.med_height,
                        photo.updated_at,
                        photo.id,
                    ),
                )

    def create_metadata(self, metadata: Metadata) -> None:
        """Insert or update metadata record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO metadata 
                       (photo_id, collection_id, captured_at, latitude, longitude, extra, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (photo_id) 
                       DO UPDATE SET 
                           created_at = EXCLUDED.created_at,
                           collection_id = EXCLUDED.collection_id,
                           captured_at = EXCLUDED.captured_at,
                           latitude = EXCLUDED.latitude,
                           longitude = EXCLUDED.longitude,
                           extra = EXCLUDED.extra""",
                    (
                        metadata.photo_id,
                        metadata.collection_id,
                        metadata.captured_at,
                        metadata.latitude,
                        metadata.longitude,
                        json.dumps(metadata.extra),
                        metadata.created_at,
                    ),
                )

    def get_metadata(self, photo_id: int) -> Optional[Metadata]:
        """Get metadata for a photo."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
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
                        status.status.value if hasattr(status.status, "value") else status.status,
                        status.processed_at,
                        status.error_message,
                    ),
                )

    def get_processing_status(self, photo_id: int, stage: str) -> Optional[ProcessingStatus]:
        """Get processing status for a specific stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT * FROM processing_status 
                       WHERE photo_id = %s AND stage = %s""",
                    (photo_id, stage),
                )
                row = cursor.fetchone()

                if row:
                    return ProcessingStatus(**dict(row))  # type: ignore[arg-type]
                return None

    def get_unprocessed_photos(self, stage: str, limit: int = 100) -> List[Photo]:
        """Get photos that haven't been processed for a specific stage.

        Uses NOT EXISTS pattern which is more efficient than LEFT JOIN anti-pattern
        as the database grows. The idx_processing_status_completed partial index
        makes this query fast by only indexing completed records.
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT p.* FROM photo p
                       WHERE NOT EXISTS (
                           SELECT 1 FROM processing_status ps
                           WHERE ps.photo_id = p.id
                             AND ps.stage = %s
                             AND ps.status = 'completed'
                       )
                       AND p.collection_id = %s
                       LIMIT %s""",
                    (stage, self.collection_id, limit),
                )
                rows = cursor.fetchall()

                return [Photo(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def get_failed_photos(self, stage: str) -> List[Dict[str, Any]]:
        """Get photos that failed processing for a specific stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT p.*, ps.error_message, ps.processed_at 
                       FROM photo p
                       JOIN processing_status ps ON p.id = ps.photo_id
                       WHERE ps.stage = %s AND ps.status = 'failed'
                         AND p.collection_id = %s""",
                    (stage, self.collection_id),
                )
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

    def get_photo_count_by_status(self, stage: str) -> Dict[str, int]:
        """Get count of photos by processing status for a stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
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
                       (photo_id, model_name, model_version, processed_at, batch_id,
                        analysis, description, objects, people_count, location_description,
                        emotional_tone, confidence_score, processing_duration_ms,
                        input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
                        error_message)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                           error_message = EXCLUDED.error_message
                       RETURNING id""",
                    (
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
                if analysis.id is None:
                    result = cursor.fetchone()
                    if result:
                        analysis.id = result[0]

    def get_llm_analysis(self, photo_id: int) -> Optional[LLMAnalysis]:
        """Get LLM analysis for a photo."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT * FROM llm_analysis WHERE photo_id = %s", (photo_id,))
                row = cursor.fetchone()

                if row:
                    # Convert objects array back to Python list if needed
                    if row.get("objects") is None:
                        row["objects"] = []
                    return LLMAnalysis(**dict(row))  # type: ignore[arg-type]
                return None

    def has_llm_analysis(self, photo_id: int) -> bool:
        """Check if a photo has LLM analysis."""
        analysis = self.get_llm_analysis(photo_id)
        return analysis is not None and analysis.error_message is None

    def get_photos_for_llm_analysis(self, limit: int = 100) -> List[Photo]:
        """Get photos that need LLM analysis (have normalized image but no analysis)."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT p.* FROM photo p
                       WHERE p.med_path IS NOT NULL
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
                    """INSERT INTO batch_job 
                       (provider_batch_id, status, submitted_at, completed_at,
                        photo_count, processed_count, failed_count, photo_ids,
                        total_input_tokens, total_output_tokens, total_cache_creation_tokens, total_cache_read_tokens,
                        estimated_cost_cents, actual_cost_cents, model_name, batch_discount_applied,
                        error_message)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (
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
                batch_job.id = cursor.fetchone()[0]

    def get_batch_job_by_provider_id(self, provider_batch_id: str) -> Optional[BatchJob]:
        """Get batch job by provider batch ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM batch_job WHERE provider_batch_id = %s", (provider_batch_id,)
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
                    """UPDATE batch_job 
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
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT * FROM batch_job 
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
                    """INSERT INTO person (collection_id, first_name, last_name, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                    (
                        person.collection_id,
                        person.first_name,
                        person.last_name,
                        person.created_at,
                        person.updated_at,
                    ),
                )
                person.id = cursor.fetchone()[0]

    def get_person_by_id(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM person WHERE id = %s AND collection_id = %s",
                    (person_id, self.collection_id),
                )
                row = cursor.fetchone()

                if row:
                    return Person(**dict(row))  # type: ignore[arg-type]
                return None

    def get_person_by_name(
        self, first_name: str, last_name: Optional[str] = None
    ) -> Optional[Person]:
        """Get person by name."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                if last_name:
                    cursor.execute(
                        """SELECT * FROM person
                           WHERE collection_id = %s AND first_name = %s AND last_name = %s""",
                        (self.collection_id, first_name, last_name),
                    )
                else:
                    cursor.execute(
                        """SELECT * FROM person
                           WHERE collection_id = %s AND first_name = %s AND last_name IS NULL""",
                        (self.collection_id, first_name),
                    )
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
                       SET first_name = %s, last_name = %s, updated_at = %s
                       WHERE id = %s""",
                    (person.first_name, person.last_name, person.updated_at, person.id),
                )

    def list_people(self) -> List[Person]:
        """List all people."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT * FROM person
                       WHERE collection_id = %s
                       ORDER BY first_name, last_name""",
                    (self.collection_id,),
                )
                rows = cursor.fetchall()

                return [Person(**dict(row)) for row in rows]  # type: ignore[arg-type]

    # Face methods

    def create_face(self, face: Face) -> None:
        """Create a new face detection record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO face 
                       (photo_id, bbox_x, bbox_y, bbox_width, bbox_height, 
                        person_id, confidence)
                       VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (
                        face.photo_id,
                        face.bbox_x,
                        face.bbox_y,
                        face.bbox_width,
                        face.bbox_height,
                        face.person_id,
                        face.confidence,
                    ),
                )
                face.id = cursor.fetchone()[0]

    def get_face_by_id(self, face_id: int) -> Optional[Face]:
        """Get face by ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT * FROM face WHERE id = %s", (face_id,))
                row = cursor.fetchone()

                if row:
                    return Face(**dict(row))  # type: ignore[arg-type]
                return None

    def get_faces_for_photo(self, photo_id: int) -> List[Face]:
        """Get all faces detected in a photo."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT * FROM face 
                       WHERE photo_id = %s 
                       ORDER BY confidence DESC""",
                    (photo_id,),
                )
                rows = cursor.fetchall()

                return [Face(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def get_faces_for_person(self, person_id: int) -> List[Face]:
        """Get all faces identified as a specific person."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT * FROM face 
                       WHERE person_id = %s 
                       ORDER BY confidence DESC""",
                    (person_id,),
                )
                rows = cursor.fetchall()

                return [Face(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def update_face_person(self, face_id: int, person_id: Optional[int]) -> None:
        """Update the person association for a face."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE face 
                       SET person_id = %s
                       WHERE id = %s""",
                    (person_id, face_id),
                )

    def get_photos_with_person(self, person_id: int) -> List[Photo]:
        """Get all photos containing a specific person."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT DISTINCT p.* FROM photo p
                       JOIN face f ON p.id = f.photo_id
                       WHERE f.person_id = %s""",
                    (person_id,),
                )
                rows = cursor.fetchall()

                return [Photo(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def get_photos_by_directory(self, directory: str) -> List[Photo]:
        """Get all photos whose filename starts with the given directory path."""
        # Ensure directory path ends with a separator to avoid partial matches
        # e.g., /foo/bar shouldn't match /foo/barbaz.jpg, only /foo/bar/baz.jpg
        if not directory.endswith(os.sep):
            directory += os.sep

        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM photo WHERE orig_path LIKE %s AND collection_id = %s",
                    (f"{directory}%", self.collection_id),
                )
                rows = cursor.fetchall()
                return [Photo(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def delete_faces_for_photo(self, photo_id: int) -> None:
        """Delete all face detections for a photo, updating cluster counts."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Decrement face_count for all affected clusters
                cursor.execute(
                    """UPDATE cluster
                       SET face_count = GREATEST(0, face_count - subq.cnt),
                           updated_at = NOW()
                       FROM (
                           SELECT cluster_id, COUNT(*) as cnt
                           FROM face
                           WHERE photo_id = %s AND cluster_id IS NOT NULL
                           GROUP BY cluster_id
                       ) subq
                       WHERE cluster.id = subq.cluster_id""",
                    (photo_id,),
                )

                # Delete the faces
                cursor.execute("DELETE FROM face WHERE photo_id = %s", (photo_id,))

    # Face embedding methods

    def save_face_embedding(self, face_id: int, embedding: List[float]) -> None:
        """Save face embedding using pgvector."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO face_embedding (face_id, embedding)
                       VALUES (%s, %s)
                       ON CONFLICT (face_id) 
                       DO UPDATE SET embedding = EXCLUDED.embedding""",
                    (face_id, embedding),
                )

    def get_face_embedding(self, face_id: int) -> Optional[List[float]]:
        """Get face embedding by face ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT embedding FROM face_embedding WHERE face_id = %s", (face_id,)
                )
                row = cursor.fetchone()
                return [float(x) for x in row[0]] if row else None

    def find_similar_faces(
        self, query_embedding: List[float], threshold: float = 0.6, limit: int = 10
    ) -> List[tuple]:
        """Find similar faces using pgvector cosine similarity."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT f.id, f.photo_id, f.confidence, 1 - (fe.embedding <=> %s) as similarity
                       FROM face f
                       JOIN face_embedding fe ON f.id = fe.face_id
                       WHERE 1 - (fe.embedding <=> %s) >= %s
                       ORDER BY similarity DESC
                       LIMIT %s""",
                    (query_embedding, query_embedding, threshold, limit),
                )
                return cursor.fetchall()

    # Clustering methods

    def find_nearest_clusters(self, embedding, limit: int = 10) -> List[tuple]:
        """Find nearest clusters using cosine distance."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT id, centroid <=> %s AS distance
                       FROM cluster
                       WHERE centroid IS NOT NULL
                         AND collection_id = %s
                       ORDER BY centroid <=> %s
                       LIMIT %s""",
                    (embedding, self.collection_id, embedding, limit),
                )
                return cursor.fetchall()

    def create_cluster(
        self,
        centroid,
        representative_face_id: int,
        medoid_face_id: int,
        face_count: int = 1,
    ) -> int:
        """Create a new cluster and return its ID."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO cluster
                       (collection_id, face_count, face_count_at_last_medoid, representative_face_id,
                        centroid, medoid_face_id, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                       RETURNING id""",
                    (
                        self.collection_id,
                        face_count,
                        face_count,
                        representative_face_id,
                        centroid,
                        medoid_face_id,
                    ),
                )
                return cursor.fetchone()[0]

    def update_face_cluster(
        self,
        face_id: int,
        cluster_id: int,
        cluster_confidence: float,
        cluster_status: str,
    ) -> bool:
        """
        Update face cluster assignment (does NOT update cluster counts).

        Only assigns if face is currently unassigned (cluster_id IS NULL).
        This prevents race conditions in parallel processing.

        Returns:
            True if face was assigned, False if already assigned to another cluster.
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Use SELECT FOR UPDATE to lock the row and check state atomically
                cursor.execute(
                    "SELECT cluster_id FROM face WHERE id = %s FOR UPDATE",
                    (face_id,),
                )
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"update_face_cluster: face {face_id} not found")
                    return False

                current_cluster_id = row[0]
                if current_cluster_id is not None:
                    logger.debug(
                        f"update_face_cluster: face {face_id} already in cluster {current_cluster_id}"
                    )
                    return False

                cursor.execute(
                    """UPDATE face
                       SET cluster_id = %s,
                           cluster_confidence = %s,
                           cluster_status = %s
                       WHERE id = %s""",
                    (cluster_id, cluster_confidence, cluster_status, face_id),
                )
                return True

    def move_face_to_cluster(
        self,
        face_id: int,
        new_cluster_id: int,
        cluster_confidence: float,
        cluster_status: str = "manual",
        delete_empty_clusters: bool = True,
    ) -> Optional[int]:
        """
        Move a face to a different cluster, updating counts atomically.

        Args:
            face_id: ID of the face to move
            new_cluster_id: ID of the destination cluster
            cluster_confidence: Confidence score for the assignment
            cluster_status: Status of the assignment (default: 'manual')
            delete_empty_clusters: If True, delete the old cluster if it becomes empty

        Returns:
            ID of deleted cluster if one was removed, None otherwise
        """
        deleted_cluster_id = None

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Get current cluster assignment
                cursor.execute(
                    "SELECT cluster_id FROM face WHERE id = %s FOR UPDATE",
                    (face_id,),
                )
                row = cursor.fetchone()
                if not row:
                    logger.error(f"Face {face_id} not found")
                    return None

                old_cluster_id = row[0]

                # Update the face
                cursor.execute(
                    """UPDATE face
                       SET cluster_id = %s,
                           cluster_confidence = %s,
                           cluster_status = %s,
                           unassigned_since = NULL
                       WHERE id = %s""",
                    (new_cluster_id, cluster_confidence, cluster_status, face_id),
                )

                # Decrement old cluster count (if face was in a cluster)
                if old_cluster_id is not None and old_cluster_id != new_cluster_id:
                    cursor.execute(
                        """UPDATE cluster
                           SET face_count = GREATEST(0, face_count - 1),
                               updated_at = NOW()
                           WHERE id = %s""",
                        (old_cluster_id,),
                    )

                    # Check if old cluster is now empty
                    if delete_empty_clusters:
                        cursor.execute(
                            "SELECT face_count FROM cluster WHERE id = %s",
                            (old_cluster_id,),
                        )
                        count_row = cursor.fetchone()
                        if count_row and count_row[0] == 0:
                            cursor.execute(
                                "DELETE FROM cluster WHERE id = %s",
                                (old_cluster_id,),
                            )
                            deleted_cluster_id = old_cluster_id
                            logger.info(f"Deleted empty cluster {old_cluster_id}")

                # Increment new cluster count
                if new_cluster_id is not None:
                    cursor.execute(
                        """UPDATE cluster
                           SET face_count = face_count + 1,
                               updated_at = NOW()
                           WHERE id = %s""",
                        (new_cluster_id,),
                    )

        return deleted_cluster_id

    def remove_face_from_cluster(
        self,
        face_id: int,
        delete_empty_cluster: bool = True,
    ) -> Optional[int]:
        """
        Remove a face from its current cluster.

        Args:
            face_id: ID of the face to remove
            delete_empty_cluster: If True, delete the cluster if it becomes empty

        Returns:
            ID of deleted cluster if one was removed, None otherwise
        """
        deleted_cluster_id = None

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Get current cluster
                cursor.execute(
                    "SELECT cluster_id FROM face WHERE id = %s FOR UPDATE",
                    (face_id,),
                )
                row = cursor.fetchone()
                if not row or row[0] is None:
                    return None

                old_cluster_id = row[0]

                # Clear face cluster assignment
                cursor.execute(
                    """UPDATE face
                       SET cluster_id = NULL,
                           cluster_status = NULL,
                           cluster_confidence = 0
                       WHERE id = %s""",
                    (face_id,),
                )

                # Decrement cluster count
                cursor.execute(
                    """UPDATE cluster
                       SET face_count = GREATEST(0, face_count - 1),
                           updated_at = NOW()
                       WHERE id = %s""",
                    (old_cluster_id,),
                )

                # Check if cluster is now empty
                if delete_empty_cluster:
                    cursor.execute(
                        "SELECT face_count FROM cluster WHERE id = %s",
                        (old_cluster_id,),
                    )
                    count_row = cursor.fetchone()
                    if count_row and count_row[0] == 0:
                        cursor.execute(
                            "DELETE FROM cluster WHERE id = %s",
                            (old_cluster_id,),
                        )
                        deleted_cluster_id = old_cluster_id
                        logger.info(f"Deleted empty cluster {old_cluster_id}")

        return deleted_cluster_id

    def set_cluster_representative(self, cluster_id: int, face_id: int) -> bool:
        """
        Set the representative face for a cluster (user-selected display photo).

        This is separate from the medoid, which is computed automatically.
        The representative face is preserved during medoid recomputation.

        Args:
            cluster_id: ID of the cluster to update
            face_id: ID of the face to set as representative

        Returns:
            True if update succeeded, False if cluster or face not found
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Verify face belongs to this cluster
                cursor.execute(
                    "SELECT cluster_id FROM face WHERE id = %s",
                    (face_id,),
                )
                row = cursor.fetchone()
                if not row or row[0] != cluster_id:
                    logger.warning(f"Face {face_id} does not belong to cluster {cluster_id}")
                    return False

                # Update the representative face
                cursor.execute(
                    """UPDATE cluster
                       SET representative_face_id = %s,
                           updated_at = NOW()
                       WHERE id = %s""",
                    (face_id, cluster_id),
                )
                return cursor.rowcount > 0

    def update_face_cluster_status(self, face_id: int, cluster_status: str) -> None:
        """Update only the cluster status for a face."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE face 
                       SET cluster_status = %s
                       WHERE id = %s""",
                    (cluster_status, face_id),
                )

    def create_face_match_candidates(self, candidates: List[tuple]) -> None:
        """Create multiple face match candidate records."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO face_match_candidate 
                       (face_id, cluster_id, collection_id, similarity, status, created_at)
                       VALUES (%s, %s, %s, %s, 'pending', NOW())""",
                    [
                        (face_id, cluster_id, self.collection_id, similarity)
                        for face_id, cluster_id, similarity in candidates
                    ],
                )

    def get_cluster_by_id(self, cluster_id: int) -> Optional[Cluster]:
        """Get cluster by ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT * FROM cluster WHERE id = %s", (cluster_id,))
                row = cursor.fetchone()

                if row:
                    return Cluster(**dict(row))  # type: ignore[arg-type]
                return None

    def update_cluster_centroid(self, cluster_id: int, centroid, face_count: int) -> None:
        """Update cluster centroid and face count."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE cluster
                       SET centroid = %s, face_count = %s, updated_at = NOW()
                       WHERE id = %s""",
                    (centroid, face_count, cluster_id),
                )

    def update_cluster_face_count(self, cluster_id: int, face_count: int) -> None:
        """Update cluster face count."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE cluster
                       SET face_count = %s,
                           face_count_at_last_medoid = %s,
                           updated_at = NOW()
                       WHERE id = %s""",
                    (face_count, face_count, cluster_id),
                )

    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete a cluster by ID. Returns True if deleted."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM cluster WHERE id = %s", (cluster_id,))
                return cursor.rowcount > 0

    def get_connection(self):
        """Get a database connection for transaction management."""
        return self.pool.get_connection()

    # Constrained clustering methods

    def find_nearest_faces(self, embedding, limit: int = 5) -> List[Dict[str, Any]]:
        """Find K nearest clustered faces using pgvector cosine distance."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT f.id, f.cluster_id, fe.embedding <=> %s AS distance
                       FROM face f
                       JOIN face_embedding fe ON f.id = fe.face_id
                       WHERE f.cluster_id IS NOT NULL
                       ORDER BY fe.embedding <=> %s
                       LIMIT %s""",
                    (embedding, embedding, limit),
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_cannot_linked_faces(self, face_id: int) -> List[Dict[str, Any]]:
        """Get faces that cannot be in same cluster as this face."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT f.id, f.cluster_id
                       FROM face f
                       JOIN (
                           SELECT face_id_2 AS linked_id FROM cannot_link WHERE face_id_1 = %s
                           UNION
                           SELECT face_id_1 AS linked_id FROM cannot_link WHERE face_id_2 = %s
                       ) cl ON f.id = cl.linked_id""",
                    (face_id, face_id),
                )
                return [dict(row) for row in cursor.fetchall()]

    def add_cannot_link(
        self, face_id_1: int, face_id_2: int, created_by: str = "human"
    ) -> Optional[int]:
        """
        Add cannot-link constraint between two faces.

        If faces are in the same cluster, marks cluster for split review.
        Returns the constraint ID or None if already exists.
        """
        # Canonical ordering
        if face_id_1 > face_id_2:
            face_id_1, face_id_2 = face_id_2, face_id_1

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO cannot_link (face_id_1, face_id_2, collection_id, created_by)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (face_id_1, face_id_2) DO NOTHING
                       RETURNING id""",
                    (face_id_1, face_id_2, self.collection_id, created_by),
                )
                result = cursor.fetchone()

                if result:
                    # Check if faces are in same cluster - mark for review
                    cursor.execute(
                        """SELECT f1.cluster_id
                           FROM face f1, face f2
                           WHERE f1.id = %s AND f2.id = %s
                             AND f1.cluster_id = f2.cluster_id
                             AND f1.cluster_id IS NOT NULL""",
                        (face_id_1, face_id_2),
                    )
                    same_cluster = cursor.fetchone()
                    if same_cluster:
                        # Mark cluster as needing split
                        cursor.execute(
                            """UPDATE cluster
                               SET verified = false, updated_at = NOW()
                               WHERE id = %s""",
                            (same_cluster[0],),
                        )
                        logger.warning(
                            f"Cluster {same_cluster[0]} needs split due to cannot-link constraint"
                        )

                return result[0] if result else None

    def add_cluster_cannot_link(self, cluster_id_1: int, cluster_id_2: int) -> Optional[int]:
        """Add cannot-link constraint between two clusters to prevent merging."""
        # Canonical ordering
        if cluster_id_1 > cluster_id_2:
            cluster_id_1, cluster_id_2 = cluster_id_2, cluster_id_1

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO cluster_cannot_link (cluster_id_1, cluster_id_2, collection_id)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (cluster_id_1, cluster_id_2) DO NOTHING
                       RETURNING id""",
                    (cluster_id_1, cluster_id_2, self.collection_id),
                )
                result = cursor.fetchone()
                return result[0] if result else None

    def has_cluster_cannot_link(self, cluster_id_1: int, cluster_id_2: int) -> bool:
        """Check if two clusters have a cannot-link constraint."""
        # Canonical ordering
        if cluster_id_1 > cluster_id_2:
            cluster_id_1, cluster_id_2 = cluster_id_2, cluster_id_1

        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT 1 FROM cluster_cannot_link
                       WHERE cluster_id_1 = %s AND cluster_id_2 = %s""",
                    (cluster_id_1, cluster_id_2),
                )
                return cursor.fetchone() is not None

    def update_face_unassigned(self, face_id: int) -> None:
        """Mark a face as unassigned (added to outlier pool)."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Get current cluster assignment
                cursor.execute(
                    "SELECT cluster_id FROM face WHERE id = %s FOR UPDATE",
                    (face_id,),
                )
                row = cursor.fetchone()
                old_cluster_id = row[0] if row else None

                # Decrement old cluster's face_count if face was in a cluster
                if old_cluster_id is not None:
                    cursor.execute(
                        """UPDATE cluster
                           SET face_count = GREATEST(0, face_count - 1),
                               updated_at = NOW()
                           WHERE id = %s""",
                        (old_cluster_id,),
                    )

                # Mark face as unassigned and clear cluster_id
                cursor.execute(
                    """UPDATE face
                       SET cluster_id = NULL,
                           cluster_status = 'unassigned',
                           cluster_confidence = 0,
                           unassigned_since = NOW()
                       WHERE id = %s""",
                    (face_id,),
                )

    def clear_face_unassigned(self, face_id: int) -> None:
        """Clear unassigned status when face is assigned to cluster."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE face
                       SET unassigned_since = NULL
                       WHERE id = %s""",
                    (face_id,),
                )

    def get_faces_in_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get all faces with embeddings in a cluster."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT f.id, fe.embedding
                       FROM face f
                       JOIN face_embedding fe ON f.id = fe.face_id
                       WHERE f.cluster_id = %s""",
                    (cluster_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

    def verify_cluster(self, cluster_id: int, verified_by: str = "human") -> None:
        """Mark a cluster as verified (human-confirmed identity)."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE cluster
                       SET verified = true,
                           verified_at = NOW(),
                           verified_by = %s,
                           updated_at = NOW()
                       WHERE id = %s""",
                    (verified_by, cluster_id),
                )

    def unverify_cluster(self, cluster_id: int) -> None:
        """Remove verified status from a cluster."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE cluster
                       SET verified = false,
                           verified_at = NULL,
                           verified_by = NULL,
                           updated_at = NOW()
                       WHERE id = %s""",
                    (cluster_id,),
                )

    def get_constraint_stats(self) -> Dict[str, int]:
        """Get statistics about constraints in the system."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM cannot_link")
                cannot_link_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cluster_cannot_link")
                cluster_cannot_link_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cluster WHERE verified = true")
                verified_clusters = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM face WHERE cluster_status = 'unassigned'")
                unassigned_faces = cursor.fetchone()[0]

                return {
                    "cannot_link_count": cannot_link_count,
                    "cluster_cannot_link_count": cluster_cannot_link_count,
                    "verified_clusters": verified_clusters,
                    "unassigned_faces": unassigned_faces,
                }

    # PersonDetection methods

    def create_person_detection(self, detection: PersonDetection) -> None:
        """Insert a new person detection record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO person_detection
                       (photo_id, collection_id, face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height,
                        face_confidence, face_path, body_bbox_x, body_bbox_y, body_bbox_width, body_bbox_height,
                        body_confidence, age_estimate, gender, gender_confidence, mivolo_output,
                        person_id, cluster_status, cluster_id, cluster_confidence,
                        detector_model, detector_version, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        detection.photo_id,
                        detection.collection_id,
                        detection.face_bbox_x,
                        detection.face_bbox_y,
                        detection.face_bbox_width,
                        detection.face_bbox_height,
                        detection.face_confidence,
                        detection.face_path,
                        detection.body_bbox_x,
                        detection.body_bbox_y,
                        detection.body_bbox_width,
                        detection.body_bbox_height,
                        detection.body_confidence,
                        detection.age_estimate,
                        detection.gender,
                        detection.gender_confidence,
                        json.dumps(detection.mivolo_output) if detection.mivolo_output else None,
                        detection.person_id,
                        detection.cluster_status,
                        detection.cluster_id,
                        detection.cluster_confidence,
                        detection.detector_model,
                        detection.detector_version,
                        detection.created_at,
                    ),
                )
                detection.id = cursor.fetchone()[0]

    def get_detections_for_photo(self, photo_id: int) -> List[PersonDetection]:
        """Get all person detections for a photo."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT * FROM person_detection
                       WHERE photo_id = %s
                       ORDER BY id""",
                    (photo_id,),
                )
                rows = cursor.fetchall()

                return [PersonDetection(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def delete_detections_for_photo(self, photo_id: int) -> int:
        """Delete all detections for a photo. Returns count deleted."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM person_detection WHERE photo_id = %s",
                    (photo_id,),
                )
                return cursor.rowcount

    def save_detection_embedding(self, detection_id: int, embedding: List[float]) -> None:
        """Save face embedding for a person detection."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO face_embedding (person_detection_id, embedding)
                       VALUES (%s, %s)
                       ON CONFLICT (person_detection_id)
                       DO UPDATE SET embedding = EXCLUDED.embedding""",
                    (detection_id, embedding),
                )

    def update_detection_age_gender(
        self,
        detection_id: int,
        age_estimate: float,
        gender: str,
        gender_confidence: float,
        mivolo_output: Dict[str, Any],
    ) -> None:
        """Update age/gender fields for a person detection."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person_detection
                       SET age_estimate = %s,
                           gender = %s,
                           gender_confidence = %s,
                           mivolo_output = %s
                       WHERE id = %s""",
                    (
                        age_estimate,
                        gender,
                        gender_confidence,
                        json.dumps(mivolo_output) if mivolo_output else None,
                        detection_id,
                    ),
                )

    def get_detections_for_person(self, person_id: int) -> List[PersonDetection]:
        """Get all detections for a person (via person_id field)."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT * FROM person_detection
                       WHERE person_id = %s
                       ORDER BY id""",
                    (person_id,),
                )
                rows = cursor.fetchall()

                return [PersonDetection(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def update_person_age_gender(
        self,
        person_id: int,
        estimated_birth_year: Optional[int],
        birth_year_stddev: Optional[float],
        gender: Optional[str],
        gender_confidence: Optional[float],
        sample_count: int,
    ) -> None:
        """Update aggregated age/gender statistics for a person."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person
                       SET estimated_birth_year = %s,
                           birth_year_stddev = %s,
                           gender = %s,
                           gender_confidence = %s,
                           age_gender_sample_count = %s,
                           age_gender_updated_at = NOW(),
                           updated_at = NOW()
                       WHERE id = %s""",
                    (
                        estimated_birth_year,
                        birth_year_stddev,
                        gender,
                        gender_confidence,
                        sample_count,
                        person_id,
                    ),
                )

    # PersonDetection clustering methods

    def get_unclustered_detections_for_photo(self, photo_id: int) -> List[Dict[str, Any]]:
        """Get all unclustered person detections for a photo with embeddings.

        Only returns detections with face bounding boxes that are:
        - Not already clustered
        - Large enough (MIN_FACE_SIZE_PX)
        - High enough confidence (MIN_FACE_CONFIDENCE)
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.*, fe.embedding
                       FROM person_detection pd
                       LEFT JOIN face_embedding fe ON pd.id = fe.person_detection_id
                       WHERE pd.photo_id = %s
                         AND pd.cluster_id IS NULL
                         AND pd.cluster_status IS NULL
                         AND pd.face_bbox_x IS NOT NULL
                         AND pd.face_confidence >= %s
                         AND pd.face_bbox_width >= %s
                         AND pd.face_bbox_height >= %s
                       ORDER BY pd.id""",
                    (photo_id, MIN_FACE_CONFIDENCE, MIN_FACE_SIZE_PX, MIN_FACE_SIZE_PX),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_all_detections_with_embeddings_for_photo(self, photo_id: int) -> List[Dict[str, Any]]:
        """Get all detections with embeddings for a photo (for force reprocessing)."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.*, fe.embedding
                       FROM person_detection pd
                       LEFT JOIN face_embedding fe ON pd.id = fe.person_detection_id
                       WHERE pd.photo_id = %s
                         AND fe.embedding IS NOT NULL
                       ORDER BY pd.id""",
                    (photo_id,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def find_nearest_detections(self, embedding, limit: int = 5) -> List[Dict[str, Any]]:
        """Find K nearest clustered detections using pgvector cosine distance."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.id, pd.cluster_id, fe.embedding <=> %s AS distance
                       FROM person_detection pd
                       JOIN face_embedding fe ON pd.id = fe.person_detection_id
                       WHERE pd.cluster_id IS NOT NULL
                         AND pd.collection_id = %s
                       ORDER BY fe.embedding <=> %s
                       LIMIT %s""",
                    (embedding, self.collection_id, embedding, limit),
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_cannot_linked_detections(self, detection_id: int) -> List[Dict[str, Any]]:
        """Get detections that cannot be in same cluster as this detection."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.id, pd.cluster_id
                       FROM person_detection pd
                       JOIN (
                           SELECT detection_id_2 AS linked_id
                           FROM cannot_link
                           WHERE detection_id_1 = %s AND collection_id = %s
                           UNION
                           SELECT detection_id_1 AS linked_id
                           FROM cannot_link
                           WHERE detection_id_2 = %s AND collection_id = %s
                        ) cl ON pd.id = cl.linked_id
                       WHERE pd.collection_id = %s""",
                    (
                        detection_id,
                        self.collection_id,
                        detection_id,
                        self.collection_id,
                        self.collection_id,
                    ),
                )
                return [dict(row) for row in cursor.fetchall()]

    def update_detection_cluster(
        self,
        detection_id: int,
        cluster_id: int,
        cluster_confidence: float,
        cluster_status: str,
    ) -> bool:
        """
        Update detection cluster assignment (does NOT update cluster counts).

        Only assigns if detection is currently unassigned (cluster_id IS NULL).
        This prevents race conditions in parallel processing.

        Returns:
            True if detection was assigned, False if already assigned to another cluster.
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Use SELECT FOR UPDATE to lock the row and check state atomically
                cursor.execute(
                    "SELECT cluster_id FROM person_detection WHERE id = %s FOR UPDATE",
                    (detection_id,),
                )
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"update_detection_cluster: detection {detection_id} not found")
                    return False

                current_cluster_id = row[0]
                if current_cluster_id is not None:
                    logger.debug(
                        f"update_detection_cluster: detection {detection_id} already in cluster {current_cluster_id}"
                    )
                    return False

                cursor.execute(
                    """UPDATE person_detection
                       SET cluster_id = %s,
                           cluster_confidence = %s,
                           cluster_status = %s
                       WHERE id = %s""",
                    (cluster_id, cluster_confidence, cluster_status, detection_id),
                )
                return True

    def update_detection_unassigned(self, detection_id: int) -> None:
        """Mark a detection as unassigned (added to outlier pool)."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Get current cluster assignment
                cursor.execute(
                    "SELECT cluster_id FROM person_detection WHERE id = %s FOR UPDATE",
                    (detection_id,),
                )
                row = cursor.fetchone()
                old_cluster_id = row[0] if row else None

                # Decrement old cluster's face_count if detection was in a cluster
                if old_cluster_id is not None:
                    cursor.execute(
                        """UPDATE cluster
                           SET face_count = GREATEST(0, face_count - 1),
                               updated_at = NOW()
                           WHERE id = %s""",
                        (old_cluster_id,),
                    )

                # Mark detection as unassigned and clear cluster_id
                cursor.execute(
                    """UPDATE person_detection
                       SET cluster_id = NULL,
                           cluster_status = 'unassigned',
                           cluster_confidence = 0,
                           unassigned_since = NOW()
                       WHERE id = %s""",
                    (detection_id,),
                )

    def clear_detection_unassigned(self, detection_id: int) -> None:
        """Clear unassigned status when detection is assigned to cluster."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person_detection
                       SET cluster_status = 'auto', unassigned_since = NULL
                       WHERE id = %s AND cluster_status = 'unassigned'""",
                    (detection_id,),
                )

    def find_similar_unassigned_detections(
        self, embedding, threshold: float, limit: int
    ) -> List[Dict[str, Any]]:
        """Find unassigned detections similar to embedding.

        Excludes detections with faces smaller than MIN_FACE_SIZE_PX pixels.
        Excludes detections with confidence below MIN_FACE_CONFIDENCE.

        Note: The query uses a subquery structure to allow the IVFFlat index to be
        used efficiently. Vector indexes (IVFFlat/HNSW) can only optimize
        ORDER BY + LIMIT, not WHERE distance < X. By moving the distance filter
        to the outer query, we fetch candidates using the index first, then filter.
        The inner LIMIT is 5x the requested limit to ensure enough candidates
        after threshold filtering.
        """
        # Overfetch factor to ensure enough results after threshold filtering
        overfetch_limit = limit * 5
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT id, distance FROM (
                           SELECT pd.id, fe.embedding <=> %s AS distance
                           FROM person_detection pd
                           JOIN face_embedding fe ON pd.id = fe.person_detection_id
                           WHERE pd.cluster_id IS NULL
                             AND pd.cluster_status = 'unassigned'
                             AND pd.face_confidence >= %s
                             AND pd.face_bbox_width >= %s
                             AND pd.face_bbox_height >= %s
                           ORDER BY fe.embedding <=> %s
                           LIMIT %s
                       ) sub
                       WHERE distance < %s
                       LIMIT %s""",
                    (
                        embedding,
                        MIN_FACE_CONFIDENCE,
                        MIN_FACE_SIZE_PX,
                        MIN_FACE_SIZE_PX,
                        embedding,
                        overfetch_limit,
                        threshold,
                        limit,
                    ),
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_detections_in_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get all detections with embeddings in a cluster."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.id, fe.embedding
                       FROM person_detection pd
                       JOIN face_embedding fe ON pd.id = fe.person_detection_id
                       WHERE pd.cluster_id = %s""",
                    (cluster_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_detection_embedding(self, detection_id: int) -> Optional[List[float]]:
        """Get face embedding by detection ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT embedding FROM face_embedding WHERE person_detection_id = %s",
                    (detection_id,),
                )
                row = cursor.fetchone()
                return [float(x) for x in row[0]] if row else None

    def remove_detection_from_cluster(
        self,
        detection_id: int,
        delete_empty_cluster: bool = True,
    ) -> Optional[int]:
        """
        Remove a detection from its current cluster.

        Args:
            detection_id: ID of the detection to remove
            delete_empty_cluster: If True, delete the cluster if it becomes empty

        Returns:
            ID of deleted cluster if one was removed, None otherwise
        """
        deleted_cluster_id = None

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Get current cluster
                cursor.execute(
                    "SELECT cluster_id FROM person_detection WHERE id = %s FOR UPDATE",
                    (detection_id,),
                )
                row = cursor.fetchone()
                if not row or row[0] is None:
                    return None

                old_cluster_id = row[0]

                # Clear detection cluster assignment
                cursor.execute(
                    """UPDATE person_detection
                       SET cluster_id = NULL,
                           cluster_status = NULL,
                           cluster_confidence = 0
                       WHERE id = %s""",
                    (detection_id,),
                )

                # Decrement cluster count
                cursor.execute(
                    """UPDATE cluster
                       SET face_count = GREATEST(0, face_count - 1),
                           updated_at = NOW()
                       WHERE id = %s""",
                    (old_cluster_id,),
                )

                # Check if cluster is now empty
                if delete_empty_cluster:
                    cursor.execute(
                        "SELECT face_count FROM cluster WHERE id = %s",
                        (old_cluster_id,),
                    )
                    count_row = cursor.fetchone()
                    if count_row and count_row[0] == 0:
                        cursor.execute(
                            "DELETE FROM cluster WHERE id = %s",
                            (old_cluster_id,),
                        )
                        deleted_cluster_id = old_cluster_id
                        logger.info(f"Deleted empty cluster {old_cluster_id}")

        return deleted_cluster_id

    def update_detection_cluster_status(self, detection_id: int, cluster_status: str) -> None:
        """Update only the cluster status for a detection."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person_detection
                       SET cluster_status = %s
                       WHERE id = %s""",
                    (cluster_status, detection_id),
                )

    def create_cluster_for_detection(
        self,
        centroid,
        representative_detection_id: int,
        medoid_detection_id: int,
        face_count: int = 1,
    ) -> int:
        """Create a new cluster with detection IDs and return its ID."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO cluster
                       (collection_id, face_count, face_count_at_last_medoid, representative_detection_id,
                        centroid, medoid_detection_id, created_at, updated_at)
                       VALUES (
                           (SELECT collection_id FROM person_detection WHERE id = %s),
                           %s, %s, %s, %s, %s, NOW(), NOW()
                       )
                       RETURNING id""",
                    (
                        representative_detection_id,
                        face_count,
                        face_count,
                        representative_detection_id,
                        centroid,
                        medoid_detection_id,
                    ),
                )
                return cursor.fetchone()[0]

    def create_detection_match_candidates(self, candidates: List[tuple]) -> None:
        """Create multiple detection match candidate records."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                candidates_with_collection = [
                    (detection_id, cluster_id, detection_id, similarity)
                    for detection_id, cluster_id, similarity in candidates
                ]
                cursor.executemany(
                    """INSERT INTO face_match_candidate
                       (detection_id, cluster_id, collection_id, similarity, status, created_at)
                       VALUES (%s, %s, (SELECT collection_id FROM person_detection WHERE id = %s), %s, 'pending', NOW())""",
                    candidates_with_collection,
                )

    # Prompt Category methods

    def get_prompt_categories(
        self, target: Optional[str] = None, active_only: bool = True
    ) -> List[PromptCategory]:
        """Get prompt categories, optionally filtered by target."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                query = "SELECT * FROM prompt_category WHERE 1=1"
                params: List[Any] = []

                if active_only:
                    query += " AND is_active = true"
                if target:
                    query += " AND target = %s"
                    params.append(target)

                query += " ORDER BY display_order"
                cursor.execute(query, params)

                return [self._row_to_prompt_category(row) for row in cursor.fetchall()]

    def get_prompt_category_by_name(self, name: str) -> Optional[PromptCategory]:
        """Get a prompt category by name."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM prompt_category WHERE name = %s", (name,))
                row = cursor.fetchone()
                return self._row_to_prompt_category(row) if row else None

    def create_prompt_category(
        self,
        name: str,
        target: Optional[str] = None,
        selection_mode: str = "single",
        min_confidence: float = 0.1,
        max_results: int = 5,
        description: Optional[str] = None,
        display_order: int = 0,
    ) -> PromptCategory:
        """Create a new prompt category."""
        # Infer target from name if not provided
        if target is None:
            if name.startswith("face_"):
                target = "face"
            else:
                target = "scene"

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO prompt_category
                    (name, target, selection_mode, min_confidence, max_results,
                     description, display_order, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, true, NOW(), NOW())
                    ON CONFLICT (name) DO UPDATE SET
                        target = EXCLUDED.target,
                        updated_at = NOW()
                    RETURNING id, name, target, selection_mode, min_confidence, max_results,
                              description, display_order, is_active, created_at, updated_at
                    """,
                    (
                        name,
                        target,
                        selection_mode,
                        min_confidence,
                        max_results,
                        description,
                        display_order,
                    ),
                )
                row = cursor.fetchone()
                return self._row_to_prompt_category(row)

    def _row_to_prompt_category(self, row) -> PromptCategory:
        return PromptCategory(
            id=row[0],
            name=row[1],
            target=row[2],
            selection_mode=row[3],
            min_confidence=row[4],
            max_results=row[5],
            description=row[6],
            display_order=row[7],
            is_active=row[8],
            created_at=row[9],
            updated_at=row[10],
        )

    # Prompt Embedding methods

    def get_prompts_by_category(
        self, category_id: int, active_only: bool = True, with_embeddings: bool = True
    ) -> List[PromptEmbedding]:
        """Get all prompts for a category."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                if with_embeddings:
                    cols = "*"
                else:
                    cols = (
                        "id, category_id, label, prompt_text, NULL as embedding, "
                        "model_name, model_version, display_name, parent_label, "
                        "confidence_boost, metadata, is_active, embedding_computed_at, "
                        "created_at, updated_at"
                    )
                query = f"SELECT {cols} FROM prompt_embedding WHERE category_id = %s"
                params: List[Any] = [category_id]

                if active_only:
                    query += " AND is_active = true"

                query += " ORDER BY label"
                cursor.execute(query, params)

                return [self._row_to_prompt_embedding(row) for row in cursor.fetchall()]

    def get_prompts_needing_embedding(self, model_name: str) -> List[PromptEmbedding]:
        """Get prompts that need embedding computation for a model."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM prompt_embedding
                    WHERE is_active = true
                    AND (embedding IS NULL OR model_name != %s)
                    ORDER BY category_id, label
                    """,
                    (model_name,),
                )
                return [self._row_to_prompt_embedding(row) for row in cursor.fetchall()]

    def upsert_prompt_embedding(self, prompt: PromptEmbedding) -> PromptEmbedding:
        """Insert or update a prompt embedding."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO prompt_embedding
                    (category_id, label, prompt_text, embedding, model_name, model_version,
                     display_name, parent_label, confidence_boost, metadata, is_active,
                     embedding_computed_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (category_id, label) DO UPDATE SET
                        prompt_text = EXCLUDED.prompt_text,
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        model_version = EXCLUDED.model_version,
                        display_name = EXCLUDED.display_name,
                        parent_label = EXCLUDED.parent_label,
                        confidence_boost = EXCLUDED.confidence_boost,
                        metadata = EXCLUDED.metadata,
                        embedding_computed_at = EXCLUDED.embedding_computed_at,
                        updated_at = EXCLUDED.updated_at
                    RETURNING id
                    """,
                    (
                        prompt.category_id,
                        prompt.label,
                        prompt.prompt_text,
                        prompt.embedding,
                        prompt.model_name,
                        prompt.model_version,
                        prompt.display_name,
                        prompt.parent_label,
                        prompt.confidence_boost,
                        json.dumps(prompt.metadata) if prompt.metadata else None,
                        prompt.is_active,
                        prompt.embedding_computed_at,
                        prompt.created_at,
                        prompt.updated_at,
                    ),
                )
                prompt.id = cursor.fetchone()[0]
        return prompt

    def update_prompt_embedding_vector(
        self,
        prompt_id: int,
        embedding: List[float],
        model_name: str,
        model_version: Optional[str] = None,
    ) -> None:
        """Update just the embedding vector for a prompt."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE prompt_embedding
                    SET embedding = %s, model_name = %s, model_version = %s,
                        embedding_computed_at = now(), updated_at = now()
                    WHERE id = %s
                    """,
                    (embedding, model_name, model_version, prompt_id),
                )

    def _row_to_prompt_embedding(self, row) -> PromptEmbedding:
        return PromptEmbedding(
            id=row[0],
            category_id=row[1],
            label=row[2],
            prompt_text=row[3],
            embedding=[float(x) for x in row[4]] if row[4] is not None else None,
            model_name=row[5],
            model_version=row[6],
            display_name=row[7],
            parent_label=row[8],
            confidence_boost=row[9],
            metadata=row[10]
            if isinstance(row[10], dict)
            else json.loads(row[10])
            if row[10]
            else None,
            is_active=row[11],
            embedding_computed_at=row[12],
            created_at=row[13],
            updated_at=row[14],
        )

    # Photo/Detection Tag methods

    def bulk_upsert_photo_tags(self, tags: List[PhotoTag]) -> None:
        """Bulk insert/update photo tags."""
        if not tags:
            return
        # Validate all tags belong to the same photo
        photo_ids = {t.photo_id for t in tags}
        if len(photo_ids) > 1:
            raise ValueError(f"All tags must belong to the same photo, got photo_ids: {photo_ids}")
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Delete existing tags for this photo in affected categories
                photo_id = tags[0].photo_id
                prompt_ids = [t.prompt_id for t in tags]
                cursor.execute(
                    """
                    DELETE FROM photo_tag
                    WHERE photo_id = %s AND prompt_id = ANY(%s)
                    """,
                    (photo_id, prompt_ids),
                )

                # Insert new tags
                for tag in tags:
                    cursor.execute(
                        """
                        INSERT INTO photo_tag
                        (photo_id, prompt_id, confidence, rank_in_category, analysis_output_id, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (photo_id, prompt_id) DO UPDATE SET
                            confidence = EXCLUDED.confidence,
                            rank_in_category = EXCLUDED.rank_in_category,
                            analysis_output_id = EXCLUDED.analysis_output_id
                        """,
                        (
                            tag.photo_id,
                            tag.prompt_id,
                            tag.confidence,
                            tag.rank_in_category,
                            tag.analysis_output_id,
                            tag.created_at,
                        ),
                    )

    def bulk_upsert_detection_tags(self, tags: List[DetectionTag]) -> None:
        """Bulk insert/update detection tags."""
        if not tags:
            return
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                for tag in tags:
                    cursor.execute(
                        """
                        INSERT INTO detection_tag
                        (detection_id, prompt_id, confidence, rank_in_category, analysis_output_id, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (detection_id, prompt_id) DO UPDATE SET
                            confidence = EXCLUDED.confidence,
                            rank_in_category = EXCLUDED.rank_in_category,
                            analysis_output_id = EXCLUDED.analysis_output_id
                        """,
                        (
                            tag.detection_id,
                            tag.prompt_id,
                            tag.confidence,
                            tag.rank_in_category,
                            tag.analysis_output_id,
                            tag.created_at,
                        ),
                    )

    def get_photo_tags(self, photo_id: int) -> List[Dict[str, Any]]:
        """Get all tags for a photo with category info."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM photo_tags_view WHERE photo_id = %s
                    """,
                    (photo_id,),
                )
                return [
                    {
                        "photo_id": row[0],
                        "category": row[1],
                        "target": row[2],
                        "label": row[3],
                        "display_name": row[4],
                        "confidence": row[5],
                        "rank": row[6],
                    }
                    for row in cursor.fetchall()
                ]

    # Analysis Output and Scene Analysis methods

    def create_analysis_output(self, output: AnalysisOutput) -> AnalysisOutput:
        """Insert analysis output record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO analysis_output
                    (photo_id, model_type, model_name, model_version, output,
                     processing_time_ms, device, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        output.photo_id,
                        output.model_type,
                        output.model_name,
                        output.model_version,
                        json.dumps(output.output),
                        output.processing_time_ms,
                        output.device,
                        output.created_at,
                    ),
                )
                output.id = cursor.fetchone()[0]
        return output

    def upsert_scene_analysis(self, analysis: SceneAnalysis) -> SceneAnalysis:
        """Insert or update scene analysis."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO scene_analysis
                    (photo_id, taxonomy_labels, taxonomy_confidences,
                     taxonomy_output_id, mobileclip_output_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (photo_id) DO UPDATE SET
                        taxonomy_labels = COALESCE(EXCLUDED.taxonomy_labels, scene_analysis.taxonomy_labels),
                        taxonomy_confidences = COALESCE(EXCLUDED.taxonomy_confidences, scene_analysis.taxonomy_confidences),
                        taxonomy_output_id = COALESCE(EXCLUDED.taxonomy_output_id, scene_analysis.taxonomy_output_id),
                        mobileclip_output_id = COALESCE(EXCLUDED.mobileclip_output_id, scene_analysis.mobileclip_output_id),
                        updated_at = EXCLUDED.updated_at
                    RETURNING id
                    """,
                    (
                        analysis.photo_id,
                        analysis.taxonomy_labels,
                        analysis.taxonomy_confidences,
                        analysis.taxonomy_output_id,
                        analysis.mobileclip_output_id,
                        analysis.created_at,
                        analysis.updated_at,
                    ),
                )
                analysis.id = cursor.fetchone()[0]
        return analysis

    # HDBSCAN clustering methods

    def get_all_embeddings_for_collection(
        self, collection_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all embeddings with their detection IDs and cluster info for HDBSCAN bootstrap.

        Only returns detections with face bounding boxes that are:
        - Large enough (MIN_FACE_SIZE_PX)
        - High enough confidence (MIN_FACE_CONFIDENCE)

        Args:
            collection_id: Optional collection ID to filter by. Uses instance default if not provided.

        Returns:
            List of dicts with detection_id, cluster_id, cluster_status, embedding
        """
        resolved_collection_id = self._resolve_collection_id(collection_id)
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.id AS detection_id, pd.cluster_id, pd.cluster_status, fe.embedding
                       FROM person_detection pd
                       JOIN face_embedding fe ON pd.id = fe.person_detection_id
                       WHERE pd.collection_id = %s
                         AND pd.face_confidence >= %s
                         AND pd.face_bbox_width >= %s
                         AND pd.face_bbox_height >= %s""",
                    (
                        resolved_collection_id,
                        MIN_FACE_CONFIDENCE,
                        MIN_FACE_SIZE_PX,
                        MIN_FACE_SIZE_PX,
                    ),
                )
                return [dict(row) for row in cursor.fetchall()]

    def mark_detection_as_core(self, detection_id: int, is_core: bool = True) -> None:
        """Set the is_core flag on a detection.

        Args:
            detection_id: ID of the detection to update
            is_core: Whether this detection is a core point
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person_detection SET is_core = %s WHERE id = %s""",
                    (is_core, detection_id),
                )

    def mark_detections_as_core(self, detection_ids: List[int], is_core: bool = True) -> None:
        """Bulk set is_core flag for multiple detections.

        Args:
            detection_ids: List of detection IDs to update
            is_core: Whether these detections are core points
        """
        if not detection_ids:
            return
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person_detection SET is_core = %s WHERE id = ANY(%s)""",
                    (is_core, detection_ids),
                )

    def find_neighbors_within_epsilon(
        self, embedding, epsilon: float, collection_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find all detections within epsilon distance of the given embedding.

        Uses pgvector's distance operators for epsilon-ball queries.

        Args:
            embedding: Query embedding vector
            epsilon: Maximum cosine distance threshold
            collection_id: Optional collection ID to filter by. Uses instance default if not provided.

        Returns:
            List of dicts with id, cluster_id, distance, is_core
        """
        resolved_collection_id = self._resolve_collection_id(collection_id)
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.id, pd.cluster_id, fe.embedding <=> %s AS distance, pd.is_core
                       FROM person_detection pd
                       JOIN face_embedding fe ON pd.id = fe.person_detection_id
                       WHERE pd.collection_id = %s
                         AND fe.embedding <=> %s < %s
                       ORDER BY distance""",
                    (embedding, resolved_collection_id, embedding, epsilon),
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_cluster_epsilon(self, cluster_id: int) -> Optional[float]:
        """Get the epsilon value for a cluster.

        Args:
            cluster_id: ID of the cluster

        Returns:
            The epsilon value, or None if not set or cluster not found
        """
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT epsilon FROM cluster WHERE id = %s""",
                    (cluster_id,),
                )
                row = cursor.fetchone()
                return row[0] if row else None

    def update_cluster_epsilon(self, cluster_id: int, epsilon: float, core_count: int) -> None:
        """Update a cluster's epsilon and core_count values.

        Args:
            cluster_id: ID of the cluster to update
            epsilon: The epsilon value (max intra-cluster distance)
            core_count: Number of core points in this cluster
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE cluster SET epsilon = %s, core_count = %s, updated_at = NOW()
                       WHERE id = %s""",
                    (epsilon, core_count, cluster_id),
                )

    def get_clusters_with_epsilon(
        self, collection_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all clusters with their epsilon values for incremental assignment.

        Returns all clusters regardless of whether epsilon is set (epsilon may be NULL
        for clusters created before HDBSCAN was implemented). The caller should use
        a fallback threshold for clusters without epsilon.

        Args:
            collection_id: Optional collection ID to filter by. Uses instance default if not provided.

        Returns:
            List of dicts with id, epsilon (may be None), centroid
        """
        resolved_collection_id = self._resolve_collection_id(collection_id)
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT id, epsilon, centroid
                       FROM cluster
                       WHERE collection_id = %s AND centroid IS NOT NULL""",
                    (resolved_collection_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

    def create_cluster_with_epsilon(
        self,
        centroid,
        representative_detection_id: int,
        medoid_detection_id: int,
        face_count: int = 1,
        epsilon: Optional[float] = None,
        core_count: int = 0,
    ) -> int:
        """Create a new cluster with epsilon and core_count values (for HDBSCAN bootstrap).

        Args:
            centroid: Cluster centroid embedding
            representative_detection_id: Detection ID for display photo
            medoid_detection_id: Detection ID closest to centroid
            face_count: Number of faces in this cluster
            epsilon: Maximum intra-cluster distance threshold
            core_count: Number of core points in this cluster

        Returns:
            The newly created cluster ID
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO cluster
                       (collection_id, face_count, face_count_at_last_medoid, representative_detection_id,
                        centroid, medoid_detection_id, epsilon, core_count, created_at, updated_at)
                       VALUES (
                           (SELECT collection_id FROM person_detection WHERE id = %s),
                           %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                       )
                       RETURNING id""",
                    (
                        representative_detection_id,
                        face_count,
                        face_count,
                        representative_detection_id,
                        centroid,
                        medoid_detection_id,
                        epsilon,
                        core_count,
                    ),
                )
                return cursor.fetchone()[0]

    def force_update_detection_cluster(
        self,
        detection_id: int,
        cluster_id: int,
        cluster_confidence: float,
        cluster_status: str,
    ) -> None:
        """
        Force update detection cluster assignment (bypasses race condition check).

        This is used during bootstrap operations where we control the entire process
        and don't need to worry about concurrent updates.

        Args:
            detection_id: Detection to update
            cluster_id: Cluster to assign to
            cluster_confidence: Confidence score
            cluster_status: Status string (e.g., 'hdbscan', 'hdbscan_core')
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person_detection
                       SET cluster_id = %s,
                           cluster_confidence = %s,
                           cluster_status = %s
                       WHERE id = %s""",
                    (cluster_id, cluster_confidence, cluster_status, detection_id),
                )

    def get_clusters_without_epsilon(
        self, min_faces: int = 3, collection_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get clusters with NULL epsilon that have at least min_faces faces.

        These are clusters that need epsilon calculation, typically manual clusters
        created before HDBSCAN was implemented.

        Args:
            min_faces: Minimum number of faces required (default 3)
            collection_id: Optional collection ID to filter by

        Returns:
            List of dicts with id, face_count, centroid
        """
        resolved_collection_id = self._resolve_collection_id(collection_id)
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT id, face_count, centroid
                       FROM cluster
                       WHERE collection_id = %s
                         AND epsilon IS NULL
                         AND face_count >= %s
                         AND centroid IS NOT NULL""",
                    (resolved_collection_id, min_faces),
                )
                return [dict(row) for row in cursor.fetchall()]

    def update_cluster_epsilon_only(self, cluster_id: int, epsilon: float) -> None:
        """Update only a cluster's epsilon value (not core_count).

        Used for calculating epsilon for manual clusters where core_count
        is not applicable.

        Args:
            cluster_id: ID of the cluster to update
            epsilon: The epsilon value (max intra-cluster distance)
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE cluster SET epsilon = %s, updated_at = NOW()
                       WHERE id = %s""",
                    (epsilon, cluster_id),
                )

    # ============================================
    # GENEALOGICAL RELATIONSHIP METHODS
    # ============================================

    def add_person_parent(
        self,
        person_id: int,
        parent_id: int,
        parent_role: Optional[str] = "parent",
        is_biological: bool = True,
        source: Optional[str] = "user",
    ) -> PersonParent:
        """Add a parent-child relationship.

        Args:
            person_id: ID of the child person
            parent_id: ID of the parent person
            parent_role: Role of the parent ('mother', 'father', 'parent')
            is_biological: Whether the relationship is biological
            source: Source of the relationship ('user', 'inferred', 'imported')

        Returns:
            The created PersonParent record

        Raises:
            ValueError: If person_id == parent_id (self-reference)
        """
        if person_id == parent_id:
            raise ValueError("A person cannot be their own parent")

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO person_parent
                       (person_id, parent_id, parent_role, is_biological, source, created_at)
                       VALUES (%s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (person_id, parent_id) DO UPDATE SET
                           parent_role = EXCLUDED.parent_role,
                           is_biological = EXCLUDED.is_biological,
                           source = EXCLUDED.source""",
                    (person_id, parent_id, parent_role, is_biological, source),
                )

        return PersonParent.create(
            person_id=person_id,
            parent_id=parent_id,
            parent_role=parent_role,
            is_biological=is_biological,
            source=source,
        )

    def remove_person_parent(self, person_id: int, parent_id: int) -> bool:
        """Remove a parent-child relationship.

        Args:
            person_id: ID of the child person
            parent_id: ID of the parent person

        Returns:
            True if a relationship was removed, False if not found
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM person_parent WHERE person_id = %s AND parent_id = %s",
                    (person_id, parent_id),
                )
                return cursor.rowcount > 0

    def get_person_parents(self, person_id: int) -> List[Person]:
        """Get all parents of a person.

        Args:
            person_id: ID of the person

        Returns:
            List of parent Person records
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT p.*
                       FROM person p
                       JOIN person_parent pp ON p.id = pp.parent_id
                       WHERE pp.person_id = %s
                       ORDER BY pp.parent_role, p.id""",
                    (person_id,),
                )
                rows = cursor.fetchall()
                return [Person(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def get_person_children(self, person_id: int) -> List[Person]:
        """Get all children of a person.

        Args:
            person_id: ID of the person

        Returns:
            List of child Person records
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT p.*
                       FROM person p
                       JOIN person_parent pp ON p.id = pp.person_id
                       WHERE pp.parent_id = %s
                       ORDER BY p.id""",
                    (person_id,),
                )
                rows = cursor.fetchall()
                return [Person(**dict(row)) for row in rows]  # type: ignore[arg-type]

    def add_person_partnership(
        self,
        person1_id: int,
        person2_id: int,
        partnership_type: Optional[str] = "partner",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        is_current: bool = True,
    ) -> PersonPartnership:
        """Add a partnership between two persons.

        Args:
            person1_id: ID of the first person
            person2_id: ID of the second person
            partnership_type: Type of partnership ('married', 'partner', 'divorced', 'separated')
            start_year: Year the partnership started
            end_year: Year the partnership ended
            is_current: Whether the partnership is current

        Returns:
            The created PersonPartnership record

        Raises:
            ValueError: If person1_id == person2_id
        """
        if person1_id == person2_id:
            raise ValueError("A person cannot have a partnership with themselves")

        # Ensure canonical ordering
        if person1_id > person2_id:
            person1_id, person2_id = person2_id, person1_id

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO person_partnership
                       (person1_id, person2_id, partnership_type, start_year, end_year, is_current, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (person1_id, person2_id, COALESCE(start_year, 0)) DO UPDATE SET
                           partnership_type = EXCLUDED.partnership_type,
                           end_year = EXCLUDED.end_year,
                           is_current = EXCLUDED.is_current
                       RETURNING id""",
                    (person1_id, person2_id, partnership_type, start_year, end_year, is_current),
                )
                result = cursor.fetchone()
                partnership_id = result[0] if result else None

        partnership = PersonPartnership.create(
            person1_id=person1_id,
            person2_id=person2_id,
            partnership_type=partnership_type,
            start_year=start_year,
            end_year=end_year,
            is_current=is_current,
        )
        partnership.id = partnership_id
        return partnership

    def get_person_partnerships(self, person_id: int) -> List[PersonPartnership]:
        """Get all partnerships for a person.

        Args:
            person_id: ID of the person

        Returns:
            List of PersonPartnership records
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT *
                       FROM person_partnership
                       WHERE person1_id = %s OR person2_id = %s
                       ORDER BY COALESCE(start_year, 0) DESC, id""",
                    (person_id, person_id),
                )
                rows = cursor.fetchall()
                return [
                    PersonPartnership(
                        id=row["id"],
                        person1_id=row["person1_id"],
                        person2_id=row["person2_id"],
                        partnership_type=row["partnership_type"],
                        start_year=row["start_year"],
                        end_year=row["end_year"],
                        is_current=row["is_current"],
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]

    def add_birth_order(
        self,
        older_person_id: int,
        younger_person_id: int,
        source: Optional[str] = "user",
    ) -> PersonBirthOrder:
        """Add a birth order relationship (older_person was born before younger_person).

        Args:
            older_person_id: ID of the older person
            younger_person_id: ID of the younger person
            source: Source ('exact_dates', 'user', 'inferred', 'photo_evidence')

        Returns:
            The created PersonBirthOrder record

        Raises:
            ValueError: If older_person_id == younger_person_id
        """
        if older_person_id == younger_person_id:
            raise ValueError("A person cannot be older than themselves")

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO person_birth_order
                       (older_person_id, younger_person_id, source, created_at)
                       VALUES (%s, %s, %s, NOW())
                       ON CONFLICT (older_person_id, younger_person_id) DO UPDATE SET
                           source = EXCLUDED.source""",
                    (older_person_id, younger_person_id, source),
                )

        return PersonBirthOrder.create(
            older_person_id=older_person_id,
            younger_person_id=younger_person_id,
            source=source,
        )

    def create_placeholder_parent(
        self,
        child_id: int,
        role: str = "parent",
    ) -> int:
        """Create a placeholder parent for a person by calling the DB function.

        Args:
            child_id: ID of the child person
            role: Role of the parent ('mother', 'father', 'parent')

        Returns:
            The ID of the newly created placeholder parent
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT create_placeholder_parent(%s, %s)",
                    (child_id, role),
                )
                result = cursor.fetchone()
                return result[0]

    def link_siblings(
        self,
        person1_id: int,
        person2_id: int,
        sibling_type: str = "full",
    ) -> int:
        """Link two people as siblings by calling the DB function.

        If they already share a parent, returns that parent's ID.
        Otherwise, creates placeholder parent(s) and links both children.
        For full siblings, creates two placeholder parents.

        Args:
            person1_id: ID of the first person
            person2_id: ID of the second person
            sibling_type: 'full' or 'half' (default: 'full')

        Returns:
            The ID of the shared parent (existing or newly created)

        Raises:
            ValueError: If person1_id == person2_id
        """
        if person1_id == person2_id:
            raise ValueError("Cannot link a person as their own sibling")

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT link_siblings(%s, %s, %s)",
                    (person1_id, person2_id, sibling_type),
                )
                result = cursor.fetchone()
                return result[0]

    def refresh_genealogy_closures(self) -> None:
        """Refresh all genealogy closure tables by calling the DB function.

        This refreshes both person_ancestor_closure and person_birth_order_closure.
        Should be called after adding/removing parent-child or birth order relationships.
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT refresh_genealogy_closures()")

    def get_family_tree(
        self,
        center_id: int,
        max_generations: int = 3,
        include_placeholders: bool = True,
    ) -> List[FamilyMember]:
        """Get the family tree centered on a person by calling the DB function.

        Args:
            center_id: ID of the center person
            max_generations: Maximum generations to traverse (default: 3)
            include_placeholders: Whether to include placeholder persons (default: True)

        Returns:
            List of FamilyMember records representing the family tree
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM get_family_tree(%s, %s, %s)",
                    (center_id, max_generations, include_placeholders),
                )
                rows = cursor.fetchall()
                return [
                    FamilyMember(
                        person_id=row["person_id"],
                        display_name=row["display_name"],
                        relation=row["relation"],
                        generation_offset=row["generation_offset"],
                        is_placeholder=row["is_placeholder"],
                    )
                    for row in rows
                ]

    def propagate_birth_year_constraints(
        self,
        min_parent_gap: int = 15,
        max_parent_gap: int = 60,
    ) -> int:
        """Propagate birth year constraints through genealogical relationships.

        Uses parent-child relationships to infer birth year min/max constraints.
        Runs iteratively until no more updates are made.

        Args:
            min_parent_gap: Minimum age difference between parent and child (default: 15)
            max_parent_gap: Maximum age difference between parent and child (default: 60)

        Returns:
            Total number of person records updated
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT propagate_birth_year_constraints(%s, %s)",
                    (min_parent_gap, max_parent_gap),
                )
                result = cursor.fetchone()
                return result[0] if result else 0

    def get_persons_older_than(self, person_id: int) -> List[Person]:
        """Get all persons who are older than the given person (from birth order closure).

        Args:
            person_id: ID of the person

        Returns:
            List of Person records who are known to be older
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT p.*, pboc.min_confidence, pboc.inference_type
                       FROM person_birth_order_closure pboc
                       JOIN person p ON p.id = pboc.older_person_id
                       WHERE pboc.younger_person_id = %s
                       ORDER BY pboc.min_confidence DESC NULLS LAST, p.id""",
                    (person_id,),
                )
                rows = cursor.fetchall()
                return [
                    Person(
                        **{
                            k: v
                            for k, v in dict(row).items()
                            if k not in ("min_confidence", "inference_type")
                        }
                    )
                    for row in rows
                ]  # type: ignore[arg-type]

    def get_person_siblings(self, person_id: int) -> List[Sibling]:
        """Get all siblings of a person from the person_siblings view.

        Args:
            person_id: ID of the person

        Returns:
            List of Sibling records
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT *
                       FROM person_siblings
                       WHERE person_id = %s
                       ORDER BY sibling_type DESC, sibling_id""",
                    (person_id,),
                )
                rows = cursor.fetchall()
                return [
                    Sibling(
                        person_id=row["person_id"],
                        sibling_id=row["sibling_id"],
                        sibling_type=row["sibling_type"],
                        shared_parent_ids=row["shared_parent_ids"] or [],
                    )
                    for row in rows
                ]

    def create_placeholder_person(
        self,
        collection_id: Optional[int] = None,
        placeholder_description: Optional[str] = None,
    ) -> Person:
        """Create a placeholder person record.

        Args:
            collection_id: Collection ID (uses default if not provided)
            placeholder_description: Description for the placeholder

        Returns:
            The created Person record with is_placeholder=True
        """
        resolved_collection_id = self._resolve_collection_id(collection_id)

        person = Person.create(
            collection_id=resolved_collection_id,
            is_placeholder=True,
            placeholder_description=placeholder_description,
        )

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO person
                       (collection_id, first_name, last_name, is_placeholder,
                        placeholder_description, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        person.collection_id,
                        person.first_name,
                        person.last_name,
                        person.is_placeholder,
                        person.placeholder_description,
                        person.created_at,
                        person.updated_at,
                    ),
                )
                person.id = cursor.fetchone()[0]

        return person

    def get_person_parent_relationships(self, person_id: int) -> List[PersonParent]:
        """Get all parent relationships for a person (with relationship details).

        Unlike get_person_parents which returns Person records, this returns
        the relationship records with parent_role, is_biological, etc.

        Args:
            person_id: ID of the person (child)

        Returns:
            List of PersonParent records
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT *
                       FROM person_parent
                       WHERE person_id = %s
                       ORDER BY parent_role, parent_id""",
                    (person_id,),
                )
                rows = cursor.fetchall()
                return [
                    PersonParent(
                        person_id=row["person_id"],
                        parent_id=row["parent_id"],
                        parent_role=row["parent_role"],
                        is_biological=row["is_biological"],
                        source=row["source"],
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]

    def get_ancestors(
        self,
        person_id: int,
        max_distance: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get all ancestors of a person from the closure table.

        Args:
            person_id: ID of the person
            max_distance: Maximum generation distance (None for unlimited)

        Returns:
            List of dicts with ancestor_id, distance
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                if max_distance is not None:
                    cursor.execute(
                        """SELECT ancestor_id, distance
                           FROM person_ancestor_closure
                           WHERE descendant_id = %s AND distance <= %s
                           ORDER BY distance, ancestor_id""",
                        (person_id, max_distance),
                    )
                else:
                    cursor.execute(
                        """SELECT ancestor_id, distance
                           FROM person_ancestor_closure
                           WHERE descendant_id = %s
                           ORDER BY distance, ancestor_id""",
                        (person_id,),
                    )
                return [dict(row) for row in cursor.fetchall()]

    def get_descendants(
        self,
        person_id: int,
        max_distance: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get all descendants of a person from the closure table.

        Args:
            person_id: ID of the person
            max_distance: Maximum generation distance (None for unlimited)

        Returns:
            List of dicts with descendant_id, distance
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                if max_distance is not None:
                    cursor.execute(
                        """SELECT descendant_id, distance
                           FROM person_ancestor_closure
                           WHERE ancestor_id = %s AND distance <= %s
                           ORDER BY distance, descendant_id""",
                        (person_id, max_distance),
                    )
                else:
                    cursor.execute(
                        """SELECT descendant_id, distance
                           FROM person_ancestor_closure
                           WHERE ancestor_id = %s
                           ORDER BY distance, descendant_id""",
                        (person_id,),
                    )
                return [dict(row) for row in cursor.fetchall()]

    # ============================================
    # NON-RELATIONSHIP TRACKING METHODS
    # ============================================

    def add_person_not_related(
        self,
        person1_id: int,
        person2_id: int,
        source: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> PersonNotRelated:
        """Add an explicit non-relationship between two persons.

        Args:
            person1_id: ID of the first person
            person2_id: ID of the second person
            source: Source of the determination ('user', 'inferred')
            notes: Optional notes about why they are not related

        Returns:
            The created PersonNotRelated record

        Raises:
            ValueError: If person1_id == person2_id
        """
        if person1_id == person2_id:
            raise ValueError("A person cannot be marked as not related to themselves")

        # Ensure canonical ordering (person1_id < person2_id)
        if person1_id > person2_id:
            person1_id, person2_id = person2_id, person1_id

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO person_not_related
                       (person1_id, person2_id, source, notes, created_at)
                       VALUES (%s, %s, %s, %s, NOW())
                       ON CONFLICT (person1_id, person2_id) DO UPDATE SET
                           source = EXCLUDED.source,
                           notes = EXCLUDED.notes""",
                    (person1_id, person2_id, source, notes),
                )

        return PersonNotRelated.create(
            person1_id=person1_id,
            person2_id=person2_id,
            source=source,
            notes=notes,
        )

    def remove_person_not_related(self, person1_id: int, person2_id: int) -> bool:
        """Remove a non-relationship record between two persons.

        Args:
            person1_id: ID of the first person
            person2_id: ID of the second person

        Returns:
            True if a record was removed, False if not found
        """
        # Ensure canonical ordering (person1_id < person2_id)
        if person1_id > person2_id:
            person1_id, person2_id = person2_id, person1_id

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM person_not_related WHERE person1_id = %s AND person2_id = %s",
                    (person1_id, person2_id),
                )
                return cursor.rowcount > 0

    def are_persons_unrelated(self, person1_id: int, person2_id: int) -> bool:
        """Check if two persons are explicitly marked as not related.

        This calls the database function are_persons_unrelated which handles
        canonical ordering internally.

        Args:
            person1_id: ID of the first person
            person2_id: ID of the second person

        Returns:
            True if the persons are marked as not related, False otherwise
        """
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT are_persons_unrelated(%s, %s)",
                    (person1_id, person2_id),
                )
                result = cursor.fetchone()
                return result[0] if result else False

    def get_unrelated_persons(self, person_id: int) -> List[int]:
        """Get all person IDs that are explicitly marked as not related to a person.

        Args:
            person_id: ID of the person

        Returns:
            List of person IDs that are not related to this person
        """
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT CASE
                           WHEN person1_id = %s THEN person2_id
                           ELSE person1_id
                       END as other_person_id
                       FROM person_not_related
                       WHERE person1_id = %s OR person2_id = %s
                       ORDER BY other_person_id""",
                    (person_id, person_id, person_id),
                )
                return [row[0] for row in cursor.fetchall()]

    # ==============================================
    # PERSON BIRTH DATE METHODS
    # ==============================================

    def set_person_birth_date(self, person_id: int, birth_date: date) -> None:
        """Set exact birth date, updating year constraints accordingly."""
        year = birth_date.year
        with self.pool.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE person
                    SET birth_date = %s,
                        birth_year_min = %s,
                        birth_year_max = %s,
                        birth_year_source = 'exact'
                    WHERE id = %s
                    """,
                    (birth_date, year, year, person_id),
                )

    def set_person_birth_year(self, person_id: int, year: int) -> None:
        """Set birth year only (no exact date known)."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE person
                    SET birth_date = NULL,
                        birth_year_min = %s,
                        birth_year_max = %s,
                        birth_year_source = 'year'
                    WHERE id = %s
                    """,
                    (year, year, person_id),
                )

    def clear_person_birth_info(self, person_id: int) -> None:
        """Clear all birth date/year information for a person."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE person
                    SET birth_date = NULL,
                        birth_year_min = NULL,
                        birth_year_max = NULL,
                        birth_year_source = NULL
                    WHERE id = %s
                    """,
                    (person_id,),
                )
