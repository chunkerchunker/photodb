from typing import Optional, List, Dict, Any
from datetime import datetime
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
    LLMAnalysis,
    Metadata,
    Person,
    PersonDetection,
    Photo,
    PhotoTag,
    ProcessingStatus,
    PromptCategory,
    PromptEmbedding,
    SceneAnalysis,
)

logger = logging.getLogger(__name__)

# Minimum face size for clustering (in pixels)
# Faces smaller than this (in either dimension) will be excluded from clustering
MIN_FACE_SIZE_PX = int(os.environ.get("MIN_FACE_SIZE_PX", 50))  # Default 50 pixels

# Minimum face detection confidence for clustering
# Faces with lower confidence will be excluded from clustering
MIN_FACE_CONFIDENCE = float(os.environ.get("MIN_FACE_CONFIDENCE", 0.9))  # Default 90%


class PhotoRepository:
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool

    def create_photo(self, photo: Photo) -> None:
        """Insert a new photo record."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO photo (filename, normalized_path, width, height, 
                                         normalized_width, normalized_height, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (
                        photo.filename,
                        photo.normalized_path,
                        photo.width,
                        photo.height,
                        photo.normalized_width,
                        photo.normalized_height,
                        photo.created_at,
                        photo.updated_at,
                    ),
                )
                photo.id = cursor.fetchone()[0]

    def get_photo_by_filename(self, filename: str) -> Optional[Photo]:
        """Get photo by filename."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT * FROM photo WHERE filename = %s", (filename,))
                row = cursor.fetchone()

                if row:
                    return Photo(**dict(row))  # type: ignore[arg-type]
                return None

    def get_photo_by_id(self, photo_id: int) -> Optional[Photo]:
        """Get photo by ID."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT * FROM photo WHERE id = %s", (photo_id,))
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
                       SET normalized_path = %s, width = %s, height = %s,
                           normalized_width = %s, normalized_height = %s, updated_at = %s
                       WHERE id = %s""",
                    (
                        photo.normalized_path,
                        photo.width,
                        photo.height,
                        photo.normalized_width,
                        photo.normalized_height,
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
                        status.status,
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
        """Get photos that haven't been processed for a specific stage."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT p.* FROM photo p
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
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT p.*, ps.error_message, ps.processed_at 
                       FROM photo p
                       JOIN processing_status ps ON p.id = ps.photo_id
                       WHERE ps.stage = %s AND ps.status = 'failed'""",
                    (stage,),
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
                       WHERE p.normalized_path IS NOT NULL
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
                    """INSERT INTO person (first_name, last_name, created_at, updated_at)
                       VALUES (%s, %s, %s, %s) RETURNING id""",
                    (
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
                cursor.execute("SELECT * FROM person WHERE id = %s", (person_id,))
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
                        "SELECT * FROM person WHERE first_name = %s AND last_name = %s",
                        (first_name, last_name),
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM person WHERE first_name = %s AND last_name IS NULL",
                        (first_name,),
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
                cursor.execute("SELECT * FROM person ORDER BY first_name, last_name")
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

    def get_unclustered_faces_for_photo(self, photo_id: int) -> List[Dict[str, Any]]:
        """Get all unclustered faces for a photo with embeddings.

        Excludes faces smaller than MIN_FACE_SIZE_PX pixels in either dimension.
        Excludes faces with detection confidence below MIN_FACE_CONFIDENCE.
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT f.*, fe.embedding
                       FROM face f
                       LEFT JOIN face_embedding fe ON f.id = fe.face_id
                       JOIN photo p ON f.photo_id = p.id
                       WHERE f.photo_id = %s
                         AND f.cluster_id IS NULL
                         AND f.cluster_status IS NULL
                         AND f.confidence >= %s
                         AND (f.bbox_width * COALESCE(p.normalized_width, p.width, 1)) >= %s
                         AND (f.bbox_height * COALESCE(p.normalized_height, p.height, 1)) >= %s
                       ORDER BY f.id""",
                    (photo_id, MIN_FACE_CONFIDENCE, MIN_FACE_SIZE_PX, MIN_FACE_SIZE_PX),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_all_faces_with_embeddings_for_photo(self, photo_id: int) -> List[Dict[str, Any]]:
        """Get all faces with embeddings for a photo (for force reprocessing)."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT f.*, fe.embedding
                       FROM face f
                       LEFT JOIN face_embedding fe ON f.id = fe.face_id
                       WHERE f.photo_id = %s 
                         AND fe.embedding IS NOT NULL
                       ORDER BY f.id""",
                    (photo_id,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def find_nearest_clusters(self, embedding, limit: int = 10) -> List[tuple]:
        """Find nearest clusters using cosine distance."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT id, centroid <=> %s AS distance
                       FROM cluster
                       WHERE centroid IS NOT NULL
                       ORDER BY centroid <=> %s
                       LIMIT %s""",
                    (embedding, embedding, limit),
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
                       (face_count, face_count_at_last_medoid, representative_face_id,
                        centroid, medoid_face_id, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                       RETURNING id""",
                    (face_count, face_count, representative_face_id, centroid, medoid_face_id),
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

                conn.commit()

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

                conn.commit()

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
                       (face_id, cluster_id, similarity, status, created_at)
                       VALUES (%s, %s, %s, 'pending', NOW())""",
                    candidates,
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

    def get_must_linked_faces(self, face_id: int) -> List[Dict[str, Any]]:
        """Get faces that must be in same cluster as this face."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT f.id, f.cluster_id
                       FROM face f
                       JOIN (
                           SELECT face_id_2 AS linked_id FROM must_link WHERE face_id_1 = %s
                           UNION
                           SELECT face_id_1 AS linked_id FROM must_link WHERE face_id_2 = %s
                       ) ml ON f.id = ml.linked_id""",
                    (face_id, face_id),
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

    def add_must_link(
        self, face_id_1: int, face_id_2: int, created_by: str = "human"
    ) -> Optional[int]:
        """
        Add must-link constraint between two faces.

        If both faces are already in different clusters, triggers a merge.
        Returns the constraint ID or None if already exists.
        """
        # Canonical ordering
        if face_id_1 > face_id_2:
            face_id_1, face_id_2 = face_id_2, face_id_1

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO must_link (face_id_1, face_id_2, created_by)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (face_id_1, face_id_2) DO NOTHING
                       RETURNING id""",
                    (face_id_1, face_id_2, created_by),
                )
                result = cursor.fetchone()
                return result[0] if result else None

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
                    """INSERT INTO cannot_link (face_id_1, face_id_2, created_by)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (face_id_1, face_id_2) DO NOTHING
                       RETURNING id""",
                    (face_id_1, face_id_2, created_by),
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
                    """INSERT INTO cluster_cannot_link (cluster_id_1, cluster_id_2)
                       VALUES (%s, %s)
                       ON CONFLICT (cluster_id_1, cluster_id_2) DO NOTHING
                       RETURNING id""",
                    (cluster_id_1, cluster_id_2),
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

    def find_similar_unassigned_faces(
        self, embedding, threshold: float, limit: int
    ) -> List[Dict[str, Any]]:
        """Find unassigned faces similar to embedding.

        Excludes faces smaller than MIN_FACE_SIZE_PX pixels in either dimension.
        Excludes faces with detection confidence below MIN_FACE_CONFIDENCE.
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT f.id, fe.embedding <=> %s AS distance
                       FROM face f
                       JOIN face_embedding fe ON f.id = fe.face_id
                       JOIN photo p ON f.photo_id = p.id
                       WHERE f.cluster_id IS NULL
                         AND f.cluster_status = 'unassigned'
                         AND f.confidence >= %s
                         AND (f.bbox_width * COALESCE(p.normalized_width, p.width, 1)) >= %s
                         AND (f.bbox_height * COALESCE(p.normalized_height, p.height, 1)) >= %s
                         AND fe.embedding <=> %s < %s
                       ORDER BY fe.embedding <=> %s
                       LIMIT %s""",
                    (
                        embedding,
                        MIN_FACE_CONFIDENCE,
                        MIN_FACE_SIZE_PX,
                        MIN_FACE_SIZE_PX,
                        embedding,
                        threshold,
                        embedding,
                        limit,
                    ),
                )
                return [dict(row) for row in cursor.fetchall()]

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
                cursor.execute("SELECT COUNT(*) FROM must_link")
                must_link_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cannot_link")
                cannot_link_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cluster_cannot_link")
                cluster_cannot_link_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cluster WHERE verified = true")
                verified_clusters = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM face WHERE cluster_status = 'unassigned'")
                unassigned_faces = cursor.fetchone()[0]

                return {
                    "must_link_count": must_link_count,
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
                       (photo_id, face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height,
                        face_confidence, body_bbox_x, body_bbox_y, body_bbox_width, body_bbox_height,
                        body_confidence, age_estimate, gender, gender_confidence, mivolo_output,
                        person_id, cluster_status, cluster_id, cluster_confidence,
                        detector_model, detector_version, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        detection.photo_id,
                        detection.face_bbox_x,
                        detection.face_bbox_y,
                        detection.face_bbox_width,
                        detection.face_bbox_height,
                        detection.face_confidence,
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
                       ORDER BY fe.embedding <=> %s
                       LIMIT %s""",
                    (embedding, embedding, limit),
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_must_linked_detections(self, detection_id: int) -> List[Dict[str, Any]]:
        """Get detections that must be in same cluster as this detection."""
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.id, pd.cluster_id
                       FROM person_detection pd
                       JOIN (
                           SELECT detection_id_2 AS linked_id FROM must_link WHERE detection_id_1 = %s
                           UNION
                           SELECT detection_id_1 AS linked_id FROM must_link WHERE detection_id_2 = %s
                       ) ml ON pd.id = ml.linked_id""",
                    (detection_id, detection_id),
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
                           SELECT detection_id_2 AS linked_id FROM cannot_link WHERE detection_id_1 = %s
                           UNION
                           SELECT detection_id_1 AS linked_id FROM cannot_link WHERE detection_id_2 = %s
                       ) cl ON pd.id = cl.linked_id""",
                    (detection_id, detection_id),
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
                           cluster_confidence = 0
                       WHERE id = %s""",
                    (detection_id,),
                )

    def clear_detection_unassigned(self, detection_id: int) -> None:
        """Clear unassigned status when detection is assigned to cluster."""
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE person_detection
                       SET cluster_status = 'auto'
                       WHERE id = %s AND cluster_status = 'unassigned'""",
                    (detection_id,),
                )

    def find_similar_unassigned_detections(
        self, embedding, threshold: float, limit: int
    ) -> List[Dict[str, Any]]:
        """Find unassigned detections similar to embedding.

        Excludes detections with faces smaller than MIN_FACE_SIZE_PX pixels.
        Excludes detections with confidence below MIN_FACE_CONFIDENCE.
        """
        with self.pool.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT pd.id, fe.embedding <=> %s AS distance
                       FROM person_detection pd
                       JOIN face_embedding fe ON pd.id = fe.person_detection_id
                       WHERE pd.cluster_id IS NULL
                         AND pd.cluster_status = 'unassigned'
                         AND pd.face_confidence >= %s
                         AND pd.face_bbox_width >= %s
                         AND pd.face_bbox_height >= %s
                         AND fe.embedding <=> %s < %s
                       ORDER BY fe.embedding <=> %s
                       LIMIT %s""",
                    (
                        embedding,
                        MIN_FACE_CONFIDENCE,
                        MIN_FACE_SIZE_PX,
                        MIN_FACE_SIZE_PX,
                        embedding,
                        threshold,
                        embedding,
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

                conn.commit()

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
                       (face_count, face_count_at_last_medoid, representative_detection_id,
                        centroid, medoid_detection_id, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                       RETURNING id""",
                    (
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
                cursor.executemany(
                    """INSERT INTO face_match_candidate
                       (detection_id, cluster_id, similarity, status, created_at)
                       VALUES (%s, %s, %s, 'pending', NOW())""",
                    candidates,
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
