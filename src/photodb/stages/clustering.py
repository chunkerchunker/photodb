import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from decimal import Decimal
from datetime import datetime

from .base import BaseStage
from ..database.models import ProcessingStatus, Photo

logger = logging.getLogger(__name__)


class ClusteringStage(BaseStage):
    """Stage for clustering faces based on embeddings."""

    stage_name = "clustering"

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        self.clustering_threshold = float(config.get("CLUSTERING_THRESHOLD", 0.45))
        logger.info(f"Initialized ClusteringStage with threshold {self.clustering_threshold}")

    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """Check if clustering is needed for faces in this photo."""
        if force:
            return True

        photo = self.repository.get_photo_by_filename(str(file_path))
        if not photo:
            return False

        # Check if photo has been processed through faces stage
        faces_status = self.repository.get_processing_status(photo.id, "faces")
        if not faces_status or faces_status.status != "completed":
            logger.debug(f"Skipping clustering for {file_path}: faces stage not completed")
            return False

        # Check if clustering has already been done
        status = self.repository.get_processing_status(photo.id, self.stage_name)
        if status and status.status == "completed":
            return False

        # Check if there are unclustered faces for this photo
        unclustered_faces = self.repository.get_unclustered_faces_for_photo(photo.id)
        return len(unclustered_faces) > 0

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process clustering for all faces in a photo."""
        try:
            # Get faces to cluster based on force flag
            if hasattr(self, 'force') and self.force:
                # If force flag is set, get all faces with embeddings for reprocessing
                faces_to_cluster = self.repository.get_all_faces_with_embeddings_for_photo(photo.id)
                logger.debug(f"Force mode: reprocessing {len(faces_to_cluster)} faces for {file_path}")
            else:
                # Normal mode: only get unclustered faces
                faces_to_cluster = self.repository.get_unclustered_faces_for_photo(photo.id)
            
            if not faces_to_cluster:
                logger.debug(f"No faces found to cluster for {file_path}")
                return True  # Success - nothing to cluster

            logger.info(f"Clustering {len(faces_to_cluster)} faces from {file_path}")

            # Process each face
            faces_processed = 0
            faces_failed = 0
            
            for face in faces_to_cluster:
                try:
                    self._cluster_single_face(face)
                    faces_processed += 1
                except Exception as e:
                    logger.error(f"Failed to cluster face {face['id']}: {e}")
                    faces_failed += 1

            # Log results
            if faces_failed > 0:
                logger.warning(f"Clustered {faces_processed} faces, {faces_failed} failed for {file_path}")
                return False  # Partial failure
            else:
                logger.info(f"Successfully clustered {faces_processed} faces from {file_path}")
                return True  # Success

        except Exception as e:
            logger.error(f"Error clustering faces for {file_path}: {e}")
            raise

    def _cluster_single_face(self, face: dict) -> None:
        """Cluster a single face based on embedding similarity."""
        face_id = face["id"]
        embedding = face.get("embedding")
        
        if embedding is None:
            logger.warning(f"No embedding found for face {face_id} - skipping clustering")
            return
        
        # pgvector-python automatically converts to numpy arrays
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding)
            except Exception as e:
                logger.error(f"Failed to convert embedding to numpy array for face {face_id}: {e}, type: {type(embedding)}")
                return
        
        # Validate embedding dimension
        try:
            embedding_dim = len(embedding)
            if embedding_dim != 512:
                logger.error(f"Invalid embedding dimension for face {face_id}: {embedding_dim} (expected 512)")
                return
        except TypeError as e:
            logger.error(f"Cannot get length of embedding for face {face_id}: {e}, embedding type: {type(embedding)}")
            return
        
        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            logger.error(f"Zero-norm embedding for face {face_id}")
            return

        # Find nearest clusters
        # pgvector-python handles numpy array conversion automatically
        nearest_clusters = self.repository.find_nearest_clusters(embedding, limit=10)
        
        # Filter clusters below threshold
        matching_clusters = [
            (cluster_id, distance) 
            for cluster_id, distance in nearest_clusters 
            if distance < self.clustering_threshold
        ]

        if len(matching_clusters) == 0:
            # Rule A: Create new cluster
            self._create_new_cluster(face_id, embedding)
        elif len(matching_clusters) == 1:
            # Rule B: Assign to single cluster
            cluster_id, distance = matching_clusters[0]
            confidence = 1.0 - distance
            self._assign_to_cluster(face_id, cluster_id, confidence, embedding)
        else:
            # Rule C: Multiple matches - mark for review
            self._mark_for_review(face_id, matching_clusters)

    def _create_new_cluster(self, face_id: int, embedding: np.ndarray) -> None:
        """Create a new cluster with this face as the first member."""
        # Create cluster with face as representative
        # pgvector-python handles numpy array conversion automatically
        cluster_id = self.repository.create_cluster(
            centroid=embedding,
            representative_face_id=face_id,
            medoid_face_id=face_id,
            face_count=1
        )
        
        # Assign face to cluster
        self.repository.update_face_cluster(
            face_id=face_id,
            cluster_id=cluster_id,
            cluster_confidence=Decimal("1.0"),
            cluster_status="auto"
        )
        
        logger.debug(f"Created new cluster {cluster_id} for face {face_id}")

    def _assign_to_cluster(self, face_id: int, cluster_id: int, confidence: float, 
                          embedding: np.ndarray) -> None:
        """Assign face to an existing cluster and update centroid."""
        # Use transaction to ensure consistency
        with self.repository.get_connection() as conn:
            with conn.cursor() as cur:
                # Lock cluster row
                cur.execute(
                    "SELECT face_count, centroid FROM cluster WHERE id = %s FOR UPDATE",
                    (cluster_id,)
                )
                row = cur.fetchone()
                if not row:
                    logger.error(f"Cluster {cluster_id} not found")
                    return
                
                face_count, current_centroid = row
                
                # Update centroid incrementally
                if current_centroid is not None:
                    # pgvector-python automatically converts to numpy arrays
                    if not isinstance(current_centroid, np.ndarray):
                        current_centroid = np.array(current_centroid)
                    new_centroid = (current_centroid * face_count + embedding) / (face_count + 1)
                else:
                    new_centroid = embedding
                
                # Update cluster
                cur.execute(
                    """
                    UPDATE cluster 
                    SET centroid = %s, face_count = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (new_centroid, face_count + 1, cluster_id)
                )
                
                # Update face
                cur.execute(
                    """
                    UPDATE face 
                    SET cluster_id = %s, cluster_confidence = %s, cluster_status = %s
                    WHERE id = %s
                    """,
                    (cluster_id, Decimal(str(confidence)), "auto", face_id)
                )
                
                conn.commit()
                logger.debug(f"Assigned face {face_id} to cluster {cluster_id} with confidence {confidence:.3f}")

    def _mark_for_review(self, face_id: int, matching_clusters: List[Tuple[int, float]]) -> None:
        """Mark face for manual review with multiple potential clusters."""
        # Create candidate records for each potential cluster
        candidates = [
            (face_id, cluster_id, 1.0 - distance)
            for cluster_id, distance in matching_clusters
        ]
        
        self.repository.create_face_match_candidates(candidates)
        
        # Mark face as pending review
        self.repository.update_face_cluster_status(face_id, "pending")
        
        logger.debug(f"Face {face_id} marked for review with {len(matching_clusters)} potential clusters")