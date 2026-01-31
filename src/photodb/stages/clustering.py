import logging
from pathlib import Path
from typing import List, Tuple, Optional, Set
import numpy as np
from decimal import Decimal

from .base import BaseStage
from ..database.models import Photo

logger = logging.getLogger(__name__)


class ClusteringStage(BaseStage):
    """
    Constrained incremental clustering stage for face embeddings.

    Implements the alternative clustering approach with:
    - Must-link/cannot-link constraint support
    - K-neighbor based confidence scoring
    - Unassigned pool for outliers
    - Verified cluster protection
    """

    stage_name = "clustering"

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        self.clustering_threshold = float(config.get("CLUSTERING_THRESHOLD", 0.45))
        self.k_neighbors = int(config.get("CLUSTERING_K_NEIGHBORS", 5))
        self.unassigned_threshold = int(config.get("UNASSIGNED_CLUSTER_THRESHOLD", 5))
        self.verified_threshold_multiplier = float(
            config.get("VERIFIED_THRESHOLD_MULTIPLIER", 0.8)
        )
        self.medoid_recompute_threshold = float(
            config.get("MEDOID_RECOMPUTE_THRESHOLD", 0.25)
        )
        logger.debug(
            f"Initialized ClusteringStage with threshold={self.clustering_threshold}, "
            f"k_neighbors={self.k_neighbors}, unassigned_threshold={self.unassigned_threshold}, "
            f"medoid_recompute_threshold={self.medoid_recompute_threshold}"
        )

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
            if hasattr(self, "force") and self.force:
                # If force flag is set, get all faces with embeddings for reprocessing
                faces_to_cluster = self.repository.get_all_faces_with_embeddings_for_photo(
                    photo.id
                )
                logger.debug(
                    f"Force mode: reprocessing {len(faces_to_cluster)} faces for {file_path}"
                )
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
                logger.warning(
                    f"Clustered {faces_processed} faces, {faces_failed} failed for {file_path}"
                )
                return False  # Partial failure
            else:
                logger.info(f"Successfully clustered {faces_processed} faces from {file_path}")
                return True  # Success

        except Exception as e:
            logger.error(f"Error clustering faces for {file_path}: {e}")
            raise

    def _normalize_embedding(self, embedding) -> Optional[np.ndarray]:
        """Normalize embedding to unit vector for cosine similarity."""
        if embedding is None:
            return None

        # pgvector-python automatically converts to numpy arrays
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding)
            except Exception as e:
                logger.error(f"Failed to convert embedding to numpy array: {e}")
                return None

        # Validate dimension
        if len(embedding) != 512:
            logger.error(f"Invalid embedding dimension: {len(embedding)} (expected 512)")
            return None

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        else:
            logger.error("Zero-norm embedding")
            return None

    def _cluster_single_face(self, face: dict) -> None:
        """
        Cluster a single face using constrained incremental clustering.

        Decision flow:
        1. Check must-link constraints first
        2. Find K nearest neighbors
        3. Calculate core distance confidence
        4. Filter by cannot-link constraints
        5. Apply decision rules (assign, create, or mark unassigned)
        """
        face_id = face["id"]
        embedding = self._normalize_embedding(face.get("embedding"))

        if embedding is None:
            logger.warning(f"No valid embedding for face {face_id} - skipping clustering")
            return

        # Step 1: Check must-link constraints first
        must_link_cluster = self._check_must_link_constraints(face_id)
        if must_link_cluster is not None:
            logger.debug(f"Face {face_id} has must-link to cluster {must_link_cluster}")
            self._assign_to_cluster(
                face_id, must_link_cluster, confidence=1.0, embedding=embedding, status="constrained"
            )
            return

        # Step 2: Find K nearest neighbors (faces, not just clusters)
        neighbors = self.repository.find_nearest_faces(embedding, limit=self.k_neighbors)

        # Step 3: Calculate core distance confidence
        if neighbors:
            distances = [n["distance"] for n in neighbors]
            core_distance = np.mean(distances)
            confidence = max(0.0, 1.0 - core_distance)
        else:
            core_distance = float("inf")
            confidence = 0.0

        # Step 4: Filter by threshold and get cluster assignments
        valid_neighbors = [n for n in neighbors if n["distance"] < self.clustering_threshold]

        if not valid_neighbors:
            # No close matches -> add to unassigned pool
            self._add_to_unassigned_pool(face_id, embedding)
            return

        # Step 5: Get unique clusters from neighbors
        neighbor_clusters: dict = {}
        for n in valid_neighbors:
            cid = n.get("cluster_id")
            if cid:
                if cid not in neighbor_clusters:
                    neighbor_clusters[cid] = []
                neighbor_clusters[cid].append(n["distance"])

        # Step 6: Filter out cannot-link clusters
        allowed_clusters = self._filter_cannot_link_clusters(face_id, list(neighbor_clusters.keys()))

        if not allowed_clusters:
            # All nearby clusters are forbidden -> unassigned
            logger.debug(f"Face {face_id}: all nearby clusters forbidden by constraints")
            self._add_to_unassigned_pool(face_id, embedding)
            return

        # Step 7: Apply decision rules
        if len(allowed_clusters) == 1:
            # Single valid cluster
            cluster_id = allowed_clusters[0]
            cluster = self.repository.get_cluster_by_id(cluster_id)

            if cluster and cluster.verified:
                # Verified cluster: use stricter threshold
                min_dist = min(neighbor_clusters[cluster_id])
                strict_threshold = self.clustering_threshold * self.verified_threshold_multiplier
                if min_dist < strict_threshold:
                    self._assign_to_cluster(face_id, cluster_id, confidence, embedding)
                else:
                    logger.debug(
                        f"Face {face_id}: too far from verified cluster {cluster_id} "
                        f"(dist={min_dist:.3f}, threshold={strict_threshold:.3f})"
                    )
                    self._add_to_unassigned_pool(face_id, embedding)
            else:
                self._assign_to_cluster(face_id, cluster_id, confidence, embedding)
        else:
            # Multiple valid clusters -> mark for review
            self._mark_for_review(face_id, allowed_clusters, neighbor_clusters)

    def _check_must_link_constraints(self, face_id: int) -> Optional[int]:
        """Check if face has must-link constraint to an already-clustered face."""
        linked_faces = self.repository.get_must_linked_faces(face_id)
        for linked_face in linked_faces:
            cluster_id = linked_face.get("cluster_id")
            if cluster_id:
                return cluster_id
        return None

    def _filter_cannot_link_clusters(
        self, face_id: int, cluster_ids: List[int]
    ) -> List[int]:
        """Remove clusters that have cannot-link constraints with this face."""
        if not cluster_ids:
            return []

        # Get faces this face cannot link to
        forbidden_faces = self.repository.get_cannot_linked_faces(face_id)
        forbidden_clusters: Set[int] = {
            f["cluster_id"] for f in forbidden_faces if f.get("cluster_id")
        }

        # Also check cluster-level constraints
        for cluster_id in cluster_ids:
            # Check if any existing face in this cluster has cannot-link with face_id
            faces_in_cluster = self.repository.get_faces_in_cluster(cluster_id)
            for face_in_cluster in faces_in_cluster:
                other_face_id = face_in_cluster["id"]
                # Check if cannot-link exists
                cannot_linked = self.repository.get_cannot_linked_faces(other_face_id)
                for cf in cannot_linked:
                    if cf["id"] == face_id:
                        forbidden_clusters.add(cluster_id)
                        break

        return [cid for cid in cluster_ids if cid not in forbidden_clusters]

    def _add_to_unassigned_pool(self, face_id: int, embedding: np.ndarray) -> None:
        """Add face to unassigned pool, potentially forming new cluster."""
        self.repository.update_face_unassigned(face_id)
        logger.debug(f"Face {face_id} added to unassigned pool")

        # Check if enough similar unassigned faces to form cluster
        similar_unassigned = self.repository.find_similar_unassigned_faces(
            embedding, threshold=self.clustering_threshold, limit=self.unassigned_threshold
        )

        logger.debug(
            f"Face {face_id}: found {len(similar_unassigned)} similar unassigned faces "
            f"(threshold={self.unassigned_threshold - 1})"
        )

        if len(similar_unassigned) >= self.unassigned_threshold - 1:  # -1 because current face is also unassigned
            # Form new cluster from unassigned pool
            # Exclude current face from similar_unassigned to avoid duplicates
            other_face_ids = [f["id"] for f in similar_unassigned if f["id"] != face_id]
            face_ids = [face_id] + other_face_ids
            logger.info(f"Attempting to create cluster from {len(face_ids)} faces: {face_ids}")
            cluster_id = self._create_cluster_from_pool(face_ids)
            if cluster_id:
                logger.info(f"Created cluster {cluster_id} from {len(face_ids)} unassigned faces")
            else:
                logger.warning(f"Failed to create cluster from faces {face_ids}")

    def _create_cluster_from_pool(self, face_ids: List[int]) -> Optional[int]:
        """Create cluster from multiple unassigned faces."""
        embeddings = []
        valid_face_ids = []

        for fid in face_ids:
            emb = self.repository.get_face_embedding(fid)
            normalized = self._normalize_embedding(emb)
            if normalized is not None:
                embeddings.append(normalized)
                valid_face_ids.append(fid)

        if not embeddings:
            return None

        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)

        # Normalize centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Find medoid (face closest to centroid)
        distances = [np.linalg.norm(e - centroid) for e in embeddings]
        medoid_idx = int(np.argmin(distances))
        medoid_face_id = valid_face_ids[medoid_idx]

        # Create cluster with initial count of 0 (will be updated as faces are assigned)
        cluster_id = self.repository.create_cluster(
            centroid=centroid,
            representative_face_id=medoid_face_id,
            medoid_face_id=medoid_face_id,
            face_count=0,
        )

        # Assign all faces with lower confidence (pool-formed clusters)
        # Track actual assignments since some may fail due to race conditions
        assigned_count = 0
        for fid in valid_face_ids:
            # Remove from old cluster if any (handles force reprocessing)
            self.repository.remove_face_from_cluster(fid, delete_empty_cluster=True)
            # Only assign if face is still unassigned (prevents race conditions)
            assigned = self.repository.update_face_cluster(
                face_id=fid,
                cluster_id=cluster_id,
                cluster_confidence=0.8,  # Lower confidence for pool-formed clusters
                cluster_status="auto",
            )
            logger.debug(f"update_face_cluster(face={fid}, cluster={cluster_id}) returned {assigned}")
            if assigned:
                assigned_count += 1
                self.repository.clear_face_unassigned(fid)

        # Update cluster with actual face count
        logger.debug(
            f"_create_cluster_from_pool: cluster {cluster_id}, "
            f"assigned {assigned_count}/{len(valid_face_ids)} faces"
        )
        if assigned_count > 0:
            self.repository.update_cluster_face_count(cluster_id, assigned_count)
            return cluster_id
        else:
            # No faces were assigned (all taken by other workers), delete empty cluster
            logger.warning(f"Deleting empty cluster {cluster_id} - no faces could be assigned")
            self.repository.delete_cluster(cluster_id)
            return None

    def _create_new_cluster(self, face_id: int, embedding: np.ndarray) -> None:
        """Create a new cluster with this face as the first member."""
        # First, remove face from old cluster if it has one
        self.repository.remove_face_from_cluster(face_id, delete_empty_cluster=True)

        # Create new cluster with count 0 initially
        cluster_id = self.repository.create_cluster(
            centroid=embedding,
            representative_face_id=face_id,
            medoid_face_id=face_id,
            face_count=0,
        )

        # Assign face to cluster (only succeeds if face is still unassigned)
        if self.repository.update_face_cluster(
            face_id=face_id,
            cluster_id=cluster_id,
            cluster_confidence=1.0,
            cluster_status="auto",
        ):
            self.repository.update_cluster_face_count(cluster_id, 1)
        else:
            # Face was taken by another worker, delete empty cluster
            self.repository.delete_cluster(cluster_id)
            logger.debug(f"Face {face_id} already assigned, deleted empty cluster {cluster_id}")

        logger.debug(f"Created new cluster {cluster_id} for face {face_id}")

    def _assign_to_cluster(
        self,
        face_id: int,
        cluster_id: int,
        confidence: float,
        embedding: np.ndarray,
        status: str = "auto",
    ) -> None:
        """Assign face to an existing cluster and update centroid."""
        # Use transaction to ensure consistency (auto-commits on successful exit)
        with self.repository.pool.transaction() as conn:
            with conn.cursor() as cur:
                # First, check if face is already in a different cluster
                cur.execute(
                    "SELECT cluster_id FROM face WHERE id = %s FOR UPDATE",
                    (face_id,),
                )
                face_row = cur.fetchone()
                old_cluster_id = face_row[0] if face_row else None

                # If face is already in this cluster, nothing to do
                if old_cluster_id == cluster_id:
                    logger.debug(f"Face {face_id} already in cluster {cluster_id}")
                    return

                # If face was in a different cluster, decrement that cluster's count
                if old_cluster_id is not None:
                    cur.execute(
                        """UPDATE cluster
                           SET face_count = GREATEST(0, face_count - 1),
                               updated_at = NOW()
                           WHERE id = %s""",
                        (old_cluster_id,),
                    )
                    # Check if old cluster is now empty and delete it
                    cur.execute(
                        "SELECT face_count FROM cluster WHERE id = %s",
                        (old_cluster_id,),
                    )
                    count_row = cur.fetchone()
                    if count_row and count_row[0] == 0:
                        cur.execute("DELETE FROM cluster WHERE id = %s", (old_cluster_id,))
                        logger.info(f"Deleted empty cluster {old_cluster_id}")

                # Lock new cluster row
                cur.execute(
                    "SELECT face_count, centroid, face_count_at_last_medoid FROM cluster WHERE id = %s FOR UPDATE",
                    (cluster_id,),
                )
                row = cur.fetchone()
                if not row:
                    logger.error(f"Cluster {cluster_id} not found")
                    return

                face_count, current_centroid, face_count_at_last_medoid = row
                face_count_at_last_medoid = face_count_at_last_medoid or 0

                # Update centroid incrementally
                if current_centroid is not None:
                    if not isinstance(current_centroid, np.ndarray):
                        current_centroid = np.array(current_centroid)
                    new_centroid = (current_centroid * face_count + embedding) / (face_count + 1)
                else:
                    new_centroid = embedding

                new_face_count = face_count + 1

                # Update cluster
                cur.execute(
                    """UPDATE cluster
                       SET centroid = %s, face_count = %s, updated_at = NOW()
                       WHERE id = %s""",
                    (new_centroid, new_face_count, cluster_id),
                )

                # Update face
                cur.execute(
                    """UPDATE face
                       SET cluster_id = %s, cluster_confidence = %s, cluster_status = %s,
                           unassigned_since = NULL
                       WHERE id = %s""",
                    (cluster_id, Decimal(str(confidence)), status, face_id),
                )

                # Check if medoid needs recomputation
                if self._should_recompute_medoid(new_face_count, face_count_at_last_medoid):
                    self._recompute_medoid(cluster_id, new_centroid, cur)

                # Transaction auto-commits on successful exit
                logger.debug(
                    f"Assigned face {face_id} to cluster {cluster_id} "
                    f"with confidence {confidence:.3f} (status={status})"
                )

    def _mark_for_review(
        self, face_id: int, cluster_ids: List[int], neighbor_clusters: dict
    ) -> None:
        """Mark face for manual review with multiple potential clusters."""
        # Create candidate records for each potential cluster
        candidates = []
        for cluster_id in cluster_ids:
            if cluster_id in neighbor_clusters:
                min_distance = min(neighbor_clusters[cluster_id])
                similarity = 1.0 - min_distance
                candidates.append((face_id, cluster_id, similarity))

        self.repository.create_face_match_candidates(candidates)

        # Mark face as pending review
        self.repository.update_face_cluster_status(face_id, "pending")

        logger.debug(
            f"Face {face_id} marked for review with {len(cluster_ids)} potential clusters"
        )

    def _should_recompute_medoid(self, face_count: int, face_count_at_last_medoid: int) -> bool:
        """Check if medoid should be recomputed based on growth threshold."""
        if face_count_at_last_medoid == 0:
            # Never computed, but will be set on cluster creation
            return False
        growth_ratio = (face_count - face_count_at_last_medoid) / face_count_at_last_medoid
        return growth_ratio >= self.medoid_recompute_threshold

    def _recompute_medoid(self, cluster_id: int, centroid: np.ndarray, cur) -> None:
        """Recompute and update medoid for a cluster inline."""
        # Get all faces in this cluster with embeddings
        cur.execute("""
            SELECT f.id, fe.embedding
            FROM face f
            JOIN face_embedding fe ON f.id = fe.face_id
            WHERE f.cluster_id = %s
        """, (cluster_id,))

        rows = cur.fetchall()
        if not rows:
            return

        face_ids = [row[0] for row in rows]
        embeddings = np.array([row[1] for row in rows])

        # Find face closest to centroid (medoid)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        medoid_idx = int(np.argmin(distances))
        medoid_face_id = face_ids[medoid_idx]

        # Update cluster with new medoid and reset tracking counter
        cur.execute("""
            UPDATE cluster
            SET medoid_face_id = %s,
                representative_face_id = %s,
                face_count_at_last_medoid = face_count,
                updated_at = NOW()
            WHERE id = %s
        """, (medoid_face_id, medoid_face_id, cluster_id))

        logger.debug(f"Recomputed medoid for cluster {cluster_id}: face {medoid_face_id}")
