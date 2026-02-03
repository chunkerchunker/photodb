import logging
from pathlib import Path
from typing import List, Optional, Set
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

    Uses PersonDetection model instead of the deprecated Face model.
    """

    stage_name = "clustering"
    force: bool = False  # Set by processor before process_photo is called

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        self.clustering_threshold = float(config.get("CLUSTERING_THRESHOLD", 0.45))
        # Pool threshold defaults to 70% of main threshold for stricter pool clustering
        self.pool_clustering_threshold = float(
            config.get("POOL_CLUSTERING_THRESHOLD", self.clustering_threshold * 0.7)
        )
        self.k_neighbors = int(config.get("CLUSTERING_K_NEIGHBORS", 5))
        self.unassigned_threshold = int(config.get("UNASSIGNED_CLUSTER_THRESHOLD", 5))
        self.verified_threshold_multiplier = float(config.get("VERIFIED_THRESHOLD_MULTIPLIER", 0.8))
        self.medoid_recompute_threshold = float(config.get("MEDOID_RECOMPUTE_THRESHOLD", 0.25))
        logger.debug(
            f"Initialized ClusteringStage with threshold={self.clustering_threshold}, "
            f"pool_threshold={self.pool_clustering_threshold}, "
            f"k_neighbors={self.k_neighbors}, unassigned_threshold={self.unassigned_threshold}, "
            f"medoid_recompute_threshold={self.medoid_recompute_threshold}"
        )

    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """Check if clustering is needed for detections in this photo."""
        if force:
            return True

        photo = self.repository.get_photo_by_filename(str(file_path))
        if not photo or photo.id is None:
            return False

        # Check if photo has been processed through detection stage
        detection_status = self.repository.get_processing_status(photo.id, "detection")
        if not detection_status or detection_status.status != "completed":
            logger.debug(f"Skipping clustering for {file_path}: detection stage not completed")
            return False

        # Check if clustering has already been done
        status = self.repository.get_processing_status(photo.id, self.stage_name)
        if status and status.status == "completed":
            return False

        # Check if there are unclustered detections for this photo
        unclustered_detections = self.repository.get_unclustered_detections_for_photo(photo.id)
        return len(unclustered_detections) > 0

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process clustering for all detections in a photo."""
        if photo.id is None:
            logger.error(f"Photo {file_path} has no ID")
            return False

        photo_id = photo.id  # Capture for type narrowing

        try:
            # Get detections to cluster based on force flag
            if self.force:
                # If force flag is set, get all detections with embeddings for reprocessing
                detections_to_cluster = (
                    self.repository.get_all_detections_with_embeddings_for_photo(photo_id)
                )
                logger.debug(
                    f"Force mode: reprocessing {len(detections_to_cluster)} detections for {file_path}"
                )
            else:
                # Normal mode: only get unclustered detections
                detections_to_cluster = self.repository.get_unclustered_detections_for_photo(
                    photo_id
                )

            if not detections_to_cluster:
                logger.debug(f"No detections found to cluster for {file_path}")
                return True  # Success - nothing to cluster

            logger.info(f"Clustering {len(detections_to_cluster)} detections from {file_path}")

            # Process each detection
            detections_processed = 0
            detections_failed = 0

            for detection in detections_to_cluster:
                try:
                    self._cluster_single_detection(detection)
                    detections_processed += 1
                except Exception as e:
                    logger.error(f"Failed to cluster detection {detection['id']}: {e}")
                    detections_failed += 1

            # Log results
            if detections_failed > 0:
                logger.warning(
                    f"Clustered {detections_processed} detections, {detections_failed} failed for {file_path}"
                )
                return False  # Partial failure
            else:
                logger.info(
                    f"Successfully clustered {detections_processed} detections from {file_path}"
                )
                return True  # Success

        except Exception as e:
            logger.error(f"Error clustering detections for {file_path}: {e}")
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

    def _cluster_single_detection(self, detection: dict) -> None:
        """
        Cluster a single detection using constrained incremental clustering.

        Decision flow:
        1. Check must-link constraints first
        2. Find K nearest neighbors
        3. Calculate core distance confidence
        4. Filter by cannot-link constraints
        5. Apply decision rules (assign, create, or mark unassigned)
        """
        detection_id = detection["id"]
        embedding = self._normalize_embedding(detection.get("embedding"))

        if embedding is None:
            logger.warning(f"No valid embedding for detection {detection_id} - skipping clustering")
            return

        # Step 1: Check must-link constraints first
        must_link_cluster = self._check_must_link_constraints(detection_id)
        if must_link_cluster is not None:
            logger.debug(f"Detection {detection_id} has must-link to cluster {must_link_cluster}")
            self._assign_to_cluster(
                detection_id,
                must_link_cluster,
                confidence=1.0,
                embedding=embedding,
                status="constrained",
            )
            return

        # Step 2: Find K nearest neighbors (detections, not just clusters)
        neighbors = self.repository.find_nearest_detections(embedding, limit=self.k_neighbors)

        # Step 3: Calculate core distance confidence
        if neighbors:
            distances = [n["distance"] for n in neighbors]
            core_distance = np.mean(distances)
            confidence = float(max(0.0, 1.0 - core_distance))
        else:
            core_distance = float("inf")
            confidence = 0.0

        # Step 4: Filter by threshold and get cluster assignments
        valid_neighbors = [n for n in neighbors if n["distance"] < self.clustering_threshold]

        if not valid_neighbors:
            # No close matches -> add to unassigned pool
            self._add_to_unassigned_pool(detection_id, embedding)
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
        allowed_clusters = self._filter_cannot_link_clusters(
            detection_id, list(neighbor_clusters.keys())
        )

        if not allowed_clusters:
            # All nearby clusters are forbidden -> unassigned
            logger.debug(f"Detection {detection_id}: all nearby clusters forbidden by constraints")
            self._add_to_unassigned_pool(detection_id, embedding)
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
                    self._assign_to_cluster(detection_id, cluster_id, confidence, embedding)
                else:
                    logger.debug(
                        f"Detection {detection_id}: too far from verified cluster {cluster_id} "
                        f"(dist={min_dist:.3f}, threshold={strict_threshold:.3f})"
                    )
                    self._add_to_unassigned_pool(detection_id, embedding)
            else:
                self._assign_to_cluster(detection_id, cluster_id, confidence, embedding)
        else:
            # Multiple valid clusters -> mark for review
            self._mark_for_review(detection_id, allowed_clusters, neighbor_clusters)

    def _check_must_link_constraints(self, detection_id: int) -> Optional[int]:
        """Check if detection has must-link constraint to an already-clustered detection."""
        linked_detections = self.repository.get_must_linked_detections(detection_id)
        for linked_detection in linked_detections:
            cluster_id = linked_detection.get("cluster_id")
            if cluster_id:
                return cluster_id
        return None

    def _filter_cannot_link_clusters(self, detection_id: int, cluster_ids: List[int]) -> List[int]:
        """Remove clusters that have cannot-link constraints with this detection."""
        if not cluster_ids:
            return []

        # Get detections this detection cannot link to
        forbidden_detections = self.repository.get_cannot_linked_detections(detection_id)
        forbidden_clusters: Set[int] = {
            d["cluster_id"] for d in forbidden_detections if d.get("cluster_id")
        }

        # Also check cluster-level constraints
        for cluster_id in cluster_ids:
            # Check if any existing detection in this cluster has cannot-link with detection_id
            detections_in_cluster = self.repository.get_detections_in_cluster(cluster_id)
            for detection_in_cluster in detections_in_cluster:
                other_detection_id = detection_in_cluster["id"]
                # Check if cannot-link exists
                cannot_linked = self.repository.get_cannot_linked_detections(other_detection_id)
                for cd in cannot_linked:
                    if cd["id"] == detection_id:
                        forbidden_clusters.add(cluster_id)
                        break

        return [cid for cid in cluster_ids if cid not in forbidden_clusters]

    def _add_to_unassigned_pool(self, detection_id: int, embedding: np.ndarray) -> None:
        """Add detection to unassigned pool, potentially forming new cluster."""
        self.repository.update_detection_unassigned(detection_id)
        logger.debug(f"Detection {detection_id} added to unassigned pool")

        # Check if enough similar unassigned detections to form cluster
        # Use stricter pool threshold to prevent chaining of dissimilar detections
        similar_unassigned = self.repository.find_similar_unassigned_detections(
            embedding, threshold=self.pool_clustering_threshold, limit=self.unassigned_threshold
        )

        logger.debug(
            f"Detection {detection_id}: found {len(similar_unassigned)} similar unassigned detections "
            f"(threshold={self.unassigned_threshold - 1})"
        )

        if (
            len(similar_unassigned) >= self.unassigned_threshold - 1
        ):  # -1 because current detection is also unassigned
            # Form new cluster from unassigned pool
            # Exclude current detection from similar_unassigned to avoid duplicates
            other_detection_ids = [d["id"] for d in similar_unassigned if d["id"] != detection_id]
            detection_ids = [detection_id] + other_detection_ids
            logger.info(
                f"Attempting to create cluster from {len(detection_ids)} detections: {detection_ids}"
            )
            cluster_id = self._create_cluster_from_pool(detection_ids)
            if cluster_id:
                logger.info(
                    f"Created cluster {cluster_id} from {len(detection_ids)} unassigned detections"
                )
            else:
                logger.warning(f"Failed to create cluster from detections {detection_ids}")

    def _create_cluster_from_pool(self, detection_ids: List[int]) -> Optional[int]:
        """Create cluster from multiple unassigned detections."""
        embeddings = []
        valid_detection_ids = []

        for did in detection_ids:
            emb = self.repository.get_detection_embedding(did)
            normalized = self._normalize_embedding(emb)
            if normalized is not None:
                embeddings.append(normalized)
                valid_detection_ids.append(did)

        if not embeddings:
            return None

        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)

        # Normalize centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Calculate distance from each detection to centroid
        distances = [np.linalg.norm(e - centroid) for e in embeddings]

        # Filter to only include detections within threshold of centroid (option 2)
        # This prevents "chaining" where A→B→C are linked but A and C are dissimilar
        filtered_detections = [
            (did, emb, dist)
            for did, emb, dist in zip(valid_detection_ids, embeddings, distances)
            if dist < self.pool_clustering_threshold
        ]

        if len(filtered_detections) < 2:
            # Not enough detections close to centroid to form a cluster
            logger.debug(
                f"Only {len(filtered_detections)} detections within threshold of centroid, "
                f"need at least 2 to form cluster"
            )
            return None

        # Recalculate centroid with only the filtered detections
        filtered_ids = [d[0] for d in filtered_detections]
        filtered_embeddings = [d[1] for d in filtered_detections]
        centroid = np.mean(filtered_embeddings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Recalculate distances and find medoid
        distances = [np.linalg.norm(e - centroid) for e in filtered_embeddings]
        medoid_idx = int(np.argmin(distances))
        medoid_detection_id = filtered_ids[medoid_idx]

        # Create cluster with initial count of 0 (will be updated as detections are assigned)
        cluster_id = self.repository.create_cluster_for_detection(
            centroid=centroid,
            representative_detection_id=medoid_detection_id,
            medoid_detection_id=medoid_detection_id,
            face_count=0,
        )

        # Assign detections with confidence based on distance to centroid (option 3)
        # Track actual assignments since some may fail due to race conditions
        assigned_count = 0
        for did, dist in zip(filtered_ids, distances):
            # Remove from old cluster if any (handles force reprocessing)
            self.repository.remove_detection_from_cluster(did, delete_empty_cluster=True)
            # Calculate confidence from distance (closer = higher confidence)
            confidence = float(max(0.0, min(1.0, 1.0 - dist)))
            # Only assign if detection is still unassigned (prevents race conditions)
            assigned = self.repository.update_detection_cluster(
                detection_id=did,
                cluster_id=cluster_id,
                cluster_confidence=confidence,
                cluster_status="auto",
            )
            logger.debug(
                f"update_detection_cluster(detection={did}, cluster={cluster_id}, "
                f"confidence={confidence:.3f}) returned {assigned}"
            )
            if assigned:
                assigned_count += 1
                self.repository.clear_detection_unassigned(did)

        # Update cluster with actual face count
        logger.debug(
            f"_create_cluster_from_pool: cluster {cluster_id}, "
            f"assigned {assigned_count}/{len(valid_detection_ids)} detections"
        )
        if assigned_count > 0:
            self.repository.update_cluster_face_count(cluster_id, assigned_count)
            return cluster_id
        else:
            # No detections were assigned (all taken by other workers), delete empty cluster
            logger.warning(f"Deleting empty cluster {cluster_id} - no detections could be assigned")
            self.repository.delete_cluster(cluster_id)
            return None

    def _create_new_cluster(self, detection_id: int, embedding: np.ndarray) -> None:
        """Create a new cluster with this detection as the first member."""
        # First, remove detection from old cluster if it has one
        self.repository.remove_detection_from_cluster(detection_id, delete_empty_cluster=True)

        # Create new cluster with count 0 initially
        cluster_id = self.repository.create_cluster_for_detection(
            centroid=embedding,
            representative_detection_id=detection_id,
            medoid_detection_id=detection_id,
            face_count=0,
        )

        # Assign detection to cluster (only succeeds if detection is still unassigned)
        if self.repository.update_detection_cluster(
            detection_id=detection_id,
            cluster_id=cluster_id,
            cluster_confidence=1.0,
            cluster_status="auto",
        ):
            self.repository.update_cluster_face_count(cluster_id, 1)
        else:
            # Detection was taken by another worker, delete empty cluster
            self.repository.delete_cluster(cluster_id)
            logger.debug(
                f"Detection {detection_id} already assigned, deleted empty cluster {cluster_id}"
            )

        logger.debug(f"Created new cluster {cluster_id} for detection {detection_id}")

    def _assign_to_cluster(
        self,
        detection_id: int,
        cluster_id: int,
        confidence: float,
        embedding: np.ndarray,
        status: str = "auto",
    ) -> None:
        """Assign detection to an existing cluster and update centroid."""
        # Use transaction to ensure consistency (auto-commits on successful exit)
        with self.repository.pool.transaction() as conn:
            with conn.cursor() as cur:
                # First, check if detection is already in a different cluster
                cur.execute(
                    "SELECT cluster_id FROM person_detection WHERE id = %s FOR UPDATE",
                    (detection_id,),
                )
                detection_row = cur.fetchone()
                old_cluster_id = detection_row[0] if detection_row else None

                # If detection is already in this cluster, nothing to do
                if old_cluster_id == cluster_id:
                    logger.debug(f"Detection {detection_id} already in cluster {cluster_id}")
                    return

                # If detection was in a different cluster, decrement that cluster's count
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

                # Update detection
                cur.execute(
                    """UPDATE person_detection
                       SET cluster_id = %s, cluster_confidence = %s, cluster_status = %s
                       WHERE id = %s""",
                    (cluster_id, Decimal(str(confidence)), status, detection_id),
                )

                # Check if medoid needs recomputation
                if self._should_recompute_medoid(new_face_count, face_count_at_last_medoid):
                    self._recompute_medoid(cluster_id, new_centroid, cur)

                # Transaction auto-commits on successful exit
                logger.debug(
                    f"Assigned detection {detection_id} to cluster {cluster_id} "
                    f"with confidence {confidence:.3f} (status={status})"
                )

    def _mark_for_review(
        self, detection_id: int, cluster_ids: List[int], neighbor_clusters: dict
    ) -> None:
        """Mark detection for manual review with multiple potential clusters."""
        # Create candidate records for each potential cluster
        candidates = []
        for cluster_id in cluster_ids:
            if cluster_id in neighbor_clusters:
                min_distance = min(neighbor_clusters[cluster_id])
                similarity = 1.0 - min_distance
                candidates.append((detection_id, cluster_id, similarity))

        self.repository.create_detection_match_candidates(candidates)

        # Mark detection as pending review
        self.repository.update_detection_cluster_status(detection_id, "pending")

        logger.debug(
            f"Detection {detection_id} marked for review with {len(cluster_ids)} potential clusters"
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
        # Get all detections in this cluster with embeddings
        cur.execute(
            """
            SELECT pd.id, fe.embedding
            FROM person_detection pd
            JOIN face_embedding fe ON pd.id = fe.person_detection_id
            WHERE pd.cluster_id = %s
        """,
            (cluster_id,),
        )

        rows = cur.fetchall()
        if not rows:
            return

        detection_ids = [row[0] for row in rows]
        embeddings = np.array([row[1] for row in rows])

        # Find detection closest to centroid (medoid)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        medoid_idx = int(np.argmin(distances))
        medoid_detection_id = detection_ids[medoid_idx]

        # Update cluster with new medoid and reset tracking counter
        # Note: Only update representative_detection_id if it's NULL (not user-set)
        cur.execute(
            """
            UPDATE cluster
            SET medoid_detection_id = %s,
                representative_detection_id = COALESCE(representative_detection_id, %s),
                face_count_at_last_medoid = face_count,
                updated_at = NOW()
            WHERE id = %s
        """,
            (medoid_detection_id, medoid_detection_id, cluster_id),
        )

        logger.debug(f"Recomputed medoid for cluster {cluster_id}: detection {medoid_detection_id}")
