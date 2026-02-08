"""
Maintenance utilities for periodic PhotoDB tasks.

This module provides utility methods for periodic maintenance tasks
that should be run on schedules to keep the database optimized.

Includes support for constrained clustering:
- Constraint violation detection
- Verified cluster protection
"""

import logging
from typing import Dict, Any, List
import numpy as np

from ..database.connection import ConnectionPool
from ..database.repository import PhotoRepository, MIN_FACE_SIZE_PX, MIN_FACE_CONFIDENCE
from .hdbscan_config import (
    create_hdbscan_clusterer,
    calculate_cluster_epsilon,
    DEFAULT_CORE_PROBABILITY_THRESHOLD,
    DEFAULT_CLUSTERING_THRESHOLD,
)

logger = logging.getLogger(__name__)


class MaintenanceUtilities:
    """Utilities for periodic maintenance tasks."""

    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.repo = PhotoRepository(connection_pool)

    def recompute_all_centroids(self) -> int:
        """
        Recompute centroids for all clusters based on current member faces.

        This should be run daily to correct for centroid drift as faces
        are added or removed from clusters.

        Returns:
            Number of clusters updated
        """
        logger.info("Starting centroid recomputation for all clusters")
        updated_count = 0

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Get all cluster IDs
                cursor.execute("SELECT id FROM cluster WHERE id > 0")
                cluster_ids = [row[0] for row in cursor.fetchall()]

                for cluster_id in cluster_ids:
                    # Compute average embedding for all faces in cluster
                    cursor.execute(
                        """
                        UPDATE cluster
                        SET centroid = (
                            SELECT AVG(fe.embedding)::vector(512)
                            FROM person_detection pd
                            JOIN face_embedding fe ON pd.id = fe.person_detection_id
                            WHERE pd.cluster_id = cluster.id
                        ),
                        updated_at = NOW()
                        WHERE id = %s
                          AND EXISTS (
                            SELECT 1 FROM person_detection pd
                            JOIN face_embedding fe ON pd.id = fe.person_detection_id
                            WHERE pd.cluster_id = %s
                          )
                    """,
                        (cluster_id, cluster_id),
                    )

                    if cursor.rowcount > 0:
                        updated_count += 1

        logger.info(f"Recomputed centroids for {updated_count} clusters")
        return updated_count

    def update_all_medoids(self) -> int:
        """
        Update medoid and representative face for all clusters.

        The medoid is the face closest to the cluster centroid, which
        serves as the best visual representation of the cluster.

        This should be run weekly.

        Returns:
            Number of clusters updated
        """
        logger.info("Starting medoid update for all clusters")
        updated_count = 0

        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get all clusters with centroids
                cursor.execute("""
                    SELECT id, centroid 
                    FROM cluster 
                    WHERE centroid IS NOT NULL
                """)
                clusters = cursor.fetchall()

                for cluster_id, centroid in clusters:
                    if self._update_cluster_medoid(cluster_id, centroid):
                        updated_count += 1

        logger.info(f"Updated medoids for {updated_count} clusters")
        return updated_count

    def _update_cluster_medoid(self, cluster_id: int, centroid) -> bool:
        """
        Update medoid for a single cluster.

        Args:
            cluster_id: ID of the cluster to update
            centroid: Current centroid embedding

        Returns:
            True if cluster was updated
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Get all faces in this cluster with embeddings
                cursor.execute(
                    """
                    SELECT pd.id, fe.embedding
                    FROM person_detection pd
                    JOIN face_embedding fe ON pd.id = fe.person_detection_id
                    WHERE pd.cluster_id = %s
                """,
                    (cluster_id,),
                )

                rows = cursor.fetchall()
                if not rows:
                    return False

                detection_ids = [row[0] for row in rows]
                embeddings = np.array([row[1] for row in rows])

                # Convert centroid to numpy array if needed
                if not isinstance(centroid, np.ndarray):
                    centroid = np.array(centroid)

                # Find face closest to centroid (medoid)
                distances = np.linalg.norm(embeddings - centroid, axis=1)
                medoid_idx = np.argmin(distances)
                medoid_detection_id = detection_ids[medoid_idx]

                # Update cluster with medoid and reset tracking counter
                # Note: Only update representative_detection_id if it's NULL (not user-set)
                cursor.execute(
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

                return cursor.rowcount > 0

    def cleanup_empty_clusters(self) -> int:
        """
        Remove clusters that have no assigned faces.

        This can happen when faces are reassigned or deleted.
        Should be run daily.

        Returns:
            Number of empty clusters removed
        """
        logger.info("Cleaning up empty clusters")

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Delete empty clusters
                cursor.execute("""
                    DELETE FROM cluster
                    WHERE id NOT IN (
                        SELECT DISTINCT cluster_id
                        FROM person_detection
                        WHERE cluster_id IS NOT NULL
                    )
                """)

                deleted_count = cursor.rowcount

        logger.info(f"Removed {deleted_count} empty clusters")
        return deleted_count

    def update_cluster_statistics(self) -> int:
        """
        Update face_count and other statistics for all clusters.

        This ensures cluster metadata is accurate.
        Should be run daily.

        Returns:
            Number of clusters updated
        """
        logger.info("Updating cluster statistics")

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE cluster
                    SET face_count = (
                        SELECT COUNT(*)
                        FROM person_detection
                        WHERE person_detection.cluster_id = cluster.id
                    ),
                    updated_at = NOW()
                    WHERE id IN (
                        SELECT DISTINCT cluster_id
                        FROM person_detection
                        WHERE cluster_id IS NOT NULL
                    )
                """)

                updated_count = cursor.rowcount

        logger.info(f"Updated statistics for {updated_count} clusters")
        return updated_count

    def get_cluster_health_stats(self) -> Dict[str, Any]:
        """
        Get health statistics for the clustering system.

        Returns:
            Dictionary with various health metrics
        """
        stats = {}

        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # Total clusters
                cursor.execute("SELECT COUNT(*) FROM cluster")
                stats["total_clusters"] = cursor.fetchone()[0]

                # Clusters without centroids
                cursor.execute("SELECT COUNT(*) FROM cluster WHERE centroid IS NULL")
                stats["clusters_without_centroids"] = cursor.fetchone()[0]

                # Clusters without medoids
                cursor.execute("SELECT COUNT(*) FROM cluster WHERE medoid_detection_id IS NULL")
                stats["clusters_without_medoids"] = cursor.fetchone()[0]

                # Empty clusters
                cursor.execute("""
                    SELECT COUNT(*) FROM cluster
                    WHERE id NOT IN (
                        SELECT DISTINCT cluster_id
                        FROM person_detection
                        WHERE cluster_id IS NOT NULL
                    )
                """)
                stats["empty_clusters"] = cursor.fetchone()[0]

                # Average cluster size
                cursor.execute("""
                    SELECT AVG(face_count), MIN(face_count), MAX(face_count)
                    FROM cluster
                    WHERE face_count > 0
                """)
                avg_size, min_size, max_size = cursor.fetchone()
                stats["avg_cluster_size"] = float(avg_size) if avg_size else 0
                stats["min_cluster_size"] = min_size or 0
                stats["max_cluster_size"] = max_size or 0

                # Unclustered faces
                cursor.execute(
                    "SELECT COUNT(*) FROM person_detection WHERE cluster_id IS NULL AND face_bbox_x IS NOT NULL"
                )
                stats["unclustered_faces"] = cursor.fetchone()[0]

                # Total faces
                cursor.execute(
                    "SELECT COUNT(*) FROM person_detection WHERE face_bbox_x IS NOT NULL"
                )
                stats["total_faces"] = cursor.fetchone()[0]

                # Constraint stats
                cursor.execute("SELECT COUNT(*) FROM cannot_link")
                stats["cannot_link_count"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cluster WHERE verified = true")
                stats["verified_clusters"] = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM person_detection WHERE cluster_status = 'unassigned'"
                )
                stats["unassigned_pool_size"] = cursor.fetchone()[0]

                # Multi-cluster person stats
                cursor.execute("""
                    SELECT COUNT(*) FROM (
                        SELECT person_id, COUNT(*) as cluster_count
                        FROM cluster
                        WHERE person_id IS NOT NULL
                        GROUP BY person_id
                        HAVING COUNT(*) > 1
                    ) multi_cluster_persons
                """)
                stats["persons_with_multiple_clusters"] = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(*) FROM cluster WHERE person_id IS NOT NULL
                """)
                stats["clusters_with_person"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM person")
                stats["total_persons"] = cursor.fetchone()[0]

        return stats

    def find_constraint_violations(self) -> List[Dict[str, Any]]:
        """
        Find clusters that violate cannot-link constraints.

        This finds cases where two faces in the same cluster have a
        cannot-link constraint between them.

        Returns:
            List of violations with constraint_id, cluster_id, face_1, face_2
        """
        logger.info("Checking for constraint violations")

        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT cl.id as constraint_id,
                           pd1.cluster_id,
                           cl.detection_id_1,
                           cl.detection_id_2
                    FROM cannot_link cl
                    JOIN person_detection pd1 ON cl.detection_id_1 = pd1.id
                    JOIN person_detection pd2 ON cl.detection_id_2 = pd2.id
                    WHERE pd1.cluster_id = pd2.cluster_id
                      AND pd1.cluster_id IS NOT NULL
                """)

                violations = [
                    {
                        "constraint_id": row[0],
                        "cluster_id": row[1],
                        "detection_1": row[2],
                        "detection_2": row[3],
                    }
                    for row in cursor.fetchall()
                ]

        if violations:
            logger.warning(f"Found {len(violations)} constraint violations")
        else:
            logger.info("No constraint violations found")

        return violations

    def cleanup_unassigned_pool(self, max_age_days: int = 30) -> int:
        """
        Create singleton clusters for old unassigned faces.

        Faces that have been in the unassigned pool for too long
        without finding similar faces get their own cluster.

        Args:
            max_age_days: Maximum days in unassigned pool before creating singleton

        Returns:
            Number of singleton clusters created
        """
        logger.info(f"Cleaning up unassigned faces older than {max_age_days} days")
        created = 0

        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pd.id, fe.embedding
                    FROM person_detection pd
                    JOIN face_embedding fe ON pd.id = fe.person_detection_id
                    WHERE pd.cluster_status = 'unassigned'
                      AND pd.unassigned_since < NOW() - INTERVAL '%s days'
                """,
                    (max_age_days,),
                )

                old_faces = cursor.fetchall()

        for detection_id, embedding in old_faces:
            if embedding is not None:
                # Normalize embedding
                emb_array = np.array(embedding)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm

                # Create singleton cluster
                cluster_id = self.repo.create_cluster_for_detection(
                    centroid=emb_array,
                    representative_detection_id=detection_id,
                    medoid_detection_id=detection_id,
                    face_count=0,
                )

                # Only succeeds if face is still unassigned
                if self.repo.update_detection_cluster(
                    detection_id=detection_id,
                    cluster_id=cluster_id,
                    cluster_confidence=0.5,  # Low confidence for singleton
                    cluster_status="auto",
                ):
                    self.repo.update_cluster_face_count(cluster_id, 1)
                    self.repo.clear_detection_unassigned(detection_id)
                    created += 1
                    logger.debug(
                        f"Created singleton cluster {cluster_id} for detection {detection_id}"
                    )
                else:
                    # Face was assigned elsewhere, delete empty cluster
                    self.repo.delete_cluster(cluster_id)

        logger.info(f"Created {created} singleton clusters from old unassigned faces")
        return created

    def cluster_unassigned_pool(self, min_cluster_size: int = 3) -> int:
        """
        Run HDBSCAN clustering on the unassigned pool to find new clusters.

        This finds faces in the unassigned pool that can form new clusters
        based on density-based clustering.

        Args:
            min_cluster_size: Minimum faces to form a cluster (default 3)

        Returns:
            Number of new clusters created
        """
        logger.info("Clustering unassigned pool with HDBSCAN")

        # Get all unassigned detections with embeddings (filtered by size/confidence)
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT pd.id, fe.embedding
                       FROM person_detection pd
                       JOIN face_embedding fe ON pd.id = fe.person_detection_id
                       WHERE pd.cluster_id IS NULL
                         AND pd.cluster_status = 'unassigned'
                         AND pd.face_confidence >= %s
                         AND pd.face_bbox_width >= %s
                         AND pd.face_bbox_height >= %s""",
                    (MIN_FACE_CONFIDENCE, MIN_FACE_SIZE_PX, MIN_FACE_SIZE_PX),
                )
                rows = cursor.fetchall()

        if len(rows) < min_cluster_size:
            logger.info(
                f"Not enough unassigned detections to cluster ({len(rows)} < {min_cluster_size})"
            )
            return 0

        # Extract detection IDs and embeddings
        detection_ids = []
        embeddings_list = []
        for detection_id, embedding in rows:
            if embedding is not None:
                # Normalize embedding
                emb_array = np.array(embedding)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                    detection_ids.append(detection_id)
                    embeddings_list.append(emb_array)

        if len(detection_ids) < min_cluster_size:
            logger.info(
                f"Not enough valid embeddings to cluster "
                f"({len(detection_ids)} < {min_cluster_size})"
            )
            return 0

        embeddings = np.array(embeddings_list)

        logger.info(
            f"Running HDBSCAN on {len(detection_ids)} unassigned detections "
            f"(min_cluster_size={min_cluster_size})"
        )

        # Run HDBSCAN using shared configuration
        clusterer = create_hdbscan_clusterer(min_cluster_size=min_cluster_size)
        clusterer.fit(embeddings)

        labels = clusterer.labels_
        probabilities = clusterer.probabilities_

        # Count clusters found
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = int(np.sum(labels == -1))

        logger.info(
            f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points in unassigned pool"
        )

        if n_clusters == 0:
            return 0

        # Group detections by label
        label_to_detections: Dict[int, List[int]] = {}
        label_to_embeddings: Dict[int, List[np.ndarray]] = {}
        label_to_probabilities: Dict[int, List[float]] = {}

        for i, detection_id in enumerate(detection_ids):
            label = int(labels[i])
            if label == -1:
                continue  # Skip noise points

            if label not in label_to_detections:
                label_to_detections[label] = []
                label_to_embeddings[label] = []
                label_to_probabilities[label] = []

            label_to_detections[label].append(detection_id)
            label_to_embeddings[label].append(embeddings[i])
            label_to_probabilities[label].append(float(probabilities[i]))

        # Create clusters
        clusters_created = 0

        for label, cluster_detection_ids in label_to_detections.items():
            cluster_embeddings = np.array(label_to_embeddings[label])
            cluster_probabilities = label_to_probabilities[label]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm

            # Calculate epsilon using shared function
            epsilon = calculate_cluster_epsilon(
                embeddings, labels, label, fallback_threshold=DEFAULT_CLUSTERING_THRESHOLD
            )

            # Find medoid (closest to centroid)
            distances_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            medoid_idx = int(np.argmin(distances_to_centroid))
            medoid_detection_id = cluster_detection_ids[medoid_idx]

            # Identify core points
            core_detection_ids = [
                did
                for did, prob in zip(cluster_detection_ids, cluster_probabilities)
                if prob >= DEFAULT_CORE_PROBABILITY_THRESHOLD
            ]

            # Create cluster with epsilon
            cluster_id = self.repo.create_cluster_with_epsilon(
                centroid=centroid,
                representative_detection_id=medoid_detection_id,
                medoid_detection_id=medoid_detection_id,
                face_count=len(cluster_detection_ids),
                epsilon=epsilon,
                core_count=len(core_detection_ids),
            )

            logger.info(
                f"Created cluster {cluster_id} from unassigned pool with "
                f"{len(cluster_detection_ids)} detections, epsilon={epsilon:.4f}, "
                f"{len(core_detection_ids)} core points"
            )

            # Assign detections to cluster
            for detection_id, prob in zip(cluster_detection_ids, cluster_probabilities):
                is_core = prob >= DEFAULT_CORE_PROBABILITY_THRESHOLD
                status = "hdbscan_core" if is_core else "hdbscan"

                # Force update (we're in maintenance, so no race conditions)
                self.repo.force_update_detection_cluster(
                    detection_id=detection_id,
                    cluster_id=cluster_id,
                    cluster_confidence=prob,
                    cluster_status=status,
                )

            # Mark core points
            if core_detection_ids:
                self.repo.mark_detections_as_core(core_detection_ids)

            clusters_created += 1

        logger.info(f"Created {clusters_created} clusters from unassigned pool")
        return clusters_created

    def revert_singleton_clusters(self) -> int:
        """
        Revert maintenance-created singleton clusters back to unassigned pool.

        Finds clusters with face_count=1 and confidence=0.5 (the markers used
        by cleanup_unassigned_pool), moves their faces back to unassigned status,
        and deletes the empty clusters.

        Returns:
            Number of singleton clusters reverted
        """
        logger.info("Reverting maintenance-created singleton clusters")
        reverted = 0

        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # Find singleton clusters created by maintenance
                # These have face_count=1 and the detection has confidence=0.5 and status='auto'
                cursor.execute("""
                    SELECT c.id as cluster_id, pd.id as detection_id
                    FROM cluster c
                    JOIN person_detection pd ON pd.cluster_id = c.id
                    WHERE c.face_count = 1
                      AND pd.cluster_confidence = 0.5
                      AND pd.cluster_status = 'auto'
                """)
                singletons = cursor.fetchall()

        for cluster_id, detection_id in singletons:
            # Move detection back to unassigned pool
            with self.pool.transaction() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE person_detection
                        SET cluster_id = NULL,
                            cluster_status = 'unassigned',
                            cluster_confidence = 0,
                            unassigned_since = NOW()
                        WHERE id = %s
                    """,
                        (detection_id,),
                    )

                    # Delete the now-empty cluster
                    cursor.execute("DELETE FROM cluster WHERE id = %s", (cluster_id,))
                    reverted += 1

        logger.info(f"Reverted {reverted} singleton clusters to unassigned pool")
        return reverted

    def calculate_missing_epsilons(self, min_faces: int = 3, percentile: float = 90.0) -> int:
        """
        Calculate epsilon for clusters that have NULL epsilon but 3+ faces.

        For each qualifying cluster:
        1. Get all face embeddings in the cluster
        2. Use the cluster's existing centroid
        3. Calculate distances from each embedding to the centroid
        4. Set epsilon = specified percentile of those distances

        This is useful for manual clusters created before HDBSCAN was implemented,
        allowing them to participate in incremental clustering.

        Args:
            min_faces: Minimum number of faces required (default 3)
            percentile: Percentile of distances to use for epsilon (default 90.0)

        Returns:
            Number of clusters updated
        """
        logger.info(
            f"Calculating missing epsilons for clusters with {min_faces}+ faces "
            f"using {percentile}th percentile"
        )
        updated_count = 0

        clusters = self.repo.get_clusters_without_epsilon(min_faces=min_faces)
        logger.info(f"Found {len(clusters)} clusters without epsilon")

        for cluster in clusters:
            cluster_id = cluster["id"]
            centroid = cluster["centroid"]

            if centroid is None:
                continue

            detections = self.repo.get_detections_in_cluster(cluster_id)
            if len(detections) < min_faces:
                continue

            embeddings = [d["embedding"] for d in detections if d["embedding"] is not None]
            if len(embeddings) < min_faces:
                continue

            embeddings_array = np.array(embeddings)
            centroid_array = np.array(centroid)

            distances = np.linalg.norm(embeddings_array - centroid_array, axis=1)

            epsilon = float(np.percentile(distances, percentile))

            self.repo.update_cluster_epsilon_only(cluster_id, epsilon)
            updated_count += 1
            logger.debug(f"Cluster {cluster_id}: epsilon = {epsilon:.4f}")

        logger.info(f"Calculated epsilon for {updated_count} clusters")
        return updated_count

    def run_daily_maintenance(self) -> Dict[str, int]:
        """
        Run all daily maintenance tasks.

        Returns:
            Dictionary with results of each task
        """
        logger.info("Starting daily maintenance tasks")
        results = {}

        try:
            results["empty_clusters_removed"] = self.cleanup_empty_clusters()
        except Exception as e:
            logger.error(f"Failed to cleanup empty clusters: {e}")
            results["empty_clusters_removed"] = 0

        try:
            results["statistics_updated"] = self.update_cluster_statistics()
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
            results["statistics_updated"] = 0

        try:
            results["epsilons_calculated"] = self.calculate_missing_epsilons()
        except Exception as e:
            logger.error(f"Failed to calculate missing epsilons: {e}")
            results["epsilons_calculated"] = 0

        try:
            violations = self.find_constraint_violations()
            results["constraint_violations"] = len(violations)
        except Exception as e:
            logger.error(f"Failed to check constraint violations: {e}")
            results["constraint_violations"] = -1

        logger.info(f"Daily maintenance completed: {results}")
        return results

    def run_weekly_maintenance(self, cluster_unassigned: bool = False) -> Dict[str, int]:
        """
        Run all weekly maintenance tasks.

        Args:
            cluster_unassigned: If True, run HDBSCAN on unassigned pool to find new clusters

        Returns:
            Dictionary with results of each task
        """
        logger.info("Starting weekly maintenance tasks")
        results = {}

        # Run daily tasks first
        results.update(self.run_daily_maintenance())

        try:
            results["centroids_recomputed"] = self.recompute_all_centroids()
        except Exception as e:
            logger.error(f"Failed to recompute centroids: {e}")
            results["centroids_recomputed"] = 0

        try:
            results["medoids_updated"] = self.update_all_medoids()
        except Exception as e:
            logger.error(f"Failed to update medoids: {e}")
            results["medoids_updated"] = 0

        if cluster_unassigned:
            try:
                results["unassigned_clusters_created"] = self.cluster_unassigned_pool()
            except Exception as e:
                logger.error(f"Failed to cluster unassigned pool: {e}")
                results["unassigned_clusters_created"] = 0

        logger.info(f"Weekly maintenance completed: {results}")
        return results
