"""
Maintenance utilities for periodic PhotoDB tasks.

This module provides utility methods for periodic maintenance tasks
that should be run on schedules to keep the database optimized.

Includes support for constrained clustering:
- Must-link/cannot-link constraint propagation
- Constraint violation detection
- Verified cluster protection
"""

import logging
from typing import Dict, Any, List
import numpy as np

from ..database.connection import ConnectionPool
from ..database.repository import PhotoRepository

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
                    cursor.execute("""
                        UPDATE cluster
                        SET centroid = (
                            SELECT AVG(fe.embedding)::vector(512)
                            FROM face f
                            JOIN face_embedding fe ON f.id = fe.face_id
                            WHERE f.cluster_id = cluster.id
                        ),
                        updated_at = NOW()
                        WHERE id = %s
                          AND EXISTS (
                            SELECT 1 FROM face f
                            JOIN face_embedding fe ON f.id = fe.face_id
                            WHERE f.cluster_id = %s
                          )
                    """, (cluster_id, cluster_id))
                    
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
                cursor.execute("""
                    SELECT f.id, fe.embedding
                    FROM face f
                    JOIN face_embedding fe ON f.id = fe.face_id
                    WHERE f.cluster_id = %s
                """, (cluster_id,))
                
                rows = cursor.fetchall()
                if not rows:
                    return False
                
                face_ids = [row[0] for row in rows]
                embeddings = np.array([row[1] for row in rows])
                
                # Convert centroid to numpy array if needed
                if not isinstance(centroid, np.ndarray):
                    centroid = np.array(centroid)
                
                # Find face closest to centroid (medoid)
                distances = np.linalg.norm(embeddings - centroid, axis=1)
                medoid_idx = np.argmin(distances)
                medoid_face_id = face_ids[medoid_idx]
                
                # Update cluster with medoid and reset tracking counter
                cursor.execute("""
                    UPDATE cluster
                    SET medoid_face_id = %s,
                        representative_face_id = %s,
                        face_count_at_last_medoid = face_count,
                        updated_at = NOW()
                    WHERE id = %s
                """, (medoid_face_id, medoid_face_id, cluster_id))

                return cursor.rowcount > 0

    def find_and_merge_similar_clusters(
        self, similarity_threshold: float = 0.3, respect_verified: bool = True
    ) -> int:
        """
        Find and merge clusters that are too similar.

        This helps consolidate clusters that have drifted together or
        were initially created as separate but represent the same person.

        This should be run weekly.

        Args:
            similarity_threshold: Maximum distance to consider clusters similar (default 0.3)
            respect_verified: If True, skip merging verified clusters (default True)

        Returns:
            Number of clusters merged
        """
        logger.info(f"Finding similar clusters with threshold {similarity_threshold}")
        merged_count = 0

        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # Find cluster pairs with high similarity
                # Exclude verified clusters if respect_verified is True
                verified_filter = "AND c1.verified = false AND c2.verified = false" if respect_verified else ""

                cursor.execute(f"""
                    WITH cluster_pairs AS (
                        SELECT c1.id as cluster1_id,
                               c2.id as cluster2_id,
                               c1.centroid <=> c2.centroid as distance,
                               (SELECT COUNT(*) FROM face WHERE cluster_id = c1.id) as count1,
                               (SELECT COUNT(*) FROM face WHERE cluster_id = c2.id) as count2
                        FROM cluster c1
                        CROSS JOIN cluster c2
                        WHERE c1.id < c2.id  -- Avoid duplicates
                          AND c1.centroid IS NOT NULL
                          AND c2.centroid IS NOT NULL
                          AND c1.centroid <=> c2.centroid < %s
                          {verified_filter}
                    )
                    SELECT cluster1_id, cluster2_id, distance, count1, count2
                    FROM cluster_pairs
                    ORDER BY distance
                """, (similarity_threshold,))

                similar_pairs = cursor.fetchall()

                # Track which clusters have been merged to avoid conflicts
                merged_clusters = set()

                for cluster1_id, cluster2_id, distance, count1, count2 in similar_pairs:
                    # Skip if either cluster was already merged
                    if cluster1_id in merged_clusters or cluster2_id in merged_clusters:
                        continue

                    # Check for cannot-link constraint between clusters
                    if self.repo.has_cluster_cannot_link(cluster1_id, cluster2_id):
                        logger.debug(
                            f"Skipping merge of clusters {cluster1_id} and {cluster2_id}: "
                            "cannot-link constraint exists"
                        )
                        continue

                    # Check for cannot-link constraints between faces in the clusters
                    if self._has_face_cannot_link_between_clusters(cluster1_id, cluster2_id):
                        logger.debug(
                            f"Skipping merge of clusters {cluster1_id} and {cluster2_id}: "
                            "face-level cannot-link constraint exists"
                        )
                        continue

                    # Merge smaller cluster into larger one
                    if count1 >= count2:
                        keep_id, merge_id = cluster1_id, cluster2_id
                    else:
                        keep_id, merge_id = cluster2_id, cluster1_id

                    logger.info(f"Merging cluster {merge_id} into {keep_id} (distance: {distance:.3f})")

                    if self._merge_clusters(keep_id, merge_id):
                        merged_clusters.add(merge_id)
                        merged_count += 1

        logger.info(f"Merged {merged_count} similar clusters")
        return merged_count

    def _has_face_cannot_link_between_clusters(
        self, cluster_id_1: int, cluster_id_2: int
    ) -> bool:
        """Check if any faces between two clusters have cannot-link constraints."""
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 1 FROM cannot_link cl
                    JOIN face f1 ON (cl.face_id_1 = f1.id OR cl.face_id_2 = f1.id)
                    JOIN face f2 ON (cl.face_id_1 = f2.id OR cl.face_id_2 = f2.id)
                    WHERE f1.id != f2.id
                      AND f1.cluster_id = %s
                      AND f2.cluster_id = %s
                    LIMIT 1
                """, (cluster_id_1, cluster_id_2))
                return cursor.fetchone() is not None

    def _merge_clusters(self, keep_cluster_id: int, merge_cluster_id: int) -> bool:
        """
        Merge one cluster into another.
        
        Args:
            keep_cluster_id: ID of cluster to keep
            merge_cluster_id: ID of cluster to merge and delete
            
        Returns:
            True if merge was successful
        """
        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Reassign all faces from merge cluster to keep cluster
                cursor.execute("""
                    UPDATE face
                    SET cluster_id = %s,
                        cluster_confidence = cluster_confidence * 0.9  -- Slightly reduce confidence
                    WHERE cluster_id = %s
                """, (keep_cluster_id, merge_cluster_id))
                
                faces_moved = cursor.rowcount
                
                # Delete the merged cluster
                cursor.execute("DELETE FROM cluster WHERE id = %s", (merge_cluster_id,))
                
                # Recompute centroid for the keeper cluster
                cursor.execute("""
                    UPDATE cluster
                    SET centroid = (
                        SELECT AVG(fe.embedding)::vector(512)
                        FROM face f
                        JOIN face_embedding fe ON f.id = fe.face_id
                        WHERE f.cluster_id = %s
                    ),
                    face_count = (
                        SELECT COUNT(*) FROM face WHERE cluster_id = %s
                    ),
                    updated_at = NOW()
                    WHERE id = %s
                """, (keep_cluster_id, keep_cluster_id, keep_cluster_id))
                
                logger.debug(f"Moved {faces_moved} faces from cluster {merge_cluster_id} to {keep_cluster_id}")
                return faces_moved > 0

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
                        FROM face 
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
                        FROM face 
                        WHERE face.cluster_id = cluster.id
                    ),
                    updated_at = NOW()
                    WHERE id IN (
                        SELECT DISTINCT cluster_id 
                        FROM face 
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
                stats['total_clusters'] = cursor.fetchone()[0]

                # Clusters without centroids
                cursor.execute("SELECT COUNT(*) FROM cluster WHERE centroid IS NULL")
                stats['clusters_without_centroids'] = cursor.fetchone()[0]

                # Clusters without medoids
                cursor.execute("SELECT COUNT(*) FROM cluster WHERE medoid_face_id IS NULL")
                stats['clusters_without_medoids'] = cursor.fetchone()[0]

                # Empty clusters
                cursor.execute("""
                    SELECT COUNT(*) FROM cluster
                    WHERE id NOT IN (
                        SELECT DISTINCT cluster_id
                        FROM face
                        WHERE cluster_id IS NOT NULL
                    )
                """)
                stats['empty_clusters'] = cursor.fetchone()[0]

                # Average cluster size
                cursor.execute("""
                    SELECT AVG(face_count), MIN(face_count), MAX(face_count)
                    FROM cluster
                    WHERE face_count > 0
                """)
                avg_size, min_size, max_size = cursor.fetchone()
                stats['avg_cluster_size'] = float(avg_size) if avg_size else 0
                stats['min_cluster_size'] = min_size or 0
                stats['max_cluster_size'] = max_size or 0

                # Unclustered faces
                cursor.execute("SELECT COUNT(*) FROM face WHERE cluster_id IS NULL")
                stats['unclustered_faces'] = cursor.fetchone()[0]

                # Total faces
                cursor.execute("SELECT COUNT(*) FROM face")
                stats['total_faces'] = cursor.fetchone()[0]

                # Constraint stats
                cursor.execute("SELECT COUNT(*) FROM must_link")
                stats['must_link_count'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cannot_link")
                stats['cannot_link_count'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM cluster WHERE verified = true")
                stats['verified_clusters'] = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM face WHERE cluster_status = 'unassigned'"
                )
                stats['unassigned_pool_size'] = cursor.fetchone()[0]

        return stats

    def propagate_must_link_constraints(self) -> int:
        """
        Propagate must-link constraints through transitive closure.

        If A~B and B~C, then A~C should also exist.
        This should be run periodically to maintain constraint consistency.

        Returns:
            Number of new constraints added
        """
        logger.info("Propagating must-link constraints")
        total_added = 0

        with self.pool.transaction() as conn:
            with conn.cursor() as cursor:
                # Keep adding transitive links until no new ones are found
                while True:
                    cursor.execute("""
                        INSERT INTO must_link (face_id_1, face_id_2, created_by)
                        SELECT DISTINCT
                            LEAST(m1.face_id_1, m2.face_id_2),
                            GREATEST(m1.face_id_1, m2.face_id_2),
                            'system'
                        FROM must_link m1
                        JOIN must_link m2 ON m1.face_id_2 = m2.face_id_1
                        WHERE m1.face_id_1 != m2.face_id_2
                          AND LEAST(m1.face_id_1, m2.face_id_2) < GREATEST(m1.face_id_1, m2.face_id_2)
                        ON CONFLICT (face_id_1, face_id_2) DO NOTHING
                    """)
                    added = cursor.rowcount
                    if added == 0:
                        break
                    total_added += added
                    logger.debug(f"Added {added} transitive must-link constraints")

        logger.info(f"Propagated {total_added} must-link constraints")
        return total_added

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
                           f1.cluster_id,
                           cl.face_id_1,
                           cl.face_id_2
                    FROM cannot_link cl
                    JOIN face f1 ON cl.face_id_1 = f1.id
                    JOIN face f2 ON cl.face_id_2 = f2.id
                    WHERE f1.cluster_id = f2.cluster_id
                      AND f1.cluster_id IS NOT NULL
                """)

                violations = [
                    {
                        "constraint_id": row[0],
                        "cluster_id": row[1],
                        "face_1": row[2],
                        "face_2": row[3],
                    }
                    for row in cursor.fetchall()
                ]

        if violations:
            logger.warning(f"Found {len(violations)} constraint violations")
        else:
            logger.info("No constraint violations found")

        return violations

    def merge_must_linked_clusters(self) -> int:
        """
        Merge clusters where faces have must-link constraints.

        If face A in cluster X has a must-link with face B in cluster Y,
        clusters X and Y should be merged.

        Returns:
            Number of cluster merges performed
        """
        logger.info("Merging clusters with must-link constraints")
        merge_count = 0

        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # Find cluster pairs that should be merged due to must-link constraints
                cursor.execute("""
                    SELECT DISTINCT
                        f1.cluster_id as cluster1,
                        f2.cluster_id as cluster2,
                        (SELECT COUNT(*) FROM face WHERE cluster_id = f1.cluster_id) as count1,
                        (SELECT COUNT(*) FROM face WHERE cluster_id = f2.cluster_id) as count2
                    FROM must_link ml
                    JOIN face f1 ON ml.face_id_1 = f1.id
                    JOIN face f2 ON ml.face_id_2 = f2.id
                    WHERE f1.cluster_id IS NOT NULL
                      AND f2.cluster_id IS NOT NULL
                      AND f1.cluster_id != f2.cluster_id
                """)

                pairs_to_merge = cursor.fetchall()

                merged_clusters = set()
                for cluster1, cluster2, count1, count2 in pairs_to_merge:
                    if cluster1 in merged_clusters or cluster2 in merged_clusters:
                        continue

                    # Check for cannot-link conflicts
                    if self._has_face_cannot_link_between_clusters(cluster1, cluster2):
                        logger.warning(
                            f"Cannot merge clusters {cluster1} and {cluster2}: "
                            "conflicting cannot-link constraint"
                        )
                        continue

                    # Merge smaller into larger
                    if count1 >= count2:
                        keep_id, merge_id = cluster1, cluster2
                    else:
                        keep_id, merge_id = cluster2, cluster1

                    logger.info(
                        f"Merging cluster {merge_id} into {keep_id} due to must-link constraint"
                    )

                    if self._merge_clusters(keep_id, merge_id):
                        merged_clusters.add(merge_id)
                        merge_count += 1

        logger.info(f"Merged {merge_count} clusters due to must-link constraints")
        return merge_count

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
                cursor.execute("""
                    SELECT f.id, fe.embedding
                    FROM face f
                    JOIN face_embedding fe ON f.id = fe.face_id
                    WHERE f.cluster_status = 'unassigned'
                      AND f.unassigned_since < NOW() - INTERVAL '%s days'
                """, (max_age_days,))

                old_faces = cursor.fetchall()

        for face_id, embedding in old_faces:
            if embedding is not None:
                # Normalize embedding
                emb_array = np.array(embedding)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm

                # Create singleton cluster
                cluster_id = self.repo.create_cluster(
                    centroid=emb_array,
                    representative_face_id=face_id,
                    medoid_face_id=face_id,
                    face_count=0,
                )

                # Only succeeds if face is still unassigned
                if self.repo.update_face_cluster(
                    face_id=face_id,
                    cluster_id=cluster_id,
                    cluster_confidence=0.5,  # Low confidence for singleton
                    cluster_status="auto",
                ):
                    self.repo.update_cluster_face_count(cluster_id, 1)
                    self.repo.clear_face_unassigned(face_id)
                    created += 1
                    logger.debug(f"Created singleton cluster {cluster_id} for face {face_id}")
                else:
                    # Face was assigned elsewhere, delete empty cluster
                    self.repo.delete_cluster(cluster_id)

        logger.info(f"Created {created} singleton clusters from old unassigned faces")
        return created

    def run_daily_maintenance(self) -> Dict[str, int]:
        """
        Run all daily maintenance tasks.

        Returns:
            Dictionary with results of each task
        """
        logger.info("Starting daily maintenance tasks")
        results = {}

        try:
            results['centroids_recomputed'] = self.recompute_all_centroids()
        except Exception as e:
            logger.error(f"Failed to recompute centroids: {e}")
            results['centroids_recomputed'] = 0

        try:
            results['empty_clusters_removed'] = self.cleanup_empty_clusters()
        except Exception as e:
            logger.error(f"Failed to cleanup empty clusters: {e}")
            results['empty_clusters_removed'] = 0

        try:
            results['statistics_updated'] = self.update_cluster_statistics()
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
            results['statistics_updated'] = 0

        try:
            results['must_link_propagated'] = self.propagate_must_link_constraints()
        except Exception as e:
            logger.error(f"Failed to propagate must-link constraints: {e}")
            results['must_link_propagated'] = 0

        try:
            results['must_link_merges'] = self.merge_must_linked_clusters()
        except Exception as e:
            logger.error(f"Failed to merge must-linked clusters: {e}")
            results['must_link_merges'] = 0

        try:
            violations = self.find_constraint_violations()
            results['constraint_violations'] = len(violations)
        except Exception as e:
            logger.error(f"Failed to check constraint violations: {e}")
            results['constraint_violations'] = -1

        logger.info(f"Daily maintenance completed: {results}")
        return results

    def run_weekly_maintenance(self, similarity_threshold: float = 0.3) -> Dict[str, int]:
        """
        Run all weekly maintenance tasks.

        Args:
            similarity_threshold: Threshold for cluster similarity

        Returns:
            Dictionary with results of each task
        """
        logger.info("Starting weekly maintenance tasks")
        results = {}

        # Run daily tasks first
        results.update(self.run_daily_maintenance())

        try:
            results['medoids_updated'] = self.update_all_medoids()
        except Exception as e:
            logger.error(f"Failed to update medoids: {e}")
            results['medoids_updated'] = 0

        try:
            results['clusters_merged'] = self.find_and_merge_similar_clusters(similarity_threshold)
        except Exception as e:
            logger.error(f"Failed to merge similar clusters: {e}")
            results['clusters_merged'] = 0

        try:
            results['unassigned_cleanup'] = self.cleanup_unassigned_pool()
        except Exception as e:
            logger.error(f"Failed to cleanup unassigned pool: {e}")
            results['unassigned_cleanup'] = 0

        logger.info(f"Weekly maintenance completed: {results}")
        return results