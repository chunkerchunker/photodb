"""
Maintenance utilities for periodic PhotoDB tasks.

This module provides utility methods for periodic maintenance tasks
that should be run on schedules to keep the database optimized.
"""

import logging
from typing import Dict, Any
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
                
                # Update cluster with medoid
                cursor.execute("""
                    UPDATE cluster
                    SET medoid_face_id = %s,
                        representative_face_id = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (medoid_face_id, medoid_face_id, cluster_id))
                
                return cursor.rowcount > 0

    def find_and_merge_similar_clusters(self, similarity_threshold: float = 0.3) -> int:
        """
        Find and merge clusters that are too similar.
        
        This helps consolidate clusters that have drifted together or
        were initially created as separate but represent the same person.
        
        This should be run weekly.
        
        Args:
            similarity_threshold: Maximum distance to consider clusters similar (default 0.3)
            
        Returns:
            Number of clusters merged
        """
        logger.info(f"Finding similar clusters with threshold {similarity_threshold}")
        merged_count = 0
        
        with self.pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # Find cluster pairs with high similarity
                cursor.execute("""
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
        
        logger.info(f"Weekly maintenance completed: {results}")
        return results

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
        
        return stats