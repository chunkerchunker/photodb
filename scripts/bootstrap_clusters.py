#!/usr/bin/env python
"""
Bootstrap HDBSCAN-based clustering on all face embeddings.

This script:
1. Backs up current cluster assignments (in memory, not destructive)
2. Runs HDBSCAN bootstrap on all embeddings
3. Reassigns detections while preserving:
   - Manual assignments (cluster_status = 'manual')
   - Verified clusters (keeps assignments, updates is_core)
   - Cannot-link constraints (stored in separate table, unchanged)
4. Calculates and stores per-cluster epsilon values

Run this after importing batches of photos through the normal pipeline
(normalize, metadata, detection, age_gender, scene_analysis).
"""

import sys
import logging
import click
from dotenv import load_dotenv

load_dotenv()

from photodb import config as defaults  # noqa: E402
from photodb.database.connection import ConnectionPool  # noqa: E402
from photodb.database.repository import PhotoRepository  # noqa: E402
from photodb.stages.clustering import ClusteringStage  # noqa: E402


@click.command()
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes")
@click.option(
    "--force", is_flag=True, help="Force re-run even if HDBSCAN columns already populated"
)
@click.option("--collection-id", type=int, default=1, help="Collection ID to process")
def main(dry_run: bool, force: bool, collection_id: int):
    """Bootstrap HDBSCAN clustering on all face embeddings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    database_url = defaults.DATABASE_URL

    logger.info("=" * 60)
    logger.info("HDBSCAN Clustering Bootstrap")
    logger.info("=" * 60)

    with ConnectionPool(connection_string=database_url) as pool:
        repository = PhotoRepository(pool, collection_id=collection_id)

        # Check current state
        logger.info(f"Processing collection ID: {collection_id}")

        # Get stats about current clustering
        embeddings = repository.get_all_embeddings_for_collection()
        logger.info(f"Total embeddings in collection: {len(embeddings)}")

        # Count current assignments
        clustered = sum(1 for e in embeddings if e.get("cluster_id") is not None)
        manual = sum(1 for e in embeddings if e.get("cluster_status") == "manual")
        logger.info(f"Currently clustered: {clustered}")
        logger.info(f"Manual assignments: {manual}")

        # Check if already bootstrapped (clusters have epsilon values)
        clusters_with_epsilon = repository.get_clusters_with_epsilon()
        epsilon_count = sum(1 for c in clusters_with_epsilon if c.get("epsilon") is not None)
        logger.info(f"Clusters with epsilon values: {epsilon_count}")

        if epsilon_count > 0 and not force:
            logger.warning(
                "Some clusters already have epsilon values. Use --force to re-run bootstrap anyway."
            )
            sys.exit(0)

        if dry_run:
            logger.info("")
            logger.info("DRY RUN MODE - No changes will be made")
            logger.info("-" * 40)
            logger.info(f"Would process {len(embeddings)} embeddings")
            logger.info(f"Would preserve {manual} manual assignments")
            logger.info("")
            logger.info("Run without --dry-run to perform the bootstrap")
            sys.exit(0)

        # Load HDBSCAN configuration from environment
        config = {
            "HDBSCAN_MIN_CLUSTER_SIZE": defaults.HDBSCAN_MIN_CLUSTER_SIZE,
            "HDBSCAN_MIN_SAMPLES": defaults.HDBSCAN_MIN_SAMPLES,
            "CORE_PROBABILITY_THRESHOLD": defaults.CORE_PROBABILITY_THRESHOLD,
            "CLUSTERING_THRESHOLD": defaults.CLUSTERING_THRESHOLD,
        }

        logger.info("")
        logger.info("HDBSCAN Configuration:")
        logger.info(f"  min_cluster_size: {config['HDBSCAN_MIN_CLUSTER_SIZE']}")
        logger.info(f"  min_samples: {config['HDBSCAN_MIN_SAMPLES']}")
        logger.info(f"  core_probability_threshold: {config['CORE_PROBABILITY_THRESHOLD']}")
        logger.info("")

        # Create clustering stage and run bootstrap
        stage = ClusteringStage(repository, config)

        logger.info("Running HDBSCAN bootstrap clustering...")
        logger.info("-" * 40)

        try:
            success = stage.run_bootstrap()

            if success:
                logger.info("")
                logger.info("=" * 60)
                logger.info("Bootstrap completed successfully!")
                logger.info("=" * 60)

                # Print summary
                new_embeddings = repository.get_all_embeddings_for_collection()
                new_clustered = sum(1 for e in new_embeddings if e.get("cluster_id") is not None)
                new_clusters = repository.get_clusters_with_epsilon()

                logger.info("")
                logger.info("Summary:")
                logger.info(f"  Embeddings clustered: {new_clustered}/{len(new_embeddings)}")
                logger.info(f"  Clusters created: {len(new_clusters)}")
                logger.info(
                    f"  Clusters with epsilon: "
                    f"{sum(1 for c in new_clusters if c.get('epsilon') is not None)}"
                )

                # Count clusters with lambda_birth set
                with repository.pool.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """SELECT COUNT(*) FROM cluster
                               WHERE collection_id = %s AND lambda_birth IS NOT NULL""",
                            (collection_id,),
                        )
                        lambda_birth_count = cursor.fetchone()[0]
                logger.info(f"  Clusters with lambda_birth: {lambda_birth_count}")

                # Check hdbscan_run was created
                active_run = repository.get_active_hdbscan_run()
                if active_run:
                    logger.info(f"  Active HDBSCAN run: #{active_run['id']}")
                    logger.info(f"  Embeddings in run: {active_run['embedding_count']}")
                    logger.info(
                        f"  Clusterer state: "
                        f"{'persisted' if active_run.get('clusterer_state') else 'not persisted'}"
                    )
            else:
                logger.error("Bootstrap failed")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Bootstrap failed with error: {e}")
            raise


if __name__ == "__main__":
    main()
