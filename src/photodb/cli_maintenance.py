#!/usr/bin/env python
"""
PhotoDB Maintenance CLI

This CLI provides access to periodic maintenance tasks for the PhotoDB system.
These tasks should be run on regular schedules to keep the database optimized.
"""

import argparse
import logging
import sys
import os
import json
from contextlib import contextmanager
from typing import Callable, Generator

from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that read them at module level
load_dotenv()

from .database.connection import ConnectionPool  # noqa: E402
from .utils.maintenance import MaintenanceUtilities  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@contextmanager
def maintenance_context() -> Generator[MaintenanceUtilities, None, None]:
    """
    Context manager for maintenance operations.

    Handles database connection pool setup and teardown automatically.

    Yields:
        MaintenanceUtilities instance with active connection pool
    """
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    try:
        yield MaintenanceUtilities(pool)
    finally:
        pool.close_all()


def run_maintenance_command(
    operation: Callable[[MaintenanceUtilities], int],
    error_message: str,
) -> int:
    """
    Execute a maintenance operation with standard error handling.

    Args:
        operation: Function that takes MaintenanceUtilities and returns exit code
        error_message: Message to log on failure

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        with maintenance_context() as maintenance:
            return operation(maintenance)
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return 1


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

    # Reduce noise from some libraries
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("psycopg").setLevel(logging.WARNING)


def run_daily(args):
    """Run daily maintenance tasks."""
    logger.info("Running daily maintenance tasks...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        results = maintenance.run_daily_maintenance()

        print("\nDaily maintenance completed:")
        print(f"  - Empty clusters removed: {results.get('empty_clusters_removed', 0)}")
        print(f"  - Statistics updated: {results.get('statistics_updated', 0)}")
        print(f"  - Epsilons calculated: {results.get('epsilons_calculated', 0)}")
        print(f"  - Constraint violations: {results.get('constraint_violations', 0)}")

        if args.json:
            print("\nJSON output:")
            print(json.dumps(results, indent=2))

        return 0

    return run_maintenance_command(operation, "Daily maintenance failed")


def run_weekly(args):
    """Run weekly maintenance tasks."""
    logger.info("Running weekly maintenance tasks...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        results = maintenance.run_weekly_maintenance(cluster_unassigned=args.cluster_unassigned)

        print("\nWeekly maintenance completed:")
        print(f"  - Empty clusters removed: {results.get('empty_clusters_removed', 0)}")
        print(f"  - Statistics updated: {results.get('statistics_updated', 0)}")
        print(f"  - Epsilons calculated: {results.get('epsilons_calculated', 0)}")
        print(f"  - Constraint violations: {results.get('constraint_violations', 0)}")
        print(f"  - Centroids recomputed: {results.get('centroids_recomputed', 0)}")
        print(f"  - Medoids updated: {results.get('medoids_updated', 0)}")
        if "unassigned_clusters_created" in results:
            print(
                f"  - Unassigned clusters created: {results.get('unassigned_clusters_created', 0)}"
            )

        if args.json:
            print("\nJSON output:")
            print(json.dumps(results, indent=2))

        return 0

    return run_maintenance_command(operation, "Weekly maintenance failed")


def recompute_centroids(args):
    """Recompute all cluster centroids."""
    logger.info("Recomputing cluster centroids...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.recompute_all_centroids()
        print(f"Recomputed centroids for {count} clusters")
        return 0

    return run_maintenance_command(operation, "Failed to recompute centroids")


def update_medoids(args):
    """Update medoids for all clusters."""
    logger.info("Updating cluster medoids...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.update_all_medoids()
        print(f"Updated medoids for {count} clusters")
        return 0

    return run_maintenance_command(operation, "Failed to update medoids")


def cleanup_empty(args):
    """Remove empty clusters."""
    logger.info("Cleaning up empty clusters...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.cleanup_empty_clusters()
        print(f"Removed {count} empty clusters")
        return 0

    return run_maintenance_command(operation, "Failed to cleanup clusters")


def update_stats(args):
    """Update cluster statistics."""
    logger.info("Updating cluster statistics...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.update_cluster_statistics()
        print(f"Updated statistics for {count} clusters")
        return 0

    return run_maintenance_command(operation, "Failed to update statistics")


def revert_singletons(args):
    """Revert maintenance-created singleton clusters to unassigned pool."""
    logger.info("Reverting singleton clusters...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.revert_singleton_clusters()
        print(f"Reverted {count} singleton clusters to unassigned pool")
        return 0

    return run_maintenance_command(operation, "Failed to revert singletons")


def cluster_unassigned(args):
    """Run HDBSCAN clustering on the unassigned pool."""
    logger.info("Clustering unassigned pool...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.cluster_unassigned_pool(min_cluster_size=args.min_cluster_size)
        print(f"Created {count} clusters from unassigned pool")
        return 0

    return run_maintenance_command(operation, "Failed to cluster unassigned pool")


def calculate_epsilons(args):
    """Calculate epsilon for clusters with NULL epsilon but 3+ faces."""
    logger.info("Calculating missing epsilons...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.calculate_missing_epsilons(
            min_faces=args.min_faces,
            percentile=args.percentile,
        )
        print(f"Calculated epsilon for {count} clusters")
        return 0

    return run_maintenance_command(operation, "Failed to calculate epsilons")


def health_check(args):
    """Check clustering system health."""
    logger.info("Checking cluster health...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        stats = maintenance.get_cluster_health_stats()

        print("\nCluster Health Report:")
        print(f"  - Total clusters: {stats['total_clusters']}")
        print(f"  - Empty clusters: {stats['empty_clusters']}")
        print(f"  - Clusters without centroids: {stats['clusters_without_centroids']}")
        print(f"  - Clusters without medoids: {stats['clusters_without_medoids']}")
        print(f"  - Average cluster size: {stats['avg_cluster_size']:.1f}")
        print(f"  - Cluster size range: {stats['min_cluster_size']} - {stats['max_cluster_size']}")
        print(f"  - Total faces: {stats['total_faces']}")
        print(f"  - Unclustered faces: {stats['unclustered_faces']}")
        print("")
        print("Person-Cluster Relationships:")
        print(f"  - Total persons: {stats.get('total_persons', 0)}")
        print(f"  - Clusters linked to a person: {stats.get('clusters_with_person', 0)}")
        print(
            f"  - Persons with multiple clusters: {stats.get('persons_with_multiple_clusters', 0)}"
        )

        if stats["total_faces"] > 0:
            clustered_pct = (1 - stats["unclustered_faces"] / stats["total_faces"]) * 100
            print(f"  - Clustering coverage: {clustered_pct:.1f}%")

        # Check for issues
        issues = []
        if stats["empty_clusters"] > 0:
            issues.append(f"{stats['empty_clusters']} empty clusters need cleanup")
        if stats["clusters_without_centroids"] > 0:
            issues.append(f"{stats['clusters_without_centroids']} clusters missing centroids")
        if stats["clusters_without_medoids"] > 0:
            issues.append(f"{stats['clusters_without_medoids']} clusters missing medoids")

        if issues:
            print("\nIssues detected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nNo issues detected")

        if args.json:
            print("\nJSON output:")
            print(json.dumps(stats, indent=2))

        return 0 if not issues else 1

    return run_maintenance_command(operation, "Health check failed")


def main():
    """Main entry point for maintenance CLI."""
    parser = argparse.ArgumentParser(
        description="PhotoDB Maintenance Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily maintenance tasks
  photodb-maintenance daily

  # Run weekly maintenance tasks
  photodb-maintenance weekly

  # Check system health
  photodb-maintenance health

  # Run specific maintenance task
  photodb-maintenance recompute-centroids
  photodb-maintenance update-medoids
  photodb-maintenance cleanup-empty

  # Get JSON output for automation
  photodb-maintenance daily --json
""",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Maintenance command to run")

    # Daily maintenance
    daily_parser = subparsers.add_parser("daily", help="Run all daily maintenance tasks")
    daily_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    daily_parser.set_defaults(func=run_daily)

    # Weekly maintenance
    weekly_parser = subparsers.add_parser("weekly", help="Run all weekly maintenance tasks")
    weekly_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    weekly_parser.add_argument(
        "--cluster-unassigned",
        action="store_true",
        help="Run HDBSCAN on unassigned pool to find new clusters",
    )
    weekly_parser.set_defaults(func=run_weekly)

    # Individual tasks
    centroids_parser = subparsers.add_parser(
        "recompute-centroids", help="Recompute all cluster centroids"
    )
    centroids_parser.set_defaults(func=recompute_centroids)

    medoids_parser = subparsers.add_parser("update-medoids", help="Update medoids for all clusters")
    medoids_parser.set_defaults(func=update_medoids)

    cleanup_parser = subparsers.add_parser("cleanup-empty", help="Remove empty clusters")
    cleanup_parser.set_defaults(func=cleanup_empty)

    stats_parser = subparsers.add_parser("update-stats", help="Update cluster statistics")
    stats_parser.set_defaults(func=update_stats)

    revert_parser = subparsers.add_parser(
        "revert-singletons", help="Revert maintenance-created singleton clusters to unassigned pool"
    )
    revert_parser.set_defaults(func=revert_singletons)

    cluster_unassigned_parser = subparsers.add_parser(
        "cluster-unassigned",
        help="Run HDBSCAN clustering on the unassigned pool to find new clusters",
    )
    cluster_unassigned_parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="Minimum faces to form a cluster (default: 3)",
    )
    cluster_unassigned_parser.set_defaults(func=cluster_unassigned)

    calculate_epsilons_parser = subparsers.add_parser(
        "calculate-epsilons", help="Calculate epsilon for clusters with NULL epsilon but 3+ faces"
    )
    calculate_epsilons_parser.add_argument(
        "--min-faces", type=int, default=3, help="Minimum faces required in cluster (default: 3)"
    )
    calculate_epsilons_parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile of distances to centroid for epsilon (default: 90.0)",
    )
    calculate_epsilons_parser.set_defaults(func=calculate_epsilons)

    # Health check
    health_parser = subparsers.add_parser("health", help="Check clustering system health")
    health_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    health_parser.set_defaults(func=health_check)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
