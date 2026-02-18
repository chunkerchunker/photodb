#!/usr/bin/env python
"""
PhotoDB Maintenance CLI

This CLI provides access to periodic maintenance tasks for the PhotoDB system.
These tasks should be run on regular schedules to keep the database optimized.
"""

import argparse
import logging
import sys
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
def maintenance_context(
    collection_id: int | None = None,
) -> Generator[MaintenanceUtilities, None, None]:
    """
    Context manager for maintenance operations.

    Handles database connection pool setup and teardown automatically.

    Args:
        collection_id: Optional collection ID to scope aggregate operations.

    Yields:
        MaintenanceUtilities instance with active connection pool
    """
    from . import config as defaults

    database_url = defaults.DATABASE_URL
    pool = ConnectionPool(connection_string=database_url)
    try:
        yield MaintenanceUtilities(pool, collection_id=collection_id)
    finally:
        pool.close_all()


def run_maintenance_command(
    operation: Callable[[MaintenanceUtilities], int],
    error_message: str,
    collection_id: int | None = None,
) -> int:
    """
    Execute a maintenance operation with standard error handling.

    Args:
        operation: Function that takes MaintenanceUtilities and returns exit code
        error_message: Message to log on failure
        collection_id: Optional collection ID to scope aggregate operations

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        with maintenance_context(collection_id=collection_id) as maintenance:
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

    return run_maintenance_command(operation, "Daily maintenance failed", args.collection_id)


def run_weekly(args):
    """Run weekly maintenance tasks."""
    logger.info("Running weekly maintenance tasks...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        results = maintenance.run_weekly_maintenance(
            cluster_unassigned=args.cluster_unassigned,
            auto_associate=args.auto_associate,
        )

        print("\nWeekly maintenance completed:")
        print(f"  - Empty clusters removed: {results.get('empty_clusters_removed', 0)}")
        print(f"  - Statistics updated: {results.get('statistics_updated', 0)}")
        print(f"  - Epsilons calculated: {results.get('epsilons_calculated', 0)}")
        print(f"  - Constraint violations: {results.get('constraint_violations', 0)}")
        print(f"  - Centroids recomputed: {results.get('centroids_recomputed', 0)}")
        print(f"  - Medoids updated: {results.get('medoids_updated', 0)}")
        assoc = results.get("auto_association", {})
        if assoc:
            print(f"  - Auto-association: {assoc.get('groups_found', 0)} groups found, "
                  f"{assoc.get('persons_created', 0)} persons created, "
                  f"{assoc.get('persons_merged', 0)} merged, "
                  f"{assoc.get('clusters_linked', 0)} clusters linked")
        if "unassigned_clusters_created" in results:
            print(
                f"  - Unassigned clusters created: {results.get('unassigned_clusters_created', 0)}"
            )

        if args.json:
            print("\nJSON output:")
            print(json.dumps(results, indent=2))

        return 0

    return run_maintenance_command(operation, "Weekly maintenance failed", args.collection_id)


def recompute_centroids(args):
    """Recompute all cluster centroids."""
    logger.info("Recomputing cluster centroids...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.recompute_all_centroids()
        print(f"Recomputed centroids for {count} clusters")
        return 0

    return run_maintenance_command(operation, "Failed to recompute centroids", args.collection_id)


def update_medoids(args):
    """Update medoids for all clusters."""
    logger.info("Updating cluster medoids...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.update_all_medoids()
        print(f"Updated medoids for {count} clusters")
        return 0

    return run_maintenance_command(operation, "Failed to update medoids", args.collection_id)


def cleanup_empty(args):
    """Remove empty clusters."""
    logger.info("Cleaning up empty clusters...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.cleanup_empty_clusters()
        print(f"Removed {count} empty clusters")
        return 0

    return run_maintenance_command(operation, "Failed to cleanup clusters", args.collection_id)


def cleanup_empty_persons(args):
    """Remove empty auto-created persons."""
    logger.info("Cleaning up empty auto-created persons...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.cleanup_empty_auto_created_persons()
        print(f"Removed {count} empty auto-created persons")
        return 0

    return run_maintenance_command(
        operation, "Failed to cleanup empty auto-created persons", args.collection_id
    )


def update_stats(args):
    """Update cluster statistics."""
    logger.info("Updating cluster statistics...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.update_cluster_statistics()
        print(f"Updated statistics for {count} clusters")
        return 0

    return run_maintenance_command(operation, "Failed to update statistics", args.collection_id)


def revert_singletons(args):
    """Revert maintenance-created singleton clusters to unassigned pool."""
    logger.info("Reverting singleton clusters...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.revert_singleton_clusters()
        print(f"Reverted {count} singleton clusters to unassigned pool")
        return 0

    return run_maintenance_command(operation, "Failed to revert singletons", args.collection_id)


def cluster_unassigned(args):
    """Run HDBSCAN clustering on the unassigned pool."""
    logger.info("Clustering unassigned pool...")

    def operation(maintenance: MaintenanceUtilities) -> int:
        count = maintenance.cluster_unassigned_pool(min_cluster_size=args.min_cluster_size)
        print(f"Created {count} clusters from unassigned pool")
        return 0

    return run_maintenance_command(
        operation, "Failed to cluster unassigned pool", args.collection_id
    )


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

    return run_maintenance_command(operation, "Failed to calculate epsilons", args.collection_id)


def auto_associate(args):
    """Auto-associate clusters to persons based on centroid similarity."""
    logger.info("Running auto-association...")

    from . import config as defaults

    threshold = args.threshold if args.threshold is not None else defaults.PERSON_ASSOCIATION_THRESHOLD

    def operation(maintenance: MaintenanceUtilities) -> int:
        result = maintenance.auto_associate_clusters(
            threshold=threshold,
            dry_run=args.dry_run,
        )

        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"\n{prefix}Auto-association completed:")
        print(f"  - Groups found: {result['groups_found']}")
        print(f"  - Persons created: {result['persons_created']}")
        print(f"  - Persons merged: {result['persons_merged']}")
        print(f"  - Clusters linked: {result['clusters_linked']}")

        if args.json:
            print("\nJSON output:")
            print(json.dumps(result, indent=2))

        return 0

    return run_maintenance_command(operation, "Auto-association failed", args.collection_id)


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

    return run_maintenance_command(operation, "Health check failed", args.collection_id)


def check_staleness(args):
    """Check if HDBSCAN bootstrap needs re-running."""
    logger.info("Checking HDBSCAN staleness...")

    def _print_staleness(result: dict) -> None:
        cid = result.get("collection_id")
        prefix = f"  [collection {cid}] " if cid is not None else "  "
        if result["active_run_id"] is not None:
            print(
                f"{prefix}Run {result['active_run_id']}: "
                f"{result['bootstrap_embedding_count']} -> {result['current_embedding_count']} "
                f"embeddings ({result['growth_ratio']:.2f}x) "
                f"{'STALE' if result['is_stale'] else 'CURRENT'}"
            )
        else:
            print(f"{prefix}No active HDBSCAN run. {result['recommendation']}")

    def operation(maintenance: MaintenanceUtilities) -> int:
        result = maintenance.check_hdbscan_staleness(threshold=args.threshold)

        # Normalize to list for uniform handling
        results = result if isinstance(result, list) else [result]

        print("\nHDBSCAN Staleness Check:")
        for r in results:
            _print_staleness(r)

        any_stale = any(r["is_stale"] for r in results)
        if any_stale:
            print("\nRecommendation: Re-run bootstrap for stale collections.")
        else:
            print("\nAll collections current. No action needed.")

        if args.json:
            print("\nJSON output:")
            print(json.dumps(result, indent=2))

        return 1 if any_stale else 0

    return run_maintenance_command(operation, "Staleness check failed", args.collection_id)


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

    # Shared parent parser so --collection-id and -v work after the subcommand name
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    common.add_argument(
        "--collection-id",
        type=int,
        default=None,
        help="Collection ID to scope operations (default: COLLECTION_ID env var or 1)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Maintenance command to run")

    # Daily maintenance
    daily_parser = subparsers.add_parser(
        "daily", parents=[common], help="Run all daily maintenance tasks"
    )
    daily_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    daily_parser.set_defaults(func=run_daily)

    # Weekly maintenance
    weekly_parser = subparsers.add_parser(
        "weekly", parents=[common], help="Run all weekly maintenance tasks"
    )
    weekly_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    weekly_parser.add_argument(
        "--cluster-unassigned",
        action="store_true",
        help="Run HDBSCAN on unassigned pool to find new clusters",
    )
    weekly_parser.add_argument(
        "--no-auto-associate",
        action="store_false",
        dest="auto_associate",
        help="Skip auto-association of clusters to persons",
    )
    weekly_parser.set_defaults(func=run_weekly, auto_associate=True)

    # Individual tasks
    centroids_parser = subparsers.add_parser(
        "recompute-centroids", parents=[common], help="Recompute all cluster centroids"
    )
    centroids_parser.set_defaults(func=recompute_centroids)

    medoids_parser = subparsers.add_parser(
        "update-medoids", parents=[common], help="Update medoids for all clusters"
    )
    medoids_parser.set_defaults(func=update_medoids)

    cleanup_parser = subparsers.add_parser(
        "cleanup-empty", parents=[common], help="Remove empty clusters"
    )
    cleanup_parser.set_defaults(func=cleanup_empty)

    cleanup_persons_parser = subparsers.add_parser(
        "cleanup-empty-persons",
        parents=[common],
        help="Remove auto-created persons with no remaining clusters",
    )
    cleanup_persons_parser.set_defaults(func=cleanup_empty_persons)

    stats_parser = subparsers.add_parser(
        "update-stats", parents=[common], help="Update cluster statistics"
    )
    stats_parser.set_defaults(func=update_stats)

    revert_parser = subparsers.add_parser(
        "revert-singletons",
        parents=[common],
        help="Revert maintenance-created singleton clusters to unassigned pool",
    )
    revert_parser.set_defaults(func=revert_singletons)

    cluster_unassigned_parser = subparsers.add_parser(
        "cluster-unassigned",
        parents=[common],
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
        "calculate-epsilons",
        parents=[common],
        help="Calculate epsilon for clusters with NULL epsilon but 3+ faces",
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

    # Auto-associate clusters to persons
    auto_assoc_parser = subparsers.add_parser(
        "auto-associate",
        parents=[common],
        help="Auto-associate clusters to persons based on centroid similarity",
    )
    auto_assoc_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine distance threshold (default: PERSON_ASSOCIATION_THRESHOLD from config)",
    )
    auto_assoc_parser.add_argument(
        "--dry-run", action="store_true", help="Show groups without making changes"
    )
    auto_assoc_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    auto_assoc_parser.set_defaults(func=auto_associate)

    # Health check
    health_parser = subparsers.add_parser(
        "health", parents=[common], help="Check clustering system health"
    )
    health_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    health_parser.set_defaults(func=health_check)

    # Staleness check
    staleness_parser = subparsers.add_parser(
        "check-staleness", parents=[common], help="Check if HDBSCAN bootstrap needs re-running"
    )
    staleness_parser.add_argument(
        "--threshold",
        type=float,
        default=1.25,
        help="Growth ratio threshold for staleness (default: 1.25 = 25%% growth)",
    )
    staleness_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    staleness_parser.set_defaults(func=check_staleness)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
