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
from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that read them at module level
load_dotenv()

from .database.connection import ConnectionPool  # noqa: E402
from .utils.maintenance import MaintenanceUtilities  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    
    # Reduce noise from some libraries
    if not verbose:
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('psycopg').setLevel(logging.WARNING)


def run_daily(args):
    """Run daily maintenance tasks."""
    logger.info("Running daily maintenance tasks...")
    
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    maintenance = MaintenanceUtilities(pool)
    
    try:
        results = maintenance.run_daily_maintenance()
        
        print("\n✅ Daily maintenance completed:")
        print(f"  • Centroids recomputed: {results.get('centroids_recomputed', 0)}")
        print(f"  • Empty clusters removed: {results.get('empty_clusters_removed', 0)}")
        print(f"  • Statistics updated: {results.get('statistics_updated', 0)}")
        
        if args.json:
            print("\nJSON output:")
            print(json.dumps(results, indent=2))
        
        return 0
    except Exception as e:
        logger.error(f"Daily maintenance failed: {e}")
        return 1
    finally:
        pool.close_all()


def run_weekly(args):
    """Run weekly maintenance tasks."""
    logger.info("Running weekly maintenance tasks...")
    
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    maintenance = MaintenanceUtilities(pool)
    
    try:
        results = maintenance.run_weekly_maintenance(
            similarity_threshold=args.similarity_threshold
        )
        
        print("\n✅ Weekly maintenance completed:")
        print(f"  • Centroids recomputed: {results.get('centroids_recomputed', 0)}")
        print(f"  • Empty clusters removed: {results.get('empty_clusters_removed', 0)}")
        print(f"  • Statistics updated: {results.get('statistics_updated', 0)}")
        print(f"  • Medoids updated: {results.get('medoids_updated', 0)}")
        print(f"  • Clusters merged: {results.get('clusters_merged', 0)}")
        
        if args.json:
            print("\nJSON output:")
            print(json.dumps(results, indent=2))
        
        return 0
    except Exception as e:
        logger.error(f"Weekly maintenance failed: {e}")
        return 1
    finally:
        pool.close_all()


def recompute_centroids(args):
    """Recompute all cluster centroids."""
    logger.info("Recomputing cluster centroids...")
    
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    maintenance = MaintenanceUtilities(pool)
    
    try:
        count = maintenance.recompute_all_centroids()
        print(f"✅ Recomputed centroids for {count} clusters")
        return 0
    except Exception as e:
        logger.error(f"Failed to recompute centroids: {e}")
        return 1
    finally:
        pool.close_all()


def update_medoids(args):
    """Update medoids for all clusters."""
    logger.info("Updating cluster medoids...")
    
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    maintenance = MaintenanceUtilities(pool)
    
    try:
        count = maintenance.update_all_medoids()
        print(f"✅ Updated medoids for {count} clusters")
        return 0
    except Exception as e:
        logger.error(f"Failed to update medoids: {e}")
        return 1
    finally:
        pool.close_all()


def merge_similar(args):
    """Find and merge similar clusters."""
    logger.info(f"Finding similar clusters (threshold: {args.similarity_threshold})...")
    
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    maintenance = MaintenanceUtilities(pool)
    
    try:
        count = maintenance.find_and_merge_similar_clusters(
            similarity_threshold=args.similarity_threshold
        )
        print(f"✅ Merged {count} similar clusters")
        return 0
    except Exception as e:
        logger.error(f"Failed to merge clusters: {e}")
        return 1
    finally:
        pool.close_all()


def cleanup_empty(args):
    """Remove empty clusters."""
    logger.info("Cleaning up empty clusters...")
    
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    maintenance = MaintenanceUtilities(pool)
    
    try:
        count = maintenance.cleanup_empty_clusters()
        print(f"✅ Removed {count} empty clusters")
        return 0
    except Exception as e:
        logger.error(f"Failed to cleanup clusters: {e}")
        return 1
    finally:
        pool.close_all()


def update_stats(args):
    """Update cluster statistics."""
    logger.info("Updating cluster statistics...")
    
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    maintenance = MaintenanceUtilities(pool)
    
    try:
        count = maintenance.update_cluster_statistics()
        print(f"✅ Updated statistics for {count} clusters")
        return 0
    except Exception as e:
        logger.error(f"Failed to update statistics: {e}")
        return 1
    finally:
        pool.close_all()


def health_check(args):
    """Check clustering system health."""
    logger.info("Checking cluster health...")
    
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    pool = ConnectionPool(connection_string=database_url)
    maintenance = MaintenanceUtilities(pool)
    
    try:
        stats = maintenance.get_cluster_health_stats()
        
        print("\n📊 Cluster Health Report:")
        print(f"  • Total clusters: {stats['total_clusters']}")
        print(f"  • Empty clusters: {stats['empty_clusters']}")
        print(f"  • Clusters without centroids: {stats['clusters_without_centroids']}")
        print(f"  • Clusters without medoids: {stats['clusters_without_medoids']}")
        print(f"  • Average cluster size: {stats['avg_cluster_size']:.1f}")
        print(f"  • Cluster size range: {stats['min_cluster_size']} - {stats['max_cluster_size']}")
        print(f"  • Total faces: {stats['total_faces']}")
        print(f"  • Unclustered faces: {stats['unclustered_faces']}")
        
        if stats['total_faces'] > 0:
            clustered_pct = (1 - stats['unclustered_faces'] / stats['total_faces']) * 100
            print(f"  • Clustering coverage: {clustered_pct:.1f}%")
        
        # Check for issues
        issues = []
        if stats['empty_clusters'] > 0:
            issues.append(f"{stats['empty_clusters']} empty clusters need cleanup")
        if stats['clusters_without_centroids'] > 0:
            issues.append(f"{stats['clusters_without_centroids']} clusters missing centroids")
        if stats['clusters_without_medoids'] > 0:
            issues.append(f"{stats['clusters_without_medoids']} clusters missing medoids")
        
        if issues:
            print("\n⚠️  Issues detected:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("\n✅ No issues detected")
        
        if args.json:
            print("\nJSON output:")
            print(json.dumps(stats, indent=2))
        
        return 0 if not issues else 1
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return 1
    finally:
        pool.close_all()


def main():
    """Main entry point for maintenance CLI."""
    parser = argparse.ArgumentParser(
        description='PhotoDB Maintenance Utilities',
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
  photodb-maintenance merge-similar --similarity-threshold 0.25
  photodb-maintenance cleanup-empty
  
  # Get JSON output for automation
  photodb-maintenance daily --json
"""
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Maintenance command to run')
    
    # Daily maintenance
    daily_parser = subparsers.add_parser('daily', help='Run all daily maintenance tasks')
    daily_parser.add_argument('--json', action='store_true',
                             help='Output results as JSON')
    daily_parser.set_defaults(func=run_daily)
    
    # Weekly maintenance  
    weekly_parser = subparsers.add_parser('weekly', help='Run all weekly maintenance tasks')
    weekly_parser.add_argument('--similarity-threshold', type=float, default=0.3,
                               help='Threshold for cluster similarity (default: 0.3)')
    weekly_parser.add_argument('--json', action='store_true',
                              help='Output results as JSON')
    weekly_parser.set_defaults(func=run_weekly)
    
    # Individual tasks
    centroids_parser = subparsers.add_parser('recompute-centroids',
                                             help='Recompute all cluster centroids')
    centroids_parser.set_defaults(func=recompute_centroids)
    
    medoids_parser = subparsers.add_parser('update-medoids',
                                           help='Update medoids for all clusters')
    medoids_parser.set_defaults(func=update_medoids)
    
    merge_parser = subparsers.add_parser('merge-similar',
                                         help='Find and merge similar clusters')
    merge_parser.add_argument('--similarity-threshold', type=float, default=0.3,
                             help='Threshold for cluster similarity (default: 0.3)')
    merge_parser.set_defaults(func=merge_similar)
    
    cleanup_parser = subparsers.add_parser('cleanup-empty',
                                           help='Remove empty clusters')
    cleanup_parser.set_defaults(func=cleanup_empty)
    
    stats_parser = subparsers.add_parser('update-stats',
                                         help='Update cluster statistics')
    stats_parser.set_defaults(func=update_stats)

    # Health check
    health_parser = subparsers.add_parser('health',
                                          help='Check clustering system health')
    health_parser.add_argument('--json', action='store_true',
                              help='Output results as JSON')
    health_parser.set_defaults(func=health_check)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())