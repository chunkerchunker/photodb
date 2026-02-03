#!/usr/bin/env python3
"""
Reset clustering data while preserving person detections.

Usage:
    uv run python scripts/reset_clustering.py
    uv run python scripts/reset_clustering.py --keep-constraints
    uv run python scripts/reset_clustering.py --keep-verified
    uv run python scripts/reset_clustering.py --dry-run
"""

import argparse
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_current_stats(pool) -> dict:
    """Get current clustering statistics."""
    with pool.get_connection() as conn:
        with conn.cursor() as cur:
            stats = {}

            cur.execute("SELECT COUNT(*) FROM person_detection WHERE face_bbox_x IS NOT NULL")
            stats["total_faces"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM person_detection WHERE cluster_id IS NOT NULL")
            stats["clustered_faces"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM cluster")
            stats["clusters"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM cluster WHERE verified = true")
            stats["verified_clusters"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM must_link")
            stats["must_links"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM cannot_link")
            stats["cannot_links"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM cluster_cannot_link")
            stats["cluster_cannot_links"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM face_match_candidate")
            stats["match_candidates"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM face_embedding")
            stats["embeddings"] = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM person_detection WHERE cluster_status = 'unassigned'"
            )
            stats["unassigned_faces"] = cur.fetchone()[0]

            return stats


def reset_clustering(
    pool,
    keep_constraints: bool = False,
    keep_verified: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Reset clustering data.

    Args:
        pool: Database connection pool
        keep_constraints: If True, preserve must_link/cannot_link constraints
        keep_verified: If True, preserve verified clusters and their assignments
        dry_run: If True, show what would be done without making changes

    Returns:
        Dictionary with counts of deleted/reset items
    """
    results = {
        "detections_reset": 0,
        "clusters_deleted": 0,
        "constraints_deleted": 0,
        "candidates_deleted": 0,
    }

    if dry_run:
        logger.info("DRY RUN - no changes will be made")

    with pool.get_connection() as conn:
        with conn.cursor() as cur:
            # Build WHERE clause for detection updates
            detection_where = "WHERE 1=1"
            if keep_verified:
                detection_where += " AND (cluster_id IS NULL OR cluster_id NOT IN (SELECT id FROM cluster WHERE verified = true))"

            # Count detections to reset
            cur.execute(f"SELECT COUNT(*) FROM person_detection {detection_where} AND cluster_id IS NOT NULL")
            results["detections_reset"] = cur.fetchone()[0]

            if not dry_run:
                # Reset detection clustering fields
                cur.execute(f"""
                    UPDATE person_detection
                    SET cluster_id = NULL,
                        cluster_status = NULL,
                        cluster_confidence = 0,
                        unassigned_since = NULL
                    {detection_where}
                """)
                logger.info(f"Reset {cur.rowcount} detection clustering assignments")

            # Delete match candidates
            cur.execute("SELECT COUNT(*) FROM face_match_candidate")
            results["candidates_deleted"] = cur.fetchone()[0]

            if not dry_run:
                cur.execute("DELETE FROM face_match_candidate")
                logger.info(f"Deleted {cur.rowcount} match candidates")

            # Handle constraints
            if not keep_constraints:
                cur.execute("SELECT COUNT(*) FROM must_link")
                ml_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM cannot_link")
                cl_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM cluster_cannot_link")
                ccl_count = cur.fetchone()[0]
                results["constraints_deleted"] = ml_count + cl_count + ccl_count

                if not dry_run:
                    cur.execute("DELETE FROM cluster_cannot_link")
                    cur.execute("DELETE FROM cannot_link")
                    cur.execute("DELETE FROM must_link")
                    logger.info(f"Deleted {results['constraints_deleted']} constraints")
            else:
                logger.info("Keeping constraints (--keep-constraints)")

            # Delete clusters
            cluster_where = ""
            if keep_verified:
                cluster_where = "WHERE verified = false"

            cur.execute(f"SELECT COUNT(*) FROM cluster {cluster_where}")
            results["clusters_deleted"] = cur.fetchone()[0]

            if not dry_run:
                cur.execute(f"DELETE FROM cluster {cluster_where}")
                logger.info(f"Deleted {cur.rowcount} clusters")

            # Reset processing status for clustering
            if not dry_run:
                if keep_verified:
                    # Only reset status for photos without verified cluster detections
                    cur.execute("""
                        DELETE FROM processing_status
                        WHERE stage = 'clustering'
                          AND photo_id NOT IN (
                              SELECT DISTINCT photo_id FROM person_detection pd
                              JOIN cluster c ON pd.cluster_id = c.id
                              WHERE c.verified = true
                          )
                    """)
                else:
                    cur.execute("DELETE FROM processing_status WHERE stage = 'clustering'")
                logger.info(f"Reset {cur.rowcount} clustering processing statuses")

            if not dry_run:
                conn.commit()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reset clustering data while preserving person detections"
    )
    parser.add_argument(
        "--keep-constraints",
        action="store_true",
        help="Preserve must_link/cannot_link constraints",
    )
    parser.add_argument(
        "--keep-verified",
        action="store_true",
        help="Preserve verified clusters and their detection assignments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Validate DATABASE_URL is set
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        logger.error("Set it with: export DATABASE_URL='postgresql://localhost/photodb'")
        sys.exit(1)

    from photodb.database.connection import ConnectionPool

    pool = ConnectionPool()

    try:
        # Show current state
        stats = get_current_stats(pool)
        print("\n=== Current State ===")
        print(f"  Total detections:   {stats['total_faces']}")
        print(f"  Face embeddings:    {stats['embeddings']}")
        print(f"  Clustered:          {stats['clustered_faces']}")
        print(f"  Unassigned:         {stats['unassigned_faces']}")
        print(f"  Clusters:           {stats['clusters']}")
        print(f"  Verified clusters:  {stats['verified_clusters']}")
        print(f"  Must-links:         {stats['must_links']}")
        print(f"  Cannot-links:       {stats['cannot_links']}")
        print(f"  Match candidates:   {stats['match_candidates']}")
        print("=====================\n")

        if stats["clusters"] == 0 and stats["clustered_faces"] == 0:
            print("Nothing to reset - no clustering data found.")
            return 0

        # Confirm
        if not args.dry_run and not args.yes:
            if args.keep_verified:
                msg = "This will reset all NON-VERIFIED clustering data."
            else:
                msg = "This will reset ALL clustering data."

            if args.keep_constraints:
                msg += " Constraints will be preserved."
            else:
                msg += " All constraints will be deleted."

            print(msg)
            response = input("\nContinue? [y/N] ")
            if response.lower() != "y":
                print("Aborted.")
                return 1

        # Do the reset
        results = reset_clustering(
            pool,
            keep_constraints=args.keep_constraints,
            keep_verified=args.keep_verified,
            dry_run=args.dry_run,
        )

        print("\n=== Results ===")
        print(f"  Detections reset:    {results['detections_reset']}")
        print(f"  Clusters deleted:    {results['clusters_deleted']}")
        print(f"  Constraints deleted: {results['constraints_deleted']}")
        print(f"  Candidates deleted:  {results['candidates_deleted']}")
        print("===============\n")

        if args.dry_run:
            print("DRY RUN - no changes were made")
        else:
            print("Ready for re-clustering:")
            print("  uv run process-local /path/to/photos --stage clustering")

        return 0

    finally:
        pool.close()


if __name__ == "__main__":
    sys.exit(main())
