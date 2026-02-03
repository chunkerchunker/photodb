#!/usr/bin/env python3
"""
Reset unassigned faces for reprocessing.

Changes cluster_status from 'unassigned' to NULL, making them eligible
for the normal clustering flow. Also resets the clustering processing
status for affected photos so they will be picked up by the clustering
stage. Cannot-link constraints are preserved and will still be respected
during reprocessing.

Usage:
    uv run python scripts/reset_unassigned.py
    uv run python scripts/reset_unassigned.py --dry-run
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
    """Get current unassigned face statistics."""
    with pool.get_connection() as conn:
        with conn.cursor() as cur:
            stats = {}

            cur.execute(
                "SELECT COUNT(*) FROM person_detection WHERE face_bbox_x IS NOT NULL"
            )
            stats["total_faces"] = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM person_detection WHERE cluster_status = 'unassigned'"
            )
            stats["unassigned_faces"] = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM person_detection WHERE cluster_id IS NOT NULL"
            )
            stats["clustered_faces"] = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(DISTINCT photo_id)
                FROM person_detection
                WHERE cluster_id IS NULL
                  AND cluster_status = 'unassigned'
                """
            )
            stats["photos_with_unassigned"] = cur.fetchone()[0]

            return stats


def reset_unassigned(pool, dry_run: bool = False) -> dict:
    """
    Reset unassigned faces for reprocessing.

    Args:
        pool: Database connection pool
        dry_run: If True, show what would be done without making changes

    Returns:
        Dictionary with 'faces_reset' and 'photos_reset' counts
    """
    results = {
        "faces_reset": 0,
        "photos_reset": 0,
    }

    if dry_run:
        logger.info("DRY RUN - no changes will be made")

    with pool.get_connection() as conn:
        with conn.cursor() as cur:
            # Get photo IDs that have unassigned faces before resetting
            cur.execute("""
                SELECT DISTINCT photo_id
                FROM person_detection
                WHERE cluster_id IS NULL
                  AND cluster_status = 'unassigned'
            """)
            photo_ids = [row[0] for row in cur.fetchall()]

            # Count faces to reset
            cur.execute("""
                SELECT COUNT(*)
                FROM person_detection
                WHERE cluster_id IS NULL
                  AND cluster_status = 'unassigned'
            """)
            results["faces_reset"] = cur.fetchone()[0]
            results["photos_reset"] = len(photo_ids)

            if not dry_run:
                # Reset face status
                cur.execute("""
                    UPDATE person_detection
                    SET cluster_status = NULL,
                        unassigned_since = NULL
                    WHERE cluster_id IS NULL
                      AND cluster_status = 'unassigned'
                """)
                logger.info(f"Reset {cur.rowcount} detection clustering statuses")

                # Reset clustering processing status for affected photos
                if photo_ids:
                    cur.execute("""
                        DELETE FROM processing_status
                        WHERE stage = 'clustering'
                          AND photo_id = ANY(%s)
                    """, (photo_ids,))
                    logger.info(f"Reset {cur.rowcount} photo processing statuses")

                conn.commit()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reset unassigned faces for reprocessing"
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
        print(f"  Total faces:        {stats['total_faces']}")
        print(f"  Clustered:          {stats['clustered_faces']}")
        print(f"  Unassigned:         {stats['unassigned_faces']}")
        print(f"  Photos affected:    {stats['photos_with_unassigned']}")
        print("=====================\n")

        if stats["unassigned_faces"] == 0:
            print("Nothing to reset - no unassigned faces found.")
            return 0

        # Confirm
        if not args.dry_run and not args.yes:
            print(
                "This will reset unassigned faces so they can be reprocessed by clustering.\n"
                "Cannot-link constraints will be preserved.\n"
            )
            response = input("Continue? [y/N] ")
            if response.lower() != "y":
                print("Aborted.")
                return 1

        # Do the reset
        results = reset_unassigned(pool, dry_run=args.dry_run)

        print("\n=== Results ===")
        print(f"  Faces reset:         {results['faces_reset']}")
        print(f"  Photos reset:        {results['photos_reset']}")
        print("===============\n")

        if args.dry_run:
            print("DRY RUN - no changes were made")
        else:
            print("Ready for re-clustering:")
            print("  uv run process-local /path/to/photos --stage clustering")

        return 0

    except Exception as e:
        logger.error(f"Reset failed: {e}")
        logger.exception("Full traceback:")
        return 1
    finally:
        pool.close_all()


if __name__ == "__main__":
    sys.exit(main())
