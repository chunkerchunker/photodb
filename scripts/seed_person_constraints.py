#!/usr/bin/env python3
"""
Backfill cluster_person_must_link rows for existing cluster-person assignments.

Usage:
    uv run python scripts/seed_person_constraints.py --dry-run   # Preview
    uv run python scripts/seed_person_constraints.py              # Run
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def seed_must_links(pool, dry_run: bool = False) -> int:
    """Insert must-link rows for all clusters that currently have a person_id."""
    with pool.get_connection() as conn:
        with conn.cursor() as cursor:
            if dry_run:
                cursor.execute(
                    """SELECT COUNT(*) FROM cluster
                       WHERE person_id IS NOT NULL AND NOT hidden
                       AND id NOT IN (SELECT cluster_id FROM cluster_person_must_link)"""
                )
                count = cursor.fetchone()[0]
                logger.info(f"Would create {count} must-link rows")
                return count

            cursor.execute(
                """INSERT INTO cluster_person_must_link (cluster_id, person_id, collection_id)
                   SELECT id, person_id, collection_id FROM cluster
                   WHERE person_id IS NOT NULL AND NOT hidden
                   ON CONFLICT (cluster_id) DO NOTHING"""
            )
            count = cursor.rowcount
            conn.commit()
            logger.info(f"Created {count} must-link rows")
            return count


def main():
    parser = argparse.ArgumentParser(description="Seed cluster-person must-link constraints")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        logger.error("Set it with: export DATABASE_URL='postgresql://localhost/photodb'")
        sys.exit(1)

    from photodb.database.connection import ConnectionPool

    pool = ConnectionPool()

    try:
        seed_must_links(pool, dry_run=args.dry_run)
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        pool.close_all()


if __name__ == "__main__":
    main()
