#!/usr/bin/env python3
"""
Reset prompts and clear associated tags.

Usage:
    uv run python scripts/reset_prompts.py
"""

import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def reset_prompts(pool):
    """Delete all prompts and associated tags."""
    with pool.transaction() as conn:
        with conn.cursor() as cursor:
            logger.info("Deleting detection tags...")
            cursor.execute("DELETE FROM detection_tag")
            
            logger.info("Deleting photo tags...")
            cursor.execute("DELETE FROM photo_tag")
            
            logger.info("Deleting prompt embeddings...")
            cursor.execute("DELETE FROM prompt_embedding")
            
            logger.info("Deleting prompt categories...")
            cursor.execute("DELETE FROM prompt_category")
            
            logger.info("All prompt data cleared.")


def main():
    # Validate DATABASE_URL is set
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        logger.error("Set it with: export DATABASE_URL='postgresql://localhost/photodb'")
        sys.exit(1)

    from photodb.database.connection import ConnectionPool

    pool = ConnectionPool()

    try:
        response = input("This will DELETE ALL PROMPTS and TAGS. Are you sure? [y/N] ")
        if response.lower() != "y":
            logger.info("Aborted.")
            return

        reset_prompts(pool)
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        sys.exit(1)
    finally:
        pool.close_all()


if __name__ == "__main__":
    main()
