#!/usr/bin/env python3
"""
Reset scene analysis data and embeddings.

Deletes all scene analysis related data from the database:
- scene_analysis records
- photo_tag and detection_tag records
- analysis_output records (for scene analysis models)
- prompt_embedding embeddings (optionally)
- processing_status for scene_analysis stage

Usage:
    uv run python scripts/reset_scene_analysis.py              # Reset processing data only
    uv run python scripts/reset_scene_analysis.py --embeddings # Also clear prompt embeddings
    uv run python scripts/reset_scene_analysis.py --all        # Reset everything including categories
    uv run python scripts/reset_scene_analysis.py --dry-run    # Show what would be deleted
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


def get_counts(cursor) -> dict:
    """Get current record counts for scene analysis tables."""
    counts = {}

    cursor.execute("SELECT COUNT(*) FROM scene_analysis")
    counts["scene_analysis"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM photo_tag")
    counts["photo_tag"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detection_tag")
    counts["detection_tag"] = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM analysis_output WHERE model_name IN ('apple_vision_classify', 'mobileclip')"
    )
    counts["analysis_output"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM processing_status WHERE stage = 'scene_analysis'")
    counts["processing_status"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM prompt_embedding WHERE embedding IS NOT NULL")
    counts["prompt_embeddings"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM prompt_embedding")
    counts["prompt_records"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM prompt_category")
    counts["prompt_categories"] = cursor.fetchone()[0]

    return counts


def reset_scene_analysis(
    pool, clear_embeddings: bool = False, clear_all: bool = False, dry_run: bool = False
):
    """Reset scene analysis data."""
    with pool.get_connection() as conn:
        with conn.cursor() as cursor:
            # Get current counts
            counts = get_counts(cursor)

            logger.info("Current record counts:")
            logger.info(f"  scene_analysis:     {counts['scene_analysis']:,}")
            logger.info(f"  photo_tag:          {counts['photo_tag']:,}")
            logger.info(f"  detection_tag:      {counts['detection_tag']:,}")
            logger.info(f"  analysis_output:    {counts['analysis_output']:,}")
            logger.info(f"  processing_status:  {counts['processing_status']:,}")
            logger.info(f"  prompt_embeddings:  {counts['prompt_embeddings']:,}")
            logger.info(f"  prompt_records:     {counts['prompt_records']:,}")
            logger.info(f"  prompt_categories:  {counts['prompt_categories']:,}")

            if dry_run:
                logger.info("\n[DRY RUN] Would delete:")
                logger.info("  - All scene_analysis records")
                logger.info("  - All photo_tag records")
                logger.info("  - All detection_tag records")
                logger.info("  - All analysis_output records for scene analysis models")
                logger.info("  - All processing_status records for scene_analysis stage")
                if clear_embeddings or clear_all:
                    logger.info("  - All prompt embeddings (set to NULL)")
                if clear_all:
                    logger.info("  - All prompt_embedding records")
                    logger.info("  - All prompt_category records")
                return

            # Delete in order respecting foreign keys
            logger.info("\nDeleting records...")

            # 1. Delete detection_tag (references prompt_embedding)
            cursor.execute("DELETE FROM detection_tag")
            logger.info(f"  Deleted {cursor.rowcount:,} detection_tag records")

            # 2. Delete photo_tag (references prompt_embedding)
            cursor.execute("DELETE FROM photo_tag")
            logger.info(f"  Deleted {cursor.rowcount:,} photo_tag records")

            # 3. Delete scene_analysis (references analysis_output)
            cursor.execute("DELETE FROM scene_analysis")
            logger.info(f"  Deleted {cursor.rowcount:,} scene_analysis records")

            # 4. Delete analysis_output for scene analysis models
            cursor.execute(
                "DELETE FROM analysis_output WHERE model_name IN ('apple_vision_classify', 'mobileclip')"
            )
            logger.info(f"  Deleted {cursor.rowcount:,} analysis_output records")

            # 5. Delete processing_status for scene_analysis stage
            cursor.execute("DELETE FROM processing_status WHERE stage = 'scene_analysis'")
            logger.info(f"  Deleted {cursor.rowcount:,} processing_status records")

            # 6. Optionally clear embeddings
            if clear_embeddings or clear_all:
                cursor.execute(
                    "UPDATE prompt_embedding SET embedding = NULL, embedding_computed_at = NULL"
                )
                logger.info(f"  Cleared {cursor.rowcount:,} prompt embeddings")

            # 7. Optionally delete all prompt data
            if clear_all:
                cursor.execute("DELETE FROM prompt_embedding")
                logger.info(f"  Deleted {cursor.rowcount:,} prompt_embedding records")

                cursor.execute("DELETE FROM prompt_category")
                logger.info(f"  Deleted {cursor.rowcount:,} prompt_category records")

            conn.commit()
            logger.info("\nReset complete.")

            # Show final counts
            final_counts = get_counts(cursor)
            logger.info("\nFinal record counts:")
            logger.info(f"  scene_analysis:     {final_counts['scene_analysis']:,}")
            logger.info(f"  photo_tag:          {final_counts['photo_tag']:,}")
            logger.info(f"  detection_tag:      {final_counts['detection_tag']:,}")
            logger.info(f"  analysis_output:    {final_counts['analysis_output']:,}")
            logger.info(f"  processing_status:  {final_counts['processing_status']:,}")
            logger.info(f"  prompt_embeddings:  {final_counts['prompt_embeddings']:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Reset scene analysis data and embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Reset processing data only (keeps embeddings):
    uv run python scripts/reset_scene_analysis.py

  Reset processing data and clear embeddings:
    uv run python scripts/reset_scene_analysis.py --embeddings

  Reset everything including categories:
    uv run python scripts/reset_scene_analysis.py --all

  Preview what would be deleted:
    uv run python scripts/reset_scene_analysis.py --dry-run
""",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Also clear prompt embeddings (requires re-running seed_prompts.py)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete everything including prompt categories (requires re-running migration)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without making changes",
    )
    args = parser.parse_args()

    # Validate DATABASE_URL
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        logger.error("Set it with: export DATABASE_URL='postgresql://localhost/photodb'")
        sys.exit(1)

    from photodb.database.connection import ConnectionPool

    pool = ConnectionPool()

    try:
        reset_scene_analysis(
            pool,
            clear_embeddings=args.embeddings,
            clear_all=args.all,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        pool.close_all()


if __name__ == "__main__":
    main()
