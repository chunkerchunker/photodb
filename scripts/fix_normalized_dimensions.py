#!/usr/bin/env python3
"""
Find and fix photos with incorrect normalized_width/normalized_height in the database.

This script detects photos where the stored dimensions don't match the actual
PNG file dimensions (typically caused by EXIF rotation not being accounted for
when storing dimensions).

Usage:
    # Dry run - show what would be fixed (scans all photos)
    uv run python scripts/fix_normalized_dimensions.py

    # Check specific photo IDs only
    uv run python scripts/fix_normalized_dimensions.py 26314 26315 26316

    # Actually fix the issues
    uv run python scripts/fix_normalized_dimensions.py --fix

    # Fix specific photos
    uv run python scripts/fix_normalized_dimensions.py --fix 26314 26315
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_connection() -> psycopg.Connection:
    """Get database connection."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    return psycopg.connect(database_url)  # type: ignore[arg-type]


def find_dimension_mismatches(conn, photo_ids: list[int] | None = None) -> list[dict]:
    """Find photos where stored dimensions don't match actual file dimensions."""
    mismatches = []

    with conn.cursor(row_factory=dict_row) as cursor:
        if photo_ids:
            # Check specific photos
            cursor.execute("""
                SELECT id, normalized_path, normalized_width, normalized_height
                FROM photo
                WHERE id = ANY(%s)
                  AND normalized_path IS NOT NULL
                ORDER BY id
            """, (photo_ids,))
        else:
            # Get all photos with normalized paths
            cursor.execute("""
                SELECT id, normalized_path, normalized_width, normalized_height
                FROM photo
                WHERE normalized_path IS NOT NULL
                  AND normalized_width IS NOT NULL
                  AND normalized_height IS NOT NULL
                ORDER BY id
            """)
        photos = cursor.fetchall()

    print(f"Checking {len(photos)} photo{'s' if len(photos) != 1 else ''}...")

    for i, photo in enumerate(photos):
        if (i + 1) % 1000 == 0:
            print(f"  Checked {i + 1}/{len(photos)}...")

        path = Path(photo["normalized_path"])
        if not path.exists():
            continue

        try:
            with Image.open(path) as img:
                actual_width = img.width
                actual_height = img.height
        except Exception as e:
            print(f"  WARNING: Could not open {path}: {e}")
            continue

        stored_width = photo["normalized_width"]
        stored_height = photo["normalized_height"]

        # Check if dimensions match
        if actual_width != stored_width or actual_height != stored_height:
            mismatches.append({
                "id": photo["id"],
                "path": str(path),
                "stored_width": stored_width,
                "stored_height": stored_height,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "is_swapped": (
                    actual_width == stored_height and actual_height == stored_width
                ),
            })

    return mismatches


def fix_dimensions(conn, mismatches: list[dict], dry_run: bool = True):
    """Fix dimension mismatches in the database."""
    if not mismatches:
        print("\nNo mismatches found - database is consistent!")
        return

    print(f"\nFound {len(mismatches)} photos with dimension mismatches:")
    print()

    swapped_count = sum(1 for m in mismatches if m["is_swapped"])
    other_count = len(mismatches) - swapped_count

    print(f"  - {swapped_count} with swapped dimensions (width/height reversed)")
    print(f"  - {other_count} with other dimension differences")
    print()

    # Show details for first few
    print("Sample mismatches:")
    for m in mismatches[:10]:
        swap_note = " [SWAPPED]" if m["is_swapped"] else ""
        print(f"  Photo {m['id']}: stored {m['stored_width']}x{m['stored_height']} "
              f"-> actual {m['actual_width']}x{m['actual_height']}{swap_note}")

    if len(mismatches) > 10:
        print(f"  ... and {len(mismatches) - 10} more")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run with --fix to update the database.")
        return

    print("\nFixing dimensions...")
    with conn.cursor() as cursor:
        for m in mismatches:
            cursor.execute("""
                UPDATE photo
                SET normalized_width = %s, normalized_height = %s, updated_at = NOW()
                WHERE id = %s
            """, (m["actual_width"], m["actual_height"], m["id"]))

    conn.commit()
    print(f"Fixed {len(mismatches)} photos.")


def main():
    parser = argparse.ArgumentParser(
        description="Find and fix photos with incorrect normalized dimensions"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Actually fix the issues (default is dry run)",
    )
    parser.add_argument(
        "photo_ids",
        nargs="*",
        type=int,
        help="Specific photo IDs to check (default: check all)",
    )
    args = parser.parse_args()

    photo_ids = args.photo_ids if args.photo_ids else None
    conn = get_connection()

    try:
        mismatches = find_dimension_mismatches(conn, photo_ids)
        fix_dimensions(conn, mismatches, dry_run=not args.fix)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
