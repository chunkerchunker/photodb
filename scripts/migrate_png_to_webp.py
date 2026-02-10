#!/usr/bin/env python3
"""
Migrate existing normalized images to WebP format in the med/ subdirectory.

This script:
1. Finds all photos with med_path in the database
2. Converts images to WebP (lossy, quality 95)
3. Moves them to a med/ subdirectory structure
4. Updates the database paths
5. Optionally deletes the original files

The new structure will be:
    {IMG_PATH}/med/{hash}.webp

Instead of the old structure:
    {IMG_PATH}/{hash}.png

Usage:
    # Dry run - show what would be converted/moved
    uv run python scripts/migrate_png_to_webp.py

    # Convert and move images (keeps originals)
    uv run python scripts/migrate_png_to_webp.py --convert

    # Convert, move, and delete originals
    uv run python scripts/migrate_png_to_webp.py --convert --delete-old

    # Convert specific photo IDs only
    uv run python scripts/migrate_png_to_webp.py --convert 1234 5678

    # Use different quality (default 95)
    uv run python scripts/migrate_png_to_webp.py --convert --quality 90

    # Parallel processing with multiple workers
    uv run python scripts/migrate_png_to_webp.py --convert --workers 8
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from PIL import Image
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Thread-local storage for database connections
_thread_local = threading.local()

# Conversion stats
stats_lock = threading.Lock()
stats = {
    "converted": 0,
    "skipped": 0,
    "failed": 0,
    "already_webp": 0,
    "bytes_saved": 0,
}


def get_connection() -> psycopg.Connection:
    """Get database connection."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    return psycopg.connect(database_url)


def get_thread_connection() -> psycopg.Connection:
    """Get thread-local database connection."""
    if not hasattr(_thread_local, "conn") or _thread_local.conn.closed:
        _thread_local.conn = get_connection()
    return _thread_local.conn


def find_photos_to_migrate(conn, photo_ids: list[int] | None = None) -> list[dict]:
    """Find photos that need migration (PNG or not in med/ subdir)."""
    with conn.cursor(row_factory=dict_row) as cursor:
        if photo_ids:
            cursor.execute(
                """
                SELECT id, med_path
                FROM photo
                WHERE id = ANY(%s)
                  AND med_path IS NOT NULL
                ORDER BY id
                """,
                (photo_ids,),
            )
        else:
            cursor.execute(
                """
                SELECT id, med_path
                FROM photo
                WHERE med_path IS NOT NULL
                ORDER BY id
                """
            )
        return cursor.fetchall()


def compute_new_path(old_path: Path) -> Path:
    """
    Compute the new path with med/ subdirectory and .webp extension.

    Examples:
        /photos/processed/abc123.png -> /photos/processed/med/abc123.webp
        /photos/processed/abc123.webp -> /photos/processed/med/abc123.webp
        /photos/processed/med/abc123.png -> /photos/processed/med/abc123.webp
        /photos/processed/med/abc123.webp -> /photos/processed/med/abc123.webp (no change)
    """
    # Get the filename without extension
    stem = old_path.stem

    # Check if already in med/ subdirectory
    if old_path.parent.name == "med":
        # Already in med/, just change extension
        new_path = old_path.with_suffix(".webp")
    else:
        # Need to add med/ subdirectory
        new_path = old_path.parent / "med" / f"{stem}.webp"

    return new_path


def needs_migration(old_path: Path, new_path: Path) -> tuple[bool, str]:
    """
    Check if migration is needed.

    Returns:
        (needs_migration, reason)
    """
    if old_path == new_path and new_path.exists():
        return False, "already migrated"

    if not old_path.exists():
        if new_path.exists():
            return False, "already migrated (source missing)"
        return False, "source not found"

    return True, "needs migration"


def convert_single_photo(
    photo: dict,
    quality: int,
    delete_old: bool,
    dry_run: bool,
) -> tuple[bool, str, int]:
    """
    Convert/move a single photo to the new structure.

    Returns:
        (success, message, bytes_saved)
    """
    photo_id = photo["id"]
    old_path = Path(photo["med_path"])
    new_path = compute_new_path(old_path)

    # Check if migration is needed
    needs_work, reason = needs_migration(old_path, new_path)

    if not needs_work:
        return False, f"Skip {photo_id}: {reason}", 0

    if dry_run:
        old_size = old_path.stat().st_size if old_path.exists() else 0
        is_png = old_path.suffix.lower() == ".png"
        # Estimate savings: ~40% for PNG->WebP, ~0% for WebP->WebP
        estimated_savings = int(old_size * 0.4) if is_png else 0
        return True, f"Would migrate: {old_path} -> {new_path}", estimated_savings

    try:
        # Get original file size
        old_size = old_path.stat().st_size

        # Create med/ subdirectory if needed
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if source is already WebP
        is_webp = old_path.suffix.lower() == ".webp"

        if is_webp and old_path.parent.name == "med":
            # Already WebP in correct location, nothing to do
            return False, f"Skip {photo_id}: already in correct location", 0

        # Convert/copy to new location
        with Image.open(old_path) as img:
            # Ensure RGB mode for WebP
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(new_path, format="WEBP", quality=quality, method=6)

        # Get new file size
        new_size = new_path.stat().st_size
        bytes_saved = old_size - new_size

        # Update database
        conn = get_thread_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE photo SET med_path = %s, updated_at = NOW() WHERE id = %s",
                (str(new_path), photo_id),
            )
        conn.commit()

        # Delete original if requested and different from new path
        if delete_old and old_path != new_path:
            old_path.unlink()

        action = "Converted" if not is_webp else "Moved"
        return True, f"{action}: {old_path.name} -> med/{new_path.name} (saved {bytes_saved:,} bytes)", bytes_saved

    except Exception as e:
        # Clean up partial file if it exists and is different from source
        if new_path.exists() and new_path != old_path:
            try:
                new_path.unlink()
            except Exception:
                pass
        return False, f"Failed {photo_id}: {e}", 0


def migrate_photos(
    photos: list[dict],
    quality: int,
    delete_old: bool,
    dry_run: bool,
    workers: int,
) -> None:
    """Migrate photos to new structure."""
    global stats

    if not photos:
        print("\nNo photos found to migrate.")
        return

    print(f"\nFound {len(photos)} photo{'s' if len(photos) != 1 else ''} with med_path")

    if dry_run:
        print("\n[DRY RUN] Showing what would be migrated:\n")
    else:
        print(f"\nMigrating to WebP in med/ subdir (quality={quality}, workers={workers})...\n")

    def process_photo(photo):
        success, message, bytes_saved = convert_single_photo(
            photo, quality, delete_old, dry_run
        )
        with stats_lock:
            if success:
                stats["converted"] += 1
                stats["bytes_saved"] += bytes_saved
            elif "already" in message.lower():
                stats["already_webp"] += 1
            elif "not found" in message.lower():
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
        return success, message

    # Process photos
    if workers == 1:
        for i, photo in enumerate(photos):
            success, message = process_photo(photo)
            if i < 10 or not success or "Failed" in message:
                print(f"  {message}")
            elif i == 10:
                print("  ...")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_photo, photo): photo for photo in photos}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                success, message = future.result()
                if completed <= 10 or "Failed" in message:
                    print(f"  [{completed}/{len(photos)}] {message}")
                elif completed == 11:
                    print("  ...")
                elif completed % 100 == 0:
                    print(f"  [{completed}/{len(photos)}] Progress...")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Migrated:        {stats['converted']}")
    print(f"  Already done:    {stats['already_webp']}")
    print(f"  Skipped:         {stats['skipped']}")
    print(f"  Failed:          {stats['failed']}")

    if stats["bytes_saved"] > 0:
        mb_saved = stats["bytes_saved"] / (1024 * 1024)
        if dry_run:
            print(f"  Estimated savings: {mb_saved:.1f} MB")
        else:
            print(f"  Space saved: {mb_saved:.1f} MB")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run with --convert to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate normalized images to WebP format in med/ subdirectory"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Actually migrate images (default is dry run)",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete old med_path files (PNGs) after successful migration. Does NOT touch orig_path source images.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="WebP quality (0-100, default 95)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default 4)",
    )
    parser.add_argument(
        "photo_ids",
        nargs="*",
        type=int,
        help="Specific photo IDs to migrate (default: all)",
    )
    args = parser.parse_args()

    if args.quality < 0 or args.quality > 100:
        print("ERROR: Quality must be between 0 and 100")
        sys.exit(1)

    photo_ids = args.photo_ids if args.photo_ids else None
    dry_run = not args.convert

    conn = get_connection()
    try:
        photos = find_photos_to_migrate(conn, photo_ids)
        migrate_photos(
            photos,
            quality=args.quality,
            delete_old=args.delete_old,
            dry_run=dry_run,
            workers=args.workers,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
