#!/usr/bin/env python3
"""
Backfill full-size WebP images for existing photos.

This script:
1. Finds all photos with orig_path but no full_path
2. Reads the original image from INGEST_PATH
3. Creates a full-size WebP (with EXIF rotation baked in)
4. Saves to {IMG_PATH}/full/{hash}.webp
5. Updates the database full_path column

Usage:
    # Dry run - show what would be processed
    uv run python scripts/backfill_full_size.py

    # Create full-size WebP images
    uv run python scripts/backfill_full_size.py --convert

    # Process specific photo IDs only
    uv run python scripts/backfill_full_size.py --convert 1234 5678

    # Use different quality (default 95)
    uv run python scripts/backfill_full_size.py --convert --quality 90

    # Parallel processing with multiple workers
    uv run python scripts/backfill_full_size.py --convert --workers 8
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pillow_heif
from PIL import Image, ImageOps
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Register HEIF opener for HEIC support
pillow_heif.register_heif_opener()

# Thread-local storage for database connections
_thread_local = threading.local()

# Conversion stats
stats_lock = threading.Lock()
stats = {
    "converted": 0,
    "skipped": 0,
    "failed": 0,
    "already_exists": 0,
    "missing_source": 0,
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


def find_photos_to_backfill(conn, photo_ids: list[int] | None = None) -> list[dict]:
    """Find photos that need full_path backfill."""
    with conn.cursor(row_factory=dict_row) as cursor:
        if photo_ids:
            cursor.execute(
                """
                SELECT id, orig_path
                FROM photo
                WHERE id = ANY(%s)
                  AND orig_path IS NOT NULL
                ORDER BY id
                """,
                (photo_ids,),
            )
        else:
            cursor.execute(
                """
                SELECT id, orig_path
                FROM photo
                WHERE orig_path IS NOT NULL
                  AND full_path IS NULL
                ORDER BY id
                """
            )
        return [dict(row) for row in cursor.fetchall()]


def generate_photo_id(orig_path: Path) -> str:
    """Generate a unique photo ID based on the original path."""
    return hashlib.sha256(str(orig_path).encode()).hexdigest()[:16]


def save_as_webp(image: Image.Image, output_path: Path, quality: int, original_path: Path) -> None:
    """Save image as WebP with EXIF rotation baked in."""
    # Apply EXIF orientation by opening original to get EXIF data
    try:
        with Image.open(original_path) as orig:
            # Get EXIF and apply orientation
            image = ImageOps.exif_transpose(image)
    except Exception:
        # If we can't read EXIF, just use the image as-is
        pass

    # Convert to RGB if needed (WebP doesn't support all modes)
    if image.mode in ("RGBA", "LA"):
        # Keep alpha for RGBA
        pass
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Save with high quality lossy compression
    image.save(
        output_path,
        "WEBP",
        quality=quality,
        method=6,  # Best compression method
    )


def process_photo(
    photo: dict,
    ingest_path: Path,
    output_dir: Path,
    quality: int,
    dry_run: bool = True,
) -> tuple[int, str, str | None]:
    """
    Process a single photo to create full-size WebP.

    Returns:
        (photo_id, status, new_path or error message)
    """
    photo_id = photo["id"]
    orig_path = photo["orig_path"]

    # Build full source path
    source_path = ingest_path / orig_path
    if not source_path.exists():
        return (photo_id, "missing_source", str(source_path))

    # Generate output path
    output_filename = f"{generate_photo_id(Path(orig_path))}.webp"
    output_path = output_dir / output_filename

    # Check if output already exists
    if output_path.exists():
        return (photo_id, "already_exists", str(output_path))

    if dry_run:
        return (photo_id, "would_convert", str(output_path))

    try:
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open and convert image
        with Image.open(source_path) as img:
            # Apply EXIF orientation
            img = ImageOps.exif_transpose(img)

            # Convert to RGB if needed (WebP doesn't support all modes)
            if img.mode in ("RGBA", "LA"):
                # Keep alpha for RGBA
                pass
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Save as WebP
            img.save(
                output_path,
                "WEBP",
                quality=quality,
                method=6,  # Best compression method
            )

        # Update database
        conn = get_thread_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE photo SET full_path = %s, updated_at = NOW() WHERE id = %s",
                (str(output_path), photo_id),
            )
        conn.commit()

        return (photo_id, "converted", str(output_path))

    except Exception as e:
        return (photo_id, "failed", str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Backfill full-size WebP images for existing photos"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Actually create full-size WebP images (default is dry run)",
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
        default=1,
        help="Number of parallel workers (default 1)",
    )
    parser.add_argument(
        "photo_ids",
        nargs="*",
        type=int,
        help="Specific photo IDs to process (default: all without full_path)",
    )

    args = parser.parse_args()
    dry_run = not args.convert

    # Get paths from environment
    ingest_path = Path(os.getenv("INGEST_PATH", "./photos/raw"))
    img_path = Path(os.getenv("IMG_PATH", "./photos/processed"))
    output_dir = img_path / "full"

    print(f"Source path: {ingest_path}")
    print(f"Output path: {output_dir}")
    print(f"Quality: {args.quality}")
    print(f"Workers: {args.workers}")
    print(f"Mode: {'DRY RUN' if dry_run else 'CONVERT'}")
    print()

    # Find photos to process
    conn = get_connection()
    photo_ids = args.photo_ids if args.photo_ids else None
    photos = find_photos_to_backfill(conn, photo_ids)
    conn.close()

    if not photos:
        print("No photos need full-size backfill")
        return

    print(f"Found {len(photos)} photos to process")
    print()

    # Process photos
    def update_stats(status: str):
        with stats_lock:
            if status == "converted":
                stats["converted"] += 1
            elif status == "would_convert":
                stats["converted"] += 1  # Count as would be converted in dry run
            elif status == "already_exists":
                stats["already_exists"] += 1
            elif status == "missing_source":
                stats["missing_source"] += 1
            elif status == "failed":
                stats["failed"] += 1
            else:
                stats["skipped"] += 1

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_photo, photo, ingest_path, output_dir, args.quality, dry_run
                ): photo
                for photo in photos
            }

            for future in as_completed(futures):
                photo_id, status, result = future.result()
                update_stats(status)

                if status == "failed":
                    print(f"  FAILED photo {photo_id}: {result}")
                elif status == "missing_source":
                    print(f"  MISSING photo {photo_id}: {result}")
                elif status == "converted":
                    print(f"  Converted photo {photo_id} -> {result}")
                elif status == "would_convert":
                    print(f"  Would convert photo {photo_id} -> {result}")
                elif status == "already_exists":
                    print(f"  Already exists photo {photo_id}: {result}")
    else:
        for photo in photos:
            photo_id, status, result = process_photo(
                photo, ingest_path, output_dir, args.quality, dry_run
            )
            update_stats(status)

            if status == "failed":
                print(f"  FAILED photo {photo_id}: {result}")
            elif status == "missing_source":
                print(f"  MISSING photo {photo_id}: {result}")
            elif status == "converted":
                print(f"  Converted photo {photo_id} -> {result}")
            elif status == "would_convert":
                print(f"  Would convert photo {photo_id} -> {result}")
            elif status == "already_exists":
                print(f"  Already exists photo {photo_id}: {result}")

    print()
    print("Summary:")
    print(f"  {'Would convert' if dry_run else 'Converted'}: {stats['converted']}")
    print(f"  Already exists: {stats['already_exists']}")
    print(f"  Missing source: {stats['missing_source']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")

    if dry_run and stats["converted"] > 0:
        print()
        print("Run with --convert to create the full-size WebP images")


if __name__ == "__main__":
    main()
