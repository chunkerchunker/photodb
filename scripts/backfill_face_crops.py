#!/usr/bin/env python3
"""
Backfill face crop images for existing person_detection records.

This script:
1. Finds all person_detection records with face bbox but no face_path
2. Reads the medium image (med_path) from IMG_PATH
3. Crops the face using stored bounding box coordinates
4. Saves to {IMG_PATH}/faces/{detection_id}.webp
5. Updates the database face_path column

Usage:
    # Dry run - show what would be processed
    uv run python scripts/backfill_face_crops.py

    # Create face crop images
    uv run python scripts/backfill_face_crops.py --process

    # Process specific detection IDs only
    uv run python scripts/backfill_face_crops.py --process 1234 5678

    # Use different quality (default 95)
    uv run python scripts/backfill_face_crops.py --process --quality 90

    # Parallel processing with multiple workers
    uv run python scripts/backfill_face_crops.py --process --workers 8
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

# Stats tracking
stats_lock = threading.Lock()
stats = {
    "processed": 0,
    "skipped": 0,
    "failed": 0,
    "already_exists": 0,
    "missing_source": 0,
    "invalid_bbox": 0,
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


def find_detections_to_backfill(conn, detection_ids: list[int] | None = None) -> list[dict]:
    """Find person_detection records that need face_path backfill."""
    with conn.cursor(row_factory=dict_row) as cursor:
        if detection_ids:
            cursor.execute(
                """
                SELECT pd.id, pd.photo_id, pd.face_bbox_x, pd.face_bbox_y,
                       pd.face_bbox_width, pd.face_bbox_height, p.med_path
                FROM person_detection pd
                JOIN photo p ON pd.photo_id = p.id
                WHERE pd.id = ANY(%s)
                  AND pd.face_bbox_x IS NOT NULL
                  AND p.med_path IS NOT NULL
                ORDER BY pd.id
                """,
                (detection_ids,),
            )
        else:
            cursor.execute(
                """
                SELECT pd.id, pd.photo_id, pd.face_bbox_x, pd.face_bbox_y,
                       pd.face_bbox_width, pd.face_bbox_height, p.med_path
                FROM person_detection pd
                JOIN photo p ON pd.photo_id = p.id
                WHERE pd.face_bbox_x IS NOT NULL
                  AND pd.face_path IS NULL
                  AND p.med_path IS NOT NULL
                ORDER BY pd.id
                """
            )
        return cursor.fetchall()


def process_detection(
    detection: dict,
    img_path: Path,
    output_dir: Path,
    quality: int,
    dry_run: bool = True,
) -> tuple[int, str, str | None]:
    """
    Process a single detection to create face crop.

    Returns:
        (detection_id, status, new_path or error message)
    """
    detection_id = detection["id"]
    med_path = detection["med_path"]
    x = detection["face_bbox_x"]
    y = detection["face_bbox_y"]
    width = detection["face_bbox_width"]
    height = detection["face_bbox_height"]

    # Build full source path
    source_path = Path(med_path)
    if not source_path.is_absolute():
        source_path = img_path / med_path
    if not source_path.exists():
        return (detection_id, "missing_source", str(source_path))

    # Generate output path using detection_id for uniqueness
    output_filename = f"{detection_id}.webp"
    output_path = output_dir / output_filename

    # Check if output already exists
    if output_path.exists():
        # Update database if file exists but path not in DB
        if not dry_run:
            conn = get_thread_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE person_detection SET face_path = %s WHERE id = %s",
                    (str(output_path), detection_id),
                )
            conn.commit()
        return (detection_id, "already_exists", str(output_path))

    if dry_run:
        return (detection_id, "would_process", str(output_path))

    try:
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open source image
        with Image.open(source_path) as img:
            # Calculate crop coordinates
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + width)
            y2 = int(y + height)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)

            # Validate crop dimensions
            if x2 <= x1 or y2 <= y1:
                return (detection_id, "invalid_bbox", f"Invalid bbox: ({x1},{y1})-({x2},{y2})")

            # Crop the face
            face_crop = img.crop((x1, y1, x2, y2))

            # Save as WebP
            face_crop.save(
                output_path,
                "WEBP",
                quality=quality,
            )

        # Update database
        conn = get_thread_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE person_detection SET face_path = %s WHERE id = %s",
                (str(output_path), detection_id),
            )
        conn.commit()

        return (detection_id, "processed", str(output_path))

    except Exception as e:
        return (detection_id, "failed", str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Backfill face crop images for existing person_detection records"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Actually create face crop images (default is dry run)",
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
        "detection_ids",
        nargs="*",
        type=int,
        help="Specific detection IDs to process (default: all without face_path)",
    )

    args = parser.parse_args()
    dry_run = not args.process

    # Get paths from environment
    img_path = Path(os.getenv("IMG_PATH", "./photos/processed"))
    output_dir = img_path / "faces"

    print(f"Image path: {img_path}")
    print(f"Output path: {output_dir}")
    print(f"Quality: {args.quality}")
    print(f"Workers: {args.workers}")
    print(f"Mode: {'DRY RUN' if dry_run else 'PROCESS'}")
    print()

    # Find detections to process
    conn = get_connection()
    detection_ids = args.detection_ids if args.detection_ids else None
    detections = find_detections_to_backfill(conn, detection_ids)
    conn.close()

    if not detections:
        print("No detections need face crop backfill")
        return

    print(f"Found {len(detections)} detections to process")
    print()

    # Process detections
    def update_stats(status: str):
        with stats_lock:
            if status == "processed":
                stats["processed"] += 1
            elif status == "would_process":
                stats["processed"] += 1  # Count as would be processed in dry run
            elif status == "already_exists":
                stats["already_exists"] += 1
            elif status == "missing_source":
                stats["missing_source"] += 1
            elif status == "invalid_bbox":
                stats["invalid_bbox"] += 1
            elif status == "failed":
                stats["failed"] += 1
            else:
                stats["skipped"] += 1

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_detection, det, img_path, output_dir, args.quality, dry_run
                ): det
                for det in detections
            }

            for future in as_completed(futures):
                detection_id, status, result = future.result()
                update_stats(status)

                if status == "failed":
                    print(f"  FAILED detection {detection_id}: {result}")
                elif status == "missing_source":
                    print(f"  MISSING detection {detection_id}: {result}")
                elif status == "invalid_bbox":
                    print(f"  INVALID detection {detection_id}: {result}")
                elif status == "processed":
                    print(f"  Processed detection {detection_id} -> {result}")
                elif status == "would_process":
                    print(f"  Would process detection {detection_id} -> {result}")
                elif status == "already_exists":
                    print(f"  Already exists detection {detection_id}: {result}")
    else:
        for det in detections:
            detection_id, status, result = process_detection(
                det, img_path, output_dir, args.quality, dry_run
            )
            update_stats(status)

            if status == "failed":
                print(f"  FAILED detection {detection_id}: {result}")
            elif status == "missing_source":
                print(f"  MISSING detection {detection_id}: {result}")
            elif status == "invalid_bbox":
                print(f"  INVALID detection {detection_id}: {result}")
            elif status == "processed":
                print(f"  Processed detection {detection_id} -> {result}")
            elif status == "would_process":
                print(f"  Would process detection {detection_id} -> {result}")
            elif status == "already_exists":
                print(f"  Already exists detection {detection_id}: {result}")

    print()
    print("Summary:")
    print(f"  {'Would process' if dry_run else 'Processed'}: {stats['processed']}")
    print(f"  Already exists: {stats['already_exists']}")
    print(f"  Missing source: {stats['missing_source']}")
    print(f"  Invalid bbox: {stats['invalid_bbox']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")

    if dry_run and stats["processed"] > 0:
        print()
        print("Run with --process to create the face crop images")


if __name__ == "__main__":
    main()
