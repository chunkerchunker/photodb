#!/usr/bin/env python3
"""Import an order from the Capture system into PhotoDB.

This script:
1. Reads an order from the 'capture' PostgreSQL database
2. Creates an app_user in photodb (first_name from orders.name, last_name = "Imported")
3. Creates a collection for that user
4. Creates albums (named "Album#{5-digit-id}") for each capture album
5. Creates photo entries for all crops in the order
6. Associates photos with their respective albums

Usage:
  python scripts/import_capture_order.py --order-id 123

  # Dry run to see what would be imported
  python scripts/import_capture_order.py --order-id 123 --dry-run

  # With custom password
  python scripts/import_capture_order.py --order-id 123 --password "userpass"

  # With custom base path for photo files
  python scripts/import_capture_order.py --order-id 123 --base-path /path/to/capture

  # Skip file existence check (import even if files are missing)
  python scripts/import_capture_order.py --order-id 123 --no-check-files

Environment:
  CAPTURE_DATABASE_URL: Connection string for capture DB (default: postgresql://localhost/capture)
  DATABASE_URL: Connection string for photodb (default: postgresql://localhost/photodb)
"""

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg


@dataclass
class CaptureOrder:
    id: int
    name: str | None
    email_address: str | None
    created_at: datetime | None


@dataclass
class CaptureAlbum:
    id: int
    order_id: int
    title: str | None


@dataclass
class CapturePage:
    id: str  # UUID as string
    album_id: int
    index: int | None
    deleted: bool


@dataclass
class CaptureCrop:
    id: str  # UUID as string
    page_id: str  # UUID as string
    album_id: int  # Denormalized from page for convenience
    deleted: bool
    # 4-corner bounding box coords (pixel values)
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    x4: int
    y4: int


def _hash_password(password: str) -> str:
    """Hash password using scrypt."""
    salt = os.urandom(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return f"scrypt${salt.hex()}${derived.hex()}"


def _generate_username(name: str | None, order_id: int) -> str:
    """Generate a username from order name and ID."""
    if name:
        # Use first word of name, lowercased, with order_id suffix
        base = name.split()[0].lower() if name.split() else "order"
        # Remove non-alphanumeric characters
        base = "".join(c for c in base if c.isalnum())
        return f"{base}_{order_id:05d}"
    return f"order_{order_id:05d}"


DEFAULT_BASE_PATH = "/Volumes/media/Pictures/capture"


def _build_photo_filename(
    order_id: int, album_id: int, page_id: str, crop_id: str, base_path: str = DEFAULT_BASE_PATH
) -> str:
    """Build the photo filename path for a capture crop."""
    return f"{base_path}/Order#{order_id:05d}/Album#{album_id:05d}/{page_id}/crops/{crop_id}_final.jpeg"


def _build_normalized_path(
    order_id: int, album_id: int, page_id: str, crop_id: str, base_path: str = DEFAULT_BASE_PATH
) -> str:
    """Build the normalized image path for a capture crop (with _sm suffix)."""
    return f"{base_path}/Order#{order_id:05d}/Album#{album_id:05d}/{page_id}/crops/{crop_id}_final_sm.jpeg"


def _euclidean_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate Euclidean distance between two points."""
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _calculate_dimensions(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    x3: int,
    y3: int,
    x4: int,
    y4: int,
    normalized_width: int = 400,
) -> tuple[int, int, int, int]:
    """Calculate original and normalized dimensions from 4-corner quadrilateral.

    Corners are ordered: (x1,y1)=top-left, (x2,y2)=top-right,
    (x3,y3)=bottom-left, (x4,y4)=bottom-right.

    Args:
        x1-x4, y1-y4: 4-corner pixel coordinates
        normalized_width: Width of the normalized image (default 400px)

    Returns:
        (width, height, normalized_width, normalized_height)
    """
    # Calculate edge lengths using Euclidean distance
    # Top edge: point 1 to point 2
    top_edge = _euclidean_distance(x1, y1, x2, y2)
    # Bottom edge: point 3 to point 4
    bottom_edge = _euclidean_distance(x3, y3, x4, y4)
    # Left edge: point 1 to point 3
    left_edge = _euclidean_distance(x1, y1, x3, y3)
    # Right edge: point 2 to point 4
    right_edge = _euclidean_distance(x2, y2, x4, y4)

    # Average opposite edges for width and height
    width = (top_edge + bottom_edge) / 2
    height = (left_edge + right_edge) / 2

    if width <= 0 or height <= 0:
        # Fallback for invalid coords
        return normalized_width, normalized_width, normalized_width, normalized_width

    aspect_ratio = width / height

    # Normalized image has fixed width, height from aspect ratio
    norm_height = round(normalized_width / aspect_ratio)

    # Original dimensions from the quadrilateral
    return round(width), round(height), normalized_width, norm_height


def fetch_order(capture_conn: psycopg.Connection, order_id: int) -> CaptureOrder | None:
    """Fetch order from capture database."""
    with capture_conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, email_address, created_at
            FROM orders
            WHERE id = %s AND deleted = false
            """,
            (order_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return CaptureOrder(
            id=row[0],
            name=row[1],
            email_address=row[2],
            created_at=row[3],
        )


def fetch_albums(capture_conn: psycopg.Connection, order_id: int) -> list[CaptureAlbum]:
    """Fetch all albums for an order."""
    with capture_conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, order_id, title
            FROM albums
            WHERE order_id = %s AND deleted = false
            ORDER BY id
            """,
            (order_id,),
        )
        return [
            CaptureAlbum(id=row[0], order_id=row[1], title=row[2]) for row in cur.fetchall()
        ]


def fetch_pages(capture_conn: psycopg.Connection, album_ids: list[int]) -> list[CapturePage]:
    """Fetch all pages for given albums."""
    if not album_ids:
        return []

    with capture_conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, album_id, index, deleted
            FROM pages
            WHERE album_id = ANY(%s) AND deleted = false
            ORDER BY album_id, index
            """,
            (album_ids,),
        )
        return [
            CapturePage(id=row[0], album_id=row[1], index=row[2], deleted=row[3])
            for row in cur.fetchall()
        ]


def fetch_crops(capture_conn: psycopg.Connection, page_ids: list[str]) -> list[CaptureCrop]:
    """Fetch all crops for given pages."""
    if not page_ids:
        return []

    with capture_conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id::text, c.page_id::text, p.album_id, c.deleted,
                   c.x1, c.y1, c.x2, c.y2, c.x3, c.y3, c.x4, c.y4
            FROM crops c
            JOIN pages p ON c.page_id = p.id
            WHERE c.page_id = ANY(%s::uuid[]) AND c.deleted = false
            ORDER BY p.album_id, p.index, c.id
            """,
            (page_ids,),
        )
        return [
            CaptureCrop(
                id=row[0],
                page_id=row[1],
                album_id=row[2],
                deleted=row[3],
                x1=row[4] or 0,
                y1=row[5] or 0,
                x2=row[6] or 0,
                y2=row[7] or 0,
                x3=row[8] or 0,
                y3=row[9] or 0,
                x4=row[10] or 0,
                y4=row[11] or 0,
            )
            for row in cur.fetchall()
        ]


def create_user_and_collection(
    photodb_conn: psycopg.Connection,
    order: CaptureOrder,
    password: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Create app_user and collection in photodb. Returns (user_id, collection_id)."""
    order_label = f"Order#{order.id:05d}"
    if order.name and order.name.strip():
        first_name = order.name.split()[0]
        collection_name = f"{order.name} - {order_label}"
    else:
        first_name = order_label
        collection_name = order_label
    last_name = "Imported"
    username = _generate_username(order.name, order.id)
    password_hash = _hash_password(password)

    if dry_run:
        print(f"Would create user: {username} ({first_name} {last_name})")
        print(f"Would create collection: {collection_name}")
        return -1, -1

    with photodb_conn.cursor() as cur:
        # Check if username already exists
        cur.execute("SELECT id FROM app_user WHERE username = %s", (username,))
        existing = cur.fetchone()
        if existing:
            print(f"User '{username}' already exists with id {existing[0]}, reusing.", file=sys.stderr)
            user_id = existing[0]
        else:
            cur.execute(
                """
                INSERT INTO app_user (username, password_hash, first_name, last_name)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (username, password_hash, first_name, last_name),
            )
            user_id = cur.fetchone()[0]
            print(f"Created user '{username}' with id {user_id}")

        # Check if collection already exists for this user
        cur.execute(
            "SELECT id FROM collection WHERE owner_user_id = %s AND name = %s",
            (user_id, collection_name),
        )
        existing_coll = cur.fetchone()
        if existing_coll:
            print(f"Collection '{collection_name}' already exists with id {existing_coll[0]}, reusing.")
            collection_id = existing_coll[0]
        else:
            cur.execute(
                """
                INSERT INTO collection (owner_user_id, name)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, collection_name),
            )
            collection_id = cur.fetchone()[0]
            print(f"Created collection '{collection_name}' with id {collection_id}")

            # Add collection member entry
            cur.execute(
                """
                INSERT INTO collection_member (collection_id, user_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (collection_id, user_id),
            )

            # Set as default collection if user doesn't have one
            cur.execute(
                """
                UPDATE app_user
                SET default_collection_id = %s
                WHERE id = %s AND default_collection_id IS NULL
                """,
                (collection_id, user_id),
            )

    return user_id, collection_id


def create_albums(
    photodb_conn: psycopg.Connection,
    collection_id: int,
    capture_albums: list[CaptureAlbum],
    dry_run: bool,
) -> dict[int, int]:
    """Create albums in photodb. Returns mapping of capture_album_id -> photodb_album_id."""
    album_id_map: dict[int, int] = {}

    if dry_run:
        print(f"\nWould create {len(capture_albums)} albums:")
        for album in capture_albums[:5]:
            album_name = f"Album#{album.id:05d}"
            print(f"  - {album_name}")
        if len(capture_albums) > 5:
            print(f"  ... and {len(capture_albums) - 5} more")
        return album_id_map

    now = datetime.now(timezone.utc)
    created = 0
    reused = 0

    with photodb_conn.cursor() as cur:
        for capture_album in capture_albums:
            album_name = f"Album#{capture_album.id:05d}"

            # Check if album already exists
            cur.execute(
                "SELECT id FROM album WHERE collection_id = %s AND name = %s",
                (collection_id, album_name),
            )
            existing = cur.fetchone()
            if existing:
                album_id_map[capture_album.id] = existing[0]
                reused += 1
            else:
                cur.execute(
                    """
                    INSERT INTO album (collection_id, name, created_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (collection_id, album_name, now, now),
                )
                album_id_map[capture_album.id] = cur.fetchone()[0]
                created += 1

    if reused:
        print(f"Reused {reused} existing albums")
    print(f"Created {created} new albums")
    return album_id_map


def create_photos(
    photodb_conn: psycopg.Connection,
    collection_id: int,
    order_id: int,
    crops: list[CaptureCrop],
    album_id_map: dict[int, int],
    dry_run: bool,
    base_path: str = DEFAULT_BASE_PATH,
    check_files: bool = True,
) -> int:
    """Create photo entries for all crops. Returns count of photos created."""
    if dry_run:
        print(f"\nWould create {len(crops)} photos:")
        for crop in crops[:5]:  # Show first 5 as example
            filename = _build_photo_filename(order_id, crop.album_id, crop.page_id, crop.id, base_path)
            normalized_path = _build_normalized_path(order_id, crop.album_id, crop.page_id, crop.id, base_path)
            width, height, norm_w, norm_h = _calculate_dimensions(
                crop.x1, crop.y1, crop.x2, crop.y2, crop.x3, crop.y3, crop.x4, crop.y4
            )
            print(f"  - {filename}")
            print(f"    normalized: {normalized_path} ({norm_w}x{norm_h})")
        if len(crops) > 5:
            print(f"  ... and {len(crops) - 5} more")
        return len(crops)

    created = 0
    skipped = 0
    skipped_missing = 0
    photo_album_created = 0
    now = datetime.now(timezone.utc)

    # Track first photo added to each album for representative
    album_first_photo: dict[int, int] = {}  # photodb_album_id -> first photo_id

    with photodb_conn.cursor() as cur:
        for crop in crops:
            filename = _build_photo_filename(order_id, crop.album_id, crop.page_id, crop.id, base_path)
            normalized_path = _build_normalized_path(order_id, crop.album_id, crop.page_id, crop.id, base_path)

            # Check if files exist on disk
            if check_files:
                missing_files = []
                if not os.path.exists(filename):
                    missing_files.append(f"filename: {filename}")
                if not os.path.exists(normalized_path):
                    missing_files.append(f"normalized: {normalized_path}")
                if missing_files:
                    print(f"WARNING: Skipping crop {crop.id} - missing files:", file=sys.stderr)
                    for mf in missing_files:
                        print(f"  {mf}", file=sys.stderr)
                    skipped_missing += 1
                    continue

            width, height, norm_w, norm_h = _calculate_dimensions(
                crop.x1, crop.y1, crop.x2, crop.y2, crop.x3, crop.y3, crop.x4, crop.y4
            )

            # Check if photo already exists
            cur.execute(
                "SELECT id FROM photo WHERE collection_id = %s AND orig_path = %s",
                (collection_id, filename),
            )
            existing = cur.fetchone()
            if existing:
                photo_id = existing[0]
                skipped += 1
            else:
                cur.execute(
                    """
                    INSERT INTO photo (
                        collection_id, orig_path, med_path,
                        width, height, med_width, med_height,
                        created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        collection_id,
                        filename,
                        normalized_path,
                        width,
                        height,
                        norm_w,
                        norm_h,
                        now,
                        now,
                    ),
                )
                photo_id = cur.fetchone()[0]
                created += 1

            # Create photo_album association
            photodb_album_id = album_id_map.get(crop.album_id)
            if photodb_album_id:
                cur.execute(
                    """
                    INSERT INTO photo_album (photo_id, album_id, added_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (photo_id, photodb_album_id, now),
                )
                if cur.rowcount > 0:
                    photo_album_created += 1
                    # Track first photo for this album
                    if photodb_album_id not in album_first_photo:
                        album_first_photo[photodb_album_id] = photo_id

        # Set representative photo for each album (first photo added)
        for album_id, first_photo_id in album_first_photo.items():
            cur.execute(
                """
                UPDATE album
                SET representative_photo_id = %s
                WHERE id = %s AND representative_photo_id IS NULL
                """,
                (first_photo_id, album_id),
            )

    if skipped_missing:
        print(f"Skipped {skipped_missing} photos due to missing files")
    if skipped:
        print(f"Skipped {skipped} photos that already exist")
    print(f"Created {created} new photos")
    print(f"Created {photo_album_created} photo-album associations")
    if album_first_photo:
        print(f"Set representative photo for {len(album_first_photo)} albums")
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a Capture order into PhotoDB")
    parser.add_argument("--order-id", type=int, required=True, help="Capture order ID to import")
    parser.add_argument("--password", type=str, default="changeme", help="Password for new user")
    parser.add_argument("--base-path", type=str, default=DEFAULT_BASE_PATH, help="Base path for photo files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Disable file existence check (by default, skips photos if files don't exist)",
    )
    args = parser.parse_args()

    capture_url = os.getenv("CAPTURE_DATABASE_URL", "postgresql://localhost/capture")
    photodb_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")

    # Connect to capture database and fetch data
    print(f"Connecting to capture database...")
    with psycopg.connect(capture_url) as capture_conn:
        order = fetch_order(capture_conn, args.order_id)
        if not order:
            print(f"Order {args.order_id} not found or deleted.", file=sys.stderr)
            return 1

        print(f"Found order: {order.id} - {order.name or '(no name)'}")

        albums = fetch_albums(capture_conn, args.order_id)
        if not albums:
            print(f"No albums found for order {args.order_id}.", file=sys.stderr)
            return 1

        print(f"Found {len(albums)} albums")

        album_ids = [a.id for a in albums]
        pages = fetch_pages(capture_conn, album_ids)
        print(f"Found {len(pages)} pages")

        page_ids = [p.id for p in pages]
        crops = fetch_crops(capture_conn, page_ids)
        print(f"Found {len(crops)} crops (photos)")

    if not crops:
        print("No crops to import.", file=sys.stderr)
        return 1

    # Filter albums to only those that have crops
    album_ids_with_crops = {crop.album_id for crop in crops}
    albums_with_photos = [a for a in albums if a.id in album_ids_with_crops]
    skipped_albums = len(albums) - len(albums_with_photos)
    if skipped_albums > 0:
        print(f"Skipping {skipped_albums} albums with no photos")
    albums = albums_with_photos

    # Connect to photodb and create records
    print(f"\nConnecting to photodb database...")
    with psycopg.connect(photodb_url) as photodb_conn:
        user_id, collection_id = create_user_and_collection(
            photodb_conn, order, args.password, args.dry_run
        )

        if not args.dry_run:
            photodb_conn.commit()

        album_id_map = create_albums(photodb_conn, collection_id, albums, args.dry_run)

        if not args.dry_run:
            photodb_conn.commit()

        create_photos(
            photodb_conn,
            collection_id,
            args.order_id,
            crops,
            album_id_map,
            args.dry_run,
            args.base_path,
            check_files=not args.no_check_files,
        )

        if not args.dry_run:
            photodb_conn.commit()

    if args.dry_run:
        print("\n[Dry run - no changes made]")
    else:
        print(f"\nImport complete!")
        print(f"User: {_generate_username(order.name, order.id)}")
        print(f"Collection ID: {collection_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
