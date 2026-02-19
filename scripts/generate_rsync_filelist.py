#!/usr/bin/env python
"""
Generate a file list for rsync --files-from containing all med/full images
for the given collection IDs.

Usage:
    uv run python scripts/generate_rsync_filelist.py 1 2 3
    uv run python scripts/generate_rsync_filelist.py 1 2 3 -o sync_list.txt
    rsync -avz --files-from=sync_list.txt $IMG_PATH remote:/destination/

Paths are output relative to IMG_PATH so rsync preserves just the med/full structure.
"""

import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from photodb import config as defaults  # noqa: E402
from photodb.database.connection import ConnectionPool  # noqa: E402


@click.command()
@click.argument("collection_ids", nargs=-1, required=True, type=int)
@click.option("-o", "--output", type=click.Path(), help="Output file (default: stdout)")
@click.option(
    "--base",
    type=click.Path(),
    default=None,
    help="Base path to strip from DB paths (default: IMG_PATH)",
)
def main(collection_ids, output, base):
    """Generate rsync --files-from list for med/full images in COLLECTION_IDS."""
    base_path = Path(base).resolve() if base else Path(defaults.IMG_PATH).resolve()

    with ConnectionPool(min_conn=1, max_conn=2) as pool:
        with pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT full_path, med_path
                    FROM photo
                    WHERE collection_id = ANY(%s)
                      AND (full_path IS NOT NULL OR med_path IS NOT NULL)
                    ORDER BY id
                    """,
                    (list(collection_ids),),
                )
                photo_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT face_path
                    FROM person_detection
                    WHERE collection_id = ANY(%s)
                      AND face_path IS NOT NULL
                    ORDER BY id
                    """,
                    (list(collection_ids),),
                )
                face_rows = cur.fetchall()

    raw_paths = []
    for full_path, med_path in photo_rows:
        if full_path:
            raw_paths.append(full_path)
        if med_path:
            raw_paths.append(med_path)
    for (face_path,) in face_rows:
        raw_paths.append(face_path)

    paths = []
    for p in raw_paths:
        try:
            paths.append(str(Path(p).resolve().relative_to(base_path)))
        except ValueError:
            paths.append(p)

    if not paths:
        click.echo(f"No images found for collection(s) {list(collection_ids)}", err=True)
        sys.exit(1)

    text = "\n".join(paths) + "\n"

    if output:
        with open(output, "w") as f:
            f.write(text)
        click.echo(f"Wrote {len(paths)} paths ({base_path}) to {output}", err=True)
    else:
        click.echo(text, nl=False)


if __name__ == "__main__":
    main()
