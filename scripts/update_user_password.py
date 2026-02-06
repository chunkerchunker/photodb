#!/usr/bin/env python3
"""Update app_user password_hash for a given user id.

Usage:
  python scripts/update_user_password.py --user-id 1 --password "newpass"

If --password is omitted, reads from STDIN.
Uses DATABASE_URL env var (defaults to postgresql://localhost/photodb).
"""

import argparse
import os
import sys
import hashlib

import psycopg


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return f"scrypt${salt.hex()}${derived.hex()}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update app_user password_hash")
    parser.add_argument("--user-id", type=int, required=True, help="app_user.id")
    parser.add_argument("--password", type=str, default=None, help="New password (omit to read from STDIN)")
    args = parser.parse_args()

    password = args.password
    if password is None:
        password = sys.stdin.read().strip()
    if not password:
        print("Password is required.", file=sys.stderr)
        return 1

    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/photodb")
    password_hash = _hash_password(password)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE app_user SET password_hash = %s WHERE id = %s", (password_hash, args.user_id))
            if cur.rowcount == 0:
                print(f"No user found with id {args.user_id}.", file=sys.stderr)
                return 1
        conn.commit()

    print(f"Updated password for user id {args.user_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
