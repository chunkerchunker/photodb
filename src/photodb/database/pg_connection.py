import psycopg2
from psycopg2 import pool
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)


class PostgresConnection:
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize PostgreSQL connection.

        Args:
            connection_string: PostgreSQL connection string. If not provided,
                             uses DATABASE_URL env var or defaults to local connection.
        """
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", "postgresql://localhost/photodb"
        )
        self._init_database()

    def _init_database(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent.parent.parent.parent / "schema_postgres.sql"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at: {schema_path}")

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                with open(schema_path, "r") as f:
                    cursor.execute(f.read())
                conn.commit()
                logger.info("Database initialized")

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            yield conn
        finally:
            if conn:
                conn.close()

    @contextmanager
    def transaction(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Execute operations within a transaction."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise


class PostgresConnectionPool:
    def __init__(
        self, connection_string: Optional[str] = None, min_conn: int = 2, max_conn: int = 100
    ):
        """Initialize PostgreSQL connection pool.

        Args:
            connection_string: PostgreSQL connection string
            min_conn: Minimum number of connections to maintain
            max_conn: Maximum number of connections allowed
        """
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", "postgresql://localhost/photodb"
        )

        # PostgreSQL can handle many more connections than SQLite
        # Typical PostgreSQL can handle 100-200 connections easily
        self.pool = pool.ThreadedConnectionPool(min_conn, max_conn, self.connection_string)

        logger.info(
            f"PostgreSQL connection pool initialized with {min_conn}-{max_conn} connections"
        )
        self._init_database()

    def _init_database(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent.parent.parent.parent / "schema_postgres.sql"

        if not schema_path.exists():
            logger.warning(f"Schema file not found at: {schema_path}, skipping initialization")
            return

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                with open(schema_path, "r") as f:
                    cursor.execute(f.read())
                conn.commit()
                logger.info("Database schema initialized")

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Get a connection from the pool with timeout and retry logic."""
        conn = None
        max_retries = 100
        retry_delay = 0.1  # 100ms

        for attempt in range(max_retries):
            try:
                # Try to get a connection from the pool
                # getconn will block if no connections are available
                conn = self.pool.getconn()
                if conn:
                    try:
                        yield conn
                    except Exception:
                        # If there's an exception, make sure we still return the connection
                        raise
                    finally:
                        # Always return connection to pool, even on exception
                        try:
                            self.pool.putconn(conn)
                        except Exception as e:
                            logger.error(f"Failed to return connection to pool: {e}")
                    return
                else:
                    raise Exception("Failed to get connection from pool")
            except pool.PoolError as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        f"Connection pool exhausted, waiting... (attempt {attempt + 1}/{max_retries})"
                    )
                    import time

                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"Could not get connection after {max_retries} attempts: {e}")

    @contextmanager
    def transaction(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Execute operations within a transaction using pooled connection."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise

    def close_all(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("All connections closed")
