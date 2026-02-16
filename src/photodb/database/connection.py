import psycopg
from psycopg_pool import ConnectionPool as PsycopgPool
from contextlib import contextmanager
from typing import Optional, Generator, Any
import logging
from pgvector.psycopg import register_vector

from .. import config as defaults

logger = logging.getLogger(__name__)


def setup_vector_adapter(conn):
    """Set up pgvector adapter for a connection."""
    register_vector(conn)


class Connection:
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize PostgreSQL connection.

        Args:
            connection_string: PostgreSQL connection string. If not provided,
                             uses DATABASE_URL env var or defaults to local connection.
        """
        self.connection_string = connection_string or defaults.DATABASE_URL

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Get a database connection with proper cleanup."""
        conn = None
        try:
            conn = psycopg.connect(self.connection_string)
            setup_vector_adapter(conn)
            yield conn
        finally:
            if conn:
                conn.close()

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Execute operations within a transaction."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise


class ConnectionPool:
    def __init__(
        self, connection_string: Optional[str] = None, min_conn: int = 2, max_conn: int = 100
    ):
        """Initialize PostgreSQL connection pool.

        Args:
            connection_string: PostgreSQL connection string
            min_conn: Minimum number of connections to maintain
            max_conn: Maximum number of connections allowed
        """
        self.connection_string = connection_string or defaults.DATABASE_URL

        self.pool = PsycopgPool(
            self.connection_string,
            min_size=min_conn,
            max_size=max_conn,
            configure=setup_vector_adapter,
        )

        logger.info(
            f"PostgreSQL connection pool initialized with {min_conn}-{max_conn} connections"
        )

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close pool."""
        self.close_all()

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Get a connection from the pool."""
        with self.pool.connection() as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Execute operations within a transaction using pooled connection."""
        with self.pool.connection() as conn:
            with conn.transaction():
                yield conn

    def close_all(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.close()
            logger.info("All connections closed")
