import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv('DB_PATH', './data/photos.db')
        self._ensure_directory()
        self._init_database()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent.parent.parent.parent / 'schema.sql'
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at: {schema_path}")
        
        with self.get_connection() as conn:
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())
            conn.commit()
            logger.info(f"Database initialized at: {self.db_path}")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup and concurrent access optimization."""
        import time
        import random
        
        max_retries = 5
        base_delay = 0.01  # 10ms
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(
                    self.db_path,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    timeout=30.0  # 30 second timeout for lock acquisition
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better concurrency
                conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes while maintaining safety
                conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
                conn.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables
                
                try:
                    yield conn
                    break
                finally:
                    conn.close()
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.01)
                    logger.warning(f"Database locked, retrying in {delay:.3f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Database connection failed after {attempt + 1} attempts: {e}")
                    raise
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Execute operations within a transaction."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise