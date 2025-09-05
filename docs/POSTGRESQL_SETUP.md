# PostgreSQL Setup for PhotoDB

## Installation

### macOS
```bash
brew install postgresql@16
brew services start postgresql@16
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

## Database Setup

1. Create the database:
```bash
createdb photodb
```

2. Set the DATABASE_URL environment variable:
```bash
export DATABASE_URL="postgresql://localhost/photodb"
```

Or for a remote database with authentication:
```bash
export DATABASE_URL="postgresql://username:password@hostname:5432/photodb"
```

3. Add to your `.env` file:
```
DATABASE_URL=postgresql://localhost/photodb
```

## Running with PostgreSQL

The application now uses PostgreSQL by default. Simply run:

```bash
uv run python src/photodb/cli.py /path/to/photos --parallel 500
```

## Benefits over SQLite

- **True parallel processing**: PostgreSQL handles hundreds of concurrent connections
- **No write locks**: Multiple writers can work simultaneously  
- **Better performance**: Optimized for concurrent workloads
- **JSONB support**: Native JSON storage and querying for metadata
- **Scalability**: Can handle millions of photos without performance degradation

## Connection Pooling

The application automatically creates a connection pool sized appropriately for your parallel workers:
- Minimum: 2 connections
- Maximum: 2x your parallel worker count (capped at 200)

With PostgreSQL, you can safely use 500+ parallel workers without database contention issues.

## Monitoring

Check active connections:
```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'photodb';
```

Check long-running queries:
```sql
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE datname = 'photodb' AND state != 'idle'
ORDER BY duration DESC;