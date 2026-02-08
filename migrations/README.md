# Database Migrations

This directory contains SQL migration scripts for upgrading the PhotoDB schema.

## Running Migrations

Migrations should be run in order. Each migration is idempotent (safe to run multiple times).

```bash
# Connect to database and run migration
psql -d photodb -f migrations/001_add_constrained_clustering.sql

# Or using environment variable
psql $DATABASE_URL -f migrations/001_add_constrained_clustering.sql
```

## Checking Migration Status

```sql
SELECT version, applied_at, description FROM schema_migrations ORDER BY version;
```

## Rolling Back

Each migration has a corresponding rollback script:

```bash
# Rollback migration 001
psql -d photodb -f migrations/001_add_constrained_clustering_rollback.sql
```

**Warning:** Rollbacks may delete data. Read the rollback script comments before running.

## Best Practices

1. **Backup first**: Always backup your database before running migrations

   ```bash
   pg_dump photodb > backup_$(date +%Y%m%d).sql
   ```

2. **Test in development**: Run migrations on a test database first

3. **Run in transaction**: Migrations use `BEGIN`/`COMMIT` for atomicity

4. **Check status after**: Verify the migration applied correctly

   ```sql
   SELECT * FROM schema_migrations;
   ```
