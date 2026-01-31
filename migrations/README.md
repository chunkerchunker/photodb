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

## Available Migrations

| Version | Description | Date |
|---------|-------------|------|
| 001 | Add constrained clustering support | 2025-01-30 |

### 001: Constrained Clustering

Adds support for the constrained incremental clustering algorithm:

**New Tables:**
- `must_link` - Forces two faces into the same cluster
- `cannot_link` - Prevents two faces from being in the same cluster
- `cluster_cannot_link` - Prevents two clusters from being merged

**Modified Tables:**
- `face` - Added `unassigned_since` column for outlier pool tracking
- `face` - Extended `cluster_status` to include 'unassigned' and 'constrained'
- `cluster` - Added `verified`, `verified_at`, `verified_by` for cluster locking

**Helper Functions:**
- `add_must_link(face_id_1, face_id_2, created_by)` - Safely add must-link constraint
- `add_cannot_link(face_id_1, face_id_2, created_by)` - Safely add cannot-link constraint
- `check_constraint_violations()` - Find cannot-link violations in same cluster

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
