# Database Migrations

This directory contains PostgreSQL migration scripts for the SUJBOT2 database.

## Migration List

### 001_fix_czech_text_search.sql

**Date:** 2025-01-11
**Status:** Required for all deployments

**Purpose:**
Fixes the "text search configuration 'czech' does not exist" error by updating the `chunks_content_tsv_trigger()` function to dynamically detect and use the Czech text search configuration if available, otherwise fallback to the 'simple' configuration.

**Changes:**
- Updates `chunks_content_tsv_trigger()` to check if 'czech' config exists
- Falls back to 'simple' configuration if Czech is not available
- Updates `hybrid_search()` function with same fallback logic (already in schema.sql)
- Regenerates all existing content_tsv values

**How to Apply:**

```bash
# Connect to your database
psql -U sujbot_app -d sujbot2 -h localhost

# Run the migration
\i database/migrations/001_fix_czech_text_search.sql

# Verify
SELECT cfgname FROM pg_ts_config WHERE cfgname IN ('czech', 'simple');
```

**For Docker deployments:**

```bash
# Copy migration to container
docker cp database/migrations/001_fix_czech_text_search.sql sujbot2_postgres_dev:/tmp/

# Execute migration
docker exec -it sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 -f /tmp/001_fix_czech_text_search.sql
```

**Optional: Install Czech Text Search Configuration**

If you want full Czech language support for full-text search:

```sql
-- Install Czech text search dictionary (if not already available)
-- On Ubuntu/Debian: apt-get install postgresql-contrib
-- On macOS with Homebrew PostgreSQL: already included

-- Create Czech configuration (simplified version)
CREATE TEXT SEARCH CONFIGURATION czech (COPY = simple);

-- Or for full Czech stemming support (requires postgresql-contrib):
-- CREATE TEXT SEARCH CONFIGURATION czech (PARSER = default);
-- ALTER TEXT SEARCH CONFIGURATION czech ADD MAPPING FOR ...
```

After installing Czech config, the trigger will automatically start using it.

## Migration History

| # | Date | Description | Status |
|---|------|-------------|--------|
| 001 | 2025-01-11 | Fix Czech text search configuration | ✅ Required |

## Best Practices

1. **Always backup before migrating:**
   ```bash
   pg_dump -U sujbot_app sujbot2 > backup_$(date +%Y%m%d_%H%M%S).sql
   ```

2. **Test migrations in development first:**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   # Apply migration to dev database
   ```

3. **Monitor migration progress:**
   For large databases, migrations that update all rows may take time.
   Use `EXPLAIN ANALYZE` to estimate duration.

4. **Verify migration success:**
   Check application logs and test document indexing after migration.

## Troubleshooting

### Migration fails with permission errors

```sql
-- Grant necessary permissions to sujbot_app user
GRANT CREATE ON SCHEMA public TO sujbot_app;
GRANT USAGE ON SCHEMA public TO sujbot_app;
```

### Migration timeout for large databases

For databases with >1M chunks, the UPDATE in migration 001 may take time.
Consider batching:

```sql
-- Update in batches of 10,000
DO $$
DECLARE
    batch_size INT := 10000;
    updated INT := 1;
BEGIN
    WHILE updated > 0 LOOP
        WITH batch AS (
            SELECT chunk_id FROM chunks
            WHERE content_tsv IS NULL
            LIMIT batch_size
        )
        UPDATE chunks
        SET content_tsv = chunks_content_tsv_trigger.content_tsv
        WHERE chunk_id IN (SELECT chunk_id FROM batch);

        GET DIAGNOSTICS updated = ROW_COUNT;
        RAISE NOTICE 'Updated % rows', updated;
        COMMIT;
    END LOOP;
END $$;
```

## Rollback

To rollback migration 001:

```sql
-- Revert to original trigger (requires Czech config)
DROP TRIGGER IF EXISTS chunks_content_tsv_update ON chunks;
DROP FUNCTION IF EXISTS chunks_content_tsv_trigger();

CREATE OR REPLACE FUNCTION chunks_content_tsv_trigger()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv = to_tsvector('czech', COALESCE(NEW.content, '') || ' ' ||
                                          COALESCE(NEW.legal_reference, '') || ' ' ||
                                          COALESCE(NEW.hierarchy_path, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunks_content_tsv_update BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_content_tsv_trigger();
```

**Warning:** Rollback will only work if Czech text search configuration is installed!
