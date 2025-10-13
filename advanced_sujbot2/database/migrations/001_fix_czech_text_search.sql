-- ============================================================================
-- Migration: Fix Czech Text Search Configuration
-- Date: 2025-01-11
-- Description: Update chunks_content_tsv_trigger to fallback to 'simple'
--              configuration when 'czech' is not available in PostgreSQL
-- ============================================================================

-- Drop and recreate the trigger function with Czech fallback
DROP TRIGGER IF EXISTS chunks_content_tsv_update ON chunks;
DROP FUNCTION IF EXISTS chunks_content_tsv_trigger();

-- Create updated trigger function with dynamic text search configuration
CREATE OR REPLACE FUNCTION chunks_content_tsv_trigger()
RETURNS TRIGGER AS $$
DECLARE
    text_config regconfig;
BEGIN
    -- Try to use Czech text search configuration, fallback to 'simple' if not available
    -- Check if 'czech' configuration exists
    SELECT oid INTO text_config
    FROM pg_ts_config
    WHERE cfgname = 'czech'
    LIMIT 1;

    -- Use 'czech' if available, otherwise use 'simple'
    IF text_config IS NULL THEN
        text_config := 'simple'::regconfig;
    ELSE
        text_config := 'czech'::regconfig;
    END IF;

    NEW.content_tsv = to_tsvector(text_config, COALESCE(NEW.content, '') || ' ' ||
                                          COALESCE(NEW.legal_reference, '') || ' ' ||
                                          COALESCE(NEW.hierarchy_path, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Recreate the trigger
CREATE TRIGGER chunks_content_tsv_update BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_content_tsv_trigger();

-- Regenerate all existing content_tsv values with the new configuration
-- The UPDATE will automatically trigger chunks_content_tsv_trigger for each row
-- We just need to touch the rows to trigger the function
UPDATE chunks SET content = content
WHERE content_tsv IS NOT NULL;

-- Note: If you have many rows, you may want to do this in batches:
-- DO $$
-- DECLARE
--     batch_size INT := 10000;
--     updated INT := 1;
-- BEGIN
--     WHILE updated > 0 LOOP
--         UPDATE chunks SET content = content
--         WHERE chunk_id IN (
--             SELECT chunk_id FROM chunks LIMIT batch_size
--         );
--         GET DIAGNOSTICS updated = ROW_COUNT;
--         RAISE NOTICE 'Updated % rows', updated;
--         COMMIT;
--     END LOOP;
-- END $$;

COMMENT ON FUNCTION chunks_content_tsv_trigger IS
'Auto-generate full-text search vector using Czech config if available, fallback to simple';

-- ============================================================================
-- Verification Query
-- ============================================================================

-- Check which text search configuration is being used
DO $$
DECLARE
    text_config TEXT;
BEGIN
    SELECT cfgname INTO text_config
    FROM pg_ts_config
    WHERE cfgname = 'czech'
    LIMIT 1;

    IF text_config IS NULL THEN
        RAISE NOTICE 'Using ''simple'' text search configuration (Czech not available)';
        RAISE NOTICE 'To enable Czech text search, install: CREATE TEXT SEARCH CONFIGURATION czech (COPY = simple);';
    ELSE
        RAISE NOTICE 'Using ''czech'' text search configuration';
    END IF;
END $$;

-- Test the trigger
-- SELECT 'Trigger updated successfully. Test with: INSERT INTO chunks (...) to verify.'::text;
