-- ============================================================================
-- Fix Missing Metadata Schema
-- ============================================================================
-- This script creates the missing metadata schema and tables that were
-- skipped during initial database initialization due to Apache AGE error.
--
-- Run this script to fix the "metadata.vector_store_stats does not exist" error.
--
-- Usage:
--   docker exec -i sujbot_postgres psql -U postgres -d sujbot < scripts/fix_metadata_schema.sql
-- ============================================================================

-- Ensure we're in the correct database
\c sujbot

-- Create metadata schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS metadata;

-- ============================================================================
-- METADATA SCHEMA: System Metadata
-- ============================================================================

-- Document registry (track indexed documents)
CREATE TABLE IF NOT EXISTS metadata.documents (
    document_id TEXT PRIMARY KEY,
    title TEXT,
    file_path TEXT,
    file_hash TEXT UNIQUE,  -- SHA256 for deduplication
    hierarchy_depth INTEGER,
    total_chunks INTEGER,

    -- Indexing metadata
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    index_version TEXT,

    -- Document metadata (flexible JSONB)
    metadata JSONB
);

-- Vector store statistics (for get_stats() API)
CREATE TABLE IF NOT EXISTS metadata.vector_store_stats (
    id SERIAL PRIMARY KEY,
    dimensions INTEGER NOT NULL DEFAULT 3072,
    layer1_count INTEGER NOT NULL DEFAULT 0,
    layer2_count INTEGER NOT NULL DEFAULT 0,
    layer3_count INTEGER NOT NULL DEFAULT 0,
    total_vectors INTEGER NOT NULL DEFAULT 0,
    document_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert initial stats row (with current counts)
INSERT INTO metadata.vector_store_stats (dimensions, layer1_count, layer2_count, layer3_count, total_vectors, document_count)
VALUES (
    3072,
    (SELECT COUNT(*) FROM vectors.layer1),
    (SELECT COUNT(*) FROM vectors.layer2),
    (SELECT COUNT(*) FROM vectors.layer3),
    (SELECT COUNT(*) FROM vectors.layer1) + (SELECT COUNT(*) FROM vectors.layer2) + (SELECT COUNT(*) FROM vectors.layer3),
    (SELECT COUNT(DISTINCT document_id) FROM vectors.layer1)
)
ON CONFLICT (id) DO UPDATE SET
    layer1_count = EXCLUDED.layer1_count,
    layer2_count = EXCLUDED.layer2_count,
    layer3_count = EXCLUDED.layer3_count,
    total_vectors = EXCLUDED.total_vectors,
    document_count = EXCLUDED.document_count,
    updated_at = NOW();

-- ============================================================================
-- METADATA INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_documents_hash ON metadata.documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_indexed ON metadata.documents(indexed_at);

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function: Update vector store statistics
CREATE OR REPLACE FUNCTION metadata.update_vector_store_stats()
RETURNS void AS $$
BEGIN
    UPDATE metadata.vector_store_stats
    SET
        layer1_count = (SELECT COUNT(*) FROM vectors.layer1),
        layer2_count = (SELECT COUNT(*) FROM vectors.layer2),
        layer3_count = (SELECT COUNT(*) FROM vectors.layer3),
        total_vectors = (SELECT COUNT(*) FROM vectors.layer1) +
                       (SELECT COUNT(*) FROM vectors.layer2) +
                       (SELECT COUNT(*) FROM vectors.layer3),
        document_count = (SELECT COUNT(DISTINCT document_id) FROM vectors.layer1),
        updated_at = NOW()
    WHERE id = 1;
END;
$$ LANGUAGE plpgsql;

-- Grant usage on metadata schema
GRANT USAGE ON SCHEMA metadata TO PUBLIC;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
DECLARE
    stats_row RECORD;
BEGIN
    -- Verify stats table was created
    SELECT * INTO stats_row FROM metadata.vector_store_stats WHERE id = 1;

    RAISE NOTICE '✓ Metadata schema fixed successfully!';
    RAISE NOTICE '  - metadata.documents table created';
    RAISE NOTICE '  - metadata.vector_store_stats table created';
    RAISE NOTICE '';
    RAISE NOTICE 'Current Statistics:';
    RAISE NOTICE '  - Layer 1: % vectors', stats_row.layer1_count;
    RAISE NOTICE '  - Layer 2: % vectors', stats_row.layer2_count;
    RAISE NOTICE '  - Layer 3: % vectors', stats_row.layer3_count;
    RAISE NOTICE '  - Total: % vectors', stats_row.total_vectors;
    RAISE NOTICE '  - Documents: % documents', stats_row.document_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Vector store backend is now fully operational!';
END $$;
