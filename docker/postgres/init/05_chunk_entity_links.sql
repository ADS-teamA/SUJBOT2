-- ============================================================================
-- Chunk-Entity Bidirectional Linking for Graphiti Integration
-- ============================================================================
-- This migration creates tables for linking PostgreSQL chunks to Neo4j entities.
-- Enables bidirectional queries:
--   - Chunk → Entities: What entities are mentioned in this chunk?
--   - Entity → Chunks: Which chunks mention this entity?
--
-- Architecture:
--   PostgreSQL (vectors.layer3)  ←→  graph.chunk_entity_mentions  ←→  Neo4j (Graphiti)
-- ============================================================================

-- Ensure we're in the correct database
\c sujbot

-- ============================================================================
-- CREATE SCHEMA FOR GRAPH-RELATED TABLES
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS graph;

COMMENT ON SCHEMA graph IS 'Schema for knowledge graph related tables (chunk-entity linking)';

-- ============================================================================
-- CHUNK-ENTITY MENTIONS TABLE
-- ============================================================================

-- Table: chunk_entity_mentions
-- Links chunks in PostgreSQL to entities in Neo4j (Graphiti)
CREATE TABLE IF NOT EXISTS graph.chunk_entity_mentions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Chunk reference (to vectors.layer3)
    chunk_id VARCHAR(255) NOT NULL,
    document_id VARCHAR(255) NOT NULL,

    -- Entity reference (Graphiti node UUID)
    entity_uuid VARCHAR(255) NOT NULL,
    entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,

    -- Mention metadata
    mention_type VARCHAR(50) DEFAULT 'mentioned',  -- mentioned, defined, primary
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Position in chunk (optional, for highlighting)
    start_offset INT,
    end_offset INT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure unique entity mention per chunk
    CONSTRAINT unique_chunk_entity UNIQUE (chunk_id, entity_uuid)
);

COMMENT ON TABLE graph.chunk_entity_mentions IS
    'Bidirectional links between PostgreSQL chunks and Neo4j Graphiti entities';

COMMENT ON COLUMN graph.chunk_entity_mentions.entity_uuid IS
    'UUID of the entity node in Neo4j/Graphiti';

COMMENT ON COLUMN graph.chunk_entity_mentions.mention_type IS
    'Type of mention: mentioned (appears), defined (first occurrence), primary (main topic)';

-- ============================================================================
-- INDEXES FOR BIDIRECTIONAL QUERIES
-- ============================================================================

-- Index for Chunk → Entities queries
-- "What entities are mentioned in chunk X?"
CREATE INDEX IF NOT EXISTS idx_chunk_entity_chunk_id
ON graph.chunk_entity_mentions(chunk_id);

-- Index for Entity → Chunks queries
-- "Which chunks mention entity Y?"
CREATE INDEX IF NOT EXISTS idx_chunk_entity_uuid
ON graph.chunk_entity_mentions(entity_uuid);

-- Index for document-level entity queries
-- "What entities appear in document Z?"
CREATE INDEX IF NOT EXISTS idx_chunk_entity_document_id
ON graph.chunk_entity_mentions(document_id);

-- Index for entity type filtering
-- "Find all regulations mentioned in chunks"
CREATE INDEX IF NOT EXISTS idx_chunk_entity_type
ON graph.chunk_entity_mentions(entity_type);

-- Index for entity name search
-- "Find chunks mentioning 'SÚJB'"
CREATE INDEX IF NOT EXISTS idx_chunk_entity_name
ON graph.chunk_entity_mentions(entity_name);

-- GIN index for fast text search on entity names
CREATE INDEX IF NOT EXISTS idx_chunk_entity_name_gin
ON graph.chunk_entity_mentions USING gin(entity_name gin_trgm_ops);

-- ============================================================================
-- ENTITY EXTRACTION JOBS TABLE
-- ============================================================================

-- Table: entity_extraction_jobs
-- Tracks Graphiti extraction progress for documents
CREATE TABLE IF NOT EXISTS graph.entity_extraction_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL UNIQUE,

    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed

    -- Statistics
    total_chunks INT DEFAULT 0,
    processed_chunks INT DEFAULT 0,
    total_entities INT DEFAULT 0,
    total_relationships INT DEFAULT 0,
    unique_entity_count INT DEFAULT 0,

    -- Error tracking
    error_message TEXT,
    failed_chunks JSONB DEFAULT '[]'::jsonb,

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    processing_time_ms FLOAT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE graph.entity_extraction_jobs IS
    'Tracks Graphiti entity extraction jobs for documents';

-- Index for status queries
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_status
ON graph.entity_extraction_jobs(status);

-- Index for document lookup
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_document
ON graph.entity_extraction_jobs(document_id);

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function: Get entities for a chunk
CREATE OR REPLACE FUNCTION graph.get_chunk_entities(
    p_chunk_id VARCHAR(255)
)
RETURNS TABLE(
    entity_uuid VARCHAR(255),
    entity_name VARCHAR(500),
    entity_type VARCHAR(100),
    mention_type VARCHAR(50),
    confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cem.entity_uuid,
        cem.entity_name,
        cem.entity_type,
        cem.mention_type,
        cem.confidence
    FROM graph.chunk_entity_mentions cem
    WHERE cem.chunk_id = p_chunk_id
    ORDER BY cem.confidence DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: Get chunks for an entity
CREATE OR REPLACE FUNCTION graph.get_entity_chunks(
    p_entity_uuid VARCHAR(255),
    p_limit INT DEFAULT 20
)
RETURNS TABLE(
    chunk_id VARCHAR(255),
    document_id VARCHAR(255),
    content TEXT,
    metadata JSONB,
    mention_type VARCHAR(50),
    confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        l3.chunk_id,
        l3.document_id,
        l3.content,
        l3.metadata,
        cem.mention_type,
        cem.confidence
    FROM vectors.layer3 l3
    JOIN graph.chunk_entity_mentions cem ON l3.chunk_id = cem.chunk_id
    WHERE cem.entity_uuid = p_entity_uuid
    ORDER BY cem.confidence DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Get entity co-occurrences
-- Find entities that frequently appear together in the same chunks
CREATE OR REPLACE FUNCTION graph.get_entity_cooccurrences(
    p_entity_uuid VARCHAR(255),
    p_limit INT DEFAULT 10
)
RETURNS TABLE(
    co_entity_uuid VARCHAR(255),
    co_entity_name VARCHAR(500),
    co_entity_type VARCHAR(100),
    cooccurrence_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cem2.entity_uuid as co_entity_uuid,
        cem2.entity_name as co_entity_name,
        cem2.entity_type as co_entity_type,
        COUNT(*) as cooccurrence_count
    FROM graph.chunk_entity_mentions cem1
    JOIN graph.chunk_entity_mentions cem2
        ON cem1.chunk_id = cem2.chunk_id
        AND cem1.entity_uuid != cem2.entity_uuid
    WHERE cem1.entity_uuid = p_entity_uuid
    GROUP BY cem2.entity_uuid, cem2.entity_name, cem2.entity_type
    ORDER BY cooccurrence_count DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Get document entity summary
CREATE OR REPLACE FUNCTION graph.get_document_entity_summary(
    p_document_id VARCHAR(255)
)
RETURNS TABLE(
    entity_type VARCHAR(100),
    entity_count BIGINT,
    example_entities TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cem.entity_type,
        COUNT(DISTINCT cem.entity_uuid) as entity_count,
        ARRAY_AGG(DISTINCT cem.entity_name ORDER BY cem.entity_name)[:5] as example_entities
    FROM graph.chunk_entity_mentions cem
    WHERE cem.document_id = p_document_id
    GROUP BY cem.entity_type
    ORDER BY entity_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: Link entities to chunk (bulk insert)
CREATE OR REPLACE FUNCTION graph.link_entities_to_chunk(
    p_chunk_id VARCHAR(255),
    p_document_id VARCHAR(255),
    p_entities JSONB  -- Array of {entity_uuid, entity_name, entity_type, confidence}
)
RETURNS INT AS $$
DECLARE
    v_inserted INT := 0;
    v_entity JSONB;
BEGIN
    FOR v_entity IN SELECT * FROM jsonb_array_elements(p_entities)
    LOOP
        INSERT INTO graph.chunk_entity_mentions (
            chunk_id, document_id, entity_uuid, entity_name, entity_type, confidence
        ) VALUES (
            p_chunk_id,
            p_document_id,
            v_entity->>'entity_uuid',
            v_entity->>'entity_name',
            v_entity->>'entity_type',
            COALESCE((v_entity->>'confidence')::FLOAT, 1.0)
        )
        ON CONFLICT (chunk_id, entity_uuid) DO UPDATE SET
            entity_name = EXCLUDED.entity_name,
            confidence = EXCLUDED.confidence,
            updated_at = NOW();

        v_inserted := v_inserted + 1;
    END LOOP;

    RETURN v_inserted;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGER FOR UPDATED_AT
-- ============================================================================

CREATE OR REPLACE FUNCTION graph.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunk_entity_mentions_updated
    BEFORE UPDATE ON graph.chunk_entity_mentions
    FOR EACH ROW
    EXECUTE FUNCTION graph.update_timestamp();

CREATE TRIGGER entity_extraction_jobs_updated
    BEFORE UPDATE ON graph.entity_extraction_jobs
    FOR EACH ROW
    EXECUTE FUNCTION graph.update_timestamp();

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '--- Chunk-Entity Linking Migration Complete ---';
    RAISE NOTICE '  Tables:';
    RAISE NOTICE '    - graph.chunk_entity_mentions (bidirectional linking)';
    RAISE NOTICE '    - graph.entity_extraction_jobs (job tracking)';
    RAISE NOTICE '  Functions:';
    RAISE NOTICE '    - graph.get_chunk_entities(chunk_id)';
    RAISE NOTICE '    - graph.get_entity_chunks(entity_uuid)';
    RAISE NOTICE '    - graph.get_entity_cooccurrences(entity_uuid)';
    RAISE NOTICE '    - graph.get_document_entity_summary(document_id)';
    RAISE NOTICE '    - graph.link_entities_to_chunk(chunk_id, document_id, entities)';
    RAISE NOTICE '';
    RAISE NOTICE 'Test queries:';
    RAISE NOTICE '  SELECT * FROM graph.get_chunk_entities(''doc_L3_c1_sec_1'');';
    RAISE NOTICE '  SELECT * FROM graph.get_entity_chunks(''graphiti_uuid_here'');';
END $$;
