-- ============================================================================
-- Entity Labeling Indexes for SUJBOT2
-- ============================================================================
-- This migration adds GIN indexes for entity-based queries on chunk metadata.
-- Entities are stored in vectors.layer3.metadata as:
-- {
--   "entities": [{"name": "SÚJB", "type": "organization", "confidence": 0.95}],
--   "entity_types": ["organization", "regulation"],
--   "topics": ["nuclear safety", "licensing"]
-- }
-- ============================================================================

-- Ensure we're in the correct database
\c sujbot

-- ============================================================================
-- GIN INDEXES FOR ENTITY QUERIES
-- ============================================================================

-- Index for entity containment queries (e.g., find chunks with specific entity)
-- Uses jsonb_path_ops for efficient @> (contains) queries
CREATE INDEX IF NOT EXISTS idx_layer3_metadata_entities
ON vectors.layer3 USING gin((metadata->'entities') jsonb_path_ops);

-- Index for entity type filtering
CREATE INDEX IF NOT EXISTS idx_layer3_metadata_entity_types
ON vectors.layer3 USING gin((metadata->'entity_types') jsonb_path_ops);

-- Index for topic filtering
CREATE INDEX IF NOT EXISTS idx_layer3_metadata_topics
ON vectors.layer3 USING gin((metadata->'topics') jsonb_path_ops);

-- Layer 1 entity indexes (document-level)
CREATE INDEX IF NOT EXISTS idx_layer1_metadata_entities
ON vectors.layer1 USING gin((metadata->'entities') jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_layer1_metadata_topics
ON vectors.layer1 USING gin((metadata->'topics') jsonb_path_ops);

-- ============================================================================
-- UTILITY FUNCTIONS FOR ENTITY SEARCH
-- ============================================================================

-- Function: Search chunks by entity type and name
-- Returns chunks containing the specified entity with similarity score
CREATE OR REPLACE FUNCTION vectors.search_by_entity(
    p_entity_type TEXT,
    p_entity_name TEXT,
    p_limit INT DEFAULT 10
)
RETURNS TABLE(
    chunk_id TEXT,
    document_id TEXT,
    content TEXT,
    metadata JSONB,
    hierarchical_path TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        l3.chunk_id,
        l3.document_id,
        l3.content,
        l3.metadata,
        l3.hierarchical_path,
        1.0::FLOAT as similarity  -- Exact match = 1.0
    FROM vectors.layer3 l3
    WHERE l3.metadata->'entities' @>
          jsonb_build_array(jsonb_build_object('type', p_entity_type, 'name', p_entity_name))
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Search chunks by entity type only
CREATE OR REPLACE FUNCTION vectors.search_by_entity_type(
    p_entity_type TEXT,
    p_limit INT DEFAULT 20
)
RETURNS TABLE(
    chunk_id TEXT,
    document_id TEXT,
    content TEXT,
    metadata JSONB,
    entity_count INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        l3.chunk_id,
        l3.document_id,
        l3.content,
        l3.metadata,
        (SELECT COUNT(*)::INT
         FROM jsonb_array_elements(l3.metadata->'entities') e
         WHERE e->>'type' = p_entity_type) as entity_count
    FROM vectors.layer3 l3
    WHERE l3.metadata->'entity_types' ? p_entity_type
    ORDER BY entity_count DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Search chunks by topic
CREATE OR REPLACE FUNCTION vectors.search_by_topic(
    p_topic TEXT,
    p_limit INT DEFAULT 20
)
RETURNS TABLE(
    chunk_id TEXT,
    document_id TEXT,
    content TEXT,
    metadata JSONB,
    hierarchical_path TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        l3.chunk_id,
        l3.document_id,
        l3.content,
        l3.metadata,
        l3.hierarchical_path
    FROM vectors.layer3 l3
    WHERE l3.metadata->'topics' ? p_topic
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Get all entities from a document
CREATE OR REPLACE FUNCTION vectors.get_document_entities(
    p_document_id TEXT
)
RETURNS TABLE(
    entity_name TEXT,
    entity_type TEXT,
    occurrence_count BIGINT,
    avg_confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e->>'name' as entity_name,
        e->>'type' as entity_type,
        COUNT(*) as occurrence_count,
        AVG((e->>'confidence')::FLOAT) as avg_confidence
    FROM vectors.layer3 l3,
         jsonb_array_elements(l3.metadata->'entities') e
    WHERE l3.document_id = p_document_id
    GROUP BY e->>'name', e->>'type'
    ORDER BY occurrence_count DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '--- Entity Labeling Indexes Migration Complete ---';
    RAISE NOTICE '  - GIN indexes for entity, topic, and entity_type queries';
    RAISE NOTICE '  - Utility functions: search_by_entity, search_by_entity_type';
    RAISE NOTICE '  - Utility functions: search_by_topic, get_document_entities';
    RAISE NOTICE '';
    RAISE NOTICE 'Test queries:';
    RAISE NOTICE '  SELECT * FROM vectors.search_by_entity(''organization'', ''SÚJB'');';
    RAISE NOTICE '  SELECT * FROM vectors.search_by_entity_type(''regulation'');';
    RAISE NOTICE '  SELECT * FROM vectors.search_by_topic(''nuclear safety'');';
END $$;
