-- ============================================================================
-- SUJBOT2 PostgreSQL Database Initialization
-- ============================================================================
-- This script creates the complete database schema for SUJBOT2:
-- 1. Extensions (pgvector, Apache AGE)
-- 2. Schemas (vectors, graphs, checkpoints, metadata)
-- 3. Tables for 3-layer vector store
-- 4. Tables for knowledge graph
-- 5. Tables for LangGraph checkpointing
-- 6. Indexes for performance (HNSW, IVFFlat, GIN, B-tree)
-- 7. Triggers for auto-updates
-- ============================================================================

-- Ensure we're in the correct database
\c sujbot

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

-- pgvector: Vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Apache AGE: Graph database (skipped for initial deployment)
-- CREATE EXTENSION IF NOT EXISTS age;

-- pg_trgm: Trigram similarity for full-text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- pg_stat_statements: Query performance tracking
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Load AGE into search path (skipped - AGE not installed)
-- LOAD 'age';
-- SET search_path = ag_catalog, "$user", public;

-- ============================================================================
-- SCHEMAS
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS vectors;
CREATE SCHEMA IF NOT EXISTS graphs;
CREATE SCHEMA IF NOT EXISTS checkpoints;
CREATE SCHEMA IF NOT EXISTS metadata;

-- ============================================================================
-- VECTORS SCHEMA: 3-Layer Embedding Tables
-- ============================================================================

-- Layer 1: Document-level embeddings (1 vector per document)
CREATE TABLE IF NOT EXISTS vectors.layer1 (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    title TEXT,

    -- Embedding (4096 dimensions for Qwen3-Embedding-8B)
    embedding vector(4096) NOT NULL,

    -- Content (for retrieval and generation)
    content TEXT NOT NULL,
    content_tsv tsvector,  -- Auto-generated via trigger

    -- Clustering metadata
    cluster_id INTEGER,
    cluster_label TEXT,
    cluster_confidence FLOAT,

    -- Hierarchical metadata
    hierarchical_path TEXT,  -- Breadcrumb: "doc_id"
    page_number INTEGER,

    -- Additional metadata (flexible JSONB)
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Layer 2: Section-level embeddings (N vectors per document)
CREATE TABLE IF NOT EXISTS vectors.layer2 (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,

    -- Section metadata
    section_id TEXT,
    section_title TEXT,
    section_path TEXT,
    section_level INTEGER,
    section_depth INTEGER,
    hierarchical_path TEXT,  -- "doc_id > section > subsection"
    page_number INTEGER,

    -- Embedding (4096 dimensions for Qwen3-Embedding-8B)
    embedding vector(4096) NOT NULL,

    -- Content
    content TEXT NOT NULL,
    content_tsv tsvector,

    -- Clustering
    cluster_id INTEGER,
    cluster_label TEXT,
    cluster_confidence FLOAT,

    -- Metadata
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Layer 3: Chunk-level embeddings (M vectors per document, PRIMARY retrieval layer)
CREATE TABLE IF NOT EXISTS vectors.layer3 (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,

    -- Section metadata (inherited from parent section)
    section_id TEXT,
    section_title TEXT,
    section_path TEXT,
    section_level INTEGER,
    section_depth INTEGER,
    hierarchical_path TEXT,
    page_number INTEGER,

    -- Chunk position
    char_start INTEGER,
    char_end INTEGER,

    -- Embedding (4096 dimensions for Qwen3-Embedding-8B)
    embedding vector(4096) NOT NULL,

    -- Content (WITHOUT summary-augmentation for generation)
    content TEXT NOT NULL,
    content_tsv tsvector,

    -- Clustering
    cluster_id INTEGER,
    cluster_label TEXT,
    cluster_confidence FLOAT,

    -- Metadata (section summaries, document summaries, etc.)
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- GRAPHS SCHEMA: Apache AGE Property Graph + Mirror Tables
-- ============================================================================

-- Create graph in Apache AGE (DISABLED - AGE not installed)
-- SELECT create_graph('knowledge_graph');

-- Entities mirror table (for efficient SQL queries without Cypher)
CREATE TABLE IF NOT EXISTS graphs.entities (
    id BIGSERIAL PRIMARY KEY,
    entity_id TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL,  -- EntityType enum (STANDARD, ORGANIZATION, etc.)
    name TEXT NOT NULL,
    normalized_name TEXT,  -- For deduplication
    confidence FLOAT NOT NULL DEFAULT 1.0,

    -- Metadata (properties like definition, aliases, etc.)
    metadata JSONB,

    -- Provenance
    source_chunk_id TEXT,
    source_document_id TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Relationships mirror table
CREATE TABLE IF NOT EXISTS graphs.relationships (
    id BIGSERIAL PRIMARY KEY,
    relationship_id TEXT NOT NULL UNIQUE,
    source_entity_id TEXT NOT NULL REFERENCES graphs.entities(entity_id) ON DELETE CASCADE,
    target_entity_id TEXT NOT NULL REFERENCES graphs.entities(entity_id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,  -- RelationshipType enum (COMPLIES_WITH, etc.)
    confidence FLOAT NOT NULL DEFAULT 1.0,

    -- Evidence text from source
    evidence TEXT,

    -- Properties (flexible JSONB)
    properties JSONB,

    -- Provenance
    source_chunk_id TEXT,
    source_document_id TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- CHECKPOINTS SCHEMA: LangGraph State Persistence
-- ============================================================================

-- LangGraph checkpoints (workflow state snapshots)
CREATE TABLE IF NOT EXISTS checkpoints.agent_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    parent_checkpoint_id TEXT,

    -- Full LangGraph state (JSONB)
    checkpoint JSONB NOT NULL,

    -- Metadata (timestamps, agent info, etc.)
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- LangGraph checkpoint writes (pending channel writes)
CREATE TABLE IF NOT EXISTS checkpoints.checkpoint_writes (
    id BIGSERIAL PRIMARY KEY,
    checkpoint_id TEXT NOT NULL REFERENCES checkpoints.agent_checkpoints(checkpoint_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

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
    dimensions INTEGER NOT NULL DEFAULT 4096,
    layer1_count INTEGER NOT NULL DEFAULT 0,
    layer2_count INTEGER NOT NULL DEFAULT 0,
    layer3_count INTEGER NOT NULL DEFAULT 0,
    total_vectors INTEGER NOT NULL DEFAULT 0,
    document_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert initial stats row
INSERT INTO metadata.vector_store_stats (dimensions, layer1_count, layer2_count, layer3_count, total_vectors, document_count)
VALUES (4096, 0, 0, 0, 0, 0)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- INDEXES: Performance Optimization
-- ============================================================================

-- ============================================================================
-- Vector Indexes (HNSW for production, IVFFlat for development)
-- ============================================================================

-- Layer 1: IVFFlat (fast build, small dataset)
CREATE INDEX IF NOT EXISTS idx_layer1_embedding_ivfflat
ON vectors.layer1 USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Layer 2: IVFFlat (medium dataset)
CREATE INDEX IF NOT EXISTS idx_layer2_embedding_ivfflat
ON vectors.layer2 USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Layer 3: HNSW (PRIMARY retrieval, highest quality)
-- NOTE: This can take 2-4 hours to build for 1M vectors!
-- Use IVFFlat for development, HNSW for production
CREATE INDEX IF NOT EXISTS idx_layer3_embedding_hnsw
ON vectors.layer3 USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat for faster builds (uncomment for development)
-- CREATE INDEX IF NOT EXISTS idx_layer3_embedding_ivfflat
-- ON vectors.layer3 USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 1000);

-- ============================================================================
-- Full-Text Search Indexes (GIN for tsvector)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_layer1_content_tsv
ON vectors.layer1 USING gin(content_tsv);

CREATE INDEX IF NOT EXISTS idx_layer2_content_tsv
ON vectors.layer2 USING gin(content_tsv);

CREATE INDEX IF NOT EXISTS idx_layer3_content_tsv
ON vectors.layer3 USING gin(content_tsv);

-- ============================================================================
-- Filtering Indexes (B-tree for document/section filtering)
-- ============================================================================

-- Layer 1
CREATE INDEX IF NOT EXISTS idx_layer1_document_id ON vectors.layer1(document_id);
CREATE INDEX IF NOT EXISTS idx_layer1_chunk_id ON vectors.layer1(chunk_id);

-- Layer 2
CREATE INDEX IF NOT EXISTS idx_layer2_document_id ON vectors.layer2(document_id);
CREATE INDEX IF NOT EXISTS idx_layer2_section_id ON vectors.layer2(section_id);
CREATE INDEX IF NOT EXISTS idx_layer2_chunk_id ON vectors.layer2(chunk_id);

-- Layer 3
CREATE INDEX IF NOT EXISTS idx_layer3_document_id ON vectors.layer3(document_id);
CREATE INDEX IF NOT EXISTS idx_layer3_section_id ON vectors.layer3(section_id);
CREATE INDEX IF NOT EXISTS idx_layer3_chunk_id ON vectors.layer3(chunk_id);
CREATE INDEX IF NOT EXISTS idx_layer3_hierarchical_path ON vectors.layer3(hierarchical_path);

-- Composite index for filtered vector search
CREATE INDEX IF NOT EXISTS idx_layer3_doc_section
ON vectors.layer3(document_id, section_id);

-- ============================================================================
-- Graph Indexes
-- ============================================================================

-- Entities
CREATE INDEX IF NOT EXISTS idx_entities_type ON graphs.entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON graphs.entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON graphs.entities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_entities_metadata ON graphs.entities USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_entities_source_doc ON graphs.entities(source_document_id);

-- Relationships
CREATE INDEX IF NOT EXISTS idx_relationships_source ON graphs.relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON graphs.relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON graphs.relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_properties ON graphs.relationships USING gin(properties);

-- Composite index for bidirectional graph traversal
CREATE INDEX IF NOT EXISTS idx_relationships_bidirectional
ON graphs.relationships(source_entity_id, target_entity_id, relationship_type);

-- ============================================================================
-- Checkpoint Indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints.agent_checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON checkpoints.agent_checkpoints(created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_checkpoint ON checkpoints.checkpoint_writes(checkpoint_id);

-- ============================================================================
-- Metadata Indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_documents_hash ON metadata.documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_indexed ON metadata.documents(indexed_at);

-- ============================================================================
-- TRIGGERS: Auto-Update Fields
-- ============================================================================

-- Function: Auto-update tsvector for full-text search
CREATE OR REPLACE FUNCTION vectors.update_content_tsv()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all layers
CREATE TRIGGER layer1_tsv_update
    BEFORE INSERT OR UPDATE OF content ON vectors.layer1
    FOR EACH ROW EXECUTE FUNCTION vectors.update_content_tsv();

CREATE TRIGGER layer2_tsv_update
    BEFORE INSERT OR UPDATE OF content ON vectors.layer2
    FOR EACH ROW EXECUTE FUNCTION vectors.update_content_tsv();

CREATE TRIGGER layer3_tsv_update
    BEFORE INSERT OR UPDATE OF content ON vectors.layer3
    FOR EACH ROW EXECUTE FUNCTION vectors.update_content_tsv();

-- Function: Auto-update timestamps
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply timestamp triggers
CREATE TRIGGER entities_update_timestamp
    BEFORE UPDATE ON graphs.entities
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER relationships_update_timestamp
    BEFORE UPDATE ON graphs.relationships
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER checkpoints_update_timestamp
    BEFORE UPDATE ON checkpoints.agent_checkpoints
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- ============================================================================
-- FUNCTIONS: Utility Functions
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

-- ============================================================================
-- GRANTS: Permissions (optional, for multi-user setup)
-- ============================================================================

-- Grant usage on schemas
GRANT USAGE ON SCHEMA vectors TO PUBLIC;
GRANT USAGE ON SCHEMA graphs TO PUBLIC;
GRANT USAGE ON SCHEMA checkpoints TO PUBLIC;
GRANT USAGE ON SCHEMA metadata TO PUBLIC;

-- Grant SELECT on all tables (read-only access)
-- Uncomment if needed for read-only users
-- GRANT SELECT ON ALL TABLES IN SCHEMA vectors TO readonly_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA graphs TO readonly_user;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✓ SUJBOT2 database initialization complete!';
    RAISE NOTICE '  - Extensions: pgvector, Apache AGE, pg_trgm';
    RAISE NOTICE '  - Schemas: vectors, graphs, checkpoints, metadata';
    RAISE NOTICE '  - Tables: 3 vector layers, graph tables, checkpoints';
    RAISE NOTICE '  - Indexes: HNSW (layer3), IVFFlat (layer1/2), GIN, B-tree';
    RAISE NOTICE '  - Triggers: Auto-update tsvector and timestamps';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '  1. Run migration: python scripts/migrate_faiss_to_postgres.py';
    RAISE NOTICE '  2. Verify data: SELECT * FROM metadata.vector_store_stats;';
    RAISE NOTICE '  3. Test search: SELECT * FROM vectors.layer3 LIMIT 1;';
END $$;
