-- ============================================================================
-- SUJBOT2 PostgreSQL Database Schema with pgvector
-- ============================================================================
-- Complete database schema for legal compliance system supporting:
-- - 100,000+ pages (300K-500K chunks)
-- - Multi-document architecture (contracts vs laws)
-- - Legal metadata (§ references, hierarchy paths)
-- - Vector embeddings (BGE-M3, 1024 dimensions)
-- - Full-text search (Czech language)
-- - Knowledge graph relationships
-- - Efficient hybrid search (vector + BM25 + metadata)
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;           -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- Trigram similarity for fuzzy search
CREATE EXTENSION IF NOT EXISTS btree_gin;        -- GIN index for multiple columns
CREATE EXTENSION IF NOT EXISTS pg_stat_statements; -- Query performance tracking

-- ============================================================================
-- DOCUMENTS TABLE
-- ============================================================================
-- Stores document-level metadata for contracts and laws

CREATE TABLE documents (
    -- Primary identification
    document_id VARCHAR(64) PRIMARY KEY,
    filename VARCHAR(512) NOT NULL,

    -- Document classification
    document_type VARCHAR(32) NOT NULL CHECK (document_type IN ('contract', 'law_code', 'regulation')),
    document_subtype VARCHAR(64),  -- e.g., 'employment_contract', 'building_code'

    -- Document content
    raw_text TEXT,                 -- Full raw text
    cleaned_text TEXT,             -- Preprocessed text

    -- Legal metadata
    title TEXT,
    document_number VARCHAR(64),   -- e.g., "89/2012"
    law_citation VARCHAR(128),     -- e.g., "Zákon č. 89/2012 Sb."
    effective_date DATE,

    -- File metadata
    file_path TEXT,
    file_format VARCHAR(16),
    file_size_bytes BIGINT,

    -- Processing status
    status VARCHAR(32) NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'processing', 'indexed', 'error')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    error_message TEXT,

    -- Statistics
    total_pages INTEGER,
    total_words INTEGER,
    total_sections INTEGER,
    total_chunks INTEGER,

    -- Timestamps
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    indexed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Additional metadata as JSONB
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Indexes
    CONSTRAINT documents_filename_not_empty CHECK (LENGTH(TRIM(filename)) > 0)
);

-- Indexes for documents table
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_uploaded_at ON documents(uploaded_at DESC);
CREATE INDEX idx_documents_document_number ON documents(document_number) WHERE document_number IS NOT NULL;
CREATE INDEX idx_documents_metadata_gin ON documents USING GIN (metadata jsonb_path_ops);
CREATE INDEX idx_documents_title_trgm ON documents USING GIN (title gin_trgm_ops);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE documents IS 'Document-level metadata for contracts, laws, and regulations';
COMMENT ON COLUMN documents.metadata IS 'Additional metadata: parties, contract_type, law_type, etc.';

-- ============================================================================
-- CHUNKS TABLE
-- ============================================================================
-- Stores chunk content, embeddings, and legal metadata

CREATE TABLE chunks (
    -- Primary identification
    chunk_id VARCHAR(128) PRIMARY KEY,
    chunk_index INTEGER NOT NULL,

    -- Document reference
    document_id VARCHAR(64) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,

    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64),  -- SHA-256 hash for deduplication

    -- Vector embedding (legal-xlm-roberta-base, 768 dimensions)
    embedding vector(768),

    -- Legal structure metadata
    hierarchy_path TEXT,           -- "Část II > Hlava III > §89"
    legal_reference VARCHAR(128),  -- "§89" or "Článek 5.2"
    structural_level VARCHAR(32),  -- 'paragraph', 'article', 'subsection', etc.

    -- Hierarchical components (for filtering)
    part_number VARCHAR(16),
    chapter_number VARCHAR(16),
    section_number VARCHAR(16),
    paragraph_number INTEGER,
    article_number INTEGER,
    subsection_number INTEGER,
    letter VARCHAR(4),
    point_number VARCHAR(16),

    -- Position in document
    start_char BIGINT,
    end_char BIGINT,
    start_line INTEGER,
    end_line INTEGER,
    page_number INTEGER,

    -- Content classification
    content_type VARCHAR(32),  -- 'obligation', 'prohibition', 'definition', 'general', 'procedure'
    contains_obligation BOOLEAN DEFAULT false,
    contains_prohibition BOOLEAN DEFAULT false,
    contains_definition BOOLEAN DEFAULT false,

    -- Statistics
    token_count INTEGER,
    word_count INTEGER,

    -- References
    references_to TEXT[],    -- Array of legal references cited
    referenced_by TEXT[],    -- Array of chunk_ids that reference this chunk

    -- Full-text search (tsvector for Czech)
    content_tsv tsvector,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Additional metadata as JSONB
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT chunks_content_not_empty CHECK (LENGTH(TRIM(content)) > 0),
    CONSTRAINT chunks_chunk_index_positive CHECK (chunk_index >= 0)
);

-- Critical indexes for chunks table

-- 1. Document lookup (most common query pattern)
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_document_id_index ON chunks(document_id, chunk_index);

-- 2. Vector similarity search (IVFFlat for 300K-500K vectors)
-- Number of lists = sqrt(num_rows) ≈ sqrt(400000) ≈ 632, rounded to 512
-- This balances index build time vs query performance
CREATE INDEX idx_chunks_embedding_ivfflat ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 512);

-- Alternative: Use HNSW for even faster queries (requires more memory)
-- CREATE INDEX idx_chunks_embedding_hnsw ON chunks
-- USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- 3. Full-text search (Czech language configuration)
CREATE INDEX idx_chunks_content_tsv_gin ON chunks USING GIN (content_tsv);

-- 4. Legal reference lookup
CREATE INDEX idx_chunks_legal_reference ON chunks(legal_reference) WHERE legal_reference IS NOT NULL;
CREATE INDEX idx_chunks_legal_reference_trgm ON chunks USING GIN (legal_reference gin_trgm_ops);

-- 5. Hierarchical filtering
CREATE INDEX idx_chunks_hierarchy_path ON chunks(hierarchy_path);
CREATE INDEX idx_chunks_hierarchy_path_trgm ON chunks USING GIN (hierarchy_path gin_trgm_ops);
CREATE INDEX idx_chunks_structural_level ON chunks(structural_level);

-- 6. Hierarchical components (for precise filtering)
CREATE INDEX idx_chunks_paragraph ON chunks(document_id, paragraph_number) WHERE paragraph_number IS NOT NULL;
CREATE INDEX idx_chunks_article ON chunks(document_id, article_number) WHERE article_number IS NOT NULL;
CREATE INDEX idx_chunks_subsection ON chunks(document_id, paragraph_number, subsection_number)
    WHERE paragraph_number IS NOT NULL AND subsection_number IS NOT NULL;

-- 7. Content classification
CREATE INDEX idx_chunks_content_type ON chunks(content_type);
CREATE INDEX idx_chunks_obligations ON chunks(document_id) WHERE contains_obligation = true;
CREATE INDEX idx_chunks_prohibitions ON chunks(document_id) WHERE contains_prohibition = true;
CREATE INDEX idx_chunks_definitions ON chunks(document_id) WHERE contains_definition = true;

-- 8. References arrays
CREATE INDEX idx_chunks_references_to_gin ON chunks USING GIN (references_to);
CREATE INDEX idx_chunks_referenced_by_gin ON chunks USING GIN (referenced_by);

-- 9. Metadata search
CREATE INDEX idx_chunks_metadata_gin ON chunks USING GIN (metadata jsonb_path_ops);

-- 10. Content hash for deduplication
CREATE INDEX idx_chunks_content_hash ON chunks(content_hash) WHERE content_hash IS NOT NULL;

-- 11. Composite index for hybrid retrieval
CREATE INDEX idx_chunks_hybrid_search ON chunks(document_id, structural_level, content_type);

-- Trigger to auto-generate content_tsv for full-text search (Czech with fallback)
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

CREATE TRIGGER chunks_content_tsv_update BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_content_tsv_trigger();

-- Trigger to auto-generate content_hash
CREATE OR REPLACE FUNCTION chunks_content_hash_trigger()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_hash = encode(sha256(NEW.content::bytea), 'hex');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunks_content_hash_update BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_content_hash_trigger();

COMMENT ON TABLE chunks IS 'Chunk content, embeddings, and legal metadata';
COMMENT ON COLUMN chunks.embedding IS 'legal-xlm-roberta-base multilingual embedding (768 dimensions)';
COMMENT ON COLUMN chunks.content_tsv IS 'Full-text search vector (Czech configuration)';
COMMENT ON COLUMN chunks.metadata IS 'Additional metadata: parties_mentioned, is_aggregated, etc.';

-- ============================================================================
-- REFERENCES TABLE
-- ============================================================================
-- Maps legal references (§, articles) to chunks for fast lookup

CREATE TABLE "references" (
    -- Primary identification
    reference_id BIGSERIAL PRIMARY KEY,

    -- Reference information
    reference_text VARCHAR(128) NOT NULL,  -- "§89", "§89 odst. 2", "Článek 5"
    reference_normalized VARCHAR(128) NOT NULL,  -- Normalized form
    reference_type VARCHAR(32) NOT NULL,  -- 'paragraph', 'article', 'subsection', etc.

    -- Components (parsed from reference_text)
    paragraph_num INTEGER,
    article_num INTEGER,
    subsection_num INTEGER,
    letter VARCHAR(4),
    point VARCHAR(16),

    -- Target chunk
    chunk_id VARCHAR(128) NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    document_id VARCHAR(64) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,

    -- Source context (where this reference appears)
    source_chunk_id VARCHAR(128) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    source_document_id VARCHAR(64) REFERENCES documents(document_id) ON DELETE CASCADE,
    source_position INTEGER,  -- Character offset in source

    -- Reference context
    context_text TEXT,  -- Surrounding text

    -- Metadata
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    is_explicit BOOLEAN DEFAULT true,  -- Explicit citation vs implicit reference

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for references table
CREATE INDEX idx_references_text ON "references"(reference_text);
CREATE INDEX idx_references_normalized ON "references"(reference_normalized);
CREATE INDEX idx_references_type ON "references"(reference_type);
CREATE INDEX idx_references_chunk_id ON "references"(chunk_id);
CREATE INDEX idx_references_document_id ON "references"(document_id);
CREATE INDEX idx_references_source_chunk ON "references"(source_chunk_id) WHERE source_chunk_id IS NOT NULL;
CREATE INDEX idx_references_paragraph ON "references"(paragraph_num) WHERE paragraph_num IS NOT NULL;
CREATE INDEX idx_references_article ON "references"(article_num) WHERE article_num IS NOT NULL;

-- Composite index for cross-document reference matching
CREATE INDEX idx_references_cross_doc ON "references"(reference_normalized, document_id, chunk_id);

-- GIN index for efficient array operations
CREATE INDEX idx_references_metadata_gin ON "references" USING GIN (metadata jsonb_path_ops);

COMMENT ON TABLE "references" IS 'Legal reference mappings (§, articles) to chunks';
COMMENT ON COLUMN "references".reference_normalized IS 'Normalized form: §89 odst. 2 → §89.2';

-- ============================================================================
-- CROSS_REFERENCES TABLE
-- ============================================================================
-- Knowledge graph edges representing relationships between chunks

CREATE TABLE cross_references (
    -- Primary identification
    edge_id BIGSERIAL PRIMARY KEY,

    -- Edge definition
    source_chunk_id VARCHAR(128) NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    target_chunk_id VARCHAR(128) NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,

    -- Edge type
    edge_type VARCHAR(32) NOT NULL CHECK (edge_type IN (
        'part_of',           -- Structural hierarchy
        'references',        -- Explicit citation
        'related_to',        -- Semantic similarity
        'conflicts_with',    -- Detected conflict
        'complies_with',     -- Compliance mapping
        'requires',          -- Dependency
        'defines'            -- Definition relationship
    )),

    -- Edge properties
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0.0),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Context
    context_text TEXT,
    explanation TEXT,  -- Human-readable explanation of relationship

    -- For conflicts
    severity VARCHAR(16) CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    risk_score FLOAT CHECK (risk_score >= 0.0 AND risk_score <= 1.0),

    -- For semantic relationships
    similarity_score FLOAT CHECK (similarity_score >= 0.0 AND similarity_score <= 1.0),

    -- Provenance
    detected_by VARCHAR(64),  -- 'reference_extractor', 'semantic_linker', 'compliance_analyzer'
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT cross_references_no_self_loop CHECK (source_chunk_id != target_chunk_id),
    UNIQUE (source_chunk_id, target_chunk_id, edge_type)
);

-- Indexes for cross_references table
CREATE INDEX idx_cross_refs_source ON cross_references(source_chunk_id);
CREATE INDEX idx_cross_refs_target ON cross_references(target_chunk_id);
CREATE INDEX idx_cross_refs_edge_type ON cross_references(edge_type);
CREATE INDEX idx_cross_refs_source_type ON cross_references(source_chunk_id, edge_type);
CREATE INDEX idx_cross_refs_target_type ON cross_references(target_chunk_id, edge_type);

-- Bidirectional lookups
CREATE INDEX idx_cross_refs_bidirectional ON cross_references(source_chunk_id, target_chunk_id);

-- Conflict queries
CREATE INDEX idx_cross_refs_conflicts ON cross_references(edge_type, severity)
    WHERE edge_type = 'conflicts_with';

-- Compliance queries
CREATE INDEX idx_cross_refs_compliance ON cross_references(edge_type)
    WHERE edge_type = 'complies_with';

-- Semantic similarity queries
CREATE INDEX idx_cross_refs_semantic ON cross_references(edge_type, similarity_score DESC)
    WHERE edge_type = 'related_to';

-- Weight-based ranking
CREATE INDEX idx_cross_refs_weight ON cross_references(source_chunk_id, weight DESC);

-- Metadata search
CREATE INDEX idx_cross_refs_metadata_gin ON cross_references USING GIN (metadata jsonb_path_ops);

COMMENT ON TABLE cross_references IS 'Knowledge graph edges (relationships between chunks)';
COMMENT ON COLUMN cross_references.edge_type IS 'Relationship type: part_of, references, conflicts_with, etc.';
COMMENT ON COLUMN cross_references.weight IS 'Edge importance weight (for graph algorithms)';

-- ============================================================================
-- COMPLIANCE_REPORTS TABLE
-- ============================================================================
-- Stores compliance check results

CREATE TABLE compliance_reports (
    -- Primary identification
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Documents being compared
    contract_document_id VARCHAR(64) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    law_document_ids VARCHAR(64)[] NOT NULL,  -- Array of law document IDs

    -- Report status
    status VARCHAR(32) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'error')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),

    -- Results summary
    total_requirements INTEGER,
    compliant_requirements INTEGER,
    missing_requirements INTEGER,
    conflicting_requirements INTEGER,

    -- Risk assessment
    overall_risk_score FLOAT CHECK (overall_risk_score >= 0.0 AND overall_risk_score <= 1.0),
    risk_level VARCHAR(16) CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),

    -- Report content
    summary TEXT,
    recommendations TEXT,
    full_report JSONB,

    -- Processing metadata
    processing_time_seconds FLOAT,
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for compliance_reports table
CREATE INDEX idx_compliance_reports_contract ON compliance_reports(contract_document_id);
CREATE INDEX idx_compliance_reports_status ON compliance_reports(status);
CREATE INDEX idx_compliance_reports_created_at ON compliance_reports(created_at DESC);
CREATE INDEX idx_compliance_reports_risk_level ON compliance_reports(risk_level);
CREATE INDEX idx_compliance_reports_law_docs_gin ON compliance_reports USING GIN (law_document_ids);

COMMENT ON TABLE compliance_reports IS 'Compliance check results comparing contracts to laws';

-- ============================================================================
-- QUERY_CACHE TABLE
-- ============================================================================
-- Cache for expensive hybrid search queries

CREATE TABLE query_cache (
    -- Primary identification
    cache_id BIGSERIAL PRIMARY KEY,

    -- Query definition
    query_text TEXT NOT NULL,
    query_embedding vector(768),
    query_hash VARCHAR(64) NOT NULL UNIQUE,  -- Hash of query parameters

    -- Query parameters
    document_ids VARCHAR(64)[],
    top_k INTEGER,
    filters JSONB,

    -- Cached results
    results JSONB NOT NULL,
    result_count INTEGER,

    -- Cache metadata
    execution_time_ms FLOAT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count INTEGER DEFAULT 1,

    -- TTL
    expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '1 hour'
);

-- Indexes for query_cache table
CREATE INDEX idx_query_cache_hash ON query_cache(query_hash);
CREATE INDEX idx_query_cache_expires ON query_cache(expires_at);
CREATE INDEX idx_query_cache_accessed ON query_cache(last_accessed_at DESC);

-- Trigger to update last_accessed_at
CREATE OR REPLACE FUNCTION query_cache_accessed_trigger()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_accessed_at = NOW();
    NEW.access_count = NEW.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER query_cache_accessed BEFORE UPDATE ON query_cache
    FOR EACH ROW EXECUTE FUNCTION query_cache_accessed_trigger();

COMMENT ON TABLE query_cache IS 'Cache for expensive hybrid search queries';

-- ============================================================================
-- STATISTICS TABLE
-- ============================================================================
-- System-wide statistics for monitoring

CREATE TABLE statistics (
    stat_id BIGSERIAL PRIMARY KEY,
    stat_name VARCHAR(64) NOT NULL,
    stat_value NUMERIC,
    stat_data JSONB,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT statistics_unique_name_time UNIQUE (stat_name, recorded_at)
);

CREATE INDEX idx_statistics_name ON statistics(stat_name);
CREATE INDEX idx_statistics_recorded_at ON statistics(recorded_at DESC);

COMMENT ON TABLE statistics IS 'System-wide statistics and monitoring data';

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Document summary with chunk counts
CREATE OR REPLACE VIEW v_document_summary AS
SELECT
    d.document_id,
    d.filename,
    d.document_type,
    d.title,
    d.document_number,
    d.status,
    d.uploaded_at,
    d.indexed_at,
    COUNT(c.chunk_id) as chunk_count,
    COUNT(DISTINCT r.reference_id) as reference_count,
    AVG(c.token_count) as avg_chunk_tokens,
    SUM(CASE WHEN c.contains_obligation THEN 1 ELSE 0 END) as obligation_count,
    SUM(CASE WHEN c.contains_prohibition THEN 1 ELSE 0 END) as prohibition_count,
    SUM(CASE WHEN c.contains_definition THEN 1 ELSE 0 END) as definition_count
FROM documents d
LEFT JOIN chunks c ON d.document_id = c.document_id
LEFT JOIN "references" r ON d.document_id = r.document_id
GROUP BY d.document_id, d.filename, d.document_type, d.title,
         d.document_number, d.status, d.uploaded_at, d.indexed_at;

COMMENT ON VIEW v_document_summary IS 'Document summary with statistics';

-- View: Chunk detail with document info
CREATE OR REPLACE VIEW v_chunk_detail AS
SELECT
    c.chunk_id,
    c.chunk_index,
    c.document_id,
    d.filename,
    d.document_type,
    c.legal_reference,
    c.hierarchy_path,
    c.structural_level,
    c.content_type,
    c.token_count,
    c.contains_obligation,
    c.contains_prohibition,
    c.contains_definition,
    ARRAY_LENGTH(c.references_to, 1) as outgoing_refs_count,
    ARRAY_LENGTH(c.referenced_by, 1) as incoming_refs_count
FROM chunks c
JOIN documents d ON c.document_id = d.document_id;

COMMENT ON VIEW v_chunk_detail IS 'Chunk details with document context';

-- View: Knowledge graph statistics
CREATE OR REPLACE VIEW v_knowledge_graph_stats AS
SELECT
    edge_type,
    COUNT(*) as edge_count,
    AVG(weight) as avg_weight,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT source_chunk_id) as unique_sources,
    COUNT(DISTINCT target_chunk_id) as unique_targets
FROM cross_references
GROUP BY edge_type
ORDER BY edge_count DESC;

COMMENT ON VIEW v_knowledge_graph_stats IS 'Knowledge graph edge statistics by type';

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Hybrid search combining vector similarity + full-text + metadata
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(768),
    doc_ids VARCHAR(64)[] DEFAULT NULL,
    limit_results INTEGER DEFAULT 20,
    vector_weight FLOAT DEFAULT 0.5,
    fulltext_weight FLOAT DEFAULT 0.3,
    metadata_filters JSONB DEFAULT NULL
)
RETURNS TABLE (
    chunk_id VARCHAR(128),
    content TEXT,
    legal_reference VARCHAR(128),
    hierarchy_path TEXT,
    document_id VARCHAR(64),
    combined_score FLOAT,
    vector_score FLOAT,
    fulltext_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        -- Vector similarity search
        SELECT
            c.chunk_id,
            c.content,
            c.legal_reference,
            c.hierarchy_path,
            c.document_id,
            1 - (c.embedding <=> query_embedding) AS vector_similarity
        FROM chunks c
        WHERE (doc_ids IS NULL OR c.document_id = ANY(doc_ids))
            AND c.embedding IS NOT NULL
            AND (metadata_filters IS NULL OR c.metadata @> metadata_filters)
        ORDER BY c.embedding <=> query_embedding
        LIMIT limit_results * 3  -- Retrieve more candidates for fusion
    ),
    fulltext_results AS (
        -- Full-text search (with dynamic config selection)
        SELECT
            c.chunk_id,
            ts_rank_cd(c.content_tsv, query_ts) AS fulltext_rank
        FROM chunks c,
             to_tsquery(
                 CASE
                     WHEN EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'czech')
                     THEN 'czech'
                     ELSE 'simple'
                 END,
                 query_text
             ) query_ts
        WHERE (doc_ids IS NULL OR c.document_id = ANY(doc_ids))
            AND c.content_tsv @@ query_ts
            AND (metadata_filters IS NULL OR c.metadata @> metadata_filters)
        ORDER BY fulltext_rank DESC
        LIMIT limit_results * 3
    )
    -- Combine results using reciprocal rank fusion
    SELECT
        v.chunk_id,
        v.content,
        v.legal_reference,
        v.hierarchy_path,
        v.document_id,
        (vector_weight * COALESCE(v.vector_similarity, 0) +
         fulltext_weight * COALESCE(f.fulltext_rank, 0)) AS combined_score,
        v.vector_similarity AS vector_score,
        COALESCE(f.fulltext_rank, 0) AS fulltext_score
    FROM vector_results v
    LEFT JOIN fulltext_results f ON v.chunk_id = f.chunk_id
    ORDER BY combined_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION hybrid_search IS 'Hybrid search combining vector similarity and full-text search';

-- Function: Get related chunks (knowledge graph traversal)
CREATE OR REPLACE FUNCTION get_related_chunks(
    start_chunk_id VARCHAR(128),
    edge_types VARCHAR(32)[] DEFAULT NULL,
    max_depth INTEGER DEFAULT 2
)
RETURNS TABLE (
    chunk_id VARCHAR(128),
    depth INTEGER,
    edge_type VARCHAR(32),
    path VARCHAR(128)[]
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE graph_traversal AS (
        -- Base case: starting chunk
        SELECT
            start_chunk_id AS chunk_id,
            0 AS depth,
            NULL::VARCHAR(32) AS edge_type,
            ARRAY[start_chunk_id] AS path

        UNION ALL

        -- Recursive case: follow edges
        SELECT
            cr.target_chunk_id AS chunk_id,
            gt.depth + 1 AS depth,
            cr.edge_type,
            gt.path || cr.target_chunk_id AS path
        FROM graph_traversal gt
        JOIN cross_references cr ON gt.chunk_id = cr.source_chunk_id
        WHERE gt.depth < max_depth
            AND (edge_types IS NULL OR cr.edge_type = ANY(edge_types))
            AND NOT (cr.target_chunk_id = ANY(gt.path))  -- Prevent cycles
    )
    SELECT DISTINCT ON (gt.chunk_id)
        gt.chunk_id,
        gt.depth,
        gt.edge_type,
        gt.path
    FROM graph_traversal gt
    WHERE gt.depth > 0  -- Exclude starting chunk
    ORDER BY gt.chunk_id, gt.depth;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_related_chunks IS 'Knowledge graph traversal to find related chunks';

-- Function: Find chunks by legal reference with fuzzy matching
CREATE OR REPLACE FUNCTION find_by_legal_reference(
    ref_pattern TEXT,
    doc_id VARCHAR(64) DEFAULT NULL,
    fuzzy_match BOOLEAN DEFAULT false
)
RETURNS TABLE (
    chunk_id VARCHAR(128),
    legal_reference VARCHAR(128),
    content TEXT,
    similarity FLOAT
) AS $$
BEGIN
    IF fuzzy_match THEN
        -- Fuzzy matching using trigrams
        RETURN QUERY
        SELECT
            c.chunk_id,
            c.legal_reference,
            c.content,
            similarity(c.legal_reference, ref_pattern) AS similarity
        FROM chunks c
        WHERE (doc_id IS NULL OR c.document_id = doc_id)
            AND c.legal_reference IS NOT NULL
            AND c.legal_reference % ref_pattern  -- Trigram similarity operator
        ORDER BY similarity DESC
        LIMIT 50;
    ELSE
        -- Exact matching
        RETURN QUERY
        SELECT
            c.chunk_id,
            c.legal_reference,
            c.content,
            1.0 AS similarity
        FROM chunks c
        WHERE (doc_id IS NULL OR c.document_id = doc_id)
            AND c.legal_reference = ref_pattern;
    END IF;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_by_legal_reference IS 'Find chunks by legal reference with optional fuzzy matching';

-- Function: Calculate document similarity
CREATE OR REPLACE FUNCTION document_similarity(
    doc_id_1 VARCHAR(64),
    doc_id_2 VARCHAR(64)
)
RETURNS FLOAT AS $$
DECLARE
    avg_similarity FLOAT;
BEGIN
    -- Calculate average cosine similarity between all chunk pairs
    SELECT AVG(1 - (c1.embedding <=> c2.embedding))
    INTO avg_similarity
    FROM chunks c1
    CROSS JOIN chunks c2
    WHERE c1.document_id = doc_id_1
        AND c2.document_id = doc_id_2
        AND c1.embedding IS NOT NULL
        AND c2.embedding IS NOT NULL
    LIMIT 1000;  -- Sample to avoid expensive computation

    RETURN COALESCE(avg_similarity, 0.0);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION document_similarity IS 'Calculate semantic similarity between two documents';

-- ============================================================================
-- MAINTENANCE FUNCTIONS
-- ============================================================================

-- Function: Clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM query_cache
    WHERE expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION clean_expired_cache IS 'Remove expired entries from query cache';

-- Function: Vacuum and analyze all tables
CREATE OR REPLACE FUNCTION maintenance_vacuum_analyze()
RETURNS VOID AS $$
BEGIN
    VACUUM ANALYZE documents;
    VACUUM ANALYZE chunks;
    VACUUM ANALYZE "references";
    VACUUM ANALYZE cross_references;
    VACUUM ANALYZE compliance_reports;
    VACUUM ANALYZE query_cache;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION maintenance_vacuum_analyze IS 'Vacuum and analyze all tables';

-- Function: Reindex all tables
CREATE OR REPLACE FUNCTION maintenance_reindex()
RETURNS VOID AS $$
BEGIN
    REINDEX TABLE documents;
    REINDEX TABLE chunks;
    REINDEX TABLE "references";
    REINDEX TABLE cross_references;
    REINDEX TABLE compliance_reports;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION maintenance_reindex IS 'Rebuild all indexes';

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default statistics
INSERT INTO statistics (stat_name, stat_value) VALUES
    ('total_documents', 0),
    ('total_chunks', 0),
    ('total_references', 0),
    ('total_edges', 0)
ON CONFLICT (stat_name, recorded_at) DO NOTHING;

-- ============================================================================
-- PERMISSIONS (example for application user)
-- ============================================================================

-- Create application role (uncomment when setting up)
-- CREATE ROLE sujbot_app WITH LOGIN PASSWORD 'your_secure_password';
-- GRANT CONNECT ON DATABASE sujbot2 TO sujbot_app;
-- GRANT USAGE ON SCHEMA public TO sujbot_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO sujbot_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO sujbot_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO sujbot_app;

-- ============================================================================
-- PERFORMANCE TUNING RECOMMENDATIONS
-- ============================================================================

/*
RECOMMENDED POSTGRESQL CONFIGURATION (postgresql.conf):

# Memory settings (adjust based on available RAM)
shared_buffers = 4GB                     # 25% of RAM
effective_cache_size = 12GB              # 75% of RAM
maintenance_work_mem = 1GB               # For index builds
work_mem = 128MB                         # Per operation

# Vector-specific settings
max_parallel_workers_per_gather = 4      # Parallel queries
max_parallel_workers = 8
max_worker_processes = 8

# IVFFlat index settings (for pgvector)
ivfflat.probes = 10                      # Balance speed vs accuracy

# Checkpoint settings
checkpoint_timeout = 15min
checkpoint_completion_target = 0.9

# WAL settings
wal_buffers = 16MB
min_wal_size = 1GB
max_wal_size = 4GB

# Query planner
random_page_cost = 1.1                   # SSD
effective_io_concurrency = 200           # SSD

# Connection settings
max_connections = 100

# Logging
log_min_duration_statement = 1000        # Log queries > 1s
*/

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
