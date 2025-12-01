-- BM25 Storage Tables
-- Store BM25 sparse retrieval data for all 3 layers

-- BM25 Store Configuration
CREATE TABLE IF NOT EXISTS bm25_config (
    id SERIAL PRIMARY KEY,
    languages TEXT[] NOT NULL,  -- e.g., ['cs', 'en']
    primary_language TEXT NOT NULL,  -- e.g., 'cs'
    format_version TEXT NOT NULL DEFAULT '3.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- BM25 Layer 1 (Document level)
CREATE TABLE IF NOT EXISTS bm25_layer1 (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    corpus TEXT NOT NULL,  -- Original text for BM25
    metadata JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- BM25 Layer 2 (Section level)
CREATE TABLE IF NOT EXISTS bm25_layer2 (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    section_id TEXT,
    corpus TEXT NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- BM25 Layer 3 (Chunk level - PRIMARY)
CREATE TABLE IF NOT EXISTS bm25_layer3 (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    section_id TEXT,
    corpus TEXT NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_bm25_layer1_document_id ON bm25_layer1(document_id);
CREATE INDEX IF NOT EXISTS idx_bm25_layer1_chunk_id ON bm25_layer1(chunk_id);

CREATE INDEX IF NOT EXISTS idx_bm25_layer2_document_id ON bm25_layer2(document_id);
CREATE INDEX IF NOT EXISTS idx_bm25_layer2_section_id ON bm25_layer2(section_id);
CREATE INDEX IF NOT EXISTS idx_bm25_layer2_chunk_id ON bm25_layer2(chunk_id);

CREATE INDEX IF NOT EXISTS idx_bm25_layer3_document_id ON bm25_layer3(document_id);
CREATE INDEX IF NOT EXISTS idx_bm25_layer3_section_id ON bm25_layer3(section_id);
CREATE INDEX IF NOT EXISTS idx_bm25_layer3_chunk_id ON bm25_layer3(chunk_id);

-- Full-text search indexes for PostgreSQL's built-in search
-- (Bonus: enables hybrid BM25 + PostgreSQL FTS if needed)
CREATE INDEX IF NOT EXISTS idx_bm25_layer1_corpus_fts ON bm25_layer1 USING GIN(to_tsvector('simple', corpus));
CREATE INDEX IF NOT EXISTS idx_bm25_layer2_corpus_fts ON bm25_layer2 USING GIN(to_tsvector('simple', corpus));
CREATE INDEX IF NOT EXISTS idx_bm25_layer3_corpus_fts ON bm25_layer3 USING GIN(to_tsvector('simple', corpus));

-- Comments
COMMENT ON TABLE bm25_config IS 'BM25 multilingual configuration';
COMMENT ON TABLE bm25_layer1 IS 'BM25 document-level sparse index';
COMMENT ON TABLE bm25_layer2 IS 'BM25 section-level sparse index';
COMMENT ON TABLE bm25_layer3 IS 'BM25 chunk-level sparse index (PRIMARY)';
