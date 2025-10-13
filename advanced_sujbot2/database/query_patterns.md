# PostgreSQL Query Patterns for SUJBOT2

This document provides optimized query patterns for common operations in the SUJBOT2 legal compliance system.

## Table of Contents

1. [Vector Similarity Search](#vector-similarity-search)
2. [Hybrid Search (Vector + Full-Text)](#hybrid-search)
3. [Legal Reference Lookups](#legal-reference-lookups)
4. [Knowledge Graph Queries](#knowledge-graph-queries)
5. [Compliance Checking](#compliance-checking)
6. [Metadata Filtering](#metadata-filtering)
7. [Aggregations and Statistics](#aggregations-and-statistics)
8. [Performance Optimization](#performance-optimization)

## Vector Similarity Search

### Basic Vector Search

```sql
-- Find top 20 most similar chunks to query embedding
SELECT
    chunk_id,
    content,
    legal_reference,
    hierarchy_path,
    document_id,
    1 - (embedding <=> $1::vector) AS similarity_score
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

### Vector Search with Document Filter

```sql
-- Search within specific documents only
SELECT
    chunk_id,
    content,
    legal_reference,
    1 - (embedding <=> $1::vector) AS similarity_score
FROM chunks
WHERE document_id = ANY($2::varchar[])  -- $2 = ['doc1', 'doc2']
    AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

### Vector Search with Structural Filter

```sql
-- Search only in paragraphs (not subsections)
SELECT
    chunk_id,
    content,
    legal_reference,
    paragraph_number,
    1 - (embedding <=> $1::vector) AS similarity_score
FROM chunks
WHERE structural_level = 'paragraph'
    AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

### Vector Search with Content Classification

```sql
-- Find obligations similar to query
SELECT
    chunk_id,
    content,
    legal_reference,
    content_type,
    1 - (embedding <=> $1::vector) AS similarity_score
FROM chunks
WHERE contains_obligation = true
    AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

## Hybrid Search

### Hybrid Search (Vector + Full-Text)

```sql
-- Use built-in hybrid_search function
SELECT * FROM hybrid_search(
    query_text := 'odpovědnost za vady',
    query_embedding := $1::vector,
    doc_ids := ARRAY['doc1', 'doc2']::varchar[],
    limit_results := 20,
    vector_weight := 0.5,
    fulltext_weight := 0.3
);
```

### Manual Reciprocal Rank Fusion

```sql
-- Combine vector and full-text search with custom fusion
WITH vector_results AS (
    SELECT
        chunk_id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS vec_rank,
        1 - (embedding <=> $1::vector) AS vec_score
    FROM chunks
    WHERE embedding IS NOT NULL
    LIMIT 100
),
fulltext_results AS (
    SELECT
        chunk_id,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(content_tsv, query) DESC) AS ft_rank,
        ts_rank_cd(content_tsv, query) AS ft_score
    FROM chunks, to_tsquery('czech', $2) AS query
    WHERE content_tsv @@ query
    LIMIT 100
)
SELECT
    c.chunk_id,
    c.content,
    c.legal_reference,
    c.hierarchy_path,
    -- Reciprocal rank fusion
    (1.0 / (60 + COALESCE(v.vec_rank, 1000)) +
     1.0 / (60 + COALESCE(f.ft_rank, 1000))) AS fused_score,
    v.vec_score,
    f.ft_score
FROM chunks c
LEFT JOIN vector_results v ON c.chunk_id = v.chunk_id
LEFT JOIN fulltext_results f ON c.chunk_id = f.chunk_id
WHERE v.chunk_id IS NOT NULL OR f.chunk_id IS NOT NULL
ORDER BY fused_score DESC
LIMIT 20;
```

### Triple Hybrid (Vector + Full-Text + Structural)

```sql
-- Add structural matching to hybrid search
WITH vector_results AS (
    SELECT
        chunk_id,
        1 - (embedding <=> $1::vector) AS vec_score
    FROM chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT 50
),
fulltext_results AS (
    SELECT
        chunk_id,
        ts_rank_cd(content_tsv, query) AS ft_score
    FROM chunks, to_tsquery('czech', $2) AS query
    WHERE content_tsv @@ query
    ORDER BY ft_score DESC
    LIMIT 50
),
structural_matches AS (
    SELECT
        chunk_id,
        CASE
            WHEN hierarchy_path ILIKE '%' || $3 || '%' THEN 1.0  -- Exact hierarchy match
            WHEN legal_reference = $3 THEN 1.0  -- Exact reference match
            WHEN legal_reference LIKE $3 || '%' THEN 0.8  -- Prefix match
            ELSE 0.0
        END AS struct_score
    FROM chunks
    WHERE hierarchy_path ILIKE '%' || $3 || '%'
        OR legal_reference ILIKE '%' || $3 || '%'
)
SELECT
    c.chunk_id,
    c.content,
    c.legal_reference,
    c.hierarchy_path,
    (0.5 * COALESCE(v.vec_score, 0) +
     0.3 * COALESCE(f.ft_score, 0) +
     0.2 * COALESCE(s.struct_score, 0)) AS combined_score
FROM chunks c
LEFT JOIN vector_results v ON c.chunk_id = v.chunk_id
LEFT JOIN fulltext_results f ON c.chunk_id = f.chunk_id
LEFT JOIN structural_matches s ON c.chunk_id = s.chunk_id
WHERE v.chunk_id IS NOT NULL
    OR f.chunk_id IS NOT NULL
    OR s.chunk_id IS NOT NULL
ORDER BY combined_score DESC
LIMIT 20;
```

## Legal Reference Lookups

### Exact Reference Lookup

```sql
-- Find chunks by exact legal reference
SELECT chunk_id, content, hierarchy_path
FROM chunks
WHERE legal_reference = '§89'
    AND document_id = $1;
```

### Fuzzy Reference Matching

```sql
-- Use trigram similarity for fuzzy matching
SELECT
    chunk_id,
    legal_reference,
    content,
    similarity(legal_reference, $1) AS sim_score
FROM chunks
WHERE legal_reference % $1  -- Trigram similarity operator
ORDER BY sim_score DESC
LIMIT 10;
```

### Reference Prefix Search

```sql
-- Find all subsections of a paragraph
SELECT chunk_id, legal_reference, content
FROM chunks
WHERE legal_reference LIKE '§89%'
    AND document_id = $1
ORDER BY legal_reference;
```

### Cross-Document Reference Resolution

```sql
-- Find target chunk in law for reference in contract
SELECT
    c.chunk_id,
    c.content,
    c.legal_reference,
    c.document_id,
    d.title AS document_title
FROM references r
JOIN chunks c ON r.chunk_id = c.chunk_id
JOIN documents d ON c.document_id = d.document_id
WHERE r.reference_normalized = $1  -- e.g., '§89.2'
    AND d.document_type = 'law_code'
LIMIT 1;
```

## Knowledge Graph Queries

### Get Direct Neighbors

```sql
-- Get all chunks directly referenced by a chunk
SELECT
    c.chunk_id,
    c.legal_reference,
    c.content,
    cr.edge_type,
    cr.weight
FROM cross_references cr
JOIN chunks c ON cr.target_chunk_id = c.chunk_id
WHERE cr.source_chunk_id = $1
ORDER BY cr.weight DESC;
```

### Multi-Hop Graph Traversal

```sql
-- Use built-in recursive function
SELECT * FROM get_related_chunks(
    start_chunk_id := 'chunk_123',
    edge_types := ARRAY['references', 'related_to']::varchar[],
    max_depth := 2
);
```

### Manual Recursive Traversal

```sql
-- Find all chunks reachable within 3 hops
WITH RECURSIVE graph_walk AS (
    -- Base case: starting chunk
    SELECT
        $1::varchar AS chunk_id,
        0 AS depth,
        ARRAY[$1::varchar] AS path

    UNION ALL

    -- Recursive case: follow edges
    SELECT
        cr.target_chunk_id AS chunk_id,
        gw.depth + 1 AS depth,
        gw.path || cr.target_chunk_id AS path
    FROM graph_walk gw
    JOIN cross_references cr ON gw.chunk_id = cr.source_chunk_id
    WHERE gw.depth < 3
        AND NOT (cr.target_chunk_id = ANY(gw.path))  -- Prevent cycles
)
SELECT
    gw.chunk_id,
    gw.depth,
    gw.path,
    c.legal_reference,
    c.content
FROM graph_walk gw
JOIN chunks c ON gw.chunk_id = c.chunk_id
WHERE gw.depth > 0  -- Exclude starting chunk
ORDER BY gw.depth, gw.chunk_id;
```

### Find Shortest Path

```sql
-- Find shortest path between two chunks (limited to 5 hops)
WITH RECURSIVE path_search AS (
    SELECT
        $1::varchar AS current_chunk,
        $2::varchar AS target_chunk,
        ARRAY[$1::varchar] AS path,
        0 AS depth

    UNION ALL

    SELECT
        cr.target_chunk_id AS current_chunk,
        ps.target_chunk,
        ps.path || cr.target_chunk_id AS path,
        ps.depth + 1 AS depth
    FROM path_search ps
    JOIN cross_references cr ON ps.current_chunk = cr.source_chunk_id
    WHERE ps.current_chunk != ps.target_chunk
        AND ps.depth < 5
        AND NOT (cr.target_chunk_id = ANY(ps.path))
)
SELECT path, depth
FROM path_search
WHERE current_chunk = target_chunk
ORDER BY depth
LIMIT 1;
```

### Find Conflicts

```sql
-- Find all conflicts for a contract clause
SELECT
    c_target.chunk_id,
    c_target.legal_reference,
    c_target.content,
    cr.severity,
    cr.risk_score,
    cr.explanation
FROM cross_references cr
JOIN chunks c_target ON cr.target_chunk_id = c_target.chunk_id
WHERE cr.source_chunk_id = $1
    AND cr.edge_type = 'conflicts_with'
ORDER BY cr.risk_score DESC;
```

## Compliance Checking

### Find Missing Requirements

```sql
-- Find law requirements not matched by contract
SELECT
    l.chunk_id AS law_chunk_id,
    l.legal_reference,
    l.content AS requirement
FROM chunks l
WHERE l.document_id = $1  -- law document
    AND l.contains_obligation = true
    AND NOT EXISTS (
        SELECT 1
        FROM cross_references cr
        WHERE cr.target_chunk_id = l.chunk_id
            AND cr.edge_type = 'complies_with'
            AND cr.source_chunk_id IN (
                SELECT chunk_id
                FROM chunks
                WHERE document_id = $2  -- contract document
            )
    )
ORDER BY l.legal_reference;
```

### Find Compliant Mappings

```sql
-- Get all contract clauses that comply with law requirements
SELECT
    c_contract.chunk_id AS contract_chunk_id,
    c_contract.legal_reference AS contract_ref,
    c_law.chunk_id AS law_chunk_id,
    c_law.legal_reference AS law_ref,
    cr.confidence,
    cr.explanation
FROM cross_references cr
JOIN chunks c_contract ON cr.source_chunk_id = c_contract.chunk_id
JOIN chunks c_law ON cr.target_chunk_id = c_law.chunk_id
WHERE cr.edge_type = 'complies_with'
    AND c_contract.document_id = $1  -- contract
    AND c_law.document_id = $2  -- law
ORDER BY c_contract.chunk_index;
```

### Calculate Compliance Score

```sql
-- Calculate overall compliance percentage
WITH law_requirements AS (
    SELECT COUNT(*) AS total_requirements
    FROM chunks
    WHERE document_id = $1  -- law document
        AND contains_obligation = true
),
compliant_clauses AS (
    SELECT COUNT(DISTINCT cr.target_chunk_id) AS compliant_count
    FROM cross_references cr
    JOIN chunks c ON cr.source_chunk_id = c.chunk_id
    WHERE cr.edge_type = 'complies_with'
        AND cr.target_chunk_id IN (
            SELECT chunk_id
            FROM chunks
            WHERE document_id = $1
                AND contains_obligation = true
        )
        AND c.document_id = $2  -- contract document
)
SELECT
    lr.total_requirements,
    cc.compliant_count,
    (cc.compliant_count::float / NULLIF(lr.total_requirements, 0) * 100)::numeric(5,2) AS compliance_percentage
FROM law_requirements lr, compliant_clauses cc;
```

## Metadata Filtering

### JSONB Contains Query

```sql
-- Find chunks with specific metadata
SELECT chunk_id, content, metadata
FROM chunks
WHERE metadata @> '{"is_aggregated": true}'::jsonb
    AND document_id = $1;
```

### JSONB Path Query

```sql
-- Find chunks where metadata contains specific party
SELECT chunk_id, content, metadata->'parties_mentioned' AS parties
FROM chunks
WHERE metadata->'parties_mentioned' ? 'dodavatel'
    AND document_type = 'contract';
```

### Array Contains Query

```sql
-- Find chunks that reference specific paragraph
SELECT chunk_id, content, references_to
FROM chunks
WHERE '§89' = ANY(references_to)
    AND document_id = $1;
```

### Complex Metadata Filter

```sql
-- Combine multiple metadata conditions
SELECT chunk_id, content, legal_reference
FROM chunks
WHERE document_id = $1
    AND structural_level = 'paragraph'
    AND contains_obligation = true
    AND token_count BETWEEN 100 AND 500
    AND metadata @> '{"content_type": "obligation"}'::jsonb
ORDER BY paragraph_number;
```

## Aggregations and Statistics

### Document Statistics

```sql
-- Get comprehensive document statistics
SELECT
    d.document_id,
    d.filename,
    d.document_type,
    COUNT(c.chunk_id) AS total_chunks,
    AVG(c.token_count)::int AS avg_chunk_tokens,
    SUM(CASE WHEN c.contains_obligation THEN 1 ELSE 0 END) AS obligations,
    SUM(CASE WHEN c.contains_prohibition THEN 1 ELSE 0 END) AS prohibitions,
    SUM(CASE WHEN c.contains_definition THEN 1 ELSE 0 END) AS definitions,
    COUNT(DISTINCT c.legal_reference) AS unique_references
FROM documents d
LEFT JOIN chunks c ON d.document_id = c.document_id
WHERE d.document_id = $1
GROUP BY d.document_id, d.filename, d.document_type;
```

### Chunk Distribution by Type

```sql
-- Analyze chunk distribution by content type
SELECT
    content_type,
    COUNT(*) AS chunk_count,
    AVG(token_count)::int AS avg_tokens,
    MIN(token_count) AS min_tokens,
    MAX(token_count) AS max_tokens
FROM chunks
WHERE document_id = $1
GROUP BY content_type
ORDER BY chunk_count DESC;
```

### Knowledge Graph Statistics

```sql
-- Analyze knowledge graph connectivity
SELECT
    edge_type,
    COUNT(*) AS edge_count,
    AVG(weight)::numeric(4,3) AS avg_weight,
    COUNT(DISTINCT source_chunk_id) AS unique_sources,
    COUNT(DISTINCT target_chunk_id) AS unique_targets
FROM cross_references
GROUP BY edge_type
ORDER BY edge_count DESC;
```

### Most Referenced Chunks

```sql
-- Find most frequently referenced chunks (hub nodes)
SELECT
    c.chunk_id,
    c.legal_reference,
    c.hierarchy_path,
    COUNT(cr.edge_id) AS incoming_references
FROM chunks c
LEFT JOIN cross_references cr ON c.chunk_id = cr.target_chunk_id
WHERE c.document_id = $1
GROUP BY c.chunk_id, c.legal_reference, c.hierarchy_path
ORDER BY incoming_references DESC
LIMIT 20;
```

## Performance Optimization

### Query Plan Analysis

```sql
-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT chunk_id, content
FROM chunks
WHERE embedding <=> $1::vector < 0.5
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

### Index Usage Check

```sql
-- Check if indexes are being used
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename IN ('chunks', 'documents', 'references', 'cross_references')
ORDER BY idx_scan DESC;
```

### Optimize Vector Search

```sql
-- Set IVFFlat probes for balance between speed and accuracy
SET ivfflat.probes = 10;  -- Default, good balance
-- SET ivfflat.probes = 20;  -- Better accuracy, slower
-- SET ivfflat.probes = 5;   -- Faster, lower accuracy

-- Then run vector search
SELECT chunk_id, 1 - (embedding <=> $1::vector) AS score
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

### Batch Insert Optimization

```sql
-- Use COPY for bulk inserts (fastest)
COPY chunks (chunk_id, chunk_index, document_id, content, embedding, ...)
FROM '/path/to/data.csv'
WITH (FORMAT csv, HEADER true);

-- Or use multi-row INSERT with prepared statement
INSERT INTO chunks (chunk_id, chunk_index, document_id, content, ...)
VALUES
    ($1, $2, $3, $4, ...),
    ($5, $6, $7, $8, ...),
    ... -- batch of 1000 rows
```

### Parallel Query Execution

```sql
-- Enable parallel queries for large datasets
SET max_parallel_workers_per_gather = 4;

-- Run expensive query
SELECT
    c.document_id,
    COUNT(*) AS chunk_count,
    AVG(1 - (c.embedding <=> $1::vector)) AS avg_similarity
FROM chunks c
WHERE c.embedding IS NOT NULL
GROUP BY c.document_id;
```

### Materialized View for Common Queries

```sql
-- Create materialized view for document summaries
CREATE MATERIALIZED VIEW mv_document_summary AS
SELECT
    d.document_id,
    d.filename,
    d.document_type,
    COUNT(c.chunk_id) AS chunk_count,
    AVG(c.token_count)::int AS avg_tokens,
    SUM(CASE WHEN c.contains_obligation THEN 1 ELSE 0 END) AS obligation_count
FROM documents d
LEFT JOIN chunks c ON d.document_id = c.document_id
GROUP BY d.document_id, d.filename, d.document_type;

-- Create index on materialized view
CREATE INDEX idx_mv_doc_summary_type ON mv_document_summary(document_type);

-- Refresh periodically
REFRESH MATERIALIZED VIEW mv_document_summary;
```

### Query Result Caching

```sql
-- Use query_cache table for expensive queries
-- Check cache first
SELECT results
FROM query_cache
WHERE query_hash = $1
    AND expires_at > NOW();

-- If cache miss, run query and store result
INSERT INTO query_cache (
    query_text, query_hash, results, execution_time_ms
) VALUES (
    $1, $2, $3::jsonb, $4
)
ON CONFLICT (query_hash) DO UPDATE
SET
    results = EXCLUDED.results,
    execution_time_ms = EXCLUDED.execution_time_ms,
    last_accessed_at = NOW(),
    access_count = query_cache.access_count + 1;
```

## Common Query Patterns by Use Case

### Use Case 1: Chat Query

```sql
-- User asks: "What are the obligations regarding warranty?"
-- 1. Hybrid search for relevant chunks
-- 2. Filter by obligations
-- 3. Return top 5 results

WITH vector_matches AS (
    SELECT chunk_id, 1 - (embedding <=> $1::vector) AS vec_score
    FROM chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT 50
),
text_matches AS (
    SELECT chunk_id, ts_rank_cd(content_tsv, query) AS text_score
    FROM chunks, to_tsquery('czech', 'odpovědnost & záruka') AS query
    WHERE content_tsv @@ query
    LIMIT 50
)
SELECT
    c.chunk_id,
    c.content,
    c.legal_reference,
    c.hierarchy_path,
    (0.6 * COALESCE(v.vec_score, 0) + 0.4 * COALESCE(t.text_score, 0)) AS score
FROM chunks c
LEFT JOIN vector_matches v ON c.chunk_id = v.chunk_id
LEFT JOIN text_matches t ON c.chunk_id = t.chunk_id
WHERE c.contains_obligation = true
    AND (v.chunk_id IS NOT NULL OR t.chunk_id IS NOT NULL)
ORDER BY score DESC
LIMIT 5;
```

### Use Case 2: Compliance Check

```sql
-- Check if contract complies with law
-- 1. Find all law requirements
-- 2. Match with contract clauses
-- 3. Identify gaps and conflicts

WITH law_requirements AS (
    SELECT chunk_id, legal_reference, content
    FROM chunks
    WHERE document_id = $1  -- law
        AND contains_obligation = true
),
matched_requirements AS (
    SELECT
        lr.chunk_id AS law_chunk_id,
        lr.legal_reference,
        cr.edge_type,
        cr.source_chunk_id AS contract_chunk_id
    FROM law_requirements lr
    LEFT JOIN cross_references cr ON lr.chunk_id = cr.target_chunk_id
        AND cr.edge_type IN ('complies_with', 'conflicts_with')
        AND cr.source_chunk_id IN (
            SELECT chunk_id FROM chunks WHERE document_id = $2  -- contract
        )
)
SELECT
    legal_reference,
    CASE
        WHEN edge_type = 'complies_with' THEN 'COMPLIANT'
        WHEN edge_type = 'conflicts_with' THEN 'CONFLICT'
        ELSE 'MISSING'
    END AS status,
    contract_chunk_id
FROM matched_requirements
ORDER BY
    CASE
        WHEN edge_type = 'conflicts_with' THEN 1
        WHEN edge_type IS NULL THEN 2
        ELSE 3
    END,
    legal_reference;
```

### Use Case 3: Find Related Provisions

```sql
-- User clicks on §89, show related provisions
-- 1. Direct references
-- 2. Semantic similarity
-- 3. Knowledge graph neighbors

WITH direct_refs AS (
    SELECT
        c.chunk_id,
        c.legal_reference,
        c.content,
        'direct_reference' AS relation_type,
        1.0 AS score
    FROM cross_references cr
    JOIN chunks c ON cr.target_chunk_id = c.chunk_id
    WHERE cr.source_chunk_id = $1
        AND cr.edge_type = 'references'
    LIMIT 10
),
semantic_similar AS (
    SELECT
        c.chunk_id,
        c.legal_reference,
        c.content,
        'semantic_similar' AS relation_type,
        1 - (c.embedding <=> (SELECT embedding FROM chunks WHERE chunk_id = $1)) AS score
    FROM chunks c
    WHERE c.chunk_id != $1
        AND c.embedding IS NOT NULL
        AND c.document_id = (SELECT document_id FROM chunks WHERE chunk_id = $1)
    ORDER BY c.embedding <=> (SELECT embedding FROM chunks WHERE chunk_id = $1)
    LIMIT 10
),
graph_neighbors AS (
    SELECT
        c.chunk_id,
        c.legal_reference,
        c.content,
        'graph_neighbor' AS relation_type,
        cr.weight AS score
    FROM cross_references cr
    JOIN chunks c ON cr.target_chunk_id = c.chunk_id
    WHERE cr.source_chunk_id = $1
        AND cr.edge_type = 'related_to'
    ORDER BY cr.weight DESC
    LIMIT 10
)
SELECT * FROM direct_refs
UNION ALL
SELECT * FROM semantic_similar
UNION ALL
SELECT * FROM graph_neighbors
ORDER BY score DESC;
```

## Best Practices

1. **Always use parameterized queries** to prevent SQL injection
2. **Use EXPLAIN ANALYZE** to understand query performance
3. **Create indexes** for frequently filtered columns
4. **Use connection pooling** (PgBouncer) for high concurrency
5. **Enable query caching** for expensive repeated queries
6. **Monitor slow queries** with pg_stat_statements
7. **Use materialized views** for complex aggregations
8. **Batch inserts** for better write performance
9. **Use prepared statements** for repeated queries
10. **Set appropriate IVFFlat probes** based on accuracy requirements
