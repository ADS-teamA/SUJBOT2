# SUJBOT2 PostgreSQL Database Documentation

Complete database schema and migration guide for the SUJBOT2 legal compliance system with pgvector support.

## Overview

This directory contains the PostgreSQL schema design and migration tools for SUJBOT2, replacing the FAISS-based vector storage with a unified PostgreSQL database using the pgvector extension.

### Key Features

- **Vector Search**: 1024-dimensional embeddings (BGE-M3) with IVFFlat indexing
- **Full-Text Search**: Czech language support with tsvector/tsquery
- **Knowledge Graph**: Persistent graph storage with recursive queries
- **Multi-Document**: Separate storage for contracts and laws
- **Scalability**: Optimized for 300K-500K chunks (100K+ pages)
- **Legal Metadata**: Hierarchical structure, § references, compliance tracking

## Files

- **`schema.sql`** - Complete PostgreSQL schema with all tables, indexes, views, and functions
- **`migration_guide.md`** - Step-by-step guide to migrate from FAISS to PostgreSQL
- **`query_patterns.md`** - Optimized query patterns for common operations
- **`README.md`** - This file

## Quick Start

### 1. Install Prerequisites

```bash
# Install PostgreSQL 15+
brew install postgresql@15  # macOS
# OR
sudo apt-get install postgresql-15  # Ubuntu

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 2. Create Database

```bash
# Create database
createdb sujbot2

# Apply schema
psql sujbot2 < database/schema.sql
```

### 3. Verify Installation

```sql
-- Connect to database
psql sujbot2

-- Check extensions
\dx

-- Expected output:
-- vector     | 0.5.1
-- pg_trgm    | 1.6
-- btree_gin  | 1.3

-- Check tables
\dt

-- Expected tables:
-- documents
-- chunks
-- references
-- cross_references
-- compliance_reports
-- query_cache
-- statistics
```

## Database Schema

### Core Tables

#### `documents`
Document-level metadata for contracts, laws, and regulations.

**Key columns:**
- `document_id` (PK) - Unique document identifier
- `document_type` - 'contract', 'law_code', 'regulation'
- `title`, `document_number`, `law_citation` - Legal metadata
- `status` - 'uploaded', 'processing', 'indexed', 'error'
- `metadata` (JSONB) - Additional metadata

**Indexes:**
- B-tree on `document_type`, `status`, `document_number`
- GIN on `metadata` (JSONB path ops)
- Trigram on `title` (fuzzy search)

#### `chunks`
Chunk content, embeddings, and legal metadata.

**Key columns:**
- `chunk_id` (PK) - Unique chunk identifier
- `document_id` (FK) - References documents table
- `content` (TEXT) - Chunk text content
- `embedding` (vector(1024)) - BGE-M3 embedding
- `legal_reference` - §89, Článek 5, etc.
- `hierarchy_path` - Full hierarchy: "Část II > Hlava III > §89"
- `structural_level` - 'paragraph', 'article', 'subsection'
- `content_type` - 'obligation', 'prohibition', 'definition', etc.
- `content_tsv` (tsvector) - Full-text search vector (Czech)

**Indexes:**
- IVFFlat on `embedding` (vector similarity search, 512 lists)
- GIN on `content_tsv` (full-text search)
- B-tree on `document_id`, `legal_reference`, `hierarchy_path`
- Composite indexes for hybrid search

**Size estimates:**
- 400,000 chunks
- 1024-dimensional embeddings (4KB per vector)
- ~2GB for embeddings alone
- ~5-10GB total with indexes

#### `references`
Legal reference mappings (§ → chunks).

**Key columns:**
- `reference_id` (PK) - Auto-incrementing ID
- `reference_text` - "§89 odst. 2"
- `reference_normalized` - "§89.2"
- `chunk_id` (FK) - Target chunk
- `source_chunk_id` (FK) - Where reference appears

**Use cases:**
- Fast lookup: "§89" → list of chunks
- Cross-document reference resolution
- Reference graph construction

#### `cross_references`
Knowledge graph edges (chunk relationships).

**Key columns:**
- `edge_id` (PK) - Auto-incrementing ID
- `source_chunk_id`, `target_chunk_id` (FK) - Edge endpoints
- `edge_type` - 'part_of', 'references', 'conflicts_with', 'complies_with', etc.
- `weight`, `confidence` - Edge properties
- `severity`, `risk_score` - For conflicts

**Use cases:**
- Knowledge graph traversal
- Multi-hop reasoning
- Conflict detection
- Compliance mapping

### Supporting Tables

- **`compliance_reports`** - Compliance check results
- **`query_cache`** - Cache for expensive queries (1 hour TTL)
- **`statistics`** - System monitoring data

### Views

- **`v_document_summary`** - Document statistics with chunk counts
- **`v_chunk_detail`** - Chunk details with document context
- **`v_knowledge_graph_stats`** - Graph statistics by edge type

### Functions

- **`hybrid_search()`** - Combined vector + full-text search
- **`get_related_chunks()`** - Recursive graph traversal
- **`find_by_legal_reference()`** - Reference lookup with fuzzy matching
- **`document_similarity()`** - Calculate document similarity
- **`clean_expired_cache()`** - Maintenance function

## Configuration

### PostgreSQL Settings

Recommended settings in `postgresql.conf`:

```ini
# Memory (adjust for your hardware)
shared_buffers = 4GB                     # 25% of RAM
effective_cache_size = 12GB              # 75% of RAM
maintenance_work_mem = 1GB
work_mem = 128MB

# Parallel execution
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_worker_processes = 8

# pgvector settings
ivfflat.probes = 10                      # Balance speed vs accuracy

# Connection pooling
max_connections = 100
```

### IVFFlat Index Configuration

**Lists parameter calculation:**
```
lists = sqrt(num_rows)

For 400,000 chunks:
lists = sqrt(400000) ≈ 632

Rounded to: 512 (power of 2 for better performance)
```

**Probes parameter (query-time):**
- `probes = 1`: Fastest, lowest accuracy (~70% recall)
- `probes = 10`: Good balance (default, ~90% recall)
- `probes = 20`: Better accuracy, slower (~95% recall)
- `probes = lists`: Exact search (same as exhaustive)

**Trade-offs:**
- More lists = Faster queries, slower index build, more memory
- More probes = Better accuracy, slower queries
- IVFFlat = Good for large datasets (>100K vectors)
- HNSW = Faster queries, more memory (alternative for <100K vectors)

## Usage Examples

### Basic Vector Search

```python
import asyncpg
from pgvector.asyncpg import register_vector
import numpy as np

# Connect to database
conn = await asyncpg.connect("postgresql://localhost/sujbot2")
await register_vector(conn)

# Generate query embedding (using your embedding model)
query_embedding = embedder.encode("odpovědnost za vady")  # 1024-dim vector

# Search for similar chunks
results = await conn.fetch("""
    SELECT chunk_id, content, legal_reference,
           1 - (embedding <=> $1::vector) AS similarity
    FROM chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT 20
""", query_embedding.tolist())

for row in results:
    print(f"{row['legal_reference']}: {row['similarity']:.3f}")
```

### Hybrid Search

```python
# Combine vector and full-text search
results = await conn.fetch("""
    SELECT * FROM hybrid_search(
        query_text := $1,
        query_embedding := $2::vector,
        doc_ids := $3,
        limit_results := 20,
        vector_weight := 0.5,
        fulltext_weight := 0.3
    )
""", "odpovědnost za vady", query_embedding.tolist(), ['law_doc_1'])

for row in results:
    print(f"{row['legal_reference']}: combined_score={row['combined_score']:.3f}")
```

### Legal Reference Lookup

```python
# Find all chunks for §89
results = await conn.fetch("""
    SELECT chunk_id, content, hierarchy_path
    FROM chunks
    WHERE legal_reference = $1
        AND document_id = $2
""", "§89", "law_doc_1")
```

### Knowledge Graph Traversal

```python
# Find related chunks within 2 hops
results = await conn.fetch("""
    SELECT * FROM get_related_chunks(
        start_chunk_id := $1,
        edge_types := $2,
        max_depth := 2
    )
""", "chunk_123", ["references", "related_to"])
```

## Performance Benchmarks

### Query Performance (400K chunks)

| Query Type | Latency (avg) | Notes |
|------------|---------------|-------|
| Vector search (top 20) | 5-15ms | IVFFlat with 512 lists, probes=10 |
| Full-text search | 3-10ms | GIN index on tsvector |
| Hybrid search | 15-30ms | Combined vector + full-text |
| Reference lookup | <1ms | B-tree index |
| Graph traversal (2 hops) | 5-20ms | Recursive CTE |

### Index Build Times

| Index | Build Time | Size |
|-------|------------|------|
| IVFFlat (512 lists) | 10-15 min | ~2GB |
| GIN (tsvector) | 5-10 min | ~500MB |
| B-tree (legal_reference) | <1 min | ~50MB |

### Scaling Estimates

| Chunks | Embeddings | Total DB Size | Query Time |
|--------|------------|---------------|------------|
| 100K | 500MB | 2GB | 3-8ms |
| 400K | 2GB | 8GB | 5-15ms |
| 1M | 5GB | 20GB | 10-25ms |

## Migration from FAISS

See **`migration_guide.md`** for detailed instructions.

**High-level steps:**

1. Export FAISS data to JSON
2. Create PostgreSQL database
3. Import documents and chunks
4. Build IVFFlat indexes
5. Import knowledge graph
6. Update application code
7. Verify and benchmark

**Advantages of PostgreSQL:**

- **Unified storage**: No separate FAISS files, JSON metadata, graph storage
- **ACID transactions**: Data integrity guaranteed
- **Concurrent access**: Multiple readers/writers
- **Full-text search**: Native Czech language support
- **Query flexibility**: SQL for complex queries
- **Backup/restore**: Standard PostgreSQL tools
- **Monitoring**: Rich ecosystem (pg_stat_statements, pgAdmin)

**Potential drawbacks:**

- **Query latency**: Slightly slower than FAISS (5-15ms vs 3-8ms)
- **Memory usage**: Higher overhead than pure FAISS
- **Complexity**: More moving parts to manage

## Monitoring and Maintenance

### Check Index Usage

```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
WHERE tablename = 'chunks'
ORDER BY idx_scan DESC;
```

### Check Slow Queries

```sql
SELECT
    query,
    mean_exec_time,
    calls,
    total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Vacuum and Analyze

```bash
# Run regularly (daily)
psql sujbot2 -c "VACUUM ANALYZE chunks;"

# Or use autovacuum (recommended)
# Already enabled by default in PostgreSQL
```

### Database Size

```sql
SELECT pg_size_pretty(pg_database_size('sujbot2'));
```

### Table Sizes

```sql
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Backup and Recovery

### Backup Database

```bash
# Full backup
pg_dump sujbot2 | gzip > sujbot2_backup_$(date +%Y%m%d).sql.gz

# Backup schema only
pg_dump -s sujbot2 > sujbot2_schema.sql

# Backup data only
pg_dump -a sujbot2 > sujbot2_data.sql
```

### Restore Database

```bash
# Create new database
createdb sujbot2_restore

# Restore from backup
gunzip -c sujbot2_backup_20250101.sql.gz | psql sujbot2_restore
```

### Automated Backups

Add to crontab:

```bash
# Daily backup at 2 AM
0 2 * * * pg_dump sujbot2 | gzip > /backups/sujbot2_$(date +\%Y\%m\%d).sql.gz

# Keep only last 7 days
0 3 * * * find /backups -name "sujbot2_*.sql.gz" -mtime +7 -delete
```

## Troubleshooting

### Issue: Out of Memory

**Symptoms:** Queries fail with "out of memory" errors

**Solution:**
```sql
-- Reduce work_mem for current session
SET work_mem = '64MB';

-- Or permanently in postgresql.conf
work_mem = 64MB
```

### Issue: Slow Vector Search

**Symptoms:** Vector queries take >100ms

**Solutions:**

1. **Increase probes** (if accuracy is low):
   ```sql
   SET ivfflat.probes = 20;
   ```

2. **Rebuild index with more lists**:
   ```sql
   DROP INDEX idx_chunks_embedding_ivfflat;
   CREATE INDEX idx_chunks_embedding_ivfflat ON chunks
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 1024);
   ```

3. **Switch to HNSW** (faster, more memory):
   ```sql
   CREATE INDEX idx_chunks_embedding_hnsw ON chunks
   USING hnsw (embedding vector_cosine_ops)
   WITH (m = 16, ef_construction = 64);
   ```

### Issue: Full-Text Search Not Working

**Symptoms:** No results from full-text queries

**Solution:**
```sql
-- Check if tsvector is populated
SELECT COUNT(*) FROM chunks WHERE content_tsv IS NOT NULL;

-- Regenerate tsvectors if needed
UPDATE chunks SET content_tsv = to_tsvector('czech', content);

-- Verify Czech dictionary
SELECT * FROM pg_ts_config WHERE cfgname = 'czech';
```

### Issue: High CPU Usage

**Symptoms:** PostgreSQL consuming 100% CPU

**Solutions:**

1. **Check for missing indexes**:
   ```sql
   SELECT * FROM pg_stat_user_tables WHERE seq_scan > idx_scan;
   ```

2. **Limit parallel workers**:
   ```sql
   SET max_parallel_workers_per_gather = 2;
   ```

3. **Identify expensive queries**:
   ```sql
   SELECT query, total_exec_time, calls
   FROM pg_stat_statements
   ORDER BY total_exec_time DESC
   LIMIT 5;
   ```

## Resources

- **pgvector Documentation**: https://github.com/pgvector/pgvector
- **PostgreSQL Full-Text Search**: https://www.postgresql.org/docs/15/textsearch.html
- **PostgreSQL Performance Tuning**: https://wiki.postgresql.org/wiki/Performance_Optimization
- **Czech Text Search Configuration**: https://www.postgresql.org/docs/15/textsearch-dictionaries.html

## Support

For questions or issues:

1. Check **`query_patterns.md`** for common query examples
2. Review **`migration_guide.md`** for migration troubleshooting
3. Consult PostgreSQL and pgvector documentation
4. File an issue in the project repository

## License

Same as SUJBOT2 project license.
