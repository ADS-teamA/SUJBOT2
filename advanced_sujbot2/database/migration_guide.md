# Migration Guide: FAISS to PostgreSQL with pgvector

This guide explains how to migrate the SUJBOT2 system from FAISS-based storage to PostgreSQL with pgvector.

## Overview

**Current Architecture:**
- FAISS indices (one per document) stored on disk
- Metadata in JSON files
- Knowledge graph in NetworkX (in-memory)

**Target Architecture:**
- PostgreSQL with pgvector for vector storage
- Full-text search with Czech language support
- Persistent knowledge graph in database
- Unified data storage

## Prerequisites

### 1. Install PostgreSQL 15+ with Extensions

```bash
# Install PostgreSQL 15 or later
brew install postgresql@15  # macOS
# OR
sudo apt-get install postgresql-15  # Ubuntu

# Start PostgreSQL
brew services start postgresql@15  # macOS
# OR
sudo systemctl start postgresql  # Linux
```

### 2. Install pgvector Extension

```bash
# Clone pgvector repository
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector

# Build and install
make
sudo make install
```

### 3. Create Database

```bash
# Create database
createdb sujbot2

# Connect and enable extensions
psql sujbot2 -f database/schema.sql
```

### 4. Install Python Dependencies

```bash
pip install psycopg2-binary  # or psycopg2
pip install pgvector
```

## Migration Steps

### Step 1: Export Existing FAISS Data

Create a migration script to export current data:

```python
# migration/export_faiss_data.py

import json
from pathlib import Path
from src.indexing import IndexPersistence, MultiDocumentVectorStore
from src.embeddings import LegalEmbedder

async def export_document_data(document_id: str, output_dir: Path):
    """Export document data from FAISS to JSON"""

    persistence = IndexPersistence(index_dir="./indexes")

    # Load document data
    index, metadata, chunks, reference_map = await persistence.load(document_id)

    # Export document metadata
    doc_export = {
        "document_id": document_id,
        "metadata": metadata,
        "chunks": [],
        "references": []
    }

    # Export chunks with embeddings
    for i, chunk in enumerate(chunks):
        # Get embedding from FAISS index
        embedding = index.reconstruct(i).tolist()

        chunk_data = {
            "chunk_id": chunk.chunk_id,
            "chunk_index": i,
            "content": chunk.content,
            "embedding": embedding,
            "document_id": chunk.document_id,
            "document_type": chunk.document_type,
            "hierarchy_path": chunk.hierarchy_path,
            "legal_reference": chunk.legal_reference,
            "structural_level": chunk.structural_level,
            "metadata": chunk.metadata
        }
        doc_export["chunks"].append(chunk_data)

    # Export references
    if reference_map:
        doc_export["references"] = {
            "ref_to_chunks": dict(reference_map.ref_to_chunks),
            "chunk_to_refs": reference_map.chunk_to_refs
        }

    # Save to JSON
    output_file = output_dir / f"{document_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(doc_export, f, ensure_ascii=False, indent=2)

    print(f"Exported {document_id}: {len(chunks)} chunks")

async def export_all_documents():
    """Export all indexed documents"""

    output_dir = Path("./migration/export")
    output_dir.mkdir(parents=True, exist_ok=True)

    persistence = IndexPersistence(index_dir="./indexes")
    document_ids = persistence.list_documents()

    print(f"Found {len(document_ids)} documents to export")

    for doc_id in document_ids:
        try:
            await export_document_data(doc_id, output_dir)
        except Exception as e:
            print(f"Error exporting {doc_id}: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(export_all_documents())
```

Run the export:

```bash
cd advanced_sujbot2
python migration/export_faiss_data.py
```

### Step 2: Import to PostgreSQL

Create an import script:

```python
# migration/import_to_postgres.py

import json
import asyncio
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from pathlib import Path

class PostgresImporter:
    """Import data from JSON to PostgreSQL"""

    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        self.conn.autocommit = False
        register_vector(self.conn)

    def import_document(self, export_file: Path):
        """Import a single document"""

        with open(export_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        cursor = self.conn.cursor()

        try:
            # 1. Insert document metadata
            doc_metadata = data["metadata"]
            cursor.execute("""
                INSERT INTO documents (
                    document_id, filename, document_type, title,
                    document_number, status, total_chunks, metadata,
                    uploaded_at, indexed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (document_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    indexed_at = EXCLUDED.indexed_at
            """, (
                data["document_id"],
                doc_metadata.get("file_path", ""),
                doc_metadata.get("document_type", "regulation"),
                doc_metadata.get("title", ""),
                doc_metadata.get("document_number"),
                "indexed",
                len(data["chunks"]),
                json.dumps(doc_metadata),
                doc_metadata.get("uploaded_at"),
                doc_metadata.get("indexed_at")
            ))

            # 2. Insert chunks
            chunk_values = []
            for chunk in data["chunks"]:
                chunk_values.append((
                    chunk["chunk_id"],
                    chunk["chunk_index"],
                    chunk["document_id"],
                    chunk["content"],
                    chunk["embedding"],  # vector type
                    chunk.get("hierarchy_path"),
                    chunk.get("legal_reference"),
                    chunk.get("structural_level"),
                    chunk["metadata"].get("part"),
                    chunk["metadata"].get("chapter"),
                    chunk["metadata"].get("paragraph"),
                    chunk["metadata"].get("subsection"),
                    chunk["metadata"].get("article"),
                    chunk["metadata"].get("content_type"),
                    chunk["metadata"].get("contains_obligation", False),
                    chunk["metadata"].get("contains_prohibition", False),
                    chunk["metadata"].get("contains_definition", False),
                    chunk["metadata"].get("token_count"),
                    chunk["metadata"].get("references_to", []),
                    json.dumps(chunk["metadata"])
                ))

            execute_values(cursor, """
                INSERT INTO chunks (
                    chunk_id, chunk_index, document_id, content, embedding,
                    hierarchy_path, legal_reference, structural_level,
                    part_number, chapter_number, paragraph_number,
                    subsection_number, article_number,
                    content_type, contains_obligation, contains_prohibition,
                    contains_definition, token_count, references_to, metadata
                ) VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING
            """, chunk_values)

            # 3. Insert references
            if "references" in data and data["references"]:
                ref_to_chunks = data["references"]["ref_to_chunks"]

                ref_values = []
                for ref_text, chunk_ids in ref_to_chunks.items():
                    for chunk_id in chunk_ids:
                        ref_values.append((
                            ref_text,
                            ref_text.lower().replace(" ", ""),  # normalized
                            self._classify_reference_type(ref_text),
                            chunk_id,
                            data["document_id"]
                        ))

                if ref_values:
                    execute_values(cursor, """
                        INSERT INTO references (
                            reference_text, reference_normalized,
                            reference_type, chunk_id, document_id
                        ) VALUES %s
                        ON CONFLICT DO NOTHING
                    """, ref_values)

            self.conn.commit()
            print(f"Imported {data['document_id']}: {len(data['chunks'])} chunks")

        except Exception as e:
            self.conn.rollback()
            print(f"Error importing {data['document_id']}: {e}")
            raise

    def _classify_reference_type(self, ref_text: str) -> str:
        """Classify reference type from text"""
        if ref_text.startswith("§"):
            if "odst." in ref_text:
                return "subsection"
            else:
                return "paragraph"
        elif "článek" in ref_text.lower():
            return "article"
        else:
            return "general"

    def import_all(self, export_dir: Path):
        """Import all exported documents"""

        export_files = list(export_dir.glob("*.json"))
        print(f"Found {len(export_files)} documents to import")

        for export_file in export_files:
            try:
                self.import_document(export_file)
            except Exception as e:
                print(f"Failed to import {export_file}: {e}")

        print("Import complete!")

    def close(self):
        self.conn.close()

async def main():
    # Connection string
    conn_string = "postgresql://localhost/sujbot2"

    # Import data
    importer = PostgresImporter(conn_string)
    try:
        importer.import_all(Path("./migration/export"))
    finally:
        importer.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Run the import:

```bash
python migration/import_to_postgres.py
```

### Step 3: Build Vector Indexes

After importing data, build the IVFFlat indexes:

```sql
-- Connect to database
psql sujbot2

-- Build IVFFlat index (this will take time for large datasets)
-- First, ensure you have enough data
SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;

-- If count > 100K, build IVFFlat index with 512 lists
CREATE INDEX CONCURRENTLY idx_chunks_embedding_ivfflat ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 512);

-- For smaller datasets (< 100K), use HNSW instead
-- CREATE INDEX CONCURRENTLY idx_chunks_embedding_hnsw ON chunks
-- USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- Verify index was created
\di+ idx_chunks_embedding_ivfflat
```

### Step 4: Import Knowledge Graph

Export and import knowledge graph edges using JSON:

```python
# migration/import_knowledge_graph.py

import json
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from src.knowledge_graph import LegalKnowledgeGraph

def export_knowledge_graph_to_json(kg_path: Path, output_file: Path):
    """Export knowledge graph to JSON (safe serialization)"""

    # Load knowledge graph from file
    kg = LegalKnowledgeGraph.load(str(kg_path))

    # Export edges
    edges = []
    for u, v, data in kg.graph.edges(data=True):
        edges.append({
            "source_chunk_id": u,
            "target_chunk_id": v,
            "edge_type": data.get("edge_type"),
            "weight": data.get("weight", 1.0),
            "confidence": data.get("confidence", 1.0),
            "metadata": data.get("metadata", {})
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(edges)} edges")

def import_knowledge_graph(export_file: Path, conn_string: str):
    """Import knowledge graph edges to PostgreSQL"""

    with open(export_file, 'r', encoding='utf-8') as f:
        edges = json.load(f)

    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()

    try:
        edge_values = []
        for edge in edges:
            edge_values.append((
                edge["source_chunk_id"],
                edge["target_chunk_id"],
                edge["edge_type"],
                edge.get("weight", 1.0),
                edge.get("confidence", 1.0),
                json.dumps(edge.get("metadata", {}))
            ))

        execute_values(cursor, """
            INSERT INTO cross_references (
                source_chunk_id, target_chunk_id, edge_type,
                weight, confidence, metadata
            ) VALUES %s
            ON CONFLICT (source_chunk_id, target_chunk_id, edge_type) DO NOTHING
        """, edge_values)

        conn.commit()
        print(f"Imported {len(edges)} edges")

    finally:
        conn.close()

# Run export and import
if __name__ == "__main__":
    # Export
    kg_path = Path("./knowledge_graphs/main.pkl")
    output_file = Path("./migration/knowledge_graph.json")

    if kg_path.exists():
        export_knowledge_graph_to_json(kg_path, output_file)

        # Import
        import_knowledge_graph(output_file, "postgresql://localhost/sujbot2")
```

### Step 5: Update Application Code

Update the RAG pipeline to use PostgreSQL instead of FAISS:

```python
# backend/app/rag/postgres_store.py

import asyncpg
import numpy as np
from typing import List, Dict, Optional
from pgvector.asyncpg import register_vector

class PostgresVectorStore:
    """PostgreSQL vector store with pgvector"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=10,
            max_size=50,
            command_timeout=60
        )

        # Register vector type
        async with self.pool.acquire() as conn:
            await register_vector(conn)

    async def add_document(
        self,
        chunks: List[Dict],
        document_id: str,
        document_type: str,
        metadata: Dict
    ):
        """Add document to store"""

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                await conn.execute("""
                    INSERT INTO documents (
                        document_id, filename, document_type,
                        title, status, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (document_id) DO UPDATE
                    SET status = EXCLUDED.status
                """, document_id, metadata.get("filename", ""),
                    document_type, metadata.get("title", ""),
                    "indexed", metadata)

                # Insert chunks
                for i, chunk in enumerate(chunks):
                    await conn.execute("""
                        INSERT INTO chunks (
                            chunk_id, chunk_index, document_id, content,
                            embedding, hierarchy_path, legal_reference,
                            structural_level, token_count, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, chunk["chunk_id"], i, document_id, chunk["content"],
                        chunk["embedding"], chunk.get("hierarchy_path"),
                        chunk.get("legal_reference"), chunk.get("structural_level"),
                        chunk.get("token_count"), chunk.get("metadata", {}))

    async def search(
        self,
        query_embedding: np.ndarray,
        document_ids: Optional[List[str]] = None,
        top_k: int = 20,
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Vector similarity search"""

        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    chunk_id, content, legal_reference, hierarchy_path,
                    document_id,
                    1 - (embedding <=> $1::vector) AS score
                FROM chunks
                WHERE embedding IS NOT NULL
            """
            params = [query_embedding.tolist()]

            if document_ids:
                query += " AND document_id = ANY($2)"
                params.append(document_ids)

            query += " ORDER BY embedding <=> $1::vector LIMIT $" + str(len(params) + 1)
            params.append(top_k)

            results = await conn.fetch(query, *params)

            return [dict(row) for row in results]

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        document_ids: Optional[List[str]] = None,
        top_k: int = 20
    ) -> List[Dict]:
        """Hybrid search using built-in function"""

        async with self.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT * FROM hybrid_search(
                    $1, $2::vector, $3, $4
                )
            """, query_text, query_embedding.tolist(), document_ids, top_k)

            return [dict(row) for row in results]

    async def get_by_reference(
        self,
        legal_reference: str,
        document_id: Optional[str] = None
    ) -> List[Dict]:
        """Get chunks by legal reference"""

        async with self.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT * FROM find_by_legal_reference($1, $2, false)
            """, legal_reference, document_id)

            return [dict(row) for row in results]

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
```

Update configuration:

```python
# backend/app/core/config.py

# Replace FAISS configuration with PostgreSQL
POSTGRES_CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost/sujbot2"
)
```

## Verification

After migration, verify the data:

```sql
-- Check document counts
SELECT document_type, COUNT(*), SUM(total_chunks)
FROM documents
GROUP BY document_type;

-- Check chunk counts
SELECT COUNT(*), AVG(token_count)
FROM chunks;

-- Check embeddings
SELECT COUNT(*)
FROM chunks
WHERE embedding IS NOT NULL;

-- Check references
SELECT reference_type, COUNT(*)
FROM references
GROUP BY reference_type;

-- Check knowledge graph
SELECT edge_type, COUNT(*)
FROM cross_references
GROUP BY edge_type;

-- Test vector search
SELECT chunk_id, legal_reference,
       1 - (embedding <=> (SELECT embedding FROM chunks LIMIT 1)) AS similarity
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> (SELECT embedding FROM chunks LIMIT 1)
LIMIT 10;
```

## Performance Comparison

Run benchmark tests to compare FAISS vs PostgreSQL:

```python
# tests/benchmark_migration.py

import time
import numpy as np
from src.indexing import MultiDocumentVectorStore
from backend.app.rag.postgres_store import PostgresVectorStore

async def benchmark_search():
    """Compare search performance"""

    # Generate test query
    query_embedding = np.random.rand(1024).astype('float32')

    # FAISS search
    faiss_store = MultiDocumentVectorStore(...)
    start = time.time()
    faiss_results = await faiss_store.search(query_embedding, top_k=20)
    faiss_time = time.time() - start

    # PostgreSQL search
    pg_store = PostgresVectorStore("postgresql://localhost/sujbot2")
    await pg_store.initialize()
    start = time.time()
    pg_results = await pg_store.search(query_embedding, top_k=20)
    pg_time = time.time() - start

    print(f"FAISS: {faiss_time:.3f}s")
    print(f"PostgreSQL: {pg_time:.3f}s")
    print(f"Speedup: {faiss_time/pg_time:.2f}x")
```

## Rollback Plan

If migration fails, rollback to FAISS:

1. Keep original FAISS indices until migration is verified
2. Store backup of exported JSON files
3. Use feature flags to switch between FAISS and PostgreSQL

```python
# config.yaml
storage:
  backend: "postgres"  # or "faiss"
  postgres_url: "postgresql://localhost/sujbot2"
  faiss_index_dir: "./indexes"
```

## Post-Migration Tasks

1. **Delete old FAISS indices** (after verification):
   ```bash
   rm -rf indexes/
   rm -rf knowledge_graphs/
   ```

2. **Set up automated backups**:
   ```bash
   # Add to crontab
   0 2 * * * pg_dump sujbot2 | gzip > /backups/sujbot2_$(date +\%Y\%m\%d).sql.gz
   ```

3. **Monitor performance**:
   ```sql
   -- Enable query logging
   ALTER DATABASE sujbot2 SET log_min_duration_statement = 1000;

   -- Check slow queries
   SELECT query, mean_exec_time, calls
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 10;
   ```

4. **Optimize indexes**:
   ```sql
   -- Vacuum and analyze
   VACUUM ANALYZE chunks;

   -- Reindex if needed
   REINDEX TABLE chunks;
   ```

## Troubleshooting

### Issue: Index build is slow

**Solution:** Build index in background:
```sql
CREATE INDEX CONCURRENTLY idx_chunks_embedding_ivfflat ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 512);
```

### Issue: Out of memory during import

**Solution:** Import in batches:
```python
BATCH_SIZE = 1000
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    # Import batch
    conn.commit()
```

### Issue: Search is slower than FAISS

**Solution:** Tune IVFFlat parameters:
```sql
-- Increase probes for better accuracy (slower)
SET ivfflat.probes = 20;

-- Or use HNSW index (faster, more memory)
CREATE INDEX idx_chunks_embedding_hnsw ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

## Next Steps

1. Update frontend to use PostgreSQL-backed API
2. Implement caching layer (Redis)
3. Set up read replicas for scaling
4. Enable connection pooling (PgBouncer)
