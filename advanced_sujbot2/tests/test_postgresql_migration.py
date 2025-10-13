#!/usr/bin/env python3
"""
Test suite for PostgreSQL + pgvector migration

Tests basic functionality, vector search, and performance.

Usage:
    pytest tests/test_postgresql_migration.py -v
    python tests/test_postgresql_migration.py  # Run standalone
"""

import asyncio
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pg_vector_store import PostgreSQLVectorStore, PostgreSQLConfig
from src.embeddings import LegalEmbedder, EmbeddingConfig
from src.chunker import LegalChunk


# Test configuration
TEST_CONFIG = PostgreSQLConfig(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("POSTGRES_DB", "sujbot2"),
    user=os.getenv("POSTGRES_USER", "sujbot_app"),
    password=os.getenv("POSTGRES_PASSWORD", "sujbot2_dev_password"),
)


@pytest.fixture
async def vector_store():
    """Create vector store for testing"""
    embedder = LegalEmbedder(EmbeddingConfig(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=8
    ))

    store = PostgreSQLVectorStore(embedder, TEST_CONFIG)
    yield store
    await store.close()


@pytest.fixture
def sample_chunks():
    """Create sample legal chunks for testing"""
    chunks = [
        LegalChunk(
            chunk_id="test_chunk_1",
            chunk_index=0,
            content="§89 Odpovědnost za vady. Dodavatel odpovídá za vady výrobku.",
            document_id="test_law_001",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §89",
            legal_reference="§89",
            structural_level="paragraph",
            metadata={
                "part": "II",
                "chapter": "III",
                "paragraph": 89,
                "content_type": "obligation",
                "token_count": 20
            }
        ),
        LegalChunk(
            chunk_id="test_chunk_2",
            chunk_index=1,
            content="§90 Záruční doba. Záruční doba činí 24 měsíců od převzetí věci.",
            document_id="test_law_001",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §90",
            legal_reference="§90",
            structural_level="paragraph",
            metadata={
                "part": "II",
                "chapter": "III",
                "paragraph": 90,
                "content_type": "definition",
                "token_count": 18
            }
        ),
        LegalChunk(
            chunk_id="test_chunk_3",
            chunk_index=2,
            content="Článek 5 Platební podmínky. Kupující uhradí cenu do 30 dnů.",
            document_id="test_contract_001",
            document_type="contract",
            hierarchy_path="Článek 5",
            legal_reference="Článek 5",
            structural_level="article",
            metadata={
                "article": 5,
                "content_type": "obligation",
                "token_count": 15
            }
        )
    ]
    return chunks


class TestPostgreSQLConnection:
    """Test basic PostgreSQL connectivity"""

    @pytest.mark.asyncio
    async def test_connection_pool(self, vector_store):
        """Test that connection pool is created successfully"""
        await vector_store._ensure_pool()
        assert vector_store._pool is not None
        print("✓ Connection pool created")

    @pytest.mark.asyncio
    async def test_database_extensions(self, vector_store):
        """Test that required PostgreSQL extensions are installed"""
        await vector_store._ensure_pool()

        async with vector_store._pool.acquire() as conn:
            result = await conn.fetch(
                """
                SELECT extname, extversion
                FROM pg_extension
                WHERE extname IN ('vector', 'pg_trgm', 'btree_gin')
                """
            )

        extensions = {row['extname'] for row in result}
        assert 'vector' in extensions, "pgvector extension not installed"
        assert 'pg_trgm' in extensions, "pg_trgm extension not installed"
        assert 'btree_gin' in extensions, "btree_gin extension not installed"

        print("✓ All required extensions installed:")
        for row in result:
            print(f"  - {row['extname']}: v{row['extversion']}")


class TestDocumentIndexing:
    """Test document indexing operations"""

    @pytest.mark.asyncio
    async def test_add_document(self, vector_store, sample_chunks):
        """Test adding document with chunks"""
        await vector_store.add_document(
            chunks=sample_chunks[:2],  # Law chunks
            document_id="test_law_001",
            document_type="law_code",
            metadata={
                "title": "Test Law",
                "filename": "test_law.pdf"
            }
        )

        # Verify document was added
        doc_info = vector_store.get_document_info("test_law_001")
        assert doc_info is not None
        assert doc_info['num_chunks'] == 2
        assert doc_info['document_type'] == 'law_code'

        print(f"✓ Document indexed: {doc_info['num_chunks']} chunks")

    @pytest.mark.asyncio
    async def test_get_document_chunks(self, vector_store, sample_chunks):
        """Test retrieving all chunks for a document"""
        # Add document first
        await vector_store.add_document(
            chunks=sample_chunks[:2],
            document_id="test_law_002",
            document_type="law_code",
            metadata={}
        )

        # Retrieve chunks
        retrieved_chunks = await vector_store.get_document_chunks("test_law_002")

        assert len(retrieved_chunks) == 2
        assert retrieved_chunks[0].chunk_id == "test_chunk_1"
        assert retrieved_chunks[1].chunk_id == "test_chunk_2"

        print(f"✓ Retrieved {len(retrieved_chunks)} chunks from database")


class TestVectorSearch:
    """Test vector similarity search"""

    @pytest.mark.asyncio
    async def test_semantic_search(self, vector_store, sample_chunks):
        """Test semantic vector search"""
        # Add test documents
        await vector_store.add_document(
            chunks=sample_chunks,
            document_id="test_doc_search",
            document_type="law_code",
            metadata={}
        )

        # Search for liability-related content
        results = await vector_store.search(
            query="odpovědnost za vady",
            document_ids=["test_doc_search"],
            top_k=2
        )

        assert len(results) > 0
        assert results[0].score > 0.5, "Similarity score too low"

        print(f"✓ Vector search returned {len(results)} results")
        print(f"  Top result: {results[0].chunk.legal_reference} (score: {results[0].score:.3f})")

    @pytest.mark.asyncio
    async def test_search_by_reference(self, vector_store, sample_chunks):
        """Test direct lookup by legal reference"""
        await vector_store.add_document(
            chunks=sample_chunks[:2],
            document_id="test_doc_ref",
            document_type="law_code",
            metadata={}
        )

        # Search for specific paragraph
        chunk = await vector_store.search_by_reference("§89", "test_doc_ref")

        assert chunk is not None
        assert chunk.legal_reference == "§89"
        assert "odpovědnost" in chunk.content.lower()

        print(f"✓ Reference lookup successful: {chunk.legal_reference}")

    @pytest.mark.asyncio
    async def test_metadata_filtering(self, vector_store, sample_chunks):
        """Test search with metadata filters"""
        await vector_store.add_document(
            chunks=sample_chunks,
            document_id="test_doc_filter",
            document_type="law_code",
            metadata={}
        )

        # Search for obligations only
        results = await vector_store.search(
            query="povinnost",
            document_ids=["test_doc_filter"],
            top_k=5,
            filter_metadata={"content_type": "obligation"}
        )

        assert len(results) > 0
        for result in results:
            assert result.chunk.metadata.get("content_type") == "obligation"

        print(f"✓ Metadata filtering successful: {len(results)} obligations found")


class TestPerformance:
    """Performance benchmarks"""

    @pytest.mark.asyncio
    async def test_batch_insert_performance(self, vector_store):
        """Test batch insert performance"""
        import time

        # Create 100 test chunks
        chunks = []
        for i in range(100):
            chunks.append(LegalChunk(
                chunk_id=f"perf_chunk_{i}",
                chunk_index=i,
                content=f"Test content paragraph {i}. This is sample legal text for performance testing.",
                document_id="perf_test_doc",
                document_type="law_code",
                hierarchy_path=f"§{i}",
                legal_reference=f"§{i}",
                structural_level="paragraph",
                metadata={"paragraph": i, "token_count": 20}
            ))

        start = time.time()
        await vector_store.add_document(
            chunks=chunks,
            document_id="perf_test_doc",
            document_type="law_code",
            metadata={}
        )
        elapsed = time.time() - start

        chunks_per_sec = len(chunks) / elapsed

        print(f"✓ Batch insert: {len(chunks)} chunks in {elapsed:.2f}s ({chunks_per_sec:.1f} chunks/sec)")

        # Should be reasonably fast
        assert elapsed < 30, f"Batch insert too slow: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_search_latency(self, vector_store, sample_chunks):
        """Test search query latency"""
        import time

        # Add test data
        await vector_store.add_document(
            chunks=sample_chunks * 10,  # 30 chunks total
            document_id="latency_test_doc",
            document_type="law_code",
            metadata={}
        )

        # Measure search latency
        start = time.time()
        results = await vector_store.search(
            query="odpovědnost dodavatel",
            document_ids=["latency_test_doc"],
            top_k=10
        )
        elapsed = (time.time() - start) * 1000  # Convert to ms

        print(f"✓ Vector search latency: {elapsed:.1f}ms for {len(results)} results")

        # Should be reasonably fast (IVFFlat target: <30ms for 100K vectors)
        assert elapsed < 100, f"Search too slow: {elapsed:.1f}ms"


class TestDataIntegrity:
    """Test data integrity and consistency"""

    @pytest.mark.asyncio
    async def test_chunk_count_consistency(self, vector_store, sample_chunks):
        """Test that chunk counts are consistent"""
        doc_id = "integrity_test_doc"

        await vector_store.add_document(
            chunks=sample_chunks,
            document_id=doc_id,
            document_type="law_code",
            metadata={}
        )

        # Check document info
        doc_info = vector_store.get_document_info(doc_id)
        assert doc_info['num_chunks'] == len(sample_chunks)

        # Check actual chunks in database
        chunks = await vector_store.get_document_chunks(doc_id)
        assert len(chunks) == len(sample_chunks)

        print(f"✓ Data consistency verified: {len(chunks)} chunks")

    @pytest.mark.asyncio
    async def test_embedding_dimensions(self, vector_store, sample_chunks):
        """Test that embeddings have correct dimensions"""
        await vector_store.add_document(
            chunks=sample_chunks[:1],
            document_id="embedding_test_doc",
            document_type="law_code",
            metadata={}
        )

        await vector_store._ensure_pool()
        async with vector_store._pool.acquire() as conn:
            result = await conn.fetchrow(
                "SELECT embedding FROM chunks WHERE document_id = $1 LIMIT 1",
                "embedding_test_doc"
            )

        embedding = result['embedding']
        # BGE-M3 has 1024 dimensions
        assert len(embedding) == 1024, f"Wrong embedding dimension: {len(embedding)}"

        print(f"✓ Embedding dimension correct: {len(embedding)}")


class TestCleanup:
    """Test cleanup operations"""

    @pytest.mark.asyncio
    async def test_delete_document(self, vector_store, sample_chunks):
        """Test document deletion"""
        doc_id = "delete_test_doc"

        # Add document
        await vector_store.add_document(
            chunks=sample_chunks[:2],
            document_id=doc_id,
            document_type="law_code",
            metadata={}
        )

        # Verify it exists
        assert doc_id in vector_store.document_info

        # Delete it
        await vector_store.delete_document(doc_id)

        # Verify it's gone from cache
        assert doc_id not in vector_store.document_info

        # Verify it's gone from database
        await vector_store._ensure_pool()
        async with vector_store._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM documents WHERE document_id = $1",
                doc_id
            )

        assert count == 0, "Document not deleted from database"

        print(f"✓ Document deleted successfully")


# Standalone test runner
async def run_all_tests():
    """Run all tests standalone (without pytest)"""
    print("=" * 70)
    print("PostgreSQL + pgvector Migration Test Suite")
    print("=" * 70)
    print()

    # Create fixtures
    embedder = LegalEmbedder(EmbeddingConfig(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=8
    ))
    store = PostgreSQLVectorStore(embedder, TEST_CONFIG)

    # Sample chunks
    sample_chunks = [
        LegalChunk(
            chunk_id="test_chunk_1",
            chunk_index=0,
            content="§89 Odpovědnost za vady. Dodavatel odpovídá za vady výrobku.",
            document_id="test_law_001",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §89",
            legal_reference="§89",
            structural_level="paragraph",
            metadata={"part": "II", "chapter": "III", "paragraph": 89, "content_type": "obligation", "token_count": 20}
        ),
        LegalChunk(
            chunk_id="test_chunk_2",
            chunk_index=1,
            content="§90 Záruční doba. Záruční doba činí 24 měsíců od převzetí věci.",
            document_id="test_law_001",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §90",
            legal_reference="§90",
            structural_level="paragraph",
            metadata={"part": "II", "chapter": "III", "paragraph": 90, "content_type": "definition", "token_count": 18}
        )
    ]

    try:
        # Test 1: Connection
        print("\n[1/6] Testing PostgreSQL connection...")
        await store._ensure_pool()
        print("✓ Connected to PostgreSQL")

        # Test 2: Extensions
        print("\n[2/6] Checking extensions...")
        async with store._pool.acquire() as conn:
            result = await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'btree_gin')"
            )
        extensions = [row['extname'] for row in result]
        assert 'vector' in extensions
        print("✓ pgvector extension found")

        # Test 3: Indexing
        print("\n[3/6] Testing document indexing...")
        await store.add_document(
            chunks=sample_chunks,
            document_id="standalone_test_doc",
            document_type="law_code",
            metadata={"title": "Test Law"}
        )
        print("✓ Document indexed successfully")

        # Test 4: Vector search
        print("\n[4/6] Testing vector search...")
        results = await store.search(
            query="odpovědnost za vady",
            document_ids=["standalone_test_doc"],
            top_k=5
        )
        assert len(results) > 0
        print(f"✓ Search returned {len(results)} results (top score: {results[0].score:.3f})")

        # Test 5: Reference lookup
        print("\n[5/6] Testing reference lookup...")
        chunk = await store.search_by_reference("§89", "standalone_test_doc")
        assert chunk is not None
        print(f"✓ Found reference: {chunk.legal_reference}")

        # Test 6: Cleanup
        print("\n[6/6] Testing cleanup...")
        await store.delete_document("standalone_test_doc")
        print("✓ Document deleted")

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await store.close()

    return True


if __name__ == "__main__":
    # Run tests standalone
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
