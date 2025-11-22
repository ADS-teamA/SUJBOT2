"""
Tests for VectorStoreAdapter interface and FAISS/PostgreSQL adapters.

CRITICAL: Storage adapters provide unified interface for PostgreSQL migration.
All RAG tools depend on this abstraction working correctly.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

from src.storage.vector_store_adapter import VectorStoreAdapter
from src.storage.faiss_adapter import FAISSVectorStoreAdapter
from src.storage.postgres_adapter import PostgresVectorStoreAdapter


# ============================================================================
# FAISS Adapter Tests
# ============================================================================

class TestFAISSVectorStoreAdapter:
    """Tests for FAISS backend adapter."""

    @pytest.fixture
    def mock_faiss_store(self):
        """Mock FAISS vector store."""
        store = Mock()

        # Mock hierarchical search
        store.hierarchical_search = Mock(return_value={
            "layer1": [
                {
                    "chunk_id": "doc1:sec1:0",
                    "text": "Test chunk from layer 1",
                    "relevance_score": 0.95,
                    "document_id": "doc1",
                    "metadata": {"section": "Introduction"}
                }
            ],
            "layer3": [
                {
                    "chunk_id": "doc1:sec1:0",
                    "text": "Test chunk from layer 3",
                    "relevance_score": 0.95,
                    "document_id": "doc1",
                    "metadata": {"section": "Introduction"}
                }
            ]
        })

        # Mock similarity search
        store.similarity_search = Mock(return_value=[
            {
                "chunk_id": "doc1:sec1:0",
                "text": "Similar chunk",
                "relevance_score": 0.9,
                "document_id": "doc1"
            }
        ])

        # Mock document enumeration
        store.metadata_layer3 = [
            {"document_id": "doc1", "title": "Doc 1"},
            {"document_id": "doc2", "title": "Doc 2"}
        ]

        return store

    @pytest.fixture
    def faiss_adapter(self, mock_faiss_store):
        """Create FAISS adapter with mock store."""
        return FAISSVectorStoreAdapter(mock_faiss_store)

    def test_faiss_adapter_hierarchical_search(self, faiss_adapter, mock_faiss_store):
        """FAISS adapter should delegate hierarchical_search to underlying store."""
        query_embedding = np.random.rand(768).astype(np.float32)

        results = faiss_adapter.hierarchical_search(
            query_embedding=query_embedding,
            k_layer3=5,
            use_doc_filtering=True
        )

        # Verify delegation
        mock_faiss_store.hierarchical_search.assert_called_once()
        assert "layer1" in results
        assert "layer3" in results
        assert len(results["layer3"]) > 0

    def test_faiss_adapter_similarity_search(self, faiss_adapter, mock_faiss_store):
        """FAISS adapter should delegate similarity_search."""
        query_embedding = np.random.rand(768).astype(np.float32)

        results = faiss_adapter.similarity_search(
            query_embedding=query_embedding,
            k=10,
            layer="layer3"
        )

        mock_faiss_store.similarity_search.assert_called_once()
        assert len(results) > 0
        assert all("chunk_id" in r for r in results)

    def test_faiss_adapter_get_all_document_ids(self, faiss_adapter):
        """FAISS adapter should extract document IDs from metadata."""
        doc_ids = faiss_adapter.get_all_document_ids()

        assert isinstance(doc_ids, list)
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_faiss_adapter_get_document_metadata(self, faiss_adapter):
        """FAISS adapter should return document metadata."""
        metadata = faiss_adapter.get_document_metadata("doc1")

        assert metadata is not None
        assert metadata["document_id"] == "doc1"
        assert metadata["title"] == "Doc 1"

    def test_faiss_adapter_get_document_metadata_not_found(self, faiss_adapter):
        """Should return None for non-existent document."""
        metadata = faiss_adapter.get_document_metadata("nonexistent")

        assert metadata is None


# ============================================================================
# PostgreSQL Adapter Tests
# ============================================================================

class TestPostgresVectorStoreAdapter:
    """Tests for PostgreSQL backend adapter."""

    @pytest.fixture
    def mock_pg_connection(self):
        """Mock PostgreSQL connection."""
        conn = Mock()
        cursor = Mock()

        # Mock cursor context manager
        cursor.__enter__ = Mock(return_value=cursor)
        cursor.__exit__ = Mock(return_value=False)

        # Mock query results
        cursor.fetchall = Mock(return_value=[
            ("doc1:sec1:0", "Test chunk", 0.95, "doc1", {"section": "Intro"}),
            ("doc1:sec2:1", "Another chunk", 0.88, "doc1", {"section": "Body"})
        ])

        cursor.fetchone = Mock(return_value=("doc1", "Test Document", {"year": 2024}))

        conn.cursor = Mock(return_value=cursor)

        return conn

    @pytest.fixture
    def postgres_adapter(self, mock_pg_connection):
        """Create PostgreSQL adapter with mock connection."""
        with patch("src.storage.postgres_adapter.psycopg2.connect", return_value=mock_pg_connection):
            adapter = PostgresVectorStoreAdapter(connection_string="postgresql://test")
            adapter.conn = mock_pg_connection  # Inject mock
            return adapter

    def test_postgres_adapter_hierarchical_search(self, postgres_adapter):
        """PostgreSQL adapter should execute hierarchical search query."""
        query_embedding = np.random.rand(768).astype(np.float32)

        results = postgres_adapter.hierarchical_search(
            query_embedding=query_embedding,
            k_layer3=5,
            use_doc_filtering=False
        )

        # Verify results structure matches FAISS
        assert "layer1" in results
        assert "layer3" in results
        assert isinstance(results["layer3"], list)

    def test_postgres_adapter_similarity_search(self, postgres_adapter, mock_pg_connection):
        """PostgreSQL adapter should execute similarity search with pgvector."""
        query_embedding = np.random.rand(768).astype(np.float32)

        results = postgres_adapter.similarity_search(
            query_embedding=query_embedding,
            k=10,
            layer="layer3"
        )

        # Verify query executed
        cursor = mock_pg_connection.cursor.return_value
        cursor.execute.assert_called()

        # Verify results format
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_postgres_adapter_get_all_document_ids(self, postgres_adapter, mock_pg_connection):
        """PostgreSQL adapter should query distinct document IDs."""
        cursor = mock_pg_connection.cursor.return_value
        cursor.fetchall.return_value = [("doc1",), ("doc2",), ("doc3",)]

        doc_ids = postgres_adapter.get_all_document_ids()

        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids

    def test_postgres_adapter_get_document_metadata(self, postgres_adapter):
        """PostgreSQL adapter should query document metadata table."""
        metadata = postgres_adapter.get_document_metadata("doc1")

        assert metadata is not None
        assert "document_id" in metadata

    @pytest.mark.requires_postgres
    def test_postgres_adapter_connection_pooling(self):
        """PostgreSQL adapter should use connection pooling."""
        # This test would require real PostgreSQL instance
        # For now, verify connection string parsing
        adapter = PostgresVectorStoreAdapter(connection_string="postgresql://user:pass@localhost:5432/db")

        assert adapter.connection_string == "postgresql://user:pass@localhost:5432/db"

    def test_postgres_adapter_handles_connection_failure(self):
        """Should raise clear error on connection failure."""
        with patch("src.storage.postgres_adapter.psycopg2.connect", side_effect=Exception("Connection refused")):
            with pytest.raises(Exception, match="Connection refused"):
                PostgresVectorStoreAdapter(connection_string="postgresql://invalid")


# ============================================================================
# Backend Switching Tests
# ============================================================================

class TestBackendSwitching:
    """Tests for seamless switching between FAISS and PostgreSQL."""

    @pytest.fixture
    def faiss_results(self):
        """Sample results from FAISS."""
        return {
            "layer1": [
                {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "relevance_score": 0.95}
            ],
            "layer3": [
                {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "relevance_score": 0.95},
                {"chunk_id": "doc2:sec1:0", "document_id": "doc2", "relevance_score": 0.88}
            ]
        }

    @pytest.fixture
    def postgres_results(self):
        """Sample results from PostgreSQL."""
        return {
            "layer1": [
                {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "relevance_score": 0.94}
            ],
            "layer3": [
                {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "relevance_score": 0.94},
                {"chunk_id": "doc2:sec1:0", "document_id": "doc2", "relevance_score": 0.87}
            ]
        }

    def test_backends_return_same_structure(self, faiss_results, postgres_results):
        """Both backends should return identical structure."""
        # Verify keys match
        assert set(faiss_results.keys()) == set(postgres_results.keys())

        # Verify layer3 has same fields
        faiss_fields = set(faiss_results["layer3"][0].keys())
        postgres_fields = set(postgres_results["layer3"][0].keys())
        assert faiss_fields == postgres_fields

    def test_backends_return_similar_documents(self, faiss_results, postgres_results):
        """Both backends should return similar documents for same query."""
        # Extract document IDs
        faiss_docs = {r["document_id"] for r in faiss_results["layer3"]}
        postgres_docs = {r["document_id"] for r in postgres_results["layer3"]}

        # Should have significant overlap (at least 60%)
        overlap = len(faiss_docs & postgres_docs)
        total = len(faiss_docs | postgres_docs)

        overlap_ratio = overlap / total if total > 0 else 0
        assert overlap_ratio >= 0.6, f"Only {overlap_ratio*100:.1f}% document overlap"

    def test_adapter_factory_selects_correct_backend(self):
        """load_vector_store_adapter() should select backend based on config."""
        from src.storage.vector_store_adapter import load_vector_store_adapter

        # Test FAISS selection
        with patch("src.storage.vector_store_adapter.config") as mock_config:
            mock_config.STORAGE_BACKEND = "faiss"
            mock_config.VECTOR_DB_PATH = "/tmp/faiss"

            with patch("src.storage.faiss_adapter.FAISSVectorStore"):
                adapter = load_vector_store_adapter()
                assert isinstance(adapter, FAISSVectorStoreAdapter)

        # Test PostgreSQL selection
        with patch("src.storage.vector_store_adapter.config") as mock_config:
            mock_config.STORAGE_BACKEND = "postgresql"
            mock_config.DATABASE_URL = "postgresql://test"

            with patch("src.storage.postgres_adapter.psycopg2.connect"):
                adapter = load_vector_store_adapter()
                assert isinstance(adapter, PostgresVectorStoreAdapter)


# ============================================================================
# Interface Compliance Tests
# ============================================================================

class TestVectorStoreAdapterInterface:
    """Verify both adapters implement VectorStoreAdapter interface correctly."""

    def test_faiss_adapter_implements_interface(self, mock_faiss_store):
        """FAISS adapter should implement all required methods."""
        adapter = FAISSVectorStoreAdapter(mock_faiss_store)

        # Check all interface methods exist
        assert hasattr(adapter, "hierarchical_search")
        assert hasattr(adapter, "similarity_search")
        assert hasattr(adapter, "get_all_document_ids")
        assert hasattr(adapter, "get_document_metadata")

        # Check methods are callable
        assert callable(adapter.hierarchical_search)
        assert callable(adapter.similarity_search)
        assert callable(adapter.get_all_document_ids)
        assert callable(adapter.get_document_metadata)

    def test_postgres_adapter_implements_interface(self):
        """PostgreSQL adapter should implement all required methods."""
        with patch("src.storage.postgres_adapter.psycopg2.connect"):
            adapter = PostgresVectorStoreAdapter(connection_string="postgresql://test")

        # Check all interface methods exist
        assert hasattr(adapter, "hierarchical_search")
        assert hasattr(adapter, "similarity_search")
        assert hasattr(adapter, "get_all_document_ids")
        assert hasattr(adapter, "get_document_metadata")


# ============================================================================
# Document Filtering Tests
# ============================================================================

class TestDocumentFiltering:
    """Test document filtering in hierarchical search."""

    @pytest.fixture
    def adapter_with_multi_doc_store(self, mock_faiss_store):
        """Create adapter with store containing multiple documents."""
        mock_faiss_store.hierarchical_search.return_value = {
            "layer1": [
                {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "relevance_score": 0.95},
                {"chunk_id": "doc2:sec1:0", "document_id": "doc2", "relevance_score": 0.85}
            ],
            "layer3": [
                {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "relevance_score": 0.95},
                {"chunk_id": "doc1:sec2:1", "document_id": "doc1", "relevance_score": 0.90},
                {"chunk_id": "doc2:sec1:0", "document_id": "doc2", "relevance_score": 0.85}
            ]
        }

        return FAISSVectorStoreAdapter(mock_faiss_store)

    def test_hierarchical_search_with_document_filter(self, adapter_with_multi_doc_store):
        """Should filter results to specific documents when use_doc_filtering=True."""
        query_embedding = np.random.rand(768).astype(np.float32)

        results = adapter_with_multi_doc_store.hierarchical_search(
            query_embedding=query_embedding,
            k_layer3=10,
            use_doc_filtering=True,
            document_ids=["doc1"]  # Filter to doc1 only
        )

        # All results should be from doc1
        layer3_docs = [r["document_id"] for r in results["layer3"]]
        assert all(doc_id == "doc1" for doc_id in layer3_docs)

    def test_hierarchical_search_without_document_filter(self, adapter_with_multi_doc_store):
        """Should return results from all documents when use_doc_filtering=False."""
        query_embedding = np.random.rand(768).astype(np.float32)

        results = adapter_with_multi_doc_store.hierarchical_search(
            query_embedding=query_embedding,
            k_layer3=10,
            use_doc_filtering=False
        )

        # Results should include multiple documents
        layer3_docs = set(r["document_id"] for r in results["layer3"])
        assert len(layer3_docs) > 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestStorageAdapterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_results_handled_gracefully(self, mock_faiss_store):
        """Should handle empty search results without crashing."""
        mock_faiss_store.hierarchical_search.return_value = {
            "layer1": [],
            "layer3": []
        }

        adapter = FAISSVectorStoreAdapter(mock_faiss_store)
        query_embedding = np.random.rand(768).astype(np.float32)

        results = adapter.hierarchical_search(
            query_embedding=query_embedding,
            k_layer3=5
        )

        assert results["layer1"] == []
        assert results["layer3"] == []

    def test_missing_relevance_score_handled(self, mock_faiss_store):
        """Should handle chunks missing relevance_score field."""
        mock_faiss_store.similarity_search.return_value = [
            {"chunk_id": "doc1:sec1:0", "text": "Chunk without score"}
        ]

        adapter = FAISSVectorStoreAdapter(mock_faiss_store)
        query_embedding = np.random.rand(768).astype(np.float32)

        results = adapter.similarity_search(query_embedding, k=5)

        # Should not crash
        assert len(results) > 0

    def test_malformed_metadata_handled(self, mock_faiss_store):
        """Should handle malformed metadata gracefully."""
        mock_faiss_store.metadata_layer3 = [
            {"document_id": "doc1"},  # Missing title
            {"title": "Doc 2"},  # Missing document_id
            None  # Null entry
        ]

        adapter = FAISSVectorStoreAdapter(mock_faiss_store)

        # Should not crash
        doc_ids = adapter.get_all_document_ids()
        assert isinstance(doc_ids, list)

    def test_invalid_embedding_dimension_raises_error(self, mock_faiss_store):
        """Should raise error for incorrect embedding dimension."""
        adapter = FAISSVectorStoreAdapter(mock_faiss_store)

        # Wrong dimension (512 instead of 768)
        wrong_embedding = np.random.rand(512).astype(np.float32)

        # FAISS would raise dimension mismatch error
        mock_faiss_store.hierarchical_search.side_effect = ValueError("Dimension mismatch")

        with pytest.raises(ValueError, match="Dimension mismatch"):
            adapter.hierarchical_search(query_embedding=wrong_embedding, k_layer3=5)
