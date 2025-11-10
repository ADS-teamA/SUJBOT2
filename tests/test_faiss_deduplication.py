"""
Unit tests for FAISS merge deduplication.

Tests the fix for the critical bug where merge() was adding duplicate vectors.

Strategy: Directly manipulate FAISS stores to test merge logic without
going through add_chunks() (which requires full Chunk objects).
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.faiss_vector_store import FAISSVectorStore


class TestFAISSDeduplication:
    """Test FAISS vector store deduplication during merge."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    def _create_store_with_metadata(self, metadata_list: list, dimensions: int = 3072):
        """
        Helper: Create FAISS store by directly adding metadata and vectors.

        Bypasses add_chunks() to avoid needing full Chunk objects.
        """
        store = FAISSVectorStore(dimensions=dimensions)

        # Create random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(len(metadata_list), dimensions).astype(np.float32)

        # Add directly to layer 3
        store.index_layer3.add(embeddings)
        store.metadata_layer3.extend(metadata_list)

        # Update doc_id_to_indices
        for idx, meta in enumerate(metadata_list):
            doc_id = meta.get("document_id")
            if doc_id:
                if doc_id not in store.doc_id_to_indices[3]:
                    store.doc_id_to_indices[3][doc_id] = []
                store.doc_id_to_indices[3][doc_id].append(idx)

        return store

    @pytest.fixture
    def sample_metadata_layer3(self):
        """Create sample L3 metadata for testing."""
        return [
            {
                "chunk_id": "doc1_L3_sec_1_chunk_0",
                "document_id": "doc1",
                "section_id": "doc1_sec_1",
                "content": "First chunk content",
            },
            {
                "chunk_id": "doc1_L3_sec_1_chunk_1",
                "document_id": "doc1",
                "section_id": "doc1_sec_1",
                "content": "Second chunk content",
            },
            {
                "chunk_id": "doc1_L3_sec_2_chunk_0",
                "document_id": "doc1",
                "section_id": "doc1_sec_2",
                "content": "Third chunk content",
            },
        ]

    def test_merge_prevents_duplicates(self, temp_dir, sample_metadata_layer3):
        """Test that merge() detects and skips duplicate chunk_ids."""
        # Create first vector store with 3 chunks
        store1 = self._create_store_with_metadata(sample_metadata_layer3)

        # Create second vector store with SAME chunks (simulates duplicate merge)
        store2 = self._create_store_with_metadata(sample_metadata_layer3)

        # Merge store2 into store1
        merge_stats = store1.merge(store2)

        # Assertions
        assert merge_stats["added"] == 0, "Should not add any vectors (all duplicates)"
        assert merge_stats["skipped"] == 3, "Should skip all 3 duplicate chunks"
        assert store1.index_layer3.ntotal == 3, "Should still have only 3 vectors"

        # Verify metadata
        chunk_ids = [meta["chunk_id"] for meta in store1.metadata_layer3]
        assert len(chunk_ids) == 3, "Should have 3 unique chunk_ids"
        assert len(set(chunk_ids)) == 3, "All chunk_ids should be unique"

    def test_merge_adds_new_chunks(self, temp_dir, sample_metadata_layer3):
        """Test that merge() adds new (non-duplicate) chunks."""
        # Create first vector store with first 2 chunks
        store1 = self._create_store_with_metadata(sample_metadata_layer3[:2])

        # Create second vector store with NEW chunk
        new_metadata = [
            {
                "chunk_id": "doc2_L3_sec_1_chunk_0",
                "document_id": "doc2",
                "section_id": "doc2_sec_1",
                "content": "New document chunk",
            }
        ]

        store2 = self._create_store_with_metadata(new_metadata)

        # Merge
        merge_stats = store1.merge(store2)

        # Assertions
        assert merge_stats["added"] == 1, "Should add 1 new vector"
        assert merge_stats["skipped"] == 0, "Should not skip any vectors"
        assert store1.index_layer3.ntotal == 3, "Should have 3 vectors total"

        # Verify metadata
        chunk_ids = [meta["chunk_id"] for meta in store1.metadata_layer3]
        assert "doc2_L3_sec_1_chunk_0" in chunk_ids, "Should contain new chunk_id"

    def test_merge_mixed_duplicates_and_new(self, temp_dir, sample_metadata_layer3):
        """Test merge with both duplicate and new chunks."""
        # Create first vector store with 3 chunks
        store1 = self._create_store_with_metadata(sample_metadata_layer3)

        # Create second vector store with 2 duplicates + 1 new
        mixed_metadata = [
            sample_metadata_layer3[0],  # Duplicate
            sample_metadata_layer3[1],  # Duplicate
            {
                "chunk_id": "doc2_L3_sec_1_chunk_0",
                "document_id": "doc2",
                "section_id": "doc2_sec_1",
                "content": "New chunk",
            },  # New
        ]

        store2 = self._create_store_with_metadata(mixed_metadata)

        # Merge
        merge_stats = store1.merge(store2)

        # Assertions
        assert merge_stats["added"] == 1, "Should add 1 new vector"
        assert merge_stats["skipped"] == 2, "Should skip 2 duplicate vectors"
        assert store1.index_layer3.ntotal == 4, "Should have 4 vectors total"

    def test_merge_32x_duplication_scenario(self, temp_dir, sample_metadata_layer3):
        """
        Test the exact scenario that caused the bug: 32 merges of same data.

        This simulates what happened in production where same documents
        were merged 32 times, creating 32x duplicates.
        """
        # Create initial vector store with 3 chunks
        store = self._create_store_with_metadata(sample_metadata_layer3)

        # Simulate 32 merges (the bug scenario)
        for i in range(32):
            # Create duplicate store
            dup_store = self._create_store_with_metadata(sample_metadata_layer3)

            # Merge (should detect duplicates after first merge)
            merge_stats = store.merge(dup_store)

            if i == 0:
                # First merge should add nothing (already in store)
                assert merge_stats["added"] == 0
                assert merge_stats["skipped"] == 3
            else:
                # All subsequent merges should also skip
                assert merge_stats["added"] == 0
                assert merge_stats["skipped"] == 3

        # Final verification
        assert store.index_layer3.ntotal == 3, "Should still have only 3 vectors (not 96!)"

        stats = store.get_stats()
        assert stats["layer3_count"] == 3, "Layer 3 should have 3 vectors"
        assert stats["documents"] == 1, "Should have 1 document"

        # Verify no duplicates in metadata
        chunk_ids = [meta["chunk_id"] for meta in store.metadata_layer3]
        assert len(chunk_ids) == 3
        assert len(set(chunk_ids)) == 3, "All chunk_ids should be unique"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
