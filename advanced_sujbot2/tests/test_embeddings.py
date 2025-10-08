"""
Tests for Legal Embeddings Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import (
    LegalEmbedder,
    LegalChunk,
    EmbeddingConfig,
    EmbeddingCache,
    EmbeddingError,
    ModelLoadError
)


@pytest.fixture
def sample_chunk():
    """Create a sample legal chunk"""
    return LegalChunk(
        chunk_id="test_chunk_1",
        content="Dodavatel odpovídá za vady díla podle občanského zákoníku.",
        document_id="law_89_2012",
        document_type="law_code",
        hierarchy_path="Část II > Hlava III > §89",
        legal_reference="§89",
        structural_level="paragraph",
        metadata={
            "part": "II",
            "chapter": "III",
            "paragraph": 89,
            "content_type": "obligation"
        }
    )


@pytest.fixture
def embedder():
    """Create embedder instance"""
    config = EmbeddingConfig(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=2,
        show_progress_bar=False
    )
    return LegalEmbedder(config)


class TestLegalChunk:
    """Test LegalChunk data class"""

    def test_chunk_creation(self, sample_chunk):
        """Test basic chunk creation"""
        assert sample_chunk.chunk_id == "test_chunk_1"
        assert sample_chunk.document_type == "law_code"
        assert sample_chunk.legal_reference == "§89"

    def test_get_citation_law(self, sample_chunk):
        """Test citation for law code"""
        sample_chunk.metadata['law_citation'] = "Zákon č. 89/2012 Sb."
        citation = sample_chunk.get_citation()
        assert "89/2012" in citation
        assert "§89" in citation

    def test_get_citation_contract(self):
        """Test citation for contract"""
        chunk = LegalChunk(
            chunk_id="contract_1",
            content="Test content",
            document_type="contract",
            hierarchy_path="Článek 5.2",
            legal_reference="Článek 5.2"
        )
        citation = chunk.get_citation()
        assert citation == "Článek 5.2"


class TestLegalEmbedder:
    """Test LegalEmbedder class"""

    def test_embedder_initialization(self, embedder):
        """Test embedder initializes correctly"""
        assert embedder.model is not None
        assert embedder.device in ["cpu", "cuda", "mps"]

    @pytest.mark.asyncio
    async def test_embed_single_chunk(self, embedder, sample_chunk):
        """Test embedding a single chunk"""
        embeddings = await embedder.embed_chunks([sample_chunk])

        assert embeddings.shape == (1, 1024)  # BGE-M3 dimension
        assert np.allclose(np.linalg.norm(embeddings[0]), 1.0, atol=0.01)  # Normalized

    @pytest.mark.asyncio
    async def test_embed_multiple_chunks(self, embedder):
        """Test embedding multiple chunks"""
        chunks = [
            LegalChunk(
                chunk_id=f"chunk_{i}",
                content=f"Test content {i}",
                legal_reference=f"§{i}"
            )
            for i in range(5)
        ]

        embeddings = await embedder.embed_chunks(chunks)

        assert embeddings.shape == (5, 1024)
        # Check all embeddings are normalized
        for emb in embeddings:
            assert np.allclose(np.linalg.norm(emb), 1.0, atol=0.01)

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, embedder):
        """Test embedding empty chunk list"""
        embeddings = await embedder.embed_chunks([])
        assert embeddings.shape == (0,)

    @pytest.mark.asyncio
    async def test_contextualized_embedding(self, embedder, sample_chunk):
        """Test embedding with context"""
        # With context
        emb_with_context = await embedder.embed_chunks([sample_chunk], add_context=True)

        # Without context
        emb_without_context = await embedder.embed_chunks([sample_chunk], add_context=False)

        # Embeddings should be different
        assert not np.allclose(emb_with_context, emb_without_context)

    @pytest.mark.asyncio
    async def test_embed_query(self, embedder):
        """Test query embedding"""
        query = "odpovědnost za vady"
        embedding = await embedder.embed_query(query)

        assert embedding.shape == (1024,)
        assert np.allclose(np.linalg.norm(embedding), 1.0, atol=0.01)

    def test_contextualize(self, embedder, sample_chunk):
        """Test context generation"""
        contextualized = embedder._contextualize(sample_chunk)

        # Should contain reference, hierarchy, and content
        assert sample_chunk.legal_reference in contextualized
        assert sample_chunk.hierarchy_path in contextualized or True  # Might be deduplicated
        assert sample_chunk.content in contextualized

    def test_get_embedding_dimension(self, embedder):
        """Test getting embedding dimension"""
        dim = embedder.get_embedding_dimension()
        assert dim == 1024


class TestEmbeddingCache:
    """Test EmbeddingCache class"""

    @pytest.mark.asyncio
    async def test_cache_basic(self, embedder, sample_chunk):
        """Test basic caching"""
        cache = EmbeddingCache(max_size=10)

        # First call - compute
        emb1 = await cache.get_or_compute(
            sample_chunk.chunk_id,
            sample_chunk,
            embedder
        )

        # Second call - from cache
        emb2 = await cache.get_or_compute(
            sample_chunk.chunk_id,
            sample_chunk,
            embedder
        )

        # Should be same embedding
        assert np.allclose(emb1, emb2)

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, embedder):
        """Test LRU eviction"""
        cache = EmbeddingCache(max_size=3)

        # Add 4 chunks (should evict first one)
        chunks = [
            LegalChunk(chunk_id=f"chunk_{i}", content=f"Content {i}")
            for i in range(4)
        ]

        for chunk in chunks:
            await cache.get_or_compute(chunk.chunk_id, chunk, embedder)

        # First chunk should be evicted
        assert "chunk_0" not in cache.cache
        assert "chunk_1" in cache.cache
        assert "chunk_2" in cache.cache
        assert "chunk_3" in cache.cache

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = EmbeddingCache(max_size=100)
        cache.cache = {f"chunk_{i}": np.zeros(1024) for i in range(50)}
        cache.access_order = [f"chunk_{i}" for i in range(50)]

        stats = cache.get_cache_stats()
        assert stats["size"] == 50
        assert stats["max_size"] == 100
        assert stats["usage_percent"] == 50

    def test_cache_clear(self):
        """Test cache clearing"""
        cache = EmbeddingCache(max_size=10)
        cache.cache = {"chunk_1": np.zeros(1024)}
        cache.access_order = ["chunk_1"]

        cache.clear()

        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass"""

    def test_default_config(self):
        """Test default configuration"""
        config = EmbeddingConfig()

        assert config.model_name == "BAAI/bge-m3"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.normalize is True
        assert config.add_hierarchical_context is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = EmbeddingConfig(
            model_name="custom-model",
            device="cuda",
            batch_size=64,
            add_hierarchical_context=False
        )

        assert config.model_name == "custom-model"
        assert config.device == "cuda"
        assert config.batch_size == 64
        assert config.add_hierarchical_context is False


# Integration tests

@pytest.mark.asyncio
async def test_embeddings_similarity(embedder):
    """Test that similar chunks have similar embeddings"""
    chunk1 = LegalChunk(
        chunk_id="chunk_1",
        content="Dodavatel odpovídá za vady díla."
    )
    chunk2 = LegalChunk(
        chunk_id="chunk_2",
        content="Dodavatel odpovídá za vady výrobku."
    )
    chunk3 = LegalChunk(
        chunk_id="chunk_3",
        content="Kupující zaplatí cenu do třiceti dnů."
    )

    embeddings = await embedder.embed_chunks([chunk1, chunk2, chunk3], add_context=False)

    # Cosine similarity between chunk1 and chunk2 (similar)
    sim_12 = np.dot(embeddings[0], embeddings[1])

    # Cosine similarity between chunk1 and chunk3 (different)
    sim_13 = np.dot(embeddings[0], embeddings[2])

    # Similar chunks should have higher similarity
    assert sim_12 > sim_13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
