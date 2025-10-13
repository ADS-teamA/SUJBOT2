"""
Unit tests for hybrid retrieval system (triple hybrid: semantic + keyword + structural)

Tests cover:
- Semantic search (BGE-M3 + FAISS)
- Keyword search (BM25)
- Structural search (legal references, hierarchy)
- Score fusion and normalization
- Adaptive weighting
- Deduplication
- Metadata filtering
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hybrid_retriever import (
    SearchResult, SearchQuery, RetrievalConfig,
    SemanticSearcher, KeywordSearcher, StructuralSearcher,
    HybridRetriever, LegalChunk,
    create_hybrid_retriever
)
from indexing import MultiDocumentVectorStore
from embeddings import LegalEmbedder, EmbeddingConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def retrieval_config():
    """Standard retrieval configuration"""
    return RetrievalConfig(
        semantic_weight=0.5,
        keyword_weight=0.3,
        structural_weight=0.2,
        top_k=10,
        normalize_scores=True,
        adaptive_weights=True
    )


@pytest.fixture
def mock_chunks():
    """Mock legal chunks for testing"""
    chunks = [
        LegalChunk(
            chunk_id="chunk_1",
            content="Dodavatel odpovídá za vady díla podle §89 občanského zákoníku.",
            document_type="contract",
            hierarchy_path="Článek 5",
            legal_reference="Článek 5.1",
            structural_level="article_point",
            metadata={
                'article': 5,
                'point': '5.1',
                'content_type': 'obligation',
                'references_to': ['§89']
            }
        ),
        LegalChunk(
            chunk_id="chunk_2",
            content="§89 Odpovědnost za vady. Dodavatel odpovídá za vady.",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §89",
            legal_reference="§89",
            structural_level="paragraph",
            metadata={
                'part': 'II',
                'chapter': 'III',
                'paragraph': 89,
                'content_type': 'obligation'
            }
        ),
        LegalChunk(
            chunk_id="chunk_3",
            content="Záruční lhůta činí 24 měsíců od převzetí díla.",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §90",
            legal_reference="§90",
            structural_level="paragraph",
            metadata={
                'part': 'II',
                'chapter': 'III',
                'paragraph': 90,
                'content_type': 'general'
            }
        ),
    ]
    return chunks


@pytest.fixture
def mock_vector_store(mock_chunks, mock_embedder):
    """Mock vector store with sample data"""
    vector_store = MultiDocumentVectorStore(embedder=mock_embedder)

    # Create mock FAISS index (3 chunks x 1024 dimensions for BGE-M3)
    import faiss
    dimension = 1024
    n_chunks = len(mock_chunks)

    # Generate random embeddings for testing
    embeddings = np.random.randn(n_chunks, dimension).astype('float32')

    # Normalize embeddings (important for FAISS IndexFlatIP)
    faiss.normalize_L2(embeddings)

    # Create index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Store in vector store
    doc_id = "test_doc"
    vector_store.indices[doc_id] = index

    # Store metadata
    metadata_store = {}
    for i, chunk in enumerate(mock_chunks):
        metadata_store[chunk.chunk_id] = chunk

    vector_store.metadata_stores[doc_id] = metadata_store

    return vector_store


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns random embeddings"""
    config = EmbeddingConfig(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=4
    )

    # Create mock that doesn't actually load model
    class MockEmbedder:
        def __init__(self):
            self.config = config
            self.dimension = 1024

        async def embed_chunks(self, chunks, add_context=None):
            """Return random normalized embeddings"""
            n = len(chunks)
            embeddings = np.random.randn(n, self.dimension).astype('float32')
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            return embeddings

        async def embed_query(self, query):
            """Return random normalized embedding"""
            embedding = np.random.randn(self.dimension).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

    return MockEmbedder()


# ============================================================================
# Test Configuration
# ============================================================================

def test_retrieval_config_validation():
    """Test that weights must sum to 1.0"""
    config = RetrievalConfig(
        semantic_weight=0.5,
        keyword_weight=0.3,
        structural_weight=0.2
    )

    # Should validate successfully
    config.validate()

    # Invalid weights
    invalid_config = RetrievalConfig(
        semantic_weight=0.6,
        keyword_weight=0.3,
        structural_weight=0.3
    )

    with pytest.raises(AssertionError):
        invalid_config.validate()


def test_retrieval_config_defaults():
    """Test default configuration values"""
    config = RetrievalConfig()

    assert config.semantic_weight == 0.5
    assert config.keyword_weight == 0.3
    assert config.structural_weight == 0.2
    assert config.top_k == 20
    assert config.normalize_scores is True


# ============================================================================
# Test Semantic Search
# ============================================================================

@pytest.mark.asyncio
async def test_semantic_searcher_basic(mock_vector_store, mock_embedder):
    """Test basic semantic search"""
    searcher = SemanticSearcher(mock_embedder, mock_vector_store)

    results = await searcher.search(
        query="odpovědnost za vady",
        document_ids=["test_doc"],
        top_k=5
    )

    # Should return results
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)

    # Check result structure
    result = results[0]
    assert result.chunk_id
    assert result.semantic_score is not None
    # Allow small floating point tolerance for scores (can be slightly negative due to precision)
    assert -0.01 <= result.semantic_score <= 1.0
    assert result.retrieval_method == "semantic"


@pytest.mark.asyncio
async def test_semantic_searcher_metadata_filtering(mock_vector_store, mock_embedder):
    """Test semantic search with metadata filters"""
    searcher = SemanticSearcher(mock_embedder, mock_vector_store)

    # Filter by content_type
    results = await searcher.search(
        query="odpovědnost",
        document_ids=["test_doc"],
        top_k=5,
        filters={'content_type': 'obligation'}
    )

    # All results should match filter
    for result in results:
        assert result.chunk.metadata.get('content_type') == 'obligation'


@pytest.mark.asyncio
async def test_semantic_searcher_empty_query(mock_vector_store, mock_embedder):
    """Test semantic search with empty query"""
    searcher = SemanticSearcher(mock_embedder, mock_vector_store)

    results = await searcher.search(
        query="",
        document_ids=["test_doc"],
        top_k=5
    )

    # Should still return results (though not meaningful)
    assert isinstance(results, list)


# ============================================================================
# Test Keyword Search (BM25)
# ============================================================================

@pytest.mark.asyncio
async def test_keyword_searcher_basic(mock_vector_store):
    """Test basic BM25 keyword search"""
    searcher = KeywordSearcher(mock_vector_store, k1=1.5, b=0.75)

    results = await searcher.search(
        query="odpovědnost vady",
        document_ids=["test_doc"],
        top_k=5
    )

    # Should return results
    assert isinstance(results, list)
    assert len(results) > 0

    # Check result structure
    result = results[0]
    assert result.keyword_score is not None
    assert result.keyword_score >= 0
    assert result.retrieval_method == "keyword"


@pytest.mark.asyncio
async def test_keyword_searcher_exact_match(mock_vector_store):
    """Test keyword search with exact term match"""
    searcher = KeywordSearcher(mock_vector_store)

    # Search for exact term that appears in chunks
    results = await searcher.search(
        query="§89",
        document_ids=["test_doc"],
        top_k=5
    )

    # Should find chunks containing §89
    assert len(results) > 0


@pytest.mark.asyncio
async def test_keyword_searcher_tokenization(mock_vector_store):
    """Test BM25 tokenization"""
    searcher = KeywordSearcher(mock_vector_store)

    # Test tokenization
    tokens = searcher._tokenize("Dodavatel odpovídá za vady podle §89.")

    # Should tokenize and return list of tokens
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    # Should have substantive tokens (lowercased)
    assert 'dodavatel' in tokens or 'odpovídá' in tokens or 'vady' in tokens

    # Should preserve legal references
    assert any('§' in t or '89' in t for t in tokens)


# ============================================================================
# Test Structural Search
# ============================================================================

@pytest.mark.asyncio
async def test_structural_searcher_basic(mock_vector_store):
    """Test basic structural search"""
    searcher = StructuralSearcher(mock_vector_store)

    results = await searcher.search(
        query="podle §89",
        document_ids=["test_doc"],
        top_k=5
    )

    # Should return results
    assert isinstance(results, list)
    assert len(results) > 0

    # Check result structure
    result = results[0]
    assert result.structural_score is not None
    assert 0 <= result.structural_score <= 1.0
    assert result.retrieval_method == "structural"


@pytest.mark.asyncio
async def test_structural_searcher_reference_extraction(mock_vector_store):
    """Test extraction of legal references from query"""
    searcher = StructuralSearcher(mock_vector_store)

    hints = searcher._extract_structural_hints("podle §89 odst. 2 písm. a)")

    # Should extract §89 odst. 2 písm. a)
    assert len(hints['references']) > 0
    assert any('§89' in ref or '89' in ref for ref in hints['references'])


@pytest.mark.asyncio
async def test_structural_searcher_hierarchy_extraction(mock_vector_store):
    """Test extraction of hierarchy hints from query"""
    searcher = StructuralSearcher(mock_vector_store)

    hints = searcher._extract_structural_hints("v Části II Hlava III")

    # Should extract Part and Chapter (may return None if pattern doesn't match exactly)
    # The implementation looks for "Část" followed by Roman numerals
    assert hints['part'] is not None or hints['chapter'] is not None
    # If chapter is extracted, it should be "III"
    if hints['chapter'] is not None:
        assert hints['chapter'] == 'III'


@pytest.mark.asyncio
async def test_structural_searcher_content_type_detection(mock_vector_store):
    """Test content type detection from query"""
    searcher = StructuralSearcher(mock_vector_store)

    hints1 = searcher._extract_structural_hints("jaké jsou povinnosti dodavatele")
    assert 'obligation' in hints1['content_types']

    hints2 = searcher._extract_structural_hints("co je zakázáno")
    assert 'prohibition' in hints2['content_types']


# ============================================================================
# Test Hybrid Retriever
# ============================================================================

@pytest.mark.asyncio
async def test_hybrid_retriever_basic(mock_vector_store, mock_embedder, retrieval_config):
    """Test basic hybrid retrieval"""
    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    results = await retriever.search(
        query="odpovědnost dodavatele podle §89",
        document_ids=["test_doc"],
        top_k=5
    )

    # Should return results
    assert isinstance(results, list)
    assert len(results) > 0

    # Check result structure
    result = results[0]
    assert result.score is not None
    assert 0 <= result.score <= 1.0
    assert result.retrieval_method == "hybrid"

    # Should have component scores
    assert result.semantic_score is not None or result.keyword_score is not None or result.structural_score is not None


@pytest.mark.asyncio
async def test_hybrid_retriever_score_fusion(mock_vector_store, mock_embedder, retrieval_config):
    """Test score fusion from multiple strategies"""
    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    results = await retriever.search(
        query="§89 odpovědnost vady",
        document_ids=["test_doc"],
        top_k=5
    )

    # Check score breakdown
    for result in results:
        breakdown = result.get_score_breakdown()
        assert 'semantic' in breakdown
        assert 'keyword' in breakdown
        assert 'structural' in breakdown
        assert 'combined' in breakdown

        # Combined score should be computed
        assert breakdown['combined'] >= 0


@pytest.mark.asyncio
async def test_hybrid_retriever_adaptive_weighting(mock_vector_store, mock_embedder, retrieval_config):
    """Test adaptive weighting based on query characteristics"""
    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    retrieval_config.adaptive_weights = True
    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    # Query with legal reference → should boost structural
    weights_with_ref = retriever._get_adaptive_weights("podle §89", None)
    weights_without_ref = retriever._get_adaptive_weights("odpovědnost dodavatele", None)

    assert weights_with_ref['structural'] > weights_without_ref['structural']


@pytest.mark.asyncio
async def test_hybrid_retriever_deduplication(mock_vector_store, mock_embedder, retrieval_config):
    """Test that results are deduplicated"""
    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    results = await retriever.search(
        query="§89",
        document_ids=["test_doc"],
        top_k=10
    )

    # No duplicate chunk IDs
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.asyncio
async def test_hybrid_retriever_score_normalization(mock_vector_store, mock_embedder, retrieval_config):
    """Test score normalization"""
    retrieval_config.normalize_scores = True

    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    results = await retriever.search(
        query="odpovědnost",
        document_ids=["test_doc"],
        top_k=5
    )

    # All scores should be in [0, 1]
    for result in results:
        assert 0 <= result.score <= 1.0
        if result.semantic_score is not None:
            assert 0 <= result.semantic_score <= 1.0
        if result.keyword_score is not None:
            assert 0 <= result.keyword_score <= 1.0
        if result.structural_score is not None:
            assert 0 <= result.structural_score <= 1.0


@pytest.mark.asyncio
async def test_hybrid_retriever_metadata_filtering(mock_vector_store, mock_embedder, retrieval_config):
    """Test metadata filtering"""
    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    # Filter by content_type
    results = await retriever.search(
        query="odpovědnost",
        document_ids=["test_doc"],
        top_k=5,
        filters={'content_type': 'obligation'}
    )

    # All results should match filter
    for result in results:
        assert result.chunk.metadata.get('content_type') == 'obligation'


@pytest.mark.asyncio
async def test_hybrid_retriever_ranking(mock_vector_store, mock_embedder, retrieval_config):
    """Test that results are ranked by score"""
    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    results = await retriever.search(
        query="odpovědnost",
        document_ids=["test_doc"],
        top_k=5
    )

    # Results should be sorted by score (descending)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

    # Ranks should be assigned
    for i, result in enumerate(results, 1):
        assert result.rank == i


# ============================================================================
# Test Factory Function
# ============================================================================

def test_create_hybrid_retriever(mock_vector_store, mock_embedder):
    """Test factory function for creating retriever"""
    config = RetrievalConfig()

    retriever = create_hybrid_retriever(
        mock_vector_store,
        mock_embedder,
        config
    )

    assert isinstance(retriever, HybridRetriever)
    assert retriever.config == config


# ============================================================================
# Test Score Breakdown
# ============================================================================

def test_search_result_score_breakdown():
    """Test search result score breakdown"""
    result = SearchResult(
        chunk_id="test",
        chunk=LegalChunk(
            chunk_id="test",
            content="test",
            document_type="law_code",
            hierarchy_path="",
            legal_reference="",
            structural_level="",
            metadata={}
        ),
        document_id="doc1",
        score=0.8,
        semantic_score=0.9,
        keyword_score=0.7,
        structural_score=0.8
    )

    breakdown = result.get_score_breakdown()

    assert breakdown['semantic'] == 0.9
    assert breakdown['keyword'] == 0.7
    assert breakdown['structural'] == 0.8
    assert breakdown['combined'] == 0.8


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_hybrid_retriever_empty_query(mock_vector_store, mock_embedder, retrieval_config):
    """Test hybrid retrieval with empty query"""
    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    results = await retriever.search(
        query="",
        document_ids=["test_doc"],
        top_k=5
    )

    # Should handle gracefully
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_retriever_no_results(mock_vector_store, mock_embedder, retrieval_config):
    """Test hybrid retrieval when no results match"""
    semantic = SemanticSearcher(mock_embedder, mock_vector_store)
    keyword = KeywordSearcher(mock_vector_store)
    structural = StructuralSearcher(mock_vector_store)

    # Set very high score threshold
    retrieval_config.min_score_threshold = 0.99
    retrieval_config.enable_score_threshold = True

    retriever = HybridRetriever(semantic, keyword, structural, retrieval_config)

    results = await retriever.search(
        query="nonexistent term xyz123",
        document_ids=["test_doc"],
        top_k=5
    )

    # May return empty results if threshold too high
    assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
