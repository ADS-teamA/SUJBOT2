"""
Unit and integration tests for DuplicateDetector.

Tests document duplicate detection using semantic similarity.
"""

import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import numpy as np

from src.duplicate_detector import DuplicateDetector, DuplicateDetectionConfig


@contextmanager
def mock_fitz_module(mock_doc_config=None):
    """
    Context manager to mock fitz module for testing.

    Args:
        mock_doc_config: Dict with 'pages', 'text_per_page' for configuring mock document
    """
    mock_fitz = Mock()
    mock_doc = MagicMock()  # MagicMock needed for __len__ and __getitem__

    if mock_doc_config:
        # Setup mock document based on config
        pages_count = mock_doc_config.get('pages', 1)
        text_per_page = mock_doc_config.get('text_per_page', "Sample text")

        mock_doc.__len__.return_value = pages_count
        if isinstance(text_per_page, list):
            # Different text per page
            mock_pages = [Mock() for _ in text_per_page]
            for page, text in zip(mock_pages, text_per_page):
                page.get_text.return_value = text
            mock_doc.__getitem__.side_effect = mock_pages
        else:
            # Same text for all pages
            mock_page = Mock()
            mock_page.get_text.return_value = text_per_page
            mock_doc.__getitem__.return_value = mock_page

    mock_fitz.open.return_value = mock_doc

    with patch.dict('sys.modules', {'fitz': mock_fitz}):
        yield mock_fitz, mock_doc


@pytest.fixture
def duplicate_config():
    """Create standard duplicate detection config."""
    return DuplicateDetectionConfig(
        enabled=True,
        similarity_threshold=0.98,
        sample_pages=1,
        cache_size=100,
    )


@pytest.fixture
def mock_embedder():
    """Create mock embedding generator."""
    embedder = Mock()
    # Return consistent 3-dimensional embeddings
    embedder.embed_texts.return_value = [np.array([0.1, 0.2, 0.3])]
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = Mock()
    store.search_layer.return_value = []  # No results by default
    return store


def test_config_validation():
    """Test DuplicateDetectionConfig validation."""
    # Valid config
    config = DuplicateDetectionConfig(similarity_threshold=0.95, sample_pages=2)
    config.validate()  # Should not raise

    # Invalid threshold (> 1.0)
    with pytest.raises(ValueError, match="similarity_threshold must be 0-1"):
        config = DuplicateDetectionConfig(similarity_threshold=1.5)
        config.validate()

    # Invalid threshold (< 0.0)
    with pytest.raises(ValueError, match="similarity_threshold must be 0-1"):
        config = DuplicateDetectionConfig(similarity_threshold=-0.1)
        config.validate()

    # Invalid sample_pages
    with pytest.raises(ValueError, match="sample_pages must be >= 1"):
        config = DuplicateDetectionConfig(sample_pages=0)
        config.validate()


def test_detector_initialization(duplicate_config):
    """Test DuplicateDetector initialization."""
    detector = DuplicateDetector(duplicate_config, vector_store_path="test_path")

    assert detector.config.similarity_threshold == 0.98
    assert detector.config.sample_pages == 1
    assert detector.vector_store_path == "test_path"
    assert detector._embedder is None  # Lazy loading
    assert detector._vector_store is None  # Lazy loading


def test_check_duplicate_disabled():
    """Test that duplicate detection can be disabled."""
    config = DuplicateDetectionConfig(enabled=False)
    detector = DuplicateDetector(config)

    is_dup, sim, match = detector.check_duplicate("test.pdf")

    assert is_dup is False
    assert sim == 0.0
    assert match is None


def test_extract_text_sample(duplicate_config):
    """Test text extraction from PDF."""
    detector = DuplicateDetector(duplicate_config)

    # Mock fitz with 5-page document
    with mock_fitz_module({'pages': 5, 'text_per_page': "Sample text from page 1"}) as (mock_fitz, mock_doc):
        text = detector._extract_text_sample("test.pdf")

    assert text == "Sample text from page 1"
    mock_fitz.open.assert_called_once_with("test.pdf")
    mock_doc.close.assert_called_once()


def test_extract_text_multiple_pages():
    """Test text extraction from multiple pages."""
    config = DuplicateDetectionConfig(sample_pages=2)
    detector = DuplicateDetector(config)

    # Mock fitz with 5-page document, different text per page
    with mock_fitz_module({'pages': 5, 'text_per_page': ["Page 1 text", "Page 2 text"]}) as (mock_fitz, mock_doc):
        text = detector._extract_text_sample("test.pdf")

    assert text == "Page 1 text\nPage 2 text"


def test_compute_file_hash(duplicate_config, tmp_path):
    """Test file hash computation."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    detector = DuplicateDetector(duplicate_config)

    # Compute hash
    hash1 = detector._compute_file_hash(str(test_file))

    # Hash should be consistent
    hash2 = detector._compute_file_hash(str(test_file))
    assert hash1 == hash2

    # Different content should produce different hash
    test_file2 = tmp_path / "test2.txt"
    test_file2.write_text("different content")
    hash3 = detector._compute_file_hash(str(test_file2))
    assert hash1 != hash3


def test_check_duplicate_no_match(duplicate_config, mock_embedder, mock_vector_store):
    """Test duplicate check when no match found."""
    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    # No matches from vector store
    mock_vector_store.search_layer.return_value = []

    # Check duplicate with mocked fitz
    with mock_fitz_module({'pages': 1, 'text_per_page': "Sample document text"}):
        is_dup, sim, match = detector.check_duplicate("test.pdf")

    assert is_dup is False
    assert sim == 0.0
    assert match is None


def test_check_duplicate_below_threshold(
    duplicate_config, mock_embedder, mock_vector_store, tmp_path
):
    """Test duplicate check when similarity is below threshold."""
    # Create temp file (needed for file hashing)
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test content")

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    # Return match with similarity below threshold (0.95 < 0.98)
    mock_vector_store.search_layer.return_value = [
        {
            "score": 0.95,
            "metadata": {"document_id": "doc1"},
        }
    ]

    # Check duplicate with mocked fitz (text must be >= 100 chars)
    long_text = "Sample document text " * 6  # 120 chars
    with mock_fitz_module({'pages': 1, 'text_per_page': long_text}):
        is_dup, sim, match = detector.check_duplicate(str(test_file))

    assert is_dup is False
    assert sim == 0.95
    assert match == "doc1"


def test_check_duplicate_above_threshold(
    duplicate_config, mock_embedder, mock_vector_store, tmp_path
):
    """Test duplicate check when similarity is above threshold."""
    # Create temp file (needed for file hashing)
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test content")

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    # Return match with similarity above threshold (0.99 >= 0.98)
    mock_vector_store.search_layer.return_value = [
        {
            "score": 0.99,
            "metadata": {"document_id": "doc1"},
        }
    ]

    # Check duplicate with mocked fitz (text must be >= 100 chars)
    long_text = "Sample document text " * 6  # 120 chars
    with mock_fitz_module({'pages': 1, 'text_per_page': long_text}):
        is_dup, sim, match = detector.check_duplicate(str(test_file))

    assert is_dup is True
    assert sim == 0.99
    assert match == "doc1"


def test_check_duplicate_excludes_self(
    duplicate_config, mock_embedder, mock_vector_store, tmp_path
):
    """Test that duplicate check excludes the document itself."""
    # Create temp file (needed for file hashing)
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test content")

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    # Return self as match (should be excluded)
    mock_vector_store.search_layer.return_value = [
        {
            "score": 1.0,  # Perfect match
            "metadata": {"document_id": "test_doc"},
        },
        {
            "score": 0.85,  # Different document
            "metadata": {"document_id": "other_doc"},
        },
    ]

    # Check duplicate with mocked fitz (text must be >= 100 chars)
    long_text = "Sample document text " * 6  # 120 chars
    with mock_fitz_module({'pages': 1, 'text_per_page': long_text}):
        is_dup, sim, match = detector.check_duplicate(str(test_file), document_id="test_doc")

    # Should skip self and use second match
    assert is_dup is False  # 0.85 < 0.98
    assert sim == 0.85
    assert match == "other_doc"


def test_embedding_cache(duplicate_config, mock_embedder, mock_vector_store, tmp_path):
    """Test that embeddings are cached."""
    # Create real test file for hashing
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test content")

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store
    mock_vector_store.search_layer.return_value = []

    # Mock fitz for both calls (text must be >= 100 chars)
    long_text = "Sample text " * 10  # 120 chars
    with mock_fitz_module({'pages': 1, 'text_per_page': long_text}):
        # First check - should call embedder
        detector.check_duplicate(str(test_file))
        assert mock_embedder.embed_texts.call_count == 1

        # Second check on same file - should use cache
        detector.check_duplicate(str(test_file))
        assert mock_embedder.embed_texts.call_count == 1  # Not called again


def test_text_too_short(duplicate_config):
    """Test handling of documents with very short text."""
    detector = DuplicateDetector(duplicate_config)

    # Mock fitz with very short text (5 chars)
    with mock_fitz_module({'pages': 1, 'text_per_page': "short"}):
        is_dup, sim, match = detector.check_duplicate("test.pdf")

    # Should skip due to short text
    assert is_dup is False
    assert sim == 0.0
    assert match is None


def test_get_stats(duplicate_config):
    """Test getting detector statistics."""
    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")

    stats = detector.get_stats()

    assert stats["enabled"] is True
    assert stats["threshold"] == 0.98
    assert stats["cache_size"] == 0
    assert stats["cache_limit"] == 100
    assert stats["embedder_loaded"] is False
    assert stats["vector_store_loaded"] is False


def test_cache_eviction(duplicate_config, tmp_path):
    """Test that cache evicts oldest entries when full."""
    config = DuplicateDetectionConfig(cache_size=2)  # Small cache
    detector = DuplicateDetector(config)

    # Add 3 entries (should evict oldest)
    for i in range(3):
        file_hash = f"hash{i}"
        embedding = np.array([i, i, i])
        detector._cache_embedding(file_hash, embedding, f"doc{i}")

    # Only last 2 should remain
    assert len(detector._embedding_cache) == 2
    assert "hash0" not in detector._embedding_cache  # Evicted
    assert "hash1" in detector._embedding_cache
    assert "hash2" in detector._embedding_cache


def test_lazy_loading_embedder(duplicate_config):
    """Test that embedder is lazily loaded."""
    detector = DuplicateDetector(duplicate_config)

    assert detector._embedder is None

    # Access embedder (would load it) - patch where it's imported FROM
    with patch("src.embedding_generator.EmbeddingGenerator") as mock_gen:
        detector._get_embedder()
        mock_gen.assert_called_once()


def test_lazy_loading_vector_store(duplicate_config):
    """Test that vector store is lazily loaded."""
    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")

    assert detector._vector_store is None

    # Access vector store (would load it) - patch where it's imported FROM
    with patch("src.hybrid_search.HybridVectorStore") as mock_store:
        detector._get_vector_store()
        mock_store.load.assert_called_once_with("test_db")


def test_no_vector_store_path(duplicate_config, mock_embedder):
    """Test behavior when no vector store path is provided."""
    detector = DuplicateDetector(duplicate_config, vector_store_path=None)
    detector._embedder = mock_embedder

    # Get vector store should return None
    store = detector._get_vector_store()
    assert store is None

    # _find_similar_document should handle None gracefully
    is_dup, sim, match = detector._find_similar_document(
        embedding=np.array([0.1, 0.2, 0.3]),
        exclude_doc_id=None,
    )

    assert is_dup is False
    assert sim == 0.0
    assert match is None
