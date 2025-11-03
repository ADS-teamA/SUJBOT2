"""
Unit and integration tests for DuplicateDetector.

Tests document duplicate detection using semantic similarity.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.duplicate_detector import DuplicateDetectionConfig, DuplicateDetector


@contextmanager
def mock_fitz_module(mock_doc_config=None):
    """
    Context manager to mock fitz module for testing.

    This mocks the PyMuPDF (fitz) library which is used to extract text from PDFs.
    The mock supports configuring:
    - Number of pages in the document
    - Text content per page (same text for all pages or different per page)

    Args:
        mock_doc_config: Dict with 'pages', 'text_per_page' for configuring mock document
            Example: {'pages': 5, 'text_per_page': "Sample text"}
            Example: {'pages': 2, 'text_per_page': ["Page 1", "Page 2"]}
    """
    mock_fitz = Mock()
    mock_doc = MagicMock()  # MagicMock needed for __len__ and __getitem__ support

    if mock_doc_config:
        pages_count = mock_doc_config.get("pages", 1)
        text_per_page = mock_doc_config.get("text_per_page", "Sample text")

        # Configure document length
        mock_doc.__len__.return_value = pages_count

        if isinstance(text_per_page, list):
            # Different text per page - create separate mock page objects
            mock_pages = [Mock() for _ in text_per_page]
            for page, text in zip(mock_pages, text_per_page):
                page.get_text.return_value = text
            mock_doc.__getitem__.side_effect = mock_pages
        else:
            # Same text for all pages - reuse same mock page
            mock_page = Mock()
            mock_page.get_text.return_value = text_per_page
            mock_doc.__getitem__.return_value = mock_page

    mock_fitz.open.return_value = mock_doc

    with patch.dict("sys.modules", {"fitz": mock_fitz}):
        yield mock_fitz, mock_doc


@pytest.fixture
def duplicate_config() -> DuplicateDetectionConfig:
    """Create standard duplicate detection config."""
    return DuplicateDetectionConfig(
        enabled=True,
        similarity_threshold=0.98,
        sample_pages=1,
        cache_size=100,
    )


@pytest.fixture
def mock_embedder() -> Mock:
    """Create mock embedding generator."""
    embedder = Mock()
    # Return consistent 3-dimensional embeddings
    embedder.embed_texts.return_value = [np.array([0.1, 0.2, 0.3])]
    return embedder


@pytest.fixture
def mock_faiss_store() -> Mock:
    """
    Create mock FAISS store with all 3 layer search methods.

    The FAISS store is the underlying storage in HybridVectorStore
    and exposes search_layer1(), search_layer2(), and search_layer3() methods.
    Each returns a list of search results with 'score' and 'metadata' keys.
    """
    faiss_store = MagicMock()
    # Configure all 3 layers to return no results by default
    faiss_store.search_layer1.return_value = []
    faiss_store.search_layer2.return_value = []
    faiss_store.search_layer3.return_value = []
    return faiss_store


@pytest.fixture
def mock_vector_store(mock_faiss_store: Mock) -> Mock:
    """
    Create mock vector store (HybridVectorStore with FAISS store).

    The HybridVectorStore exposes a faiss_store attribute which is the
    underlying FAISS store used for searching embeddings.
    """
    store = Mock()
    store.faiss_store = mock_faiss_store
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
    with mock_fitz_module({"pages": 5, "text_per_page": "Sample text from page 1"}) as (
        mock_fitz,
        mock_doc,
    ):
        text = detector._extract_text_sample("test.pdf")

    assert text == "Sample text from page 1"
    mock_fitz.open.assert_called_once_with("test.pdf")
    mock_doc.close.assert_called_once()


def test_extract_text_multiple_pages():
    """Test text extraction from multiple pages."""
    config = DuplicateDetectionConfig(sample_pages=2)
    detector = DuplicateDetector(config)

    # Mock fitz with 5-page document, different text per page
    with mock_fitz_module({"pages": 5, "text_per_page": ["Page 1 text", "Page 2 text"]}) as (
        mock_fitz,
        mock_doc,
    ):
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


def _setup_detector_with_mocks(
    config: DuplicateDetectionConfig,
    embedder: Mock,
    vector_store: Mock,
    vector_store_path: str = "test_db",
) -> DuplicateDetector:
    """
    Setup DuplicateDetector with injected mock dependencies.

    Helper function to reduce boilerplate in tests. This is a common pattern
    where tests need to inject mocks for embedder and vector store.

    Args:
        config: Duplicate detection configuration
        embedder: Mock embedder instance
        vector_store: Mock vector store instance
        vector_store_path: Path to vector store (for logging)

    Returns:
        Configured DuplicateDetector instance with mocks injected
    """
    detector = DuplicateDetector(config, vector_store_path=vector_store_path)
    detector._embedder = embedder
    detector._vector_store = vector_store
    return detector


def test_check_duplicate_no_match(duplicate_config, mock_embedder, mock_vector_store):
    """Test duplicate check when no match found."""
    detector = _setup_detector_with_mocks(duplicate_config, mock_embedder, mock_vector_store)

    # No matches from vector store (already default in fixture)
    assert mock_vector_store.faiss_store.search_layer1.return_value == []

    # Check duplicate with mocked fitz
    with mock_fitz_module({"pages": 1, "text_per_page": "Sample document text"}):
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

    detector = _setup_detector_with_mocks(duplicate_config, mock_embedder, mock_vector_store)

    # Return match with similarity below threshold (0.95 < 0.98)
    # FAISS returns flat structure with document_id at top level
    mock_vector_store.faiss_store.search_layer1.return_value = [
        {
            "score": 0.95,
            "document_id": "doc1",
        }
    ]

    # Check duplicate with mocked fitz (text must be >= 100 chars)
    long_text = "Sample document text " * 6  # 120 chars
    with mock_fitz_module({"pages": 1, "text_per_page": long_text}):
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

    detector = _setup_detector_with_mocks(duplicate_config, mock_embedder, mock_vector_store)

    # Return match with similarity above threshold (0.99 >= 0.98)
    # FAISS returns flat structure with document_id at top level
    mock_vector_store.faiss_store.search_layer1.return_value = [
        {
            "score": 0.99,
            "document_id": "doc1",
        }
    ]

    # Check duplicate with mocked fitz (text must be >= 100 chars)
    long_text = "Sample document text " * 6  # 120 chars
    with mock_fitz_module({"pages": 1, "text_per_page": long_text}):
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

    detector = _setup_detector_with_mocks(duplicate_config, mock_embedder, mock_vector_store)

    # Return self as match (should be excluded by check_duplicate)
    # FAISS returns flat structure with document_id at top level
    mock_vector_store.faiss_store.search_layer1.return_value = [
        {
            "score": 1.0,  # Perfect match (self)
            "document_id": "test_doc",
        },
        {
            "score": 0.85,  # Different document
            "document_id": "other_doc",
        },
    ]

    # Check duplicate with mocked fitz (text must be >= 100 chars)
    long_text = "Sample document text " * 6  # 120 chars
    with mock_fitz_module({"pages": 1, "text_per_page": long_text}):
        is_dup, sim, match = detector.check_duplicate(str(test_file), document_id="test_doc")

    # Should skip self and use second match (0.85 < 0.98 threshold)
    assert is_dup is False
    assert sim == 0.85
    assert match == "other_doc"


def test_embedding_cache(duplicate_config, mock_embedder, mock_vector_store, tmp_path):
    """Test that embeddings are cached across multiple checks."""
    # Create real test file for hashing
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test content")

    detector = _setup_detector_with_mocks(duplicate_config, mock_embedder, mock_vector_store)

    # Mock fitz for both calls (text must be >= 100 chars)
    long_text = "Sample text " * 10  # 120 chars
    with mock_fitz_module({"pages": 1, "text_per_page": long_text}):
        # First check - should call embedder and cache result
        detector.check_duplicate(str(test_file))
        assert mock_embedder.embed_texts.call_count == 1

        # Second check on same file - should use cache (not call embedder again)
        detector.check_duplicate(str(test_file))
        assert mock_embedder.embed_texts.call_count == 1  # Still 1, not 2


def test_text_too_short(duplicate_config):
    """Test handling of documents with very short text."""
    detector = DuplicateDetector(duplicate_config)

    # Mock fitz with very short text (5 chars)
    with mock_fitz_module({"pages": 1, "text_per_page": "short"}):
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
