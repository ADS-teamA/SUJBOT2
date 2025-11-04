"""
Integration test for DuplicateDetector metadata structure fix.

This test validates the critical fix: FAISS returns flat structure with
document_id at top level, NOT nested inside a "metadata" key.
"""

import pytest
from unittest.mock import Mock

from src.duplicate_detector import DuplicateDetectionConfig, DuplicateDetector


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return DuplicateDetectionConfig(
        enabled=True,
        similarity_threshold=0.90,
        sample_pages=1,
        cache_size=100,
    )


def test_nested_metadata_would_fail(integration_config, tmp_path):
    """
    CRITICAL INTEGRATION TEST: Verify that nested metadata structure fails.

    This test proves that the fix was necessary - if FAISS returned nested
    metadata, the NEW code (using flat access) would fail to extract document_id.

    This is a regression test to prevent reverting to the buggy nested structure.
    """
    detector = DuplicateDetector(
        config=integration_config,
        vector_store_path="test_db"
    )

    import numpy as np
    mock_embedder = Mock()
    mock_embedder.embed_texts.return_value = [np.array([0.1, 0.2, 0.3])]

    mock_vector_store = Mock()
    mock_faiss_store = Mock()

    # Simulate OLD nested structure (what the code INCORRECTLY expected before)
    mock_faiss_store.search_layer1.return_value = [
        {
            "score": 0.95,
            "metadata": {"document_id": "doc1"},  # NESTED structure (OLD/WRONG)
        }
    ]

    mock_vector_store.faiss_store = mock_faiss_store

    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    test_file = tmp_path / "test.pdf"
    test_file.write_text("test content")

    # Mock PyMuPDF
    import sys
    from unittest.mock import MagicMock

    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_page = MagicMock()

    long_text = "Sample document text " * 6
    mock_page.get_text.return_value = long_text
    mock_doc.__len__.return_value = 1
    mock_doc.__iter__.return_value = iter([mock_page])
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value.__enter__.return_value = mock_doc

    sys.modules["fitz"] = mock_fitz

    try:
        is_duplicate, similarity, matched_doc = detector.check_duplicate(
            str(test_file),
            document_id="test_doc"
        )

        # With nested structure and NEW code (flat access), doc_id extraction fails
        # result.get("document_id") returns None when structure is nested
        assert matched_doc is None, \
            "REGRESSION: Nested structure should fail with new flat-access code!"

    finally:
        if "fitz" in sys.modules:
            del sys.modules["fitz"]
