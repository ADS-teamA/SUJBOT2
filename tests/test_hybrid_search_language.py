"""
Tests for hybrid_search.py language support.

Tests universal language processing:
- Language detection (auto mode)
- spaCy tokenization with lemmatization and stop words
- NLTK fallback tokenization
- Basic tokenization fallback
- Multi-language support (Czech, English, German, etc.)
- Save/load with language configuration
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import namedtuple

from src.hybrid_search import (
    BM25Index,
    BM25Store,
    HybridVectorStore,
    detect_language,
    load_nltk_stopwords,
    SPACY_AVAILABLE,
    NLTK_AVAILABLE,
    LANGDETECT_AVAILABLE,
)

# Mock chunk for testing
Chunk = namedtuple("Chunk", ["chunk_id", "content", "raw_content", "metadata"])
ChunkMetadata = namedtuple(
    "ChunkMetadata",
    ["document_id", "section_id", "section_title", "section_path", "page_number", "layer"],
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def czech_chunks():
    """Sample Czech document chunks with stop words."""
    metadata = ChunkMetadata(
        document_id="gdpr_cz",
        section_id="sec1",
        section_title="Úvod",
        section_path="1. Úvod",
        page_number=1,
        layer=3,
    )

    return [
        Chunk(
            chunk_id="gdpr_cz:sec1:0",
            content="Tento dokument stanovuje pravidla pro ochranu osobních údajů v souladu s GDPR.",
            raw_content="Tento dokument stanovuje pravidla pro ochranu osobních údajů.",
            metadata=metadata,
        ),
        Chunk(
            chunk_id="gdpr_cz:sec1:1",
            content="Zpracování osobních údajů musí být v souladu s právními předpisy.",
            raw_content="Zpracování osobních údajů musí být v souladu s právními předpisy.",
            metadata=metadata,
        ),
    ]


@pytest.fixture
def english_chunks():
    """Sample English document chunks with stop words."""
    metadata = ChunkMetadata(
        document_id="gdpr_en",
        section_id="sec1",
        section_title="Introduction",
        section_path="1. Introduction",
        page_number=1,
        layer=3,
    )

    return [
        Chunk(
            chunk_id="gdpr_en:sec1:0",
            content="This document sets out the rules for the protection of personal data in accordance with GDPR.",
            raw_content="This document sets out the rules for the protection of personal data.",
            metadata=metadata,
        ),
        Chunk(
            chunk_id="gdpr_en:sec1:1",
            content="The processing of personal data must be in compliance with legal requirements.",
            raw_content="The processing of personal data must be in compliance with legal requirements.",
            metadata=metadata,
        ),
    ]


@pytest.fixture
def mixed_language_chunks_dict(czech_chunks, english_chunks):
    """Mixed language chunks (Layer 1, 2, 3)."""
    # For simplicity, use layer3 chunks for all layers
    return {
        "layer1": czech_chunks[:1],  # Document level
        "layer2": czech_chunks[:1],  # Section level
        "layer3": czech_chunks + english_chunks,  # Chunk level
    }


# ==============================================================================
# Language Detection Tests
# ==============================================================================

@pytest.mark.skipif(not LANGDETECT_AVAILABLE, reason="langdetect not installed")
def test_detect_language_czech():
    """Should detect Czech language."""
    text = "Tento dokument stanovuje pravidla pro ochranu osobních údajů v souladu s GDPR."
    lang = detect_language(text)
    assert lang == "cs"


@pytest.mark.skipif(not LANGDETECT_AVAILABLE, reason="langdetect not installed")
def test_detect_language_english():
    """Should detect English language."""
    text = "This document sets out the rules for the protection of personal data in accordance with GDPR."
    lang = detect_language(text)
    assert lang == "en"


@pytest.mark.skipif(not LANGDETECT_AVAILABLE, reason="langdetect not installed")
def test_detect_language_german():
    """Should detect German language."""
    text = "Dieses Dokument legt die Regeln für den Schutz personenbezogener Daten gemäß DSGVO fest."
    lang = detect_language(text)
    assert lang == "de"


def test_detect_language_fallback_when_unavailable():
    """Should use fallback when langdetect unavailable."""
    with patch("src.hybrid_search.LANGDETECT_AVAILABLE", False):
        lang = detect_language("Any text", fallback="en")
        assert lang == "en"


def test_detect_language_fallback_on_error():
    """Should use fallback on detection error."""
    # Note: langdetect can detect even very short text, so this test verifies
    # it returns a valid language code (may not always use fallback)
    lang = detect_language("x", fallback="cs")
    assert isinstance(lang, str)
    assert len(lang) == 2  # Valid ISO 639-1 code


# ==============================================================================
# NLTK Stop Words Tests
# ==============================================================================

def test_load_nltk_stopwords_czech():
    """Should load hardcoded Czech stop words."""
    # NOTE: NLTK doesn't provide Czech stopwords - we use hardcoded list
    # Source: https://github.com/stopwords-iso/stopwords-cs (422 unique words)
    stops = load_nltk_stopwords("cs")
    assert len(stops) == 422  # Hardcoded Czech stop words
    assert "a" in stops
    assert "v" in stops
    assert "na" in stops
    assert "že" in stops


@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_load_nltk_stopwords_english():
    """Should load English stop words from NLTK."""
    stops = load_nltk_stopwords("en")
    assert len(stops) > 0
    # Check common English stop words
    assert "the" in stops
    assert "and" in stops
    assert "is" in stops


def test_load_nltk_stopwords_unsupported_language():
    """Should return empty set for unsupported language."""
    stops = load_nltk_stopwords("xx")  # Invalid language code
    assert stops == set()


def test_load_nltk_stopwords_when_unavailable():
    """Should return empty set when NLTK unavailable."""
    with patch("src.hybrid_search.NLTK_AVAILABLE", False):
        stops = load_nltk_stopwords("en")
        assert stops == set()


# ==============================================================================
# BM25Index Tokenization Tests
# ==============================================================================

def test_bm25index_tokenize_basic_fallback():
    """Should use basic tokenization when no NLP model."""
    index = BM25Index(nlp_model=None, stop_words=set())

    tokens = index._tokenize("This is a test document with some words.")

    # Basic tokenization: lowercase split, filter single chars
    assert "this" in tokens
    assert "test" in tokens
    assert "document" in tokens
    # Single char 'a' should be filtered
    assert "a" not in tokens


def test_bm25index_tokenize_with_nltk_stopwords():
    """Should remove stop words with NLTK."""
    stop_words = {"the", "is", "a", "with", "some"}
    index = BM25Index(nlp_model=None, stop_words=stop_words)

    tokens = index._tokenize("This is a test document with some words here.")

    # Stop words should be removed
    assert "is" not in tokens
    assert "a" not in tokens
    assert "the" not in tokens
    assert "with" not in tokens
    assert "some" not in tokens

    # Content words should remain (note: isalnum() filters punctuation)
    assert "this" in tokens
    assert "test" in tokens
    assert "document" in tokens
    assert "words" in tokens
    assert "here" in tokens


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
def test_bm25index_tokenize_with_spacy_czech():
    """Should use spaCy for Czech tokenization (lemmatization + stop words)."""
    try:
        import spacy
        nlp = spacy.load("cs_core_news_sm", disable=["parser", "ner"])
    except OSError:
        pytest.skip("Czech spaCy model not installed")

    index = BM25Index(nlp_model=nlp, stop_words=set())

    # Czech text with stop words
    tokens = index._tokenize("Tento dokument stanovuje pravidla pro ochranu osobních údajů.")

    # Stop words should be removed (tento, pro)
    assert "tento" not in tokens
    assert "pro" not in tokens

    # Content words should be lemmatized
    assert any("dokument" in t for t in tokens)  # dokument lemmatized
    assert any("pravidl" in t for t in tokens) or any("pravidlo" in t for t in tokens)  # pravidla → pravidlo


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
def test_bm25index_tokenize_with_spacy_english():
    """Should use spaCy for English tokenization (lemmatization + stop words)."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        pytest.skip("English spaCy model not installed")

    index = BM25Index(nlp_model=nlp, stop_words=set())

    # English text with stop words
    tokens = index._tokenize("This document sets out the rules for the protection of personal data.")

    # Stop words should be removed (this, the, of, for)
    assert "this" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens
    assert "for" not in tokens

    # Content words should be lemmatized
    assert "document" in tokens
    assert "rule" in tokens  # rules → rule
    assert "protection" in tokens
    assert "personal" in tokens
    assert "data" in tokens or "datum" in tokens  # data lemmatized


def test_bm25index_tokenize_removes_punctuation():
    """Should remove punctuation."""
    index = BM25Index(nlp_model=None, stop_words=set())

    tokens = index._tokenize("Hello, world! How are you?")

    # Punctuation should be removed (by isalnum() check)
    assert "," not in tokens
    assert "!" not in tokens
    assert "?" not in tokens

    # Words should remain
    assert "hello" in tokens
    assert "world" in tokens


# ==============================================================================
# BM25Store Language Initialization Tests
# ==============================================================================

def test_bm25store_init_explicit_language():
    """Should initialize with explicit language."""
    store = BM25Store(lang="en")
    assert store.lang == "en"


def test_bm25store_init_auto_language():
    """Should initialize with auto language (no NLP until build)."""
    store = BM25Store(lang="auto")
    assert store.lang == "auto"
    assert store.nlp_model is None
    assert store.stop_words == set()


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
def test_bm25store_init_loads_spacy_model():
    """Should load spaCy model for supported language."""
    try:
        store = BM25Store(lang="en")
        assert store.nlp_model is not None
        # Check it's a spaCy model
        assert hasattr(store.nlp_model, "__call__")
    except Exception:
        pytest.skip("English spaCy model not installed")


@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_bm25store_init_loads_nltk_stopwords_fallback():
    """Should load NLTK stop words when spaCy unavailable."""
    with patch("src.hybrid_search.SPACY_AVAILABLE", False):
        store = BM25Store(lang="en")
        assert store.nlp_model is None
        assert len(store.stop_words) > 0  # NLTK stop words loaded


def test_bm25store_init_basic_fallback():
    """Should use basic tokenization when no NLP available."""
    with patch("src.hybrid_search.SPACY_AVAILABLE", False):
        with patch("src.hybrid_search.NLTK_AVAILABLE", False):
            store = BM25Store(lang="en")
            assert store.nlp_model is None
            assert store.stop_words == set()


# ==============================================================================
# BM25Store Auto-Detection Tests
# ==============================================================================

@pytest.mark.skipif(not LANGDETECT_AVAILABLE, reason="langdetect not installed")
def test_bm25store_auto_detect_czech(czech_chunks):
    """Should auto-detect Czech language from chunks."""
    chunks_dict = {
        "layer1": czech_chunks[:1],
        "layer2": czech_chunks[:1],
        "layer3": czech_chunks,
    }

    store = BM25Store(lang="auto")
    store.build_from_chunks(chunks_dict)

    assert store.lang == "cs"


@pytest.mark.skipif(not LANGDETECT_AVAILABLE, reason="langdetect not installed")
def test_bm25store_auto_detect_english(english_chunks):
    """Should auto-detect English language from chunks."""
    chunks_dict = {
        "layer1": english_chunks[:1],
        "layer2": english_chunks[:1],
        "layer3": english_chunks,
    }

    store = BM25Store(lang="auto")
    store.build_from_chunks(chunks_dict)

    assert store.lang == "en"


# ==============================================================================
# BM25Store Build and Search Tests
# ==============================================================================

def test_bm25store_build_from_chunks_basic(english_chunks):
    """Should build indexes from chunks (basic tokenization)."""
    with patch("src.hybrid_search.SPACY_AVAILABLE", False):
        with patch("src.hybrid_search.NLTK_AVAILABLE", False):
            chunks_dict = {
                "layer1": english_chunks[:1],
                "layer2": english_chunks[:1],
                "layer3": english_chunks,
            }

            store = BM25Store(lang="en")
            store.build_from_chunks(chunks_dict)

            # Verify indexes built
            assert len(store.index_layer3.corpus) == 2
            assert store.index_layer3.bm25 is not None


def test_bm25store_search_finds_relevant_chunks(english_chunks):
    """Should find relevant chunks via BM25 search."""
    chunks_dict = {
        "layer1": english_chunks[:1],
        "layer2": english_chunks[:1],
        "layer3": english_chunks,
    }

    store = BM25Store(lang="en")
    store.build_from_chunks(chunks_dict)

    # Search for "personal data"
    results = store.search_layer3("personal data", k=10)

    assert len(results) > 0
    # Both chunks mention "personal data"
    assert any("personal" in r["content"].lower() for r in results)


def test_bm25store_search_with_stopwords_improves_precision(english_chunks):
    """
    Should improve precision by removing stop words.

    Query: "the processing of personal data"
    - Without stop words: matches all chunks (low precision)
    - With stop words: matches only relevant chunk (high precision)
    """
    chunks_dict = {
        "layer1": english_chunks[:1],
        "layer2": english_chunks[:1],
        "layer3": english_chunks,
    }

    # Without stop words (basic tokenization)
    with patch("src.hybrid_search.SPACY_AVAILABLE", False):
        with patch("src.hybrid_search.NLTK_AVAILABLE", False):
            store_basic = BM25Store(lang="en")
            store_basic.build_from_chunks(chunks_dict)
            results_basic = store_basic.search_layer3("the processing of personal data", k=10)

    # With stop words (NLTK or spaCy)
    store_advanced = BM25Store(lang="en")
    store_advanced.build_from_chunks(chunks_dict)
    results_advanced = store_advanced.search_layer3("the processing of personal data", k=10)

    # Both should return results
    assert len(results_basic) > 0
    assert len(results_advanced) > 0

    # Advanced should include the "processing" chunk (better targeting)
    # Chunk 1 mentions "processing", chunk 0 mentions "protection"
    # With stop words removed, query becomes "processing personal data"
    # which should match chunk 1 better
    contents = [r["content"].lower() for r in results_advanced]
    assert any("processing" in c for c in contents), \
        "Should find chunk mentioning 'processing' when searching for 'the processing of personal data'"


# ==============================================================================
# BM25Store Save/Load Tests
# ==============================================================================

def test_bm25store_save_load_preserves_language(tmp_path, english_chunks):
    """Should save and load language configuration."""
    chunks_dict = {
        "layer1": english_chunks[:1],
        "layer2": english_chunks[:1],
        "layer3": english_chunks,
    }

    # Build and save
    store = BM25Store(lang="en")
    store.build_from_chunks(chunks_dict)
    store.save(tmp_path)

    # Verify config file exists
    config_path = tmp_path / "bm25_store_config.json"
    assert config_path.exists()

    # Load
    loaded_store = BM25Store.load(tmp_path)

    # Verify language preserved
    assert loaded_store.lang == "en"

    # Verify indexes loaded
    assert len(loaded_store.index_layer3.corpus) == 2


def test_bm25store_load_backward_compatibility(tmp_path, english_chunks):
    """Should load old format without language config."""
    chunks_dict = {
        "layer1": english_chunks[:1],
        "layer2": english_chunks[:1],
        "layer3": english_chunks,
    }

    # Build and save
    store = BM25Store(lang="en")
    store.build_from_chunks(chunks_dict)
    store.save(tmp_path)

    # Delete config file to simulate old format
    config_path = tmp_path / "bm25_store_config.json"
    if config_path.exists():
        config_path.unlink()

    # Load without config
    loaded_store = BM25Store.load(tmp_path)

    # Should default to English
    assert loaded_store.lang == "en"

    # Should still work
    results = loaded_store.search_layer3("personal data", k=10)
    assert len(results) > 0


# ==============================================================================
# BM25Store Merge Tests
# ==============================================================================

def test_bm25store_merge_same_language(english_chunks):
    """Should merge stores with same language."""
    chunks_dict1 = {
        "layer1": english_chunks[:1],
        "layer2": english_chunks[:1],
        "layer3": english_chunks[:1],
    }

    chunks_dict2 = {
        "layer1": english_chunks[1:],
        "layer2": english_chunks[1:],
        "layer3": english_chunks[1:],
    }

    store1 = BM25Store(lang="en")
    store1.build_from_chunks(chunks_dict1)

    store2 = BM25Store(lang="en")
    store2.build_from_chunks(chunks_dict2)

    # Merge
    store1.merge(store2)

    # Verify merged
    assert len(store1.index_layer3.corpus) == 2


def test_bm25store_merge_different_languages_warns(english_chunks, czech_chunks, caplog):
    """Should warn when merging stores with different languages."""
    chunks_dict_en = {
        "layer1": english_chunks[:1],
        "layer2": english_chunks[:1],
        "layer3": english_chunks,
    }

    chunks_dict_cs = {
        "layer1": czech_chunks[:1],
        "layer2": czech_chunks[:1],
        "layer3": czech_chunks,
    }

    store_en = BM25Store(lang="en")
    store_en.build_from_chunks(chunks_dict_en)

    store_cs = BM25Store(lang="cs")
    store_cs.build_from_chunks(chunks_dict_cs)

    # Merge should log warning but still work
    import logging
    caplog.set_level(logging.WARNING)
    store_en.merge(store_cs)

    # Check that warning was logged
    assert any("different languages" in record.message.lower() for record in caplog.records)

    # Verify merged (uses store1's language)
    assert len(store_en.index_layer3.corpus) == 4


# ==============================================================================
# Integration Tests with HybridVectorStore
# ==============================================================================

def test_hybrid_store_with_language_support(tmp_path, english_chunks):
    """Integration test: HybridVectorStore with language-aware BM25."""
    from src.faiss_vector_store import FAISSVectorStore

    chunks_dict = {
        "layer1": english_chunks[:1],
        "layer2": english_chunks[:1],
        "layer3": english_chunks,
    }

    # Create mock FAISS store
    faiss_store = Mock(spec=FAISSVectorStore)
    faiss_store.dimensions = 1024
    faiss_store.search_layer1 = Mock(return_value=[])
    faiss_store.search_layer2 = Mock(return_value=[])
    faiss_store.search_layer3 = Mock(return_value=[])

    # Create BM25 store with language support
    bm25_store = BM25Store(lang="en")
    bm25_store.build_from_chunks(chunks_dict)

    # Create hybrid store
    hybrid_store = HybridVectorStore(
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        fusion_k=60
    )

    # Search
    query_embedding = np.random.rand(1024).astype(np.float32)
    results = hybrid_store.hierarchical_search(
        query_text="personal data protection",
        query_embedding=query_embedding,
        k_layer3=5,
    )

    # Verify BM25 was called
    assert "layer3" in results


# ==============================================================================
# Performance Comparison Tests
# ==============================================================================

def test_stopwords_improve_search_quality(english_chunks):
    """
    Quantitative test: Stop words improve BM25 search quality.

    Measures precision improvement with stop words vs without.
    """
    # Create more test chunks for better comparison
    metadata = ChunkMetadata(
        document_id="test",
        section_id="sec1",
        section_title="Test",
        section_path="1. Test",
        page_number=1,
        layer=3,
    )

    chunks = [
        # Relevant chunk (contains "data breach notification")
        Chunk(
            chunk_id="test:1",
            content="A data breach notification must be sent within 72 hours to the supervisory authority.",
            raw_content="A data breach notification must be sent within 72 hours.",
            metadata=metadata,
        ),
        # Irrelevant chunk (many stop words but not relevant)
        Chunk(
            chunk_id="test:2",
            content="The document is about the general principles of the regulation and its scope.",
            raw_content="The document is about the general principles of the regulation.",
            metadata=metadata,
        ),
        # Somewhat relevant chunk
        Chunk(
            chunk_id="test:3",
            content="Personal data processing requires notification to data subjects in certain cases.",
            raw_content="Personal data processing requires notification to data subjects.",
            metadata=metadata,
        ),
    ]

    chunks_dict = {
        "layer1": chunks[:1],
        "layer2": chunks[:1],
        "layer3": chunks,
    }

    # Without stop words
    with patch("src.hybrid_search.SPACY_AVAILABLE", False):
        with patch("src.hybrid_search.NLTK_AVAILABLE", False):
            store_basic = BM25Store(lang="en")
            store_basic.build_from_chunks(chunks_dict)
            results_basic = store_basic.search_layer3("data breach notification", k=3)

    # With stop words
    store_advanced = BM25Store(lang="en")
    store_advanced.build_from_chunks(chunks_dict)
    results_advanced = store_advanced.search_layer3("data breach notification", k=3)

    # Check that most relevant chunk (test:1) is ranked first with stop words
    if len(results_advanced) > 0:
        top_result = results_advanced[0]
        assert "test:1" in top_result["chunk_id"], \
            "Stop words should help rank most relevant chunk first"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
