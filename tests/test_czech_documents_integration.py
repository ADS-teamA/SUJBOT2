"""
Integration test for Czech document processing with stop words.

Tests that Czech language support works correctly:
1. Language detection identifies Czech text
2. Czech stop words are loaded (422 words)
3. Tokenization filters Czech stop words correctly
4. BM25 tokenization improves search precision
"""

import pytest

from src.hybrid_search import (
    BM25Index,
    CZECH_STOP_WORDS,
    detect_language,
    load_nltk_stopwords
)


def test_language_detection_on_czech_document():
    """Test that Czech language is properly detected."""
    # Sample Czech text from legal domain
    czech_text = "Tento zákon upravuje zpracování osobních údajů a ochranu práv fyzických osob."

    # Detect language
    detected_lang = detect_language(czech_text)

    # Should detect Czech
    assert detected_lang == "cs"


def test_czech_stop_words_are_loaded():
    """Test that Czech stop words are available and comprehensive."""
    # Load Czech stop words
    czech_stops = load_nltk_stopwords("cs")

    # Should have 422 Czech stop words
    assert len(czech_stops) == 422

    # Check common Czech stop words
    common_stops = {"a", "v", "na", "že", "je", "to", "se", "s"}
    assert common_stops.issubset(czech_stops)

    # Verify it matches the hardcoded set
    assert czech_stops == CZECH_STOP_WORDS


def test_czech_tokenization_filters_stop_words():
    """Test that Czech stop words are filtered during tokenization."""
    # Load Czech stop words
    czech_stops = load_nltk_stopwords("cs")

    # Create BM25Index with Czech stop words
    bm25_index = BM25Index(nlp_model=None, stop_words=czech_stops)

    # Query with ONLY stop words
    query_only_stops = "a to je na ten pro ze že v"
    tokens = bm25_index._tokenize(query_only_stops)

    # All tokens should be filtered
    assert len(tokens) == 0

    # Query with mixed content (stop words + meaningful terms)
    query_mixed = "zpracování údajů a právo na ochranu"
    tokens_mixed = bm25_index._tokenize(query_mixed)

    # Meaningful terms should remain
    assert "zpracování" in tokens_mixed
    assert "údajů" in tokens_mixed
    assert "právo" in tokens_mixed
    assert "ochranu" in tokens_mixed

    # Stop words should be filtered
    assert "a" not in tokens_mixed
    assert "na" not in tokens_mixed


def test_czech_stop_words_improve_precision():
    """
    Test that Czech stop words improve search precision.

    Without stop words, BM25 matches on common words like "a", "v", "na".
    With stop words, BM25 focuses on meaningful terms.
    """
    # Load Czech stop words
    czech_stops = load_nltk_stopwords("cs")

    # Create BM25Index with Czech stop words
    bm25_index = BM25Index(nlp_model=None, stop_words=czech_stops)

    # Two documents - one relevant, one not
    relevant_doc = "Zpracování osobních údajů vyžaduje souhlas subjektu."
    irrelevant_doc = "To je zcela jiná věc a nemá nic společného s tím."

    # Query looking for data processing
    query = "Jak funguje zpracování údajů a co je to za podmínky?"

    # Tokenize all text
    query_tokens = bm25_index._tokenize(query)
    relevant_tokens = bm25_index._tokenize(relevant_doc)
    irrelevant_tokens = bm25_index._tokenize(irrelevant_doc)

    # Calculate overlap
    relevant_overlap = set(query_tokens) & set(relevant_tokens)
    irrelevant_overlap = set(query_tokens) & set(irrelevant_tokens)

    # Should have more overlap with relevant doc on meaningful terms
    assert len(relevant_overlap) > 0  # "zpracování", "údajů"
    assert len(relevant_overlap) >= len(irrelevant_overlap)

    # Verify that meaningful terms match
    assert "zpracování" in relevant_overlap
    assert "údajů" in relevant_overlap


def test_czech_text_complete_processing_flow():
    """
    Test complete processing flow for Czech text:
    1. Detect language
    2. Load stop words
    3. Tokenize with stop word filtering
    4. Verify quality
    """
    # Sample Czech legal text
    czech_text = """
    Zákon o ochraně osobních údajů upravuje zpracování osobních údajů
    a ochranu práv fyzických osob. Každá osoba má právo na přístup
    ke svým osobním údajům a na jejich opravu.
    """

    # Step 1: Detect language
    detected_lang = detect_language(czech_text)
    assert detected_lang == "cs"

    # Step 2: Load stop words based on detected language
    stop_words = load_nltk_stopwords(detected_lang)
    assert len(stop_words) == 422

    # Step 3: Create BM25Index and tokenize
    bm25_index = BM25Index(nlp_model=None, stop_words=stop_words)
    tokens = bm25_index._tokenize(czech_text)

    # Step 4: Verify quality
    # Meaningful terms should be present
    assert "zákon" in tokens
    assert "ochraně" in tokens or "ochranu" in tokens  # Different forms
    assert "osobních" in tokens
    assert "údajů" in tokens
    assert "zpracování" in tokens
    assert "práv" in tokens or "právo" in tokens

    # Common stop words should be filtered
    assert "a" not in tokens
    assert "na" not in tokens
    assert "o" not in tokens
    assert "ke" not in tokens


def test_czech_vs_english_different_stop_words():
    """Test that Czech and English use different stop word sets."""
    czech_stops = load_nltk_stopwords("cs")
    english_stops = load_nltk_stopwords("en")

    # Should have different sizes
    assert len(czech_stops) != len(english_stops)

    # Czech-specific stop words not in English
    czech_only = {"že", "která", "který", "jsem", "aby"}
    for word in czech_only:
        assert word in czech_stops
        assert word not in english_stops

    # English-specific stop words not in Czech
    english_only = {"the", "of", "and", "or", "but"}
    for word in english_only:
        assert word in english_stops
        assert word not in czech_stops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
