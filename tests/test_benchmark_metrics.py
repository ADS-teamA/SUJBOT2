"""
Tests for benchmark metrics.

Tests all evaluation metrics including:
- Exact Match (EM)
- F1 Score
- Precision/Recall
- Embedding Similarity (cosine similarity in embedding space)
- Combined F1
"""

import pytest
from src.benchmark.metrics import (
    calculate_exact_match,
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    calculate_embedding_similarity,
    calculate_combined_f1,
    compute_all_metrics,
    format_metrics,
    normalize_text,
    tokenize,
)


class TestNormalization:
    """Test text normalization utilities."""

    def test_normalize_text_lowercase(self):
        """Normalize should lowercase text."""
        assert normalize_text("HELLO World") == "hello world"

    def test_normalize_text_punctuation(self):
        """Normalize should remove punctuation."""
        assert normalize_text("Hello, World!") == "hello world"

    def test_normalize_text_whitespace(self):
        """Normalize should collapse whitespace."""
        assert normalize_text("hello   world  ") == "hello world"

    def test_tokenize_simple(self):
        """Tokenize should split on whitespace."""
        assert tokenize("hello world") == {"hello", "world"}

    def test_tokenize_with_punctuation(self):
        """Tokenize should handle punctuation."""
        assert tokenize("Hello, world!") == {"hello", "world"}


class TestExactMatch:
    """Test exact match metric."""

    def test_exact_match_perfect(self):
        """Perfect match should return 1.0."""
        assert calculate_exact_match("hello world", ["hello world"]) == 1.0

    def test_exact_match_case_insensitive(self):
        """Match should be case-insensitive."""
        assert calculate_exact_match("Hello World", ["hello world"]) == 1.0

    def test_exact_match_no_match(self):
        """No match should return 0.0."""
        assert calculate_exact_match("hello", ["world"]) == 0.0

    def test_exact_match_multiple_refs(self):
        """Should match any reference."""
        assert calculate_exact_match("hello", ["world", "hello", "test"]) == 1.0


class TestF1Score:
    """Test F1 score metric."""

    def test_f1_perfect_match(self):
        """Perfect match should give F1=1.0."""
        assert calculate_f1_score("hello world", ["hello world"]) == 1.0

    def test_f1_partial_match(self):
        """Partial match should give F1 between 0 and 1."""
        # Predicted: {hello}
        # Reference: {hello, world}
        # Precision: 1/1 = 1.0
        # Recall: 1/2 = 0.5
        # F1: 2*1.0*0.5/(1.0+0.5) = 0.6666...
        f1 = calculate_f1_score("hello", ["hello world"])
        assert 0.66 < f1 < 0.67

    def test_f1_no_overlap(self):
        """No overlap should give F1=0.0."""
        assert calculate_f1_score("hello", ["world"]) == 0.0

    def test_f1_empty_prediction(self):
        """Empty prediction should give F1=0.0."""
        assert calculate_f1_score("", ["hello world"]) == 0.0


class TestPrecisionRecall:
    """Test precision and recall metrics."""

    def test_precision_perfect(self):
        """All predicted tokens correct -> precision=1.0."""
        assert calculate_precision("hello", ["hello world"]) == 1.0

    def test_precision_partial(self):
        """Some incorrect predictions -> precision < 1.0."""
        # Predicted: {hello, test}
        # Reference: {hello, world}
        # Correct: {hello}
        # Precision: 1/2 = 0.5
        assert calculate_precision("hello test", ["hello world"]) == 0.5

    def test_recall_perfect(self):
        """All reference tokens found -> recall=1.0."""
        assert calculate_recall("hello world", ["hello"]) == 1.0

    def test_recall_partial(self):
        """Some reference tokens missing -> recall < 1.0."""
        # Predicted: {hello}
        # Reference: {hello, world}
        # Recall: 1/2 = 0.5
        assert calculate_recall("hello", ["hello world"]) == 0.5


class TestEmbeddingSimilarity:
    """Test embedding-based semantic similarity metric."""

    def test_embedding_similarity_perfect(self):
        """Identical text should have similarity ~1.0."""
        sim = calculate_embedding_similarity("hello world", ["hello world"])
        assert sim > 0.95  # Allow small floating point differences

    def test_embedding_similarity_semantic(self):
        """Semantically similar text should have high similarity."""
        # "waste disposal" vs "garbage removal" - similar meaning
        sim = calculate_embedding_similarity(
            "waste disposal requirements", ["garbage removal regulations"]
        )
        # Should be > 0.6 (semantic similarity, not exact match)
        assert sim > 0.6

    def test_embedding_similarity_different(self):
        """Semantically different text should have low similarity."""
        sim = calculate_embedding_similarity("waste disposal", ["quantum physics"])
        # Should be < 0.5 (very different topics)
        assert sim < 0.5

    def test_embedding_similarity_empty(self):
        """Empty prediction should return 0.0."""
        assert calculate_embedding_similarity("", ["hello world"]) == 0.0

    def test_embedding_similarity_multiple_refs(self):
        """Should return max similarity across references."""
        sim = calculate_embedding_similarity(
            "waste disposal", ["quantum physics", "garbage removal", "astronomy"]
        )
        # Should match "garbage removal" best (semantic similarity ~0.68)
        assert sim > 0.65


class TestCombinedF1:
    """Test combined F1 metric."""

    def test_combined_f1_single_ref(self):
        """With single reference, should match regular F1."""
        combined = calculate_combined_f1("hello world", ["hello world"])
        regular = calculate_f1_score("hello world", ["hello world"])
        assert combined == regular

    def test_combined_f1_multiple_refs(self):
        """With multiple refs, should penalize missing parts."""
        # Predicted: {username, photos}
        # Ref 1: {ip, address, cookies}
        # Ref 2: {username, photos}
        # Combined ref: {ip, address, cookies, username, photos}
        # Predicted only has 2/5 tokens -> lower F1
        combined = calculate_combined_f1(
            "username photos", ["ip address cookies", "username photos"]
        )
        # Should be < 1.0 because missing some tokens from combined ref
        assert combined < 1.0


class TestComputeAllMetrics:
    """Test compute_all_metrics aggregation."""

    def test_compute_all_metrics_structure(self):
        """Should return dict with all metric keys."""
        metrics = compute_all_metrics("hello world", ["hello world"])

        expected_keys = {
            "exact_match",
            "f1_score",
            "precision",
            "recall",
            "embedding_similarity",
            "combined_f1",
        }
        assert set(metrics.keys()) == expected_keys

    def test_compute_all_metrics_perfect(self):
        """Perfect match should give all metrics ~1.0."""
        metrics = compute_all_metrics("hello world", ["hello world"])

        assert metrics["exact_match"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["embedding_similarity"] > 0.95
        assert metrics["combined_f1"] == 1.0


class TestFormatMetrics:
    """Test metric formatting."""

    def test_format_metrics_abbreviations(self):
        """Should use correct abbreviations."""
        metrics = {
            "exact_match": 0.75,
            "f1_score": 0.8,
            "embedding_similarity": 0.85,
        }
        formatted = format_metrics(metrics, precision=2)

        assert "EM: 0.75" in formatted
        assert "F1: 0.80" in formatted
        assert "EMB: 0.85" in formatted
        assert "|" in formatted  # Should separate with pipes

    def test_format_metrics_precision(self):
        """Should respect precision parameter."""
        metrics = {"f1_score": 0.123456}

        # 2 decimals
        assert "0.12" in format_metrics(metrics, precision=2)

        # 4 decimals
        assert "0.1235" in format_metrics(metrics, precision=4)
