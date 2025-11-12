"""
Unit tests for QualityDetector.

Tests all 4 quality metrics and decision logic.
"""

import pytest
import numpy as np
from src.multi_agent.hitl.config import HITLConfig
from src.multi_agent.hitl.quality_detector import QualityDetector, QualityMetrics


@pytest.fixture
def default_config():
    """Default HITL configuration for testing."""
    return HITLConfig()


@pytest.fixture
def detector(default_config):
    """QualityDetector instance with default config."""
    return QualityDetector(default_config)


class TestRetrievalScore:
    """Test retrieval score metric."""

    def test_high_relevance_passes(self, detector):
        """High relevance scores should pass."""
        results = [
            {"relevance_score": 0.9},
            {"relevance_score": 0.85},
            {"relevance_score": 0.8}
        ]
        score = detector._calc_retrieval_score(results)
        assert score > 0.8
        assert score == pytest.approx(0.85, abs=0.01)

    def test_low_relevance_fails(self, detector):
        """Low relevance scores should fail threshold."""
        results = [
            {"relevance_score": 0.3},
            {"relevance_score": 0.4},
            {"relevance_score": 0.5}
        ]
        score = detector._calc_retrieval_score(results)
        assert score < 0.65  # Below default threshold
        assert score == pytest.approx(0.4, abs=0.01)

    def test_empty_results(self, detector):
        """Empty results should return 0."""
        assert detector._calc_retrieval_score([]) == 0.0

    def test_mixed_types(self, detector):
        """Handle both dict and object types."""
        class MockDoc:
            relevance_score = 0.7

        results = [
            {"relevance_score": 0.8},
            MockDoc()
        ]
        score = detector._calc_retrieval_score(results)
        assert 0.7 < score < 0.8


class TestSemanticCoherence:
    """Test semantic coherence metric."""

    def test_high_coherence_similar_embeddings(self, detector):
        """Similar embeddings = high coherence."""
        base_emb = np.array([1.0, 0.0, 0.0])
        results = [
            {"embedding": base_emb},
            {"embedding": base_emb + np.array([0.1, 0.0, 0.0])},
            {"embedding": base_emb + np.array([0.0, 0.1, 0.0])}
        ]
        score = detector._calc_semantic_coherence(results)
        assert score > 0.7  # High coherence

    def test_low_coherence_scattered_embeddings(self, detector):
        """Scattered embeddings = low coherence."""
        results = [
            {"embedding": np.array([1.0, 0.0, 0.0])},
            {"embedding": np.array([0.0, 1.0, 0.0])},
            {"embedding": np.array([0.0, 0.0, 1.0])}
        ]
        score = detector._calc_semantic_coherence(results)
        assert score < 0.5  # Low coherence (orthogonal vectors)

    def test_single_result_high_coherence(self, detector):
        """Single result is coherent by definition."""
        results = [{"embedding": np.array([1.0, 0.0, 0.0])}]
        assert detector._calc_semantic_coherence(results) == 1.0

    def test_no_embeddings_returns_moderate(self, detector):
        """No embeddings should return 0.5 (unknown)."""
        results = [{"relevance_score": 0.8}, {"relevance_score": 0.7}]
        assert detector._calc_semantic_coherence(results) == 0.5


class TestQueryPatternScore:
    """Test query pattern analysis metric."""

    def test_vague_query_low_score(self, detector):
        """Vague queries should score low."""
        vague_queries = [
            "What are the rules?",
            "Tell me about everything",
            "How does this work?",
            "Give me all information"
        ]
        for query in vague_queries:
            score = detector._calc_query_pattern_score(query)
            assert score < 0.5, f"Query '{query}' should be vague (score={score})"

    def test_specific_query_high_score(self, detector):
        """Specific queries should score high."""
        specific_queries = [
            "GDPR Article 5 requirements for 2024",
            'What are "ISO 27001" compliance standards?',
            "Analyze contract Section 3.2 dated 2023-01-15"
        ]
        for query in specific_queries:
            score = detector._calc_query_pattern_score(query)
            assert score > 0.6, f"Query '{query}' should be specific (score={score})"

    def test_czech_vague_keywords(self, detector):
        """Czech vague keywords should be detected."""
        query = "Něco o všech pravidlech"
        score = detector._calc_query_pattern_score(query)
        assert score < 0.5

    def test_entity_recognition(self, detector):
        """Known entities should boost score."""
        query = "GDPR and HIPAA requirements"
        score = detector._calc_query_pattern_score(query)
        assert score > 0.6  # Has entities


class TestDocumentDiversity:
    """Test document diversity metric."""

    def test_single_document_low_diversity(self, detector):
        """All results from single document."""
        results = [
            {"document_id": "doc1"},
            {"document_id": "doc1"},
            {"document_id": "doc1"}
        ]
        assert detector._calc_document_diversity(results) == 1

    def test_high_diversity_many_docs(self, detector):
        """Results span many documents."""
        results = [
            {"document_id": f"doc{i}"}
            for i in range(10)
        ]
        assert detector._calc_document_diversity(results) == 10

    def test_missing_doc_ids(self, detector):
        """Handle missing document_id fields."""
        results = [
            {"relevance_score": 0.8},
            {"doc_id": "doc1"},  # Alternative field name
            {}
        ]
        assert detector._calc_document_diversity(results) == 1


class TestWeightedQuality:
    """Test weighted overall quality calculation."""

    def test_all_metrics_high(self, detector):
        """All metrics passing should give high quality."""
        metrics = {
            "retrieval_score": 0.9,
            "semantic_coherence": 0.85,
            "query_pattern_score": 0.8,
            "document_diversity": 2  # Low diversity is good
        }
        quality = detector._calc_weighted_quality(metrics)
        assert quality > 0.8

    def test_all_metrics_low(self, detector):
        """All metrics failing should give low quality."""
        metrics = {
            "retrieval_score": 0.3,
            "semantic_coherence": 0.2,
            "query_pattern_score": 0.3,
            "document_diversity": 15  # High diversity is bad
        }
        quality = detector._calc_weighted_quality(metrics)
        assert quality < 0.4

    def test_weighted_average(self, detector):
        """Quality should respect configured weights."""
        # Retrieval score has 0.30 weight
        metrics = {
            "retrieval_score": 1.0,  # Perfect (30% contribution)
            "semantic_coherence": 0.0,
            "query_pattern_score": 0.0,
            "document_diversity": 0  # Normalized to 1.0
        }
        quality = detector._calc_weighted_quality(metrics)
        # Should be roughly 0.3 (30% from retrieval, 20% from diversity)
        assert 0.4 < quality < 0.6


class TestClarificationDecision:
    """Test clarification trigger logic."""

    def test_high_quality_no_clarification(self, detector):
        """High quality should not trigger clarification."""
        results = [
            {"relevance_score": 0.9, "document_id": "doc1", "embedding": np.array([1.0, 0.0, 0.0])},
            {"relevance_score": 0.85, "document_id": "doc1", "embedding": np.array([0.9, 0.1, 0.0])}
        ]
        should_clarify, metrics = detector.evaluate(
            query="GDPR Article 5 requirements for 2024",
            search_results=results,
            complexity_score=50
        )
        assert not should_clarify
        assert metrics.overall_quality > 0.6

    def test_low_quality_triggers_clarification(self, detector):
        """Low quality should trigger clarification."""
        results = [
            {"relevance_score": 0.3, "document_id": f"doc{i}", "embedding": np.random.randn(3)}
            for i in range(10)
        ]
        should_clarify, metrics = detector.evaluate(
            query="What are the rules?",  # Vague
            search_results=results,
            complexity_score=50
        )
        assert should_clarify
        assert metrics.overall_quality < 0.6
        assert len(metrics.failing_metrics) >= 2  # Multiple failures required

    def test_zero_results_always_clarifies(self, detector):
        """Zero results should always trigger clarification."""
        should_clarify, metrics = detector.evaluate(
            query="Nonexistent query",
            search_results=[],
            complexity_score=50
        )
        assert should_clarify
        assert metrics.overall_quality == 0.0

    def test_low_complexity_skips_clarification(self, detector):
        """Low complexity queries should skip clarification."""
        results = [{"relevance_score": 0.3}]  # Low quality
        should_clarify, metrics = detector.evaluate(
            query="What are the rules?",
            search_results=results,
            complexity_score=30  # Below default threshold of 40
        )
        assert not should_clarify  # Skipped due to low complexity

    def test_multiple_failures_required(self, detector):
        """Policy: require at least 2 failing metrics."""
        # Only 1 metric fails (retrieval score)
        results = [
            {"relevance_score": 0.3, "document_id": "doc1", "embedding": np.array([1.0, 0.0, 0.0])},
            {"relevance_score": 0.35, "document_id": "doc1", "embedding": np.array([0.9, 0.1, 0.0])}
        ]
        should_clarify, metrics = detector.evaluate(
            query="GDPR Article 5",  # Specific query
            search_results=results,
            complexity_score=50
        )
        # Should NOT clarify (only 1 failing metric)
        failing_count = len(metrics.failing_metrics)
        if detector.config.require_multiple_failures and failing_count < detector.config.min_failing_metrics:
            assert not should_clarify


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nan_in_embeddings(self, detector):
        """Handle NaN values in embeddings."""
        results = [
            {"embedding": np.array([np.nan, 0.0, 0.0])},
            {"embedding": np.array([1.0, 0.0, 0.0])}
        ]
        # Should not crash
        score = detector._calc_semantic_coherence(results)
        assert isinstance(score, float)

    def test_extremely_long_query(self, detector):
        """Handle very long queries."""
        query = "What " * 1000  # 1000 repetitions
        score = detector._calc_query_pattern_score(query)
        assert 0.0 <= score <= 1.0

    def test_empty_query(self, detector):
        """Handle empty query."""
        should_clarify, metrics = detector.evaluate(
            query="",
            search_results=[],
            complexity_score=50
        )
        assert should_clarify  # Zero results triggers

    def test_none_values_in_results(self, detector):
        """Handle None values gracefully."""
        results = [
            {"relevance_score": None},
            {"relevance_score": 0.7}
        ]
        score = detector._calc_retrieval_score(results)
        assert score == 0.35  # (0 + 0.7) / 2
