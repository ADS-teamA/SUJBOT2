"""
Quality Detector - Multi-metric retrieval quality assessment.

Implements 4 metrics to detect poorly-specified queries:
1. Retrieval Score: Average relevance of retrieved chunks
2. Semantic Coherence: Variance in chunk embeddings
3. Query Pattern Analysis: Vague keyword detection
4. Document Diversity: Number of distinct documents

Weighted scoring determines if clarification is needed.
"""

import re
import statistics
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from .config import HITLConfig

logger = logging.getLogger(__name__)


class QualityMetrics(BaseModel):
    """Quality assessment metrics for retrieval results."""

    retrieval_score: float = Field(..., ge=0.0, le=1.0, description="Average chunk relevance")
    semantic_coherence: float = Field(..., ge=0.0, le=1.0, description="Embedding similarity (low variance)")
    query_pattern_score: float = Field(..., ge=0.0, le=1.0, description="Query specificity")
    document_diversity: int = Field(..., ge=0, description="Distinct document count")
    overall_quality: float = Field(..., ge=0.0, le=1.0, description="Weighted overall score")

    failing_metrics: List[str] = Field(default_factory=list, description="Metrics below threshold")
    should_clarify: bool = Field(default=False, description="Trigger clarification?")


class QualityDetector:
    """
    Detect poor retrieval quality using multiple metrics.

    Usage:
        detector = QualityDetector(hitl_config)
        should_clarify, metrics = detector.evaluate(
            query="What are the rules?",
            search_results=[...],
            complexity_score=50
        )
    """

    def __init__(self, config: HITLConfig):
        """
        Initialize quality detector.

        Args:
            config: HITL configuration with metric thresholds
        """
        self.config = config
        logger.info("QualityDetector initialized with 4 metrics")

    def evaluate(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        complexity_score: int,
        unified_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, QualityMetrics]:
        """
        Evaluate retrieval quality and determine if clarification needed.

        Args:
            query: Original user query
            search_results: List of retrieved chunks/documents
            complexity_score: Query complexity (0-100)
            unified_analysis: Optional unified analysis from orchestrator (for LLM-based vagueness)

        Returns:
            (should_clarify, quality_metrics)
        """
        # Policy check: complexity too low?
        if not self.config.should_trigger_for_complexity(complexity_score):
            logger.debug(f"Query complexity {complexity_score} below threshold, skipping clarification")
            return False, self._create_passing_metrics()

        # Check unified analysis for needs_clarification flag (LLM-based decision)
        if unified_analysis and unified_analysis.get("needs_clarification"):
            logger.info("LLM unified analysis indicates clarification needed")
            # Still calculate metrics for completeness
            # Fall through to metric calculation

        # Policy check: zero results always trigger if enabled
        if len(search_results) == 0 and self.config.always_ask_if_zero_results:
            logger.warning("Zero results retrieved, triggering clarification")
            return True, self._create_zero_results_metrics()

        # Calculate individual metrics (using unified_analysis if available)
        retrieval_score = self._calc_retrieval_score(search_results)
        semantic_coherence = self._calc_semantic_coherence(search_results)
        query_pattern_score = self._calc_query_pattern_score(query, unified_analysis)
        document_diversity = self._calc_document_diversity(search_results)

        # Identify failing metrics
        failing_metrics = self._identify_failing_metrics({
            "retrieval_score": retrieval_score,
            "semantic_coherence": semantic_coherence,
            "query_pattern_score": query_pattern_score,
            "document_diversity": document_diversity
        })

        # Calculate weighted overall quality
        overall_quality = self._calc_weighted_quality({
            "retrieval_score": retrieval_score,
            "semantic_coherence": semantic_coherence,
            "query_pattern_score": query_pattern_score,
            "document_diversity": document_diversity
        })

        # Decision: trigger clarification?
        should_clarify = self._should_trigger_clarification(
            overall_quality=overall_quality,
            failing_count=len(failing_metrics)
        )

        metrics = QualityMetrics(
            retrieval_score=retrieval_score,
            semantic_coherence=semantic_coherence,
            query_pattern_score=query_pattern_score,
            document_diversity=document_diversity,
            overall_quality=overall_quality,
            failing_metrics=failing_metrics,
            should_clarify=should_clarify
        )

        if should_clarify:
            logger.info(
                f"Clarification triggered: quality={overall_quality:.2f}, "
                f"failing_metrics={len(failing_metrics)}, "
                f"metrics={failing_metrics}"
            )
        else:
            logger.debug(f"Quality check passed: quality={overall_quality:.2f}")

        return should_clarify, metrics

    def _calc_retrieval_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate average relevance score of retrieved chunks.

        Args:
            results: Search results with relevance_score field

        Returns:
            Average relevance (0-1)
        """
        if not results:
            return 0.0

        scores = []
        for result in results:
            # Handle both dict and DocumentMetadata objects
            if isinstance(result, dict):
                score = result.get("relevance_score", 0.0)
            else:
                score = getattr(result, "relevance_score", 0.0)
            scores.append(float(score))

        if not scores:
            return 0.0

        avg_score = sum(scores) / len(scores)
        logger.debug(f"Retrieval score: {avg_score:.3f} (n={len(scores)})")
        return avg_score

    def _calc_semantic_coherence(self, results: List[Dict[str, Any]]) -> float:
        """
        Measure semantic coherence via embedding similarity.

        High coherence = chunks are semantically similar (low variance).
        Low coherence = chunks are scattered (high variance).

        Args:
            results: Search results with optional embedding field

        Returns:
            Coherence score (0-1), where 1 = high coherence
        """
        if len(results) < 2:
            # Single result is coherent by definition
            return 1.0

        # Extract embeddings (if available)
        embeddings = []
        for result in results:
            if isinstance(result, dict):
                emb = result.get("embedding")
            else:
                emb = getattr(result, "embedding", None)

            if emb is not None:
                embeddings.append(np.array(emb))

        if len(embeddings) < 2:
            # No embeddings available → assume moderate coherence
            logger.debug("No embeddings available for coherence check, assuming 0.5")
            return 0.5

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        if not similarities:
            return 0.5

        # High average similarity + low variance = high coherence
        avg_similarity = sum(similarities) / len(similarities)
        variance = statistics.variance(similarities) if len(similarities) > 1 else 0.0

        # Coherence formula: high similarity, low variance
        coherence = avg_similarity * (1 - min(variance, 1.0))

        logger.debug(f"Semantic coherence: {coherence:.3f} (avg_sim={avg_similarity:.3f}, var={variance:.3f})")
        return float(coherence)

    def _calc_query_pattern_score(
        self, query: str, unified_analysis: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Get query specificity score.

        PREFERS: LLM-based vagueness_score from unified analysis (more accurate).
        FALLBACK: Keyword-based scoring (legacy).

        Args:
            query: User query string
            unified_analysis: Unified analysis from orchestrator (if available)

        Returns:
            Specificity score (0-1), where 1=specific, 0=vague
        """
        # PREFER: Use unified analysis from orchestrator (LLM-based, more accurate)
        if unified_analysis and "vagueness_score" in unified_analysis:
            vagueness = unified_analysis["vagueness_score"]
            # Invert: vagueness (0=specific, 1=vague) -> specificity (0=vague, 1=specific)
            specificity = 1.0 - float(vagueness)
            logger.debug(
                f"Query pattern score (LLM): {specificity:.3f} "
                f"(vagueness_score={vagueness}, semantic_type={unified_analysis.get('semantic_type', 'N/A')})"
            )
            return specificity

        # FALLBACK: Legacy keyword-based scoring
        return self._legacy_query_pattern_score(query)

    def _legacy_query_pattern_score(self, query: str) -> float:
        """
        Legacy keyword-based query specificity scoring.

        Lower score = more vague (generic keywords, no specifics).
        Higher score = more specific (entities, numbers, quotes).

        Args:
            query: User query string

        Returns:
            Specificity score (0-1)
        """
        query_lower = query.lower()

        # Vague indicators (negative signals)
        vague_keywords = [
            "what", "how", "why", "all", "everything", "anything",
            "general", "overview", "tell me about", "information about",
            "něco", "všechno", "cokoliv"  # Czech vague terms
        ]
        vague_count = sum(1 for kw in vague_keywords if kw in query_lower)

        # Specific indicators (positive signals)
        has_numbers = bool(re.search(r'\d+', query))
        has_quotes = '"' in query or "'" in query
        has_caps = bool(re.search(r'\b[A-Z]{2,}', query))  # Acronyms/proper nouns
        word_count = len(query.split())

        # Check for known entities (GDPR, ISO, Article, etc.)
        entity_pattern = r'\b(GDPR|CCPA|ISO|HIPAA|Article\s+\d+|Section\s+\d+)\b'
        has_entities = bool(re.search(entity_pattern, query, re.IGNORECASE))

        # Scoring
        vague_penalty = min(vague_count * 0.15, 0.6)
        specificity_bonus = (
            has_numbers * 0.15 +
            has_quotes * 0.10 +
            has_caps * 0.10 +
            has_entities * 0.20 +
            (word_count > 10) * 0.10
        )

        base_score = 0.5
        score = max(0.0, min(1.0, base_score - vague_penalty + specificity_bonus))

        logger.debug(
            f"Query pattern score (legacy): {score:.3f} "
            f"(vague={vague_count}, nums={has_numbers}, entities={has_entities})"
        )
        return score

    def _calc_document_diversity(self, results: List[Dict[str, Any]]) -> int:
        """
        Count distinct documents in results.

        Too many distinct docs = scattered retrieval (low focus).

        Args:
            results: Search results with document_id or doc_id

        Returns:
            Number of distinct documents
        """
        doc_ids = set()
        for result in results:
            if isinstance(result, dict):
                doc_id = result.get("document_id") or result.get("doc_id")
            else:
                doc_id = getattr(result, "document_id", None) or getattr(result, "doc_id", None)

            if doc_id:
                doc_ids.add(doc_id)

        diversity = len(doc_ids)
        logger.debug(f"Document diversity: {diversity} distinct docs")
        return diversity

    def _identify_failing_metrics(self, metric_values: Dict[str, float]) -> List[str]:
        """Identify which metrics failed their thresholds."""
        failing = []

        # Retrieval score
        if (self.config.retrieval_score_metric.enabled and
            metric_values["retrieval_score"] < self.config.retrieval_score_metric.threshold):
            failing.append("retrieval_score")

        # Semantic coherence
        if (self.config.semantic_coherence_metric.enabled and
            metric_values["semantic_coherence"] < self.config.semantic_coherence_metric.threshold):
            failing.append("semantic_coherence")

        # Query pattern
        if (self.config.query_pattern_metric.enabled and
            metric_values["query_pattern_score"] < self.config.query_pattern_metric.threshold):
            failing.append("query_pattern_score")

        # Document diversity (inverse: too many docs is bad)
        if (self.config.document_diversity_metric.enabled and
            metric_values["document_diversity"] > self.config.document_diversity_metric.threshold):
            failing.append("document_diversity")

        return failing

    def _calc_weighted_quality(self, metric_values: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        total_weight = 0.0
        weighted_sum = 0.0

        # Retrieval score
        if self.config.retrieval_score_metric.enabled:
            weighted_sum += metric_values["retrieval_score"] * self.config.retrieval_score_metric.weight
            total_weight += self.config.retrieval_score_metric.weight

        # Semantic coherence
        if self.config.semantic_coherence_metric.enabled:
            weighted_sum += metric_values["semantic_coherence"] * self.config.semantic_coherence_metric.weight
            total_weight += self.config.semantic_coherence_metric.weight

        # Query pattern
        if self.config.query_pattern_metric.enabled:
            weighted_sum += metric_values["query_pattern_score"] * self.config.query_pattern_metric.weight
            total_weight += self.config.query_pattern_metric.weight

        # Document diversity (normalize to 0-1, inverse)
        if self.config.document_diversity_metric.enabled:
            diversity = metric_values["document_diversity"]
            # Inverse: more docs = lower score (scattered retrieval)
            normalized = max(0.0, 1.0 - (diversity / 10.0))
            weighted_sum += normalized * self.config.document_diversity_metric.weight
            total_weight += self.config.document_diversity_metric.weight

        if total_weight == 0:
            return 0.5  # Default if all metrics disabled

        return weighted_sum / total_weight

    def _should_trigger_clarification(self, overall_quality: float, failing_count: int) -> bool:
        """Determine if clarification should be triggered."""
        # Check overall quality threshold
        if overall_quality >= self.config.quality_threshold:
            return False

        # Check multiple failures requirement
        if self.config.require_multiple_failures:
            return failing_count >= self.config.min_failing_metrics

        # Single metric failure is enough
        return True

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _create_passing_metrics(self) -> QualityMetrics:
        """Create metrics object for passing quality (no clarification)."""
        return QualityMetrics(
            retrieval_score=1.0,
            semantic_coherence=1.0,
            query_pattern_score=1.0,
            document_diversity=1,
            overall_quality=1.0,
            failing_metrics=[],
            should_clarify=False
        )

    def _create_zero_results_metrics(self) -> QualityMetrics:
        """Create metrics object for zero results (always clarify)."""
        return QualityMetrics(
            retrieval_score=0.0,
            semantic_coherence=0.0,
            query_pattern_score=0.0,
            document_diversity=0,
            overall_quality=0.0,
            failing_metrics=["retrieval_score", "document_diversity"],
            should_clarify=True
        )
