"""
RAG Confidence Scoring for Retrieval Quality Assessment.

Evaluates confidence in retrieved chunks using multiple signals:
1. Retrieval scores (BM25, Dense, RRF, Rerank)
2. Score distribution (gap, spread, consensus)
3. Retrieval method agreement (BM25-Dense correlation)
4. Context quality (diversity, redundancy)

Based on:
- RAGAS framework (context precision/recall)
- Legal compliance research (confidence thresholds)
- Hybrid search best practices
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RAGConfidenceScore:
    """
    Confidence assessment for RAG retrieval results.

    Attributes:
        overall_confidence: Overall confidence (0-1, higher = more confident)
        top_score: Best retrieval score
        score_gap: Gap between top and second result
        score_spread: Standard deviation of scores
        consensus_count: Number of high-confidence chunks
        bm25_dense_agreement: Correlation between BM25 and dense scores
        reranker_impact: How much reranker changed ranking
        graph_support: Whether knowledge graph supports results
        document_diversity: Diversity of source documents
        interpretation: Human-readable confidence level
        should_flag: Whether to flag for review
        details: Detailed breakdown for debugging
    """

    overall_confidence: float
    top_score: float
    score_gap: float
    score_spread: float
    consensus_count: int
    bm25_dense_agreement: float
    reranker_impact: float
    graph_support: bool
    document_diversity: float
    interpretation: str
    should_flag: bool
    details: Dict

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization with validation."""

        def validate_score(
            value: float, min_val: float = 0.0, max_val: float = 1.0, field_name: str = "score"
        ) -> float:
            """Validate and clamp a score value."""
            if not isinstance(value, (int, float)):
                logger.warning(f"Invalid {field_name} type: {type(value)}. Using 0.0.")
                return 0.0

            if np.isnan(value) or np.isinf(value):
                logger.warning(
                    f"Invalid {field_name} value: {value} (NaN or infinity). Using {min_val}."
                )
                return min_val

            # Clamp to range
            if value < min_val or value > max_val:
                logger.warning(
                    f"{field_name} {value} out of range [{min_val}, {max_val}]. Clamping."
                )
                return max(min_val, min(max_val, value))

            return value

        return {
            "overall_confidence": round(
                validate_score(self.overall_confidence, 0.0, 1.0, "overall_confidence"), 3
            ),
            "top_score": round(validate_score(self.top_score, 0.0, float("inf"), "top_score"), 3),
            "score_gap": round(validate_score(self.score_gap, 0.0, float("inf"), "score_gap"), 3),
            "score_spread": round(
                validate_score(self.score_spread, 0.0, float("inf"), "score_spread"), 3
            ),
            "consensus_count": max(0, int(self.consensus_count)),  # Non-negative integer
            "bm25_dense_agreement": round(
                validate_score(self.bm25_dense_agreement, 0.0, 1.0, "bm25_dense_agreement"), 3
            ),
            "reranker_impact": round(
                validate_score(self.reranker_impact, 0.0, 1.0, "reranker_impact"), 3
            ),
            "graph_support": bool(self.graph_support),
            "document_diversity": round(
                validate_score(self.document_diversity, 0.0, 1.0, "document_diversity"), 3
            ),
            "interpretation": str(self.interpretation) if self.interpretation else "Unknown",
            "should_flag_for_review": bool(self.should_flag),
            "details": self.details or {},
        }


class RAGConfidenceScorer:
    """
    Confidence scorer for RAG retrieval results.

    Evaluates retrieval quality using multiple signals:
    - Score-based: Top score, gap, spread, consensus
    - Agreement-based: BM25-Dense correlation, reranker impact
    - Context-based: Document diversity, section diversity

    Thresholds (based on legal compliance research):
    - ≥0.85: High confidence (automated)
    - 0.70-0.84: Medium confidence (review recommended)
    - 0.50-0.69: Low confidence (mandatory review)
    - <0.50: Very low confidence (expert required)
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.85,
        medium_confidence_threshold: float = 0.70,
        low_confidence_threshold: float = 0.50,
        consensus_threshold: float = 0.75,  # Score threshold for "high confidence" chunk
    ):
        """
        Initialize RAG confidence scorer.

        Args:
            high_confidence_threshold: Threshold for high confidence (default: 0.85)
            medium_confidence_threshold: Threshold for medium confidence (default: 0.70)
            low_confidence_threshold: Threshold for low confidence (default: 0.50)
            consensus_threshold: Score threshold for consensus counting (default: 0.75)
        """
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.low_threshold = low_confidence_threshold
        self.consensus_threshold = consensus_threshold

    def score_retrieval(
        self, chunks: List[Dict], query: Optional[str] = None
    ) -> RAGConfidenceScore:
        """
        Score confidence of RAG retrieval results.

        Args:
            chunks: Retrieved chunks with scores (from search tool)
            query: Original query (optional, for logging)

        Returns:
            RAGConfidenceScore with detailed breakdown
        """
        if not chunks:
            return self._empty_result()

        # Extract scores
        scores = self._extract_scores(chunks)

        # 1. Score-based metrics
        top_score = scores[0] if scores else 0.0
        score_gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
        score_spread = float(np.std(scores)) if len(scores) > 1 else 0.0
        consensus_count = sum(1 for s in scores if s >= self.consensus_threshold)

        # 2. Retrieval method agreement
        bm25_dense_agreement = self._calculate_bm25_dense_agreement(chunks)
        reranker_impact = self._calculate_reranker_impact(chunks)

        # 3. Knowledge graph support
        graph_support = any(chunk.get("graph_boost", 0.0) > 0 for chunk in chunks)

        # 4. Context quality
        document_diversity = self._calculate_document_diversity(chunks)

        # 5. Combine into overall confidence
        overall_confidence = self._calculate_overall_confidence(
            top_score=top_score,
            score_gap=score_gap,
            score_spread=score_spread,
            consensus_count=consensus_count,
            total_chunks=len(chunks),
            bm25_dense_agreement=bm25_dense_agreement,
            reranker_impact=reranker_impact,
            graph_support=graph_support,
            document_diversity=document_diversity,
        )

        # 6. Interpret confidence level
        interpretation, should_flag = self._interpret_confidence(overall_confidence)

        # 7. Build detailed breakdown
        details = {
            "total_chunks": len(chunks),
            "score_distribution": {
                "min": round(float(min(scores)), 3) if scores else 0.0,
                "max": round(float(max(scores)), 3) if scores else 0.0,
                "mean": round(float(np.mean(scores)), 3) if scores else 0.0,
                "median": round(float(np.median(scores)), 3) if scores else 0.0,
            },
            "retrieval_methods": self._analyze_retrieval_methods(chunks),
            "source_diversity": {
                "unique_documents": len(set(c.get("document_id") for c in chunks)),
                "unique_sections": len(set(c.get("section_id") for c in chunks)),
            },
        }

        if query:
            logger.info(
                f"RAG Confidence for query '{query[:50]}...': "
                f"{interpretation} ({overall_confidence:.3f})"
            )

        return RAGConfidenceScore(
            overall_confidence=overall_confidence,
            top_score=top_score,
            score_gap=score_gap,
            score_spread=score_spread,
            consensus_count=consensus_count,
            bm25_dense_agreement=bm25_dense_agreement,
            reranker_impact=reranker_impact,
            graph_support=graph_support,
            document_diversity=document_diversity,
            interpretation=interpretation,
            should_flag=should_flag,
            details=details,
        )

    def _extract_scores(self, chunks: List[Dict]) -> List[float]:
        """
        Extract primary scores from chunks in priority order.

        Priority (highest to lowest):
        1. rerank_score: Cross-encoder reranker score (best for ranking)
        2. boosted_score: Graph-boosted RRF score (includes KG relevance)
        3. rrf_score: Reciprocal Rank Fusion score (BM25 + Dense fusion)
        4. score: Fallback generic score (BM25 for search with search_method='bm25_only')

        Returns:
            List of normalized scores (0-1) from chunks, defaults to 0.0 if missing
        """
        scores = []
        for chunk in chunks:
            score = (
                chunk.get("rerank_score")
                or chunk.get("boosted_score")
                or chunk.get("rrf_score")
                or chunk.get("score")
                or 0.0
            )
            scores.append(float(score))

        return scores

    def _calculate_bm25_dense_agreement(self, chunks: List[Dict]) -> float:
        """
        Calculate correlation between BM25 and dense scores.

        High correlation = both methods agree (high confidence)
        Low correlation = methods disagree (lower confidence)
        """
        bm25_scores = []
        dense_scores = []

        for chunk in chunks:
            bm25 = chunk.get("bm25_score")
            dense = chunk.get("dense_score")

            if bm25 is not None and dense is not None:
                bm25_scores.append(float(bm25))
                dense_scores.append(float(dense))

        if len(bm25_scores) < 2:
            return 0.5  # Neutral (not enough data)

        # Calculate Pearson correlation
        correlation = float(np.corrcoef(bm25_scores, dense_scores)[0, 1])

        # Handle NaN (can occur if all scores are identical)
        if np.isnan(correlation):
            return 1.0  # Perfect agreement (all scores identical)

        # Convert to 0-1 range (correlation is -1 to 1)
        agreement = (correlation + 1) / 2

        return agreement

    def _calculate_reranker_impact(self, chunks: List[Dict]) -> float:
        """
        Calculate how much reranker changed the ranking.

        High impact = reranker significantly changed order (could be good or bad)
        Low impact = reranker agreed with initial ranking (high confidence)
        """
        rerank_scores = []
        rrf_scores = []

        for chunk in chunks:
            rerank = chunk.get("rerank_score")
            rrf = chunk.get("rrf_score")

            if rerank is not None and rrf is not None:
                rerank_scores.append(float(rerank))
                rrf_scores.append(float(rrf))

        if len(rerank_scores) < 2:
            return 0.0  # No reranking applied

        # Validate no NaN/inf in input
        if any(np.isnan(s) or np.isinf(s) for s in rerank_scores) or any(
            np.isnan(s) or np.isinf(s) for s in rrf_scores
        ):
            logger.warning(
                "Reranker impact calculation has NaN/inf in scores. Returning 0.0 (no impact)."
            )
            return 0.0

        # Try scipy first (better algorithm)
        try:
            from scipy.stats import spearmanr

            try:
                correlation, p_value = spearmanr(rerank_scores, rrf_scores)
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    f"spearmanr failed with {type(e).__name__}: {e}. Falling back to Pearson correlation."
                )
                correlation = None

            if correlation is None or np.isnan(correlation) or np.isinf(correlation):
                logger.warning(
                    f"Spearman correlation invalid: {correlation}. Falling back to Pearson."
                )
                correlation = None
        except ImportError:
            logger.debug("scipy.stats.spearmanr not available. Using Pearson correlation fallback.")
            correlation = None

        # Fallback: Pearson correlation
        if correlation is None:
            try:
                correlation_matrix = np.corrcoef(rerank_scores, rrf_scores)
                correlation = float(correlation_matrix[0, 1])

                if np.isnan(correlation) or np.isinf(correlation):
                    logger.warning(
                        f"Pearson correlation invalid: {correlation}. Using 0.0 (no impact)."
                    )
                    return 0.0

            except (ValueError, RuntimeError, IndexError) as e:
                logger.error(
                    f"Both correlation methods failed: {type(e).__name__}: {e}. Using 0.0 (no impact)."
                )
                return 0.0

        # Calculate impact
        try:
            impact = 1.0 - abs(correlation)

            # Validate result
            if not (0.0 <= impact <= 1.0):
                logger.warning(f"Calculated impact {impact} out of range [0, 1]. Clamping.")
                impact = max(0.0, min(1.0, impact))

            return float(impact)

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to calculate impact from correlation: {e}. Using 0.0.")
            return 0.0

    def _calculate_document_diversity(self, chunks: List[Dict]) -> float:
        """
        Calculate diversity of source documents.

        High diversity = results from multiple documents (could indicate uncertainty)
        Low diversity = results from single document (high confidence if scores are high)
        """
        document_ids = [chunk.get("document_id") for chunk in chunks]
        unique_docs = len(set(document_ids))
        total_chunks = len(chunks)

        diversity = unique_docs / total_chunks if total_chunks > 0 else 0.0

        return diversity

    def _calculate_overall_confidence(
        self,
        top_score: float,
        score_gap: float,
        score_spread: float,
        consensus_count: int,
        total_chunks: int,
        bm25_dense_agreement: float,
        reranker_impact: float,
        graph_support: bool,
        document_diversity: float,
    ) -> float:
        """
        Calculate overall confidence using weighted combination.

        Weights based on legal compliance research and RAGAS framework.
        """
        # Validate all inputs are finite
        inputs = {
            "top_score": top_score,
            "score_gap": score_gap,
            "score_spread": score_spread,
            "consensus_count": consensus_count,
            "total_chunks": total_chunks,
            "bm25_dense_agreement": bm25_dense_agreement,
            "reranker_impact": reranker_impact,
            "document_diversity": document_diversity,
        }

        for name, value in inputs.items():
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    logger.error(
                        f"Invalid confidence input '{name}': {value} (NaN or infinity). "
                        f"Clamping to 0.0 to continue."
                    )
                    inputs[name] = 0.0

        # Extract validated inputs
        top_score = inputs["top_score"]
        score_gap = inputs["score_gap"]
        score_spread = inputs["score_spread"]
        consensus_count = inputs["consensus_count"]
        total_chunks = inputs["total_chunks"]
        bm25_dense_agreement = inputs["bm25_dense_agreement"]
        reranker_impact = inputs["reranker_impact"]
        document_diversity = inputs["document_diversity"]

        # Normalize inputs to 0-1 range (with safeguards)

        # 1. Top score (weight: 0.30) - Most important
        top_score_norm = min(1.0, max(0.0, top_score))

        # 2. Score gap (weight: 0.20) - Clear winner?
        # Typical gap: 0.0-0.3, normalize to 0-1, gaps > 0.3 clamp to 1.0
        score_gap_norm = min(1.0, max(0.0, score_gap / 0.3))

        # 3. Consensus (weight: 0.15) - Multiple high-confidence chunks?
        consensus_norm = max(
            0.0, min(1.0, consensus_count / total_chunks if total_chunks > 0 else 0.0)
        )

        # 4. BM25-Dense agreement (weight: 0.15) - Methods agree?
        agreement_norm = bm25_dense_agreement

        # 5. Low score spread (weight: 0.10) - Consistent scores?
        # Typical spread: 0.0-0.2, invert (low spread = high confidence)
        spread_norm = max(0.0, 1.0 - (score_spread / 0.2))

        # 6. Graph support (weight: 0.05) - Knowledge graph confirms?
        graph_norm = 1.0 if graph_support else 0.5

        # 7. Document diversity (weight: 0.05) - Single source?
        # Low diversity = high confidence (for legal docs)
        diversity_norm = 1.0 - document_diversity

        # Clamp all normalized values to [0, 1]
        normalized_inputs = {
            "top_score_norm": top_score_norm,
            "score_gap_norm": score_gap_norm,
            "consensus_norm": consensus_norm,
            "agreement_norm": agreement_norm,
            "spread_norm": spread_norm,
            "graph_norm": graph_norm,
            "diversity_norm": diversity_norm,
        }

        for name, value in normalized_inputs.items():
            if np.isnan(value) or np.isinf(value):
                logger.error(f"Invalid normalized value '{name}': {value}. Using 0.5 (neutral).")
                normalized_inputs[name] = 0.5
            else:
                # Clamp to [0, 1]
                normalized_inputs[name] = max(0.0, min(1.0, value))

        # Weighted combination
        confidence = (
            0.30 * normalized_inputs["top_score_norm"]
            + 0.20 * normalized_inputs["score_gap_norm"]
            + 0.15 * normalized_inputs["consensus_norm"]
            + 0.15 * normalized_inputs["agreement_norm"]
            + 0.10 * normalized_inputs["spread_norm"]
            + 0.05 * normalized_inputs["graph_norm"]
            + 0.05 * normalized_inputs["diversity_norm"]
        )

        # Final validation - ensure result is in [0, 1]
        if np.isnan(confidence) or np.isinf(confidence):
            logger.error(
                f"Confidence calculation produced invalid result: {confidence}. Using 0.5 (neutral)."
            )
            confidence = 0.5

        return max(0.0, min(1.0, confidence))

    def _interpret_confidence(self, confidence: float) -> Tuple[str, bool]:
        """
        Interpret confidence score into human-readable level.

        Returns:
            (interpretation, should_flag_for_review)
        """
        if confidence >= self.high_threshold:
            return "HIGH - Strong retrieval confidence", False
        elif confidence >= self.medium_threshold:
            return "MEDIUM - Moderate confidence, review recommended", True
        elif confidence >= self.low_threshold:
            return "LOW - Weak retrieval, mandatory review", True
        else:
            return "VERY LOW - Poor retrieval, expert review required", True

    def _analyze_retrieval_methods(self, chunks: List[Dict]) -> Dict:
        """Analyze which retrieval methods contributed."""
        methods = {
            "hybrid_search": any(c.get("rrf_score") is not None for c in chunks),
            "reranking": any(c.get("rerank_score") is not None for c in chunks),
            "graph_boost": any(c.get("graph_boost", 0.0) > 0 for c in chunks),
            "bm25_only": all(
                c.get("bm25_score") is not None and c.get("dense_score") is None for c in chunks
            ),
            "dense_only": all(
                c.get("dense_score") is not None and c.get("bm25_score") is None for c in chunks
            ),
        }
        return methods

    def _empty_result(self) -> RAGConfidenceScore:
        """Return confidence score for empty results."""
        return RAGConfidenceScore(
            overall_confidence=0.0,
            top_score=0.0,
            score_gap=0.0,
            score_spread=0.0,
            consensus_count=0,
            bm25_dense_agreement=0.0,
            reranker_impact=0.0,
            graph_support=False,
            document_diversity=0.0,
            interpretation="NO RESULTS - No relevant chunks found",
            should_flag=True,
            details={"total_chunks": 0},
        )
