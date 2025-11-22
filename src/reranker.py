"""
PHASE 5C: Cross-Encoder Reranking

Implements two-stage retrieval:
1. Fast retrieval: Hybrid search (FAISS + BM25 + RRF) → 50 candidates
2. Precise reranking: Cross-encoder → Top 6 results

Based on research:
- Two-stage retrieval: +25% accuracy improvement
- Cross-encoders: Deeper semantic understanding than bi-encoders
- Legal documents: Test multiple models (Cohere reranker failed in LegalBench-RAG)
- Optimal candidate set: 50-75 documents for reranking

Architecture:
- CrossEncoderReranker: Main reranking class
- Multiple model support: ms-marco, BGE, custom models
- Batch processing for efficiency
- Performance monitoring and statistics

Usage:
    from reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(
        model_name='bge-reranker-large'  # SOTA accuracy
    )

    reranked_results = reranker.rerank(
        query="waste disposal requirements",
        candidates=candidates,
        top_k=6
    )
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError(
        "sentence-transformers required for reranking. "
        "Install with: pip install sentence-transformers"
    )

# Import ModelRegistry for centralized model management
from src.utils.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class RerankingStats:
    """Statistics from reranking operation."""

    candidates: int
    final_results: int
    rerank_time_ms: float
    rank_changes: int  # How many chunks changed position
    score_correlation: Optional[float] = None  # Correlation between old and new scores
    model_name: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "candidates": self.candidates,
            "final_results": self.final_results,
            "rerank_time_ms": round(self.rerank_time_ms, 2),
            "rank_changes": self.rank_changes,
            "score_correlation": (
                round(self.score_correlation, 3) if self.score_correlation else None
            ),
            "model_name": self.model_name,
        }


class CrossEncoderReranker:
    """
    Cross-encoder reranker for two-stage retrieval.

    Cross-encoders process query-document pairs jointly, achieving deeper
    semantic understanding than bi-encoders (which encode independently).

    Expected improvement: +25% accuracy over hybrid search alone.

    IMPORTANT: Test on legal documents! Cohere reranker failed in LegalBench-RAG research.
    """

    def __init__(
        self,
        model_name: str = "bge-reranker-large",  # SOTA accuracy (was: ms-marco-mini)
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Model name or alias from ModelRegistry.RERANKER_MODELS
            device: Device for inference ('cpu', 'cuda', 'mps')
            max_length: Maximum sequence length (default: 512)
            batch_size: Batch size for inference (default: 32)
        """
        # Resolve model alias using centralized ModelRegistry
        self.model_name = ModelRegistry.resolve_reranker(model_name)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Loading cross-encoder: {self.model_name} on {device}")

        # Load model
        try:
            self.model = CrossEncoder(self.model_name, device=device, max_length=max_length)
            logger.info(f"✓ Cross-encoder loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise

        # Statistics
        self.total_reranks = 0
        self.total_time_ms = 0.0

    def rerank(
        self, query: str, candidates: List[Dict], top_k: int = 6, return_stats: bool = False
    ) -> List[Dict] | Tuple[List[Dict], RerankingStats]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Query string
            candidates: List of candidate dicts (from hybrid search)
                Each must have 'content' field for reranking
            top_k: Number of final results to return (default: 6)
            return_stats: Return statistics along with results

        Returns:
            List of reranked dicts with 'rerank_score' added
            Or tuple (results, stats) if return_stats=True
        """
        if not candidates:
            logger.warning("No candidates to rerank")
            if return_stats:
                stats = RerankingStats(
                    candidates=0,
                    final_results=0,
                    rerank_time_ms=0.0,
                    rank_changes=0,
                    model_name=self.model_name,
                )
                return [], stats
            return []

        start_time = time.time()

        # Store original rankings for comparison
        original_ids = [c.get("chunk_id", f"unknown_{i}") for i, c in enumerate(candidates)]
        original_scores = [c.get("rrf_score", c.get("score", 0.0)) for c in candidates]

        logger.info(f"Reranking {len(candidates)} candidates to top {top_k}")

        # Prepare query-document pairs
        pairs = []
        for candidate in candidates:
            content = candidate.get("content", "")
            if not content:
                logger.warning(
                    f"Candidate missing 'content': {candidate.get('chunk_id', 'unknown')}"
                )
                content = ""
            pairs.append([query, content])

        # Score all pairs with cross-encoder
        try:
            scores = self.model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
            )
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            # Fallback: return original candidates
            if return_stats:
                stats = RerankingStats(
                    candidates=len(candidates),
                    final_results=len(candidates[:top_k]),
                    rerank_time_ms=0.0,
                    rank_changes=0,
                    score_correlation=1.0,
                    model_name=self.model_name,
                )
                return candidates[:top_k], stats
            return candidates[:top_k]

        # Add rerank scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
            # Preserve original scores for analysis
            candidate["original_score"] = candidate.get("rrf_score", candidate.get("score", 0.0))

        # Sort by rerank score (ascending: lowest confidence first, highest last)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=False)[:top_k]

        # Calculate statistics
        elapsed_ms = (time.time() - start_time) * 1000
        self.total_reranks += 1
        self.total_time_ms += elapsed_ms

        # Compute rank changes
        reranked_ids = [c.get("chunk_id", f"unknown_{i}") for i, c in enumerate(reranked)]
        rank_changes = len(set(reranked_ids) - set(original_ids[:top_k]))

        # Compute score correlation
        if len(original_scores) > 1 and len(scores) > 1:
            correlation = float(np.corrcoef(original_scores, scores)[0, 1])
        else:
            correlation = None

        logger.info(
            f"Reranking complete: {len(candidates)} → {len(reranked)} "
            f"in {elapsed_ms:.1f}ms, rank_changes={rank_changes}"
        )

        if return_stats:
            stats = RerankingStats(
                candidates=len(candidates),
                final_results=len(reranked),
                rerank_time_ms=elapsed_ms,
                rank_changes=rank_changes,
                score_correlation=correlation,
                model_name=self.model_name,
            )
            return reranked, stats

        return reranked

    def rerank_with_threshold(
        self, query: str, candidates: List[Dict], min_score: float = 0.0, top_k: int = 6
    ) -> List[Dict]:
        """
        Rerank with minimum score threshold.

        Args:
            query: Query string
            candidates: Candidate dicts
            min_score: Minimum rerank score to include (default: 0.0)
            top_k: Maximum results to return

        Returns:
            Filtered and reranked results
        """
        reranked = self.rerank(query, candidates, top_k=len(candidates))

        # Filter by threshold
        filtered = [c for c in reranked if c["rerank_score"] >= min_score]

        logger.info(
            f"Threshold filtering: {len(reranked)} → {len(filtered)} " f"(min_score={min_score})"
        )

        return filtered[:top_k]

    def get_stats(self) -> Dict:
        """Get reranker statistics."""
        avg_time_ms = self.total_time_ms / self.total_reranks if self.total_reranks > 0 else 0

        return {
            "model_name": self.model_name,
            "device": self.device,
            "total_reranks": self.total_reranks,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(avg_time_ms, 2),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.total_reranks = 0
        self.total_time_ms = 0.0
        logger.info("Reranker statistics reset")


# Convenience function for one-shot reranking
def rerank_results(
    query: str,
    candidates: List[Dict],
    top_k: int = 6,
    model_name: str = "default",
    device: str = "cpu",
) -> List[Dict]:
    """
    Convenience function for one-shot reranking.

    Args:
        query: Query string
        candidates: Candidate results
        top_k: Number of results to return
        model_name: Reranker model name or alias
        device: Device for inference

    Returns:
        Reranked results
    """
    reranker = CrossEncoderReranker(model_name=model_name, device=device)
    return reranker.rerank(query, candidates, top_k=top_k)


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    print("=== PHASE 5C: Cross-Encoder Reranking Example ===\n")

    # Example candidates (from hybrid search)
    candidates = [
        {
            "chunk_id": "chunk_1",
            "content": "The waste disposal requirements specify proper handling of hazardous materials.",
            "rrf_score": 0.031,
        },
        {
            "chunk_id": "chunk_2",
            "content": "Safety equipment must be worn at all times in the facility.",
            "rrf_score": 0.029,
        },
        {
            "chunk_id": "chunk_3",
            "content": "Proper waste disposal procedures include segregation, labeling, and secure storage.",
            "rrf_score": 0.028,
        },
        {
            "chunk_id": "chunk_4",
            "content": "The company provides training on environmental regulations.",
            "rrf_score": 0.025,
        },
    ]

    print("1. Initialize cross-encoder reranker:")
    print("   reranker = CrossEncoderReranker(model_name='bge-reranker-large')")
    print("")

    # Initialize reranker
    reranker = CrossEncoderReranker(model_name="bge-reranker-large")

    print("2. Rerank candidates:")
    query = "waste disposal requirements"
    print(f"   Query: '{query}'")
    print(f"   Candidates: {len(candidates)}")
    print("")

    # Rerank with statistics
    reranked, stats = reranker.rerank(
        query=query, candidates=candidates, top_k=3, return_stats=True
    )

    print("3. Results:")
    print(f"   Original ranking (by RRF score):")
    for i, c in enumerate(candidates, 1):
        print(f"      {i}. (RRF: {c['rrf_score']:.4f}) {c['content'][:60]}...")
    print("")

    print(f"   Reranked (by cross-encoder):")
    for i, c in enumerate(reranked, 1):
        print(f"      {i}. (Rerank: {c['rerank_score']:.4f}, RRF: {c['original_score']:.4f})")
        print(f"         {c['content'][:60]}...")
    print("")

    print("4. Statistics:")
    for key, value in stats.to_dict().items():
        print(f"   {key}: {value}")
    print("")

    print("5. Model registry:")
    print("   Available models:")
    for alias, model in ModelRegistry.RERANKER_MODELS.items():
        print(f"      {alias:20s} → {model}")
    print("")

    print("=== Implementation complete! ===")
