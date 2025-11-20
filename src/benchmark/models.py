"""
Benchmark data models - Shared dataclasses for benchmark results.

These models are used by both legacy and multi-agent benchmark runners.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class QueryResult:
    """
    Result for a single query evaluation.

    Attributes:
        query_id: Unique query identifier
        query: Question text
        predicted_answer: Model's answer
        ground_truth_answers: List of expected answers
        metrics: Dict of metric scores
        retrieval_time_ms: Time to retrieve + generate answer
        cost_usd: API cost for this query
        rag_confidence: RAG confidence score (0-1) or None if unavailable
    """

    query_id: int
    query: str
    predicted_answer: str
    ground_truth_answers: List[str]
    metrics: Dict[str, float]
    retrieval_time_ms: float
    cost_usd: float
    rag_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        result = {
            "query_id": self.query_id,
            "query": self.query,
            "predicted_answer": self.predicted_answer,
            "ground_truth_answers": self.ground_truth_answers,
            "metrics": self.metrics,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
        }

        # Add RAG confidence if available
        if self.rag_confidence is not None:
            result["rag_confidence"] = round(self.rag_confidence, 4)

        return result


@dataclass
class BenchmarkResult:
    """
    Complete benchmark evaluation results.

    Attributes:
        dataset_name: Name of dataset evaluated
        total_queries: Number of queries evaluated
        aggregate_metrics: Mean scores per metric
        query_results: Per-query detailed results
        total_time_seconds: Total evaluation time
        total_cost_usd: Total API cost
        config: Benchmark configuration
        timestamp: When evaluation was run
    """

    dataset_name: str
    total_queries: int
    aggregate_metrics: Dict[str, float]
    query_results: List[QueryResult]
    total_time_seconds: float
    total_cost_usd: float
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "dataset_name": self.dataset_name,
            "total_queries": self.total_queries,
            "aggregate_metrics": self.aggregate_metrics,
            "query_results": [qr.to_dict() for qr in self.query_results],
            "total_time_seconds": round(self.total_time_seconds, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_time_per_query_ms": (
                round((self.total_time_seconds * 1000) / self.total_queries, 2)
                if self.total_queries > 0
                else 0
            ),
            "cost_per_query_usd": (
                round(self.total_cost_usd / self.total_queries, 6) if self.total_queries > 0 else 0
            ),
            "config": self.config,
            "timestamp": self.timestamp,
        }
