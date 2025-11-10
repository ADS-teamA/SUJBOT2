"""
Benchmark configuration.

Follows existing SUJBOT2 config pattern with Pydantic dataclasses.
"""

import os
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark evaluation.

    Follows IndexingConfig pattern from src/config.py with from_env() factory.
    """

    # Dataset paths
    dataset_path: str = "benchmark_dataset/privacy_qa.json"
    documents_dir: str = "benchmark_dataset/privacy_qa"

    # Vector store
    vector_store_path: str = "benchmark_db"

    # Retrieval parameters (CRITICAL for apples-to-apples comparison)
    k: int = 5  # Number of chunks to retrieve
    enable_reranking: bool = True
    enable_graph_boost: bool = False
    enable_hybrid_search: bool = True

    # Execution control
    max_queries: Optional[int] = None  # Limit queries for quick tests (None = all)
    debug_mode: bool = False  # Save per-query JSON files
    fail_fast: bool = True  # Stop on first agent error
    rate_limit_delay: float = 0.0  # Seconds to wait between queries (for API rate limits)

    # Agent settings
    agent_model: str = "claude-haiku-4-5"  # Fast model for evaluation
    agent_temperature: float = 0.0  # Deterministic for benchmarking
    enable_prompt_caching: bool = True  # Cost savings

    # Metrics to compute
    metrics: List[str] = None  # Will default in __post_init__

    # Output
    output_dir: str = "benchmark_results"
    save_markdown: bool = True
    save_json: bool = True
    save_per_query: bool = False  # Debug mode: save individual query JSONs

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Default metrics (LegalBench-RAG standard + extras)
        if self.metrics is None:
            self.metrics = [
                "exact_match",  # EM (binary)
                "f1_score",  # Token-level F1
                "precision",  # Token precision
                "recall",  # Token recall
            ]

        # Validate paths exist
        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                f"Expected privacy_qa.json at {dataset_path.absolute()}"
            )

        documents_dir = Path(self.documents_dir)
        if not documents_dir.exists():
            logger.warning(
                f"Documents directory not found: {documents_dir}\n"
                f"This is OK if vector store already indexed."
            )

        # Validate vector store exists (should be indexed before eval)
        vector_store_path = Path(self.vector_store_path)
        if not vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found: {vector_store_path}\n"
                f"Please run indexing first:\n"
                f"  uv run python scripts/index_benchmark_docs.py"
            )

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Debug mode implies save_per_query
        if self.debug_mode:
            self.save_per_query = True

        logger.info(f"Benchmark config validated: {len(self.metrics)} metrics, k={self.k}")

    @classmethod
    def from_env(cls, **overrides) -> "BenchmarkConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            BENCHMARK_DATASET: Path to QA JSON file
            BENCHMARK_VECTOR_STORE: Path to indexed vector store
            BENCHMARK_K: Number of chunks to retrieve
            BENCHMARK_MAX_QUERIES: Limit queries (for testing)
            BENCHMARK_DEBUG: Enable debug mode

        Args:
            **overrides: Override specific config values

        Returns:
            BenchmarkConfig instance

        Example:
            # Load from .env with override
            config = BenchmarkConfig.from_env(max_queries=10)
        """
        # Load from environment
        config_dict = {
            "dataset_path": os.getenv("BENCHMARK_DATASET", "benchmark_dataset/privacy_qa.json"),
            "vector_store_path": os.getenv("BENCHMARK_VECTOR_STORE", "benchmark_db"),
            "k": int(os.getenv("BENCHMARK_K", "5")),
            "max_queries": (
                int(os.getenv("BENCHMARK_MAX_QUERIES"))
                if os.getenv("BENCHMARK_MAX_QUERIES")
                else None
            ),
            "debug_mode": os.getenv("BENCHMARK_DEBUG", "false").lower() == "true",
            "enable_reranking": os.getenv("BENCHMARK_RERANKING", "true").lower() == "true",
            "enable_hybrid_search": os.getenv("BENCHMARK_HYBRID_SEARCH", "true").lower() == "true",
            "agent_model": os.getenv("BENCHMARK_AGENT_MODEL", "claude-haiku-4-5"),
            "rate_limit_delay": float(os.getenv("BENCHMARK_RATE_LIMIT_DELAY", "0.0")),
        }

        # Apply overrides
        config_dict.update(overrides)

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """
        Export configuration as dictionary (for JSON serialization).

        Returns:
            Dict with all config values
        """
        return asdict(self)
