"""
Configuration utilities for evaluation scripts.

This module centralizes path management and configuration defaults
to eliminate hardcoded values across scripts.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


class EvalPaths:
    """Centralized path management for evaluation scripts."""

    # Base directories
    APP_ROOT = Path("/app")
    RESULTS_DIR = APP_ROOT / "results"
    BENCHMARK_DIR = APP_ROOT / "benchmark_dataset"
    VECTOR_DB_DIR = APP_ROOT / "vector_db"

    # Standard files
    RETRIEVAL_DATASET = BENCHMARK_DIR / "retrieval.json"
    CONFIG_FILE = APP_ROOT / "config.json"

    # Grid search specific
    GRID_SEARCH_RESULTS = RESULTS_DIR / "grid_search_k100"

    @classmethod
    def ensure_dirs_exist(cls):
        """Ensure all required directories exist."""
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_result_path(cls, experiment_name: str, k: int, timestamp: Optional[str] = None) -> Path:
        """
        Generate a result file path.

        Args:
            experiment_name: Name of the experiment
            k: Value of k parameter
            timestamp: Optional timestamp string

        Returns:
            Path object for the result file
        """
        if timestamp:
            filename = f"{experiment_name}_k{k}_{timestamp}.json"
        else:
            filename = f"{experiment_name}_k{k}.json"
        return cls.RESULTS_DIR / filename


class DefaultConfig:
    """Default configuration values for evaluation."""

    # Retrieval parameters
    K_DEFAULT = 100
    USE_HYDE_DEFAULT = False
    NUM_EXPANDS_DEFAULT = 0
    ENABLE_GRAPH_BOOST_DEFAULT = False
    SEARCH_METHOD_DEFAULT = "hybrid"

    # Grid search parameters
    GRID_SEARCH_HYDE_VALUES = [True, False]
    GRID_SEARCH_NUM_EXPANDS_VALUES = [0, 1, 2]
    GRID_SEARCH_SEARCH_METHOD_VALUES = ["hybrid", "dense_only", "bm25_only"]

    # Model defaults (can be overridden by environment)
    DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    @classmethod
    def get_api_keys(cls) -> Dict[str, Optional[str]]:
        """
        Get API keys from environment variables.

        Returns:
            Dictionary of API keys
        """
        return {
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "voyage_api_key": os.getenv("VOYAGE_API_KEY"),
        }

    @classmethod
    def validate_api_keys(cls) -> bool:
        """
        Validate that at least one API key is set.

        Returns:
            True if at least one API key is available
        """
        keys = cls.get_api_keys()
        return any(v for v in keys.values() if v)


def get_experiment_config(
    name: str,
    k: int = None,
    use_hyde: bool = None,
    num_expands: int = None,
    enable_graph_boost: bool = None,
    search_method: str = None
) -> Dict[str, Any]:
    """
    Get configuration for an experiment with defaults.

    Args:
        name: Experiment name
        k: Number of top results
        use_hyde: Whether to use HyDE
        num_expands: Number of query expansions
        enable_graph_boost: Whether to enable graph boosting
        search_method: Search method to use

    Returns:
        Configuration dictionary
    """
    return {
        "name": name,
        "k": k if k is not None else DefaultConfig.K_DEFAULT,
        "use_hyde": use_hyde if use_hyde is not None else DefaultConfig.USE_HYDE_DEFAULT,
        "num_expands": num_expands if num_expands is not None else DefaultConfig.NUM_EXPANDS_DEFAULT,
        "enable_graph_boost": enable_graph_boost if enable_graph_boost is not None else DefaultConfig.ENABLE_GRAPH_BOOST_DEFAULT,
        "search_method": search_method if search_method is not None else DefaultConfig.SEARCH_METHOD_DEFAULT,
    }


def format_config_name(use_hyde: bool, num_expands: int, search_method: str) -> str:
    """
    Generate a standardized configuration name.

    Args:
        use_hyde: Whether HyDE is enabled
        num_expands: Number of query expansions
        search_method: Search method used

    Returns:
        Formatted configuration name
    """
    hyde_str = "hyde" if use_hyde else "nohyde"
    exp_str = f"exp{num_expands}"
    return f"{search_method}_{hyde_str}_{exp_str}"