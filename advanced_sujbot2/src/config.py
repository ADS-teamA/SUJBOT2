"""
Configuration Management for SUJBOT2

This module handles loading, validation, and management of system configuration
from YAML files, environment variables, and programmatic dictionaries.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union
import yaml

from .exceptions import ConfigurationError, APIKeyError


logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for SUJBOT2.

    Supports loading from:
    1. YAML config file
    2. Dictionary passed programmatically
    3. Environment variables (override config values)
    4. Default values

    Priority (highest to lowest):
    1. Environment variables
    2. Programmatic dictionary config
    3. YAML config file
    4. Default config
    """

    # Default configuration
    DEFAULT_CONFIG = {
        "api": {
            "version": "1.0.0",
            "environment": "development",
            "python_api": {
                "enable_progress_callbacks": True,
                "max_parallel_tasks": 5,
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "output": "stdout",
            },
        },
        "document_processing": {
            "max_file_size_mb": 500,
            "supported_formats": ["pdf", "docx"],
            "temp_storage_path": "/tmp/sujbot2",
        },
        "indexing": {
            "chunk_size": 512,
            "chunk_overlap": 0.15,
            "semantic_chunking": True,
            "min_chunk_size": 128,
            "max_chunk_size": 1024,
            "batch_size": 32,
            "max_workers": 8,
        },
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "auto",  # auto (cuda > mps > cpu) | cuda | mps | cpu
            "normalize": True,
            "batch_size": 32,
        },
        "retrieval": {
            "hybrid_alpha": 0.7,
            "top_k": 20,
            "rerank_top_k": 5,
            "min_score": 0.1,
            "enable_reranking": True,
            "enable_query_decomposition": True,
            "enable_query_expansion": True,
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
            },
            "confidence_scoring": {
                "cross_encoder_min": -15,
                "cross_encoder_max": 10,
            },
        },
        "cross_document": {
            "similarity_threshold": 0.75,
            "max_cross_references": 10,
            "enable_provision_linking": True,
        },
        "compliance": {
            "default_mode": "exhaustive",
            "generate_recommendations": True,
            "severity_thresholds": {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.5,
                "low": 0.3,
            },
        },
        "llm": {
            "main_model": "claude-sonnet-4-5-20250929",
            "sub_model": "claude-3-5-haiku-20241022",
            "temperature": 0.1,
            "max_tokens": 4000,
            "timeout": 120,
            "max_retries": 3,
        },
        "knowledge_graph": {
            "enable": True,
            "semantic_link_threshold": 0.75,
            "max_links_per_node": 20,
            "include_cross_document_links": True,
        },
        "batch_processing": {
            "max_parallel": 3,
            "timeout_per_task": 600,
            "continue_on_error": True,
        },
    }

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML config file
            config: Dictionary config (overrides file)

        Raises:
            ConfigurationError: If config file is malformed or invalid
            APIKeyError: If Claude API key is missing
        """
        # Start with defaults
        self._config = self._deep_copy(self.DEFAULT_CONFIG)

        # Load from file if provided
        if config_path:
            self._load_from_file(config_path)

        # Override with programmatic config
        if config:
            self._merge_config(config)

        # Override with environment variables
        self._load_from_env()

        # Validate configuration
        self._validate()

        logger.info("Configuration loaded successfully")

    def _deep_copy(self, d: Dict) -> Dict:
        """Deep copy a dictionary."""
        import copy
        return copy.deepcopy(d)

    def _load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML file

        Raises:
            ConfigurationError: If file not found or malformed
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                config_path=str(config_path)
            )

        try:
            with open(path) as f:
                file_config = yaml.safe_load(f)

            if file_config:
                self._merge_config(file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                config_path=str(config_path)
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error loading configuration file: {e}",
                config_path=str(config_path)
            )

    def _merge_config(self, new_config: Dict) -> None:
        """
        Merge new configuration into existing config.

        Args:
            new_config: New configuration dictionary
        """
        self._config = self._deep_merge(self._config, new_config)

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _load_from_env(self) -> None:
        """Load configuration overrides from environment variables."""
        # ===== LLM Configuration =====
        api_key = os.getenv("CLAUDE_API_KEY")
        if api_key:
            if "llm" not in self._config:
                self._config["llm"] = {}
            self._config["llm"]["api_key"] = api_key

        main_model = os.getenv("MAIN_AGENT_MODEL")
        if main_model:
            self._config["llm"]["main_model"] = main_model

        sub_model = os.getenv("SUBAGENT_MODEL")
        if sub_model:
            self._config["llm"]["sub_model"] = sub_model

        self._load_env_float("LLM_TEMPERATURE", "llm.temperature")
        self._load_env_int("LLM_MAX_TOKENS", "llm.max_tokens")
        self._load_env_int("LLM_TIMEOUT", "llm.timeout")

        # ===== Retrieval Configuration =====
        self._load_env_float("RETRIEVAL_SEMANTIC_WEIGHT", "retrieval.semantic_weight")
        self._load_env_float("RETRIEVAL_KEYWORD_WEIGHT", "retrieval.keyword_weight")
        self._load_env_float("RETRIEVAL_STRUCTURAL_WEIGHT", "retrieval.structural_weight")
        self._load_env_int("RETRIEVAL_TOP_K", "retrieval.top_k")
        self._load_env_float("RETRIEVAL_CANDIDATE_MULTIPLIER", "retrieval.candidate_multiplier")
        self._load_env_bool("RETRIEVAL_NORMALIZE_SCORES", "retrieval.normalize_scores")
        self._load_env_str("RETRIEVAL_NORMALIZATION_METHOD", "retrieval.normalization_method")
        self._load_env_float("RETRIEVAL_MIN_SCORE_THRESHOLD", "retrieval.min_score_threshold")
        self._load_env_float("RETRIEVAL_BM25_K1", "retrieval.bm25.k1")
        self._load_env_float("RETRIEVAL_BM25_B", "retrieval.bm25.b")
        self._load_env_bool("RETRIEVAL_ADAPTIVE_WEIGHTS", "retrieval.adaptive_weights")
        self._load_env_float("RETRIEVAL_REFERENCE_BOOST", "retrieval.reference_boost")
        self._load_env_bool("RETRIEVAL_ENABLE_CACHING", "retrieval.enable_caching")
        self._load_env_bool("RETRIEVAL_PARALLEL", "retrieval.parallel_retrieval")

        # ===== Embedding Configuration =====
        self._load_env_str("EMBEDDING_MODEL", "embeddings.model")
        self._load_env_str("EMBEDDING_DEVICE", "embeddings.device")
        self._load_env_int("EMBEDDING_BATCH_SIZE", "embeddings.batch_size")
        self._load_env_int("EMBEDDING_MAX_SEQ_LENGTH", "embeddings.max_sequence_length")
        self._load_env_bool("EMBEDDING_NORMALIZE", "embeddings.normalize")

        # ===== Reranking Configuration =====
        self._load_env_str("RERANKING_MODEL", "reranking.cross_encoder_model")
        self._load_env_str("RERANKING_DEVICE", "reranking.cross_encoder_device")
        self._load_env_int("RERANKING_BATCH_SIZE", "reranking.cross_encoder_batch_size")
        self._load_env_int("RERANKING_MAX_LENGTH", "reranking.cross_encoder_max_length")
        self._load_env_float("RERANKING_SCORE_MIN", "reranking.cross_encoder_score_min")
        self._load_env_float("RERANKING_SCORE_MAX", "reranking.cross_encoder_score_max")
        self._load_env_float("RERANKING_GRAPH_PROXIMITY_WEIGHT", "reranking.graph_proximity_weight")
        self._load_env_float("RERANKING_GRAPH_CENTRALITY_WEIGHT", "reranking.graph_centrality_weight")
        self._load_env_float("RERANKING_GRAPH_AUTHORITY_WEIGHT", "reranking.graph_authority_weight")
        self._load_env_int("RERANKING_MAX_HOP_DISTANCE", "reranking.max_hop_distance")
        self._load_env_float("RERANKING_PRECEDENCE_CONSTITUTIONAL", "reranking.precedence_weights.constitutional")
        self._load_env_float("RERANKING_PRECEDENCE_STATUTORY", "reranking.precedence_weights.statutory")
        self._load_env_float("RERANKING_PRECEDENCE_REGULATORY", "reranking.precedence_weights.regulatory")
        self._load_env_float("RERANKING_PRECEDENCE_CONTRACTUAL", "reranking.precedence_weights.contractual")
        self._load_env_float("RERANKING_PRECEDENCE_GUIDANCE", "reranking.precedence_weights.guidance")
        self._load_env_float("RERANKING_TEMPORAL_DECAY", "reranking.temporal_decay_factor")
        self._load_env_float("RERANKING_ENSEMBLE_CROSS_ENCODER", "reranking.ensemble_weights.cross_encoder")
        self._load_env_float("RERANKING_ENSEMBLE_GRAPH", "reranking.ensemble_weights.graph")
        self._load_env_float("RERANKING_ENSEMBLE_PRECEDENCE", "reranking.ensemble_weights.precedence")
        self._load_env_int("RERANKING_FINAL_TOP_K", "reranking.final_top_k")
        self._load_env_float("RERANKING_MIN_CONFIDENCE", "reranking.min_confidence_threshold")
        self._load_env_bool("RERANKING_EXPLAIN", "reranking.explain_reranking")

        # ===== Cross-Document Retrieval =====
        self._load_env_float("CROSS_DOC_EXPLICIT_WEIGHT", "cross_document.explicit_weight")
        self._load_env_float("CROSS_DOC_SEMANTIC_WEIGHT", "cross_document.semantic_weight")
        self._load_env_float("CROSS_DOC_STRUCTURAL_WEIGHT", "cross_document.structural_weight")
        self._load_env_float("CROSS_DOC_SEMANTIC_THRESHOLD", "cross_document.semantic_similarity_threshold")
        self._load_env_float("CROSS_DOC_STRUCTURAL_THRESHOLD", "cross_document.structural_similarity_threshold")
        self._load_env_float("CROSS_DOC_MIN_SIMILARITY", "cross_document.min_similarity")
        self._load_env_int("CROSS_DOC_MAX_CROSS_REFS", "cross_document.max_cross_references")
        self._load_env_int("CROSS_DOC_MAX_SIMILAR_PROVISIONS", "cross_document.max_similar_provisions")
        self._load_env_int("CROSS_DOC_TOP_K", "cross_document.top_k_per_source")
        self._load_env_bool("CROSS_DOC_COMPUTE_SIMILARITY_MATRIX", "cross_document.compute_similarity_matrix")

        # ===== Knowledge Graph =====
        self._load_env_bool("ENABLE_KNOWLEDGE_GRAPH", "knowledge_graph.enable")
        self._load_env_float("SEMANTIC_LINK_THRESHOLD", "knowledge_graph.semantic_link_threshold")
        self._load_env_bool("KNOWLEDGE_GRAPH_BUILD_ON_INDEXING", "knowledge_graph.build_on_indexing")
        self._load_env_bool("KNOWLEDGE_GRAPH_INCLUDE_CROSS_DOC_LINKS", "knowledge_graph.include_cross_document_links")
        self._load_env_int("KNOWLEDGE_GRAPH_MAX_LINKS", "knowledge_graph.max_links_per_node")
        self._load_env_int("KNOWLEDGE_GRAPH_MAX_HOPS", "knowledge_graph.max_hops")
        self._load_env_float("KNOWLEDGE_GRAPH_BOOST_FACTOR", "knowledge_graph.graph_boost_factor")

        # ===== Chunking =====
        self._load_env_int("CHUNKING_MIN_SIZE", "chunking.min_chunk_size")
        self._load_env_int("CHUNKING_MAX_SIZE", "chunking.max_chunk_size")
        self._load_env_int("CHUNKING_TARGET_SIZE", "chunking.target_chunk_size")
        self._load_env_float("CHUNKING_OVERLAP", "chunking.chunk_overlap")

        # ===== Logging Configuration =====
        log_level = os.getenv("LOG_LEVEL") or os.getenv("VERBOSE_LOGGING")
        if log_level:
            if log_level.lower() in ["true", "1", "yes"]:
                self._config["api"]["logging"]["level"] = "DEBUG"
            elif log_level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                self._config["api"]["logging"]["level"] = log_level.upper()

        # ===== Performance =====
        max_parallel = os.getenv("MAX_PARALLEL_AGENTS")
        if max_parallel:
            try:
                self._config["api"]["python_api"]["max_parallel_tasks"] = int(max_parallel)
            except ValueError:
                logger.warning(f"Invalid MAX_PARALLEL_AGENTS value: {max_parallel}")

        # ===== Feature Flags =====
        enable_decomp = os.getenv("ENABLE_QUESTION_DECOMPOSITION")
        if enable_decomp:
            self._config["retrieval"]["enable_query_decomposition"] = enable_decomp.lower() in ["true", "1", "yes"]

        # ===== Tokenizers =====
        tokenizers_parallel = os.getenv("TOKENIZERS_PARALLELISM")
        if tokenizers_parallel:
            os.environ["TOKENIZERS_PARALLELISM"] = tokenizers_parallel

    def _load_env_str(self, env_key: str, config_path: str) -> None:
        """Load string value from environment variable."""
        value = os.getenv(env_key)
        if value:
            self.set(config_path, value)

    def _load_env_int(self, env_key: str, config_path: str) -> None:
        """Load integer value from environment variable."""
        value = os.getenv(env_key)
        if value:
            try:
                self.set(config_path, int(value))
            except ValueError:
                logger.warning(f"Invalid integer value for {env_key}: {value}")

    def _load_env_float(self, env_key: str, config_path: str) -> None:
        """Load float value from environment variable."""
        value = os.getenv(env_key)
        if value:
            try:
                self.set(config_path, float(value))
            except ValueError:
                logger.warning(f"Invalid float value for {env_key}: {value}")

    def _load_env_bool(self, env_key: str, config_path: str) -> None:
        """Load boolean value from environment variable."""
        value = os.getenv(env_key)
        if value:
            self.set(config_path, value.lower() in ["true", "1", "yes", "on"])

    def _validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ConfigurationError: If configuration is invalid
            APIKeyError: If API key is missing
        """
        invalid_fields = []

        # Check required fields
        if "llm" not in self._config or "api_key" not in self._config["llm"]:
            raise APIKeyError()

        # Validate chunk size
        if self._config["indexing"]["chunk_size"] < 1:
            invalid_fields.append("indexing.chunk_size (must be positive)")

        # Validate chunk overlap
        overlap = self._config["indexing"]["chunk_overlap"]
        if not (0 <= overlap < 1):
            invalid_fields.append("indexing.chunk_overlap (must be in [0, 1))")

        # Validate hybrid alpha
        alpha = self._config["retrieval"]["hybrid_alpha"]
        if not (0 <= alpha <= 1):
            invalid_fields.append("retrieval.hybrid_alpha (must be in [0, 1])")

        # Validate top_k values
        if self._config["retrieval"]["top_k"] < 1:
            invalid_fields.append("retrieval.top_k (must be positive)")

        if self._config["retrieval"]["rerank_top_k"] < 1:
            invalid_fields.append("retrieval.rerank_top_k (must be positive)")

        if self._config["retrieval"]["rerank_top_k"] > self._config["retrieval"]["top_k"]:
            invalid_fields.append("retrieval.rerank_top_k (must be <= top_k)")

        # Validate temperature
        temp = self._config["llm"]["temperature"]
        if not (0 <= temp <= 1):
            invalid_fields.append("llm.temperature (must be in [0, 1])")

        # Validate max_tokens
        if self._config["llm"]["max_tokens"] < 1:
            invalid_fields.append("llm.max_tokens (must be positive)")

        if invalid_fields:
            raise ConfigurationError(
                "Invalid configuration values",
                invalid_fields=invalid_fields
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "retrieval.hybrid_alpha")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config.get("llm.main_model")
            "claude-sonnet-4-5-20250929"
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "retrieval.hybrid_alpha")
            value: New value

        Example:
            >>> config.set("llm.temperature", 0.2)
        """
        keys = key.split(".")
        target = self._config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def to_dict(self) -> Dict:
        """
        Get complete configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._deep_copy(self._config)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output file path

        Raises:
            ConfigurationError: If save fails
        """
        try:
            # Don't save sensitive data
            safe_config = self._deep_copy(self._config)
            if "llm" in safe_config and "api_key" in safe_config["llm"]:
                safe_config["llm"]["api_key"] = "***REDACTED***"

            with open(path, "w") as f:
                yaml.dump(safe_config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            raise ConfigurationError(
                f"Error saving configuration: {e}",
                config_path=str(path)
            )

    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value using dict-like syntax.

        Args:
            key: Top-level key

        Returns:
            Configuration value
        """
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._config

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(environment={self.get('api.environment')}, model={self.get('llm.main_model')})"


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict] = None,
) -> Config:
    """
    Load configuration from file or dict.

    Convenience function for creating Config instance.

    Args:
        config_path: Path to YAML config file
        config: Dictionary config (overrides file)

    Returns:
        Config instance

    Raises:
        ConfigurationError: If config is invalid
        APIKeyError: If API key is missing

    Example:
        >>> config = load_config("config.yaml")
        >>> config = load_config(config={"llm": {"temperature": 0.2}})
    """
    return Config(config_path=config_path, config=config)


def get_default_config() -> Dict:
    """
    Get default configuration dictionary.

    Returns:
        Default configuration

    Example:
        >>> defaults = get_default_config()
        >>> defaults["llm"]["main_model"]
        "claude-sonnet-4-5-20250929"
    """
    import copy
    return copy.deepcopy(Config.DEFAULT_CONFIG)


__all__ = [
    "Config",
    "load_config",
    "get_default_config",
]
