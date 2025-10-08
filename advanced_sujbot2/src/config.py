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
            "device": "cpu",
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
        # Claude API key
        api_key = os.getenv("CLAUDE_API_KEY")
        if api_key:
            if "llm" not in self._config:
                self._config["llm"] = {}
            self._config["llm"]["api_key"] = api_key

        # Main model
        main_model = os.getenv("MAIN_AGENT_MODEL")
        if main_model:
            self._config["llm"]["main_model"] = main_model

        # Sub model
        sub_model = os.getenv("SUBAGENT_MODEL")
        if sub_model:
            self._config["llm"]["sub_model"] = sub_model

        # Logging level
        log_level = os.getenv("LOG_LEVEL") or os.getenv("VERBOSE_LOGGING")
        if log_level:
            if log_level.lower() in ["true", "1", "yes"]:
                self._config["api"]["logging"]["level"] = "DEBUG"
            elif log_level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                self._config["api"]["logging"]["level"] = log_level.upper()

        # Max parallel agents
        max_parallel = os.getenv("MAX_PARALLEL_AGENTS")
        if max_parallel:
            try:
                self._config["api"]["python_api"]["max_parallel_tasks"] = int(max_parallel)
            except ValueError:
                logger.warning(f"Invalid MAX_PARALLEL_AGENTS value: {max_parallel}")

        # Enable question decomposition
        enable_decomp = os.getenv("ENABLE_QUESTION_DECOMPOSITION")
        if enable_decomp:
            self._config["retrieval"]["enable_query_decomposition"] = enable_decomp.lower() in ["true", "1", "yes"]

        # Tokenizers parallelism
        tokenizers_parallel = os.getenv("TOKENIZERS_PARALLELISM")
        if tokenizers_parallel:
            # Just set it back to environment for libraries to use
            os.environ["TOKENIZERS_PARALLELISM"] = tokenizers_parallel

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
