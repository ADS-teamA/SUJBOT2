"""
RAG Configuration for Backend.

This module bridges the backend settings with RAG component configuration.
It loads settings from environment variables and provides them to the RAG pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def load_rag_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load RAG configuration from YAML file or use defaults.

    Args:
        config_path: Optional path to config.yaml file

    Returns:
        Configuration dictionary for RAG components
    """
    # Default configuration
    default_config = {
        "api": {
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "logging": {
                "level": "DEBUG" if settings.VERBOSE_LOGGING else "INFO",
                "format": "json",
                "output": "stdout",
            },
        },
        "document_processing": {
            "max_file_size_mb": settings.MAX_UPLOAD_SIZE // (1024 * 1024),
            "supported_formats": ["pdf", "docx", "txt", "md", "odt", "rtf"],
            "temp_storage_path": str(Path(settings.UPLOAD_DIR).parent / "temp"),
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
            "model": "BAAI/bge-m3",  # Multilingual BGE-M3 for Czech + English
            "device": "cpu",  # Override with "cuda" or "mps" in production
            "normalize": True,
            "batch_size": 32,
        },
        "retrieval": {
            "hybrid_alpha": 0.7,  # 70% semantic, 30% BM25
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
        "reranker": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "device": "cpu",
            "batch_size": 16,
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
        "knowledge_graph": {
            "enable": True,
            "max_entities_per_chunk": 20,
            "relation_threshold": 0.6,
        },
        "claude": {
            "api_key": settings.CLAUDE_API_KEY or "",
            "main_model": settings.MAIN_AGENT_MODEL,
            "subagent_model": settings.SUBAGENT_MODEL,
            "max_parallel_agents": settings.MAX_PARALLEL_AGENTS,
            "timeout": 300,
            "max_retries": 3,
        },
        "storage": {
            "index_dir": settings.INDEX_DIR,
            "upload_dir": settings.UPLOAD_DIR,
            "cache_dir": str(Path(settings.INDEX_DIR).parent / "cache"),
        },
    }

    # Load from YAML if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                # Deep merge YAML config with defaults
                default_config = _deep_merge(default_config, yaml_config)
                logger.info(f"Loaded RAG config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")

    # Environment variable overrides
    _apply_env_overrides(default_config)

    return default_config


def _deep_merge(base: Dict, override: Dict) -> Dict:
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
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: Dict[str, Any]):
    """
    Apply environment variable overrides to config.

    Environment variables follow pattern: RAG_{SECTION}_{KEY}
    Example: RAG_EMBEDDINGS_MODEL, RAG_RETRIEVAL_HYBRID_ALPHA
    """
    # Embeddings overrides
    if model := os.getenv("RAG_EMBEDDINGS_MODEL"):
        config["embeddings"]["model"] = model

    if device := os.getenv("RAG_EMBEDDINGS_DEVICE"):
        config["embeddings"]["device"] = device

    # Retrieval overrides
    if alpha := os.getenv("RAG_RETRIEVAL_HYBRID_ALPHA"):
        try:
            config["retrieval"]["hybrid_alpha"] = float(alpha)
        except ValueError:
            logger.warning(f"Invalid RAG_RETRIEVAL_HYBRID_ALPHA: {alpha}")

    if top_k := os.getenv("RAG_RETRIEVAL_TOP_K"):
        try:
            config["retrieval"]["top_k"] = int(top_k)
        except ValueError:
            logger.warning(f"Invalid RAG_RETRIEVAL_TOP_K: {top_k}")

    if rerank_top_k := os.getenv("RAG_RETRIEVAL_RERANK_TOP_K"):
        try:
            config["retrieval"]["rerank_top_k"] = int(rerank_top_k)
        except ValueError:
            logger.warning(f"Invalid RAG_RETRIEVAL_RERANK_TOP_K: {rerank_top_k}")

    # Indexing overrides
    if chunk_size := os.getenv("RAG_INDEXING_CHUNK_SIZE"):
        try:
            config["indexing"]["chunk_size"] = int(chunk_size)
        except ValueError:
            logger.warning(f"Invalid RAG_INDEXING_CHUNK_SIZE: {chunk_size}")

    # Claude overrides (already handled by settings, but allow explicit override)
    if api_key := os.getenv("CLAUDE_API_KEY"):
        config["claude"]["api_key"] = api_key

    if main_model := os.getenv("MAIN_AGENT_MODEL"):
        config["claude"]["main_model"] = main_model

    if subagent_model := os.getenv("SUBAGENT_MODEL"):
        config["claude"]["subagent_model"] = subagent_model

    logger.debug("Applied environment variable overrides to RAG config")


def get_default_rag_config() -> Dict[str, Any]:
    """
    Get default RAG configuration for production backend.

    Returns:
        Default configuration dictionary
    """
    return load_rag_config()


# Preload configuration on module import
DEFAULT_RAG_CONFIG = get_default_rag_config()
