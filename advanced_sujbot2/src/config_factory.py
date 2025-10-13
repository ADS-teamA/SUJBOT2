"""
Configuration Factory for RAG Components

This module provides factory functions to create component-specific configuration objects
from the central Config instance. This bridges the gap between the centralized YAML/env
configuration and the dataclass configurations used by individual components.

Usage:
    ```python
    from src.config import load_config
    from src.config_factory import create_retrieval_config, create_embedding_config

    # Load central config
    config = load_config("config.yaml")

    # Create component configs
    retrieval_config = create_retrieval_config(config)
    embedding_config = create_embedding_config(config)
    ```
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def create_retrieval_config(config: 'Config') -> 'RetrievalConfig':
    """
    Create RetrievalConfig from central Config.

    Args:
        config: Central configuration object

    Returns:
        RetrievalConfig dataclass instance
    """
    from .hybrid_retriever import RetrievalConfig

    return RetrievalConfig(
        # Weights
        semantic_weight=config.get("retrieval.semantic_weight", 0.5),
        keyword_weight=config.get("retrieval.keyword_weight", 0.3),
        structural_weight=config.get("retrieval.structural_weight", 0.2),

        # Top-K parameters
        top_k=config.get("retrieval.top_k", 20),
        candidate_multiplier=config.get("retrieval.candidate_multiplier", 1.5),

        # Score normalization
        normalize_scores=config.get("retrieval.normalize_scores", True),
        normalization_method=config.get("retrieval.normalization_method", "min-max"),

        # Filters
        enable_metadata_filtering=config.get("retrieval.enable_metadata_filtering", True),
        enable_score_threshold=config.get("retrieval.enable_score_threshold", True),
        min_score_threshold=config.get("retrieval.min_score_threshold", 0.1),

        # Performance
        enable_caching=config.get("retrieval.enable_caching", True),
        parallel_retrieval=config.get("retrieval.parallel_retrieval", True),

        # BM25 parameters
        bm25_k1=config.get("retrieval.bm25.k1", 1.5),
        bm25_b=config.get("retrieval.bm25.b", 0.75),

        # Adaptive weighting
        adaptive_weights=config.get("retrieval.adaptive_weights", True),
        reference_boost=config.get("retrieval.reference_boost", 0.2),

        # Query expansion
        enable_query_expansion=config.get("retrieval.enable_query_expansion", False),
        max_expansions=config.get("retrieval.max_expansions", 3),
    )


def create_embedding_config(config: 'Config') -> 'EmbeddingConfig':
    """
    Create EmbeddingConfig from central Config.

    Args:
        config: Central configuration object

    Returns:
        EmbeddingConfig dataclass instance
    """
    from .embeddings import EmbeddingConfig

    return EmbeddingConfig(
        model_name=config.get("embeddings.model", "joelniklaus/legal-xlm-roberta-base"),
        device=config.get("embeddings.device", "auto"),
        batch_size=config.get("embeddings.batch_size", 32),
        max_sequence_length=config.get("embeddings.max_sequence_length", 512),
        normalize=config.get("embeddings.normalize", True),
        add_hierarchical_context=config.get("embeddings.add_context", True),
        show_progress_bar=config.get("embeddings.show_progress_bar", True),
    )


def create_reranking_config(config: 'Config') -> 'RerankingConfig':
    """
    Create RerankingConfig from central Config (uses RerankingConfig.from_yaml).

    Args:
        config: Central configuration object

    Returns:
        RerankingConfig dataclass instance
    """
    from .reranker import RerankingConfig

    # reranker.py already has from_yaml method, so we use it
    return RerankingConfig.from_yaml(config.to_dict())


def create_cross_doc_config(config: 'Config') -> Dict[str, Any]:
    """
    Create cross-document retrieval configuration dict from central Config.

    Args:
        config: Central configuration object

    Returns:
        Configuration dictionary for ComparativeRetriever
    """
    return {
        # Strategy weights
        "explicit_weight": config.get("cross_document.explicit_weight", 0.5),
        "semantic_weight": config.get("cross_document.semantic_weight", 0.3),
        "structural_weight": config.get("cross_document.structural_weight", 0.2),

        # Thresholds
        "semantic_similarity_threshold": config.get("cross_document.semantic_similarity_threshold", 0.75),
        "structural_similarity_threshold": config.get("cross_document.structural_similarity_threshold", 0.6),
        "min_similarity": config.get("cross_document.min_similarity", 0.5),

        # Limits
        "max_cross_references": config.get("cross_document.max_cross_references", 10),
        "max_similar_provisions": config.get("cross_document.max_similar_provisions", 5),
        "top_k_per_source": config.get("cross_document.top_k_per_source", 10),

        # Performance
        "compute_similarity_matrix": config.get("cross_document.compute_similarity_matrix", False),
    }


def create_knowledge_graph_config(config: 'Config') -> Dict[str, Any]:
    """
    Create knowledge graph configuration dict from central Config.

    Args:
        config: Central configuration object

    Returns:
        Configuration dictionary for LegalKnowledgeGraph
    """
    return {
        "enable": config.get("knowledge_graph.enable", True),
        "build_on_indexing": config.get("knowledge_graph.build_on_indexing", True),
        "include_cross_document_links": config.get("knowledge_graph.include_cross_document_links", True),
        "semantic_link_threshold": config.get("knowledge_graph.semantic_link_threshold", 0.75),
        "max_links_per_node": config.get("knowledge_graph.max_links_per_node", 20),
        "enable_multi_hop": config.get("knowledge_graph.enable_multi_hop", True),
        "max_hops": config.get("knowledge_graph.max_hops", 3),
        "enable_graph_retrieval": config.get("knowledge_graph.enable_graph_retrieval", True),
        "proximity_scoring": config.get("knowledge_graph.proximity_scoring", True),
        "graph_boost_factor": config.get("knowledge_graph.graph_boost_factor", 1.2),
    }


def create_chunking_config(config: 'Config') -> Dict[str, Any]:
    """
    Create chunking configuration dict from central Config.

    Args:
        config: Central configuration object

    Returns:
        Configuration dictionary for LegalChunker
    """
    return {
        "min_chunk_size": config.get("chunking.min_chunk_size", 128),
        "max_chunk_size": config.get("chunking.max_chunk_size", 1024),
        "target_chunk_size": config.get("chunking.target_chunk_size", 512),
        "chunk_overlap": config.get("chunking.chunk_overlap", 0.15),
        "include_context": config.get("chunking.include_context", True),
        "aggregate_small_chunks": config.get("chunking.aggregate_small_chunks", True),
    }


def create_compliance_config(config: 'Config') -> Dict[str, Any]:
    """
    Create compliance checking configuration dict from central Config.

    Args:
        config: Central configuration object

    Returns:
        Configuration dictionary for ComplianceAnalyzer
    """
    return {
        "default_mode": config.get("compliance.default_mode", "exhaustive"),
        "enable_requirement_extraction": config.get("compliance.enable_requirement_extraction", True),
        "enable_clause_mapping": config.get("compliance.enable_clause_mapping", True),
        "enable_conflict_detection": config.get("compliance.enable_conflict_detection", True),
        "enable_gap_analysis": config.get("compliance.enable_gap_analysis", True),
        "enable_deviation_assessment": config.get("compliance.enable_deviation_assessment", True),
        "enable_risk_scoring": config.get("compliance.enable_risk_scoring", True),
        "severity_thresholds": {
            "critical": config.get("compliance.severity_thresholds.critical", 0.9),
            "high": config.get("compliance.severity_thresholds.high", 0.7),
            "medium": config.get("compliance.severity_thresholds.medium", 0.5),
            "low": config.get("compliance.severity_thresholds.low", 0.3),
        },
        "generate_recommendations": config.get("compliance.generate_recommendations", True),
        "max_recommendations": config.get("compliance.max_recommendations", 10),
        "include_evidence": config.get("compliance.include_evidence", True),
        "include_citations": config.get("compliance.include_citations", True),
        "detailed_explanations": config.get("compliance.detailed_explanations", True),
    }


def create_llm_config(config: 'Config') -> Dict[str, Any]:
    """
    Create LLM configuration dict from central Config.

    Args:
        config: Central configuration object

    Returns:
        Configuration dictionary for LLM clients
    """
    return {
        "api_key": config.get("llm.api_key"),
        "main_model": config.get("llm.main_model", "claude-sonnet-4-5-20250929"),
        "sub_model": config.get("llm.sub_model", "claude-3-5-haiku-20241022"),
        "temperature": config.get("llm.temperature", 0.1),
        "max_tokens": config.get("llm.max_tokens", 4000),
        "timeout": config.get("llm.timeout", 120),
        "max_retries": config.get("llm.max_retries", 3),
        "retry_delay": config.get("llm.retry_delay", 1.0),
        "max_requests_per_minute": config.get("llm.max_requests_per_minute", 60),
        "max_tokens_per_minute": config.get("llm.max_tokens_per_minute", 100000),
    }


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_weights(weights: Dict[str, float], name: str = "weights") -> None:
    """
    Validate that weights sum to approximately 1.0.

    Args:
        weights: Dictionary of weight values
        name: Name for error messages

    Raises:
        ValueError: If weights don't sum to 1.0
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"{name} must sum to 1.0, got {total}. "
            f"Weights: {weights}"
        )


def validate_config(config: 'Config') -> bool:
    """
    Validate the entire configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    errors = []

    # Validate retrieval weights
    try:
        retrieval_weights = {
            "semantic": config.get("retrieval.semantic_weight", 0.5),
            "keyword": config.get("retrieval.keyword_weight", 0.3),
            "structural": config.get("retrieval.structural_weight", 0.2),
        }
        validate_weights(retrieval_weights, "retrieval weights")
    except ValueError as e:
        errors.append(str(e))

    # Validate cross-doc weights
    try:
        cross_doc_weights = {
            "explicit": config.get("cross_document.explicit_weight", 0.5),
            "semantic": config.get("cross_document.semantic_weight", 0.3),
            "structural": config.get("cross_document.structural_weight", 0.2),
        }
        validate_weights(cross_doc_weights, "cross-document weights")
    except ValueError as e:
        errors.append(str(e))

    # Validate reranking ensemble weights
    try:
        ensemble_weights = {
            "cross_encoder": config.get("reranking.ensemble_weights.cross_encoder", 0.5),
            "graph": config.get("reranking.ensemble_weights.graph", 0.3),
            "precedence": config.get("reranking.ensemble_weights.precedence", 0.2),
        }
        validate_weights(ensemble_weights, "reranking ensemble weights")
    except ValueError as e:
        errors.append(str(e))

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

    return True


__all__ = [
    "create_retrieval_config",
    "create_embedding_config",
    "create_reranking_config",
    "create_cross_doc_config",
    "create_knowledge_graph_config",
    "create_chunking_config",
    "create_compliance_config",
    "create_llm_config",
    "validate_weights",
    "validate_config",
]
