"""
SUJBOT2 - Legal Compliance Checking System

This package provides tools for automated legal compliance checking using
hybrid retrieval, semantic analysis, and Claude AI.

Main API:
    ComplianceChecker - Main orchestrator for compliance checks
    BatchProcessor - Batch processing of multiple documents

Quick Start:
    >>> from advanced_sujbot2 import ComplianceChecker, ComplianceCheckRequest
    >>> import asyncio
    >>>
    >>> async def main():
    ...     checker = ComplianceChecker(config_path="config.yaml")
    ...     request = ComplianceCheckRequest(
    ...         contract_path="contract.pdf",
    ...         law_paths=["law.pdf"]
    ...     )
    ...     report = await checker.check_compliance(request)
    ...     print(f"Compliance score: {report.overall_compliance_score:.2%}")
    >>>
    >>> asyncio.run(main())
"""

__version__ = "1.0.0"
__author__ = "SUJBOT2 Team"

# Core API
from .api import ComplianceChecker
from .batch_processor import BatchProcessor

# Configuration
from .config import Config, load_config, get_default_config

# Data Models
from .models import (
    # Enums
    ComplianceMode,
    DocumentType,
    IndexingStage,
    AnalysisStage,
    SeverityLevel,
    RetrievalStrategy,

    # Request models
    ComplianceCheckRequest,

    # Progress models
    IndexingProgress,
    AnalysisProgress,

    # Response models
    Source,
    ProcessedQuery,
    QueryResponse,
    ComplianceIssue,
    ComplianceReport,
    BatchResult,

    # Document models (from existing models.py)
    LegalDocument,
    DocumentMetadata,
    StructuralElement,
    Part,
    Chapter,
    Section,
    Paragraph,
    Subsection,
    Letter,
    Article,
    Point,
    LegalReference,
    DocumentStructure,
)

# Exceptions
from .exceptions import (
    # Base
    SUJBOT2Error,

    # Document errors
    DocumentNotFoundError,
    DocumentParsingError,
    UnsupportedDocumentTypeError,

    # Indexing errors
    IndexingError,
    ChunkingError,
    EmbeddingError,
    IndexStorageError,

    # Retrieval errors
    RetrievalError,
    VectorSearchError,
    BM25SearchError,
    RerankingError,

    # Compliance errors
    ComplianceCheckError,
    ProvisionMappingError,
    GapAnalysisError,

    # Query errors
    QueryProcessingError,
    QueryDecompositionError,

    # Knowledge graph errors
    KnowledgeGraphError,
    GraphBuildError,
    ReferenceLinkingError,
    SemanticLinkingError,

    # Configuration errors
    ConfigurationError,
    APIKeyError,

    # Model errors
    ModelError,
    ModelLoadError,
    ModelInferenceError,

    # LLM errors
    LLMAPIError,
    RateLimitError,
    TokenLimitError,

    # Batch errors
    BatchProcessingError,

    # Validation errors
    ValidationError,
)


# Public API
__all__ = [
    # Version
    "__version__",
    "__author__",

    # Core API
    "ComplianceChecker",
    "BatchProcessor",

    # Configuration
    "Config",
    "load_config",
    "get_default_config",

    # Enums
    "ComplianceMode",
    "DocumentType",
    "IndexingStage",
    "AnalysisStage",
    "SeverityLevel",
    "RetrievalStrategy",

    # Request models
    "ComplianceCheckRequest",

    # Progress models
    "IndexingProgress",
    "AnalysisProgress",

    # Response models
    "Source",
    "ProcessedQuery",
    "QueryResponse",
    "ComplianceIssue",
    "ComplianceReport",
    "BatchResult",

    # Document models
    "LegalDocument",
    "DocumentMetadata",
    "StructuralElement",
    "Part",
    "Chapter",
    "Section",
    "Paragraph",
    "Subsection",
    "Letter",
    "Article",
    "Point",
    "LegalReference",
    "DocumentStructure",

    # Exceptions
    "SUJBOT2Error",
    "DocumentNotFoundError",
    "DocumentParsingError",
    "UnsupportedDocumentTypeError",
    "IndexingError",
    "ChunkingError",
    "EmbeddingError",
    "IndexStorageError",
    "RetrievalError",
    "VectorSearchError",
    "BM25SearchError",
    "RerankingError",
    "ComplianceCheckError",
    "ProvisionMappingError",
    "GapAnalysisError",
    "QueryProcessingError",
    "QueryDecompositionError",
    "KnowledgeGraphError",
    "GraphBuildError",
    "ReferenceLinkingError",
    "SemanticLinkingError",
    "ConfigurationError",
    "APIKeyError",
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "LLMAPIError",
    "RateLimitError",
    "TokenLimitError",
    "BatchProcessingError",
    "ValidationError",
]


# Package-level convenience functions

def create_checker(config_path: str = None, **config_overrides) -> ComplianceChecker:
    """
    Create a ComplianceChecker instance with optional config overrides.

    Args:
        config_path: Path to config file
        **config_overrides: Config values to override

    Returns:
        ComplianceChecker instance

    Example:
        >>> checker = create_checker("config.yaml", retrieval={"hybrid_alpha": 0.8})
    """
    if config_overrides:
        config = load_config(config_path=config_path)
        for key, value in config_overrides.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    config.set(f"{key}.{subkey}", subvalue)
            else:
                config.set(key, value)
        return ComplianceChecker(config=config)
    return ComplianceChecker(config_path=config_path)


def get_version() -> str:
    """Get package version."""
    return __version__
