"""
SUJBOT2 Exception Hierarchy

This module defines all custom exceptions used throughout the system,
providing clear error types for different failure scenarios.
"""

from typing import Optional, Any


class SUJBOT2Error(Exception):
    """
    Base exception for all SUJBOT2 errors.

    All custom exceptions in the system inherit from this base class,
    making it easy to catch any SUJBOT2-specific error.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize SUJBOT2Error.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation of error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# Document-related errors

class DocumentNotFoundError(SUJBOT2Error):
    """
    Document file not found.

    Raised when attempting to read or index a document that doesn't exist
    at the specified path.
    """

    def __init__(self, path: str, details: Optional[dict] = None):
        """
        Initialize DocumentNotFoundError.

        Args:
            path: Path to the missing document
            details: Optional additional context
        """
        message = f"Document not found: {path}"
        if details is None:
            details = {}
        details["path"] = path
        super().__init__(message, details)


class DocumentParsingError(SUJBOT2Error):
    """
    Error parsing document.

    Raised when a document exists but cannot be parsed correctly.
    This could be due to corrupted file, unsupported format, or
    encoding issues.
    """

    def __init__(self, path: str, reason: str, details: Optional[dict] = None):
        """
        Initialize DocumentParsingError.

        Args:
            path: Path to the problematic document
            reason: Explanation of what went wrong
            details: Optional additional context
        """
        message = f"Error parsing document '{path}': {reason}"
        if details is None:
            details = {}
        details.update({"path": path, "reason": reason})
        super().__init__(message, details)


class UnsupportedDocumentTypeError(SUJBOT2Error):
    """
    Document type not supported.

    Raised when attempting to process a document with an unsupported
    file format or extension.
    """

    def __init__(self, path: str, document_type: str, supported_types: list):
        """
        Initialize UnsupportedDocumentTypeError.

        Args:
            path: Path to the document
            document_type: The unsupported type
            supported_types: List of supported types
        """
        message = f"Unsupported document type '{document_type}' for '{path}'. Supported types: {', '.join(supported_types)}"
        details = {
            "path": path,
            "document_type": document_type,
            "supported_types": supported_types
        }
        super().__init__(message, details)


# Indexing errors

class IndexingError(SUJBOT2Error):
    """
    Error during indexing.

    Raised when document indexing fails at any stage (chunking,
    embedding generation, or index storage).
    """

    def __init__(self, message: str, stage: Optional[str] = None, details: Optional[dict] = None):
        """
        Initialize IndexingError.

        Args:
            message: Error description
            stage: Which indexing stage failed (chunking, embedding, indexing)
            details: Optional additional context
        """
        if details is None:
            details = {}
        if stage:
            details["stage"] = stage
        super().__init__(message, details)


class ChunkingError(IndexingError):
    """
    Error during document chunking.

    Raised when semantic chunking or token-based splitting fails.
    """

    def __init__(self, message: str, document_id: Optional[str] = None):
        """
        Initialize ChunkingError.

        Args:
            message: Error description
            document_id: ID of the document being chunked
        """
        details = {}
        if document_id:
            details["document_id"] = document_id
        super().__init__(message, stage="chunking", details=details)


class EmbeddingError(IndexingError):
    """
    Error generating embeddings.

    Raised when embedding model fails to generate vectors for text chunks.
    """

    def __init__(self, message: str, chunk_count: Optional[int] = None):
        """
        Initialize EmbeddingError.

        Args:
            message: Error description
            chunk_count: Number of chunks that failed
        """
        details = {}
        if chunk_count:
            details["chunk_count"] = chunk_count
        super().__init__(message, stage="embedding", details=details)


class IndexStorageError(IndexingError):
    """
    Error storing index.

    Raised when FAISS index or metadata cannot be saved to disk.
    """

    def __init__(self, message: str, index_path: Optional[str] = None):
        """
        Initialize IndexStorageError.

        Args:
            message: Error description
            index_path: Path where index storage failed
        """
        details = {}
        if index_path:
            details["index_path"] = index_path
        super().__init__(message, stage="storage", details=details)


# Retrieval errors

class RetrievalError(SUJBOT2Error):
    """
    Error during retrieval.

    Raised when hybrid retrieval, vector search, or BM25 search fails.
    """

    def __init__(self, message: str, query: Optional[str] = None, details: Optional[dict] = None):
        """
        Initialize RetrievalError.

        Args:
            message: Error description
            query: The query that failed
            details: Optional additional context
        """
        if details is None:
            details = {}
        if query:
            details["query"] = query
        super().__init__(message, details)


class VectorSearchError(RetrievalError):
    """
    Error during vector search.

    Raised when FAISS vector search fails.
    """
    pass


class BM25SearchError(RetrievalError):
    """
    Error during BM25 keyword search.

    Raised when BM25 search fails.
    """
    pass


class RerankingError(RetrievalError):
    """
    Error during cross-encoder reranking.

    Raised when reranking model fails to score results.
    """

    def __init__(self, message: str, candidate_count: Optional[int] = None):
        """
        Initialize RerankingError.

        Args:
            message: Error description
            candidate_count: Number of candidates being reranked
        """
        details = {}
        if candidate_count:
            details["candidate_count"] = candidate_count
        super().__init__(message, details=details)


# Compliance analysis errors

class ComplianceCheckError(SUJBOT2Error):
    """
    Error during compliance check.

    Raised when compliance analysis fails, including provision mapping,
    gap analysis, or report generation.
    """

    def __init__(self, message: str, stage: Optional[str] = None, details: Optional[dict] = None):
        """
        Initialize ComplianceCheckError.

        Args:
            message: Error description
            stage: Which analysis stage failed
            details: Optional additional context
        """
        if details is None:
            details = {}
        if stage:
            details["stage"] = stage
        super().__init__(message, details)


class ProvisionMappingError(ComplianceCheckError):
    """
    Error during provision mapping.

    Raised when automatic mapping between contract and law provisions fails.
    """

    def __init__(self, message: str, contract_id: Optional[str] = None, law_id: Optional[str] = None):
        """
        Initialize ProvisionMappingError.

        Args:
            message: Error description
            contract_id: Contract document ID
            law_id: Law document ID
        """
        details = {}
        if contract_id:
            details["contract_id"] = contract_id
        if law_id:
            details["law_id"] = law_id
        super().__init__(message, stage="provision_mapping", details=details)


class GapAnalysisError(ComplianceCheckError):
    """
    Error during gap analysis.

    Raised when identifying missing requirements or conflicts fails.
    """
    pass


# Query processing errors

class QueryProcessingError(SUJBOT2Error):
    """
    Error processing query.

    Raised when query decomposition, classification, or expansion fails.
    """

    def __init__(self, message: str, query: Optional[str] = None, details: Optional[dict] = None):
        """
        Initialize QueryProcessingError.

        Args:
            message: Error description
            query: The problematic query
            details: Optional additional context
        """
        if details is None:
            details = {}
        if query:
            details["query"] = query
        super().__init__(message, details)


class QueryDecompositionError(QueryProcessingError):
    """
    Error decomposing complex query.

    Raised when query decomposition into sub-questions fails.
    """
    pass


# Knowledge graph errors

class KnowledgeGraphError(SUJBOT2Error):
    """
    Error building or querying knowledge graph.

    Raised when graph construction, reference linking, or semantic
    linking fails.
    """

    def __init__(self, message: str, stage: Optional[str] = None, details: Optional[dict] = None):
        """
        Initialize KnowledgeGraphError.

        Args:
            message: Error description
            stage: Which graph operation failed
            details: Optional additional context
        """
        if details is None:
            details = {}
        if stage:
            details["stage"] = stage
        super().__init__(message, details)


class GraphBuildError(KnowledgeGraphError):
    """
    Error building graph structure.

    Raised when initial graph construction fails.
    """

    def __init__(self, message: str, document_ids: Optional[list] = None):
        """
        Initialize GraphBuildError.

        Args:
            message: Error description
            document_ids: Documents being processed
        """
        details = {}
        if document_ids:
            details["document_ids"] = document_ids
        super().__init__(message, stage="build", details=details)


class ReferenceLinkingError(KnowledgeGraphError):
    """
    Error linking references.

    Raised when automatic reference detection and linking fails.
    """
    pass


class SemanticLinkingError(KnowledgeGraphError):
    """
    Error creating semantic links.

    Raised when semantic similarity linking fails.
    """
    pass


# Configuration errors

class ConfigurationError(SUJBOT2Error):
    """
    Invalid configuration.

    Raised when configuration file is missing, malformed, or contains
    invalid values.
    """

    def __init__(self, message: str, config_path: Optional[str] = None, invalid_fields: Optional[list] = None):
        """
        Initialize ConfigurationError.

        Args:
            message: Error description
            config_path: Path to configuration file
            invalid_fields: List of invalid configuration fields
        """
        details = {}
        if config_path:
            details["config_path"] = config_path
        if invalid_fields:
            details["invalid_fields"] = invalid_fields
        super().__init__(message, details)


class APIKeyError(SUJBOT2Error):
    """
    Missing or invalid API key.

    Raised when Claude API key is missing from environment or config,
    or is invalid.
    """

    def __init__(self, message: str = "Claude API key missing or invalid"):
        """
        Initialize APIKeyError.

        Args:
            message: Error description
        """
        super().__init__(message, {"hint": "Set CLAUDE_API_KEY environment variable"})


# Model errors

class ModelError(SUJBOT2Error):
    """
    Error with ML model.

    Raised when embedding model, reranking model, or LLM fails to load
    or execute.
    """

    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[dict] = None):
        """
        Initialize ModelError.

        Args:
            message: Error description
            model_name: Name of the problematic model
            details: Optional additional context
        """
        if details is None:
            details = {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, details)


class ModelLoadError(ModelError):
    """
    Error loading model.

    Raised when a model fails to load from disk or HuggingFace.
    """
    pass


class ModelInferenceError(ModelError):
    """
    Error during model inference.

    Raised when a loaded model fails during inference.
    """
    pass


# LLM API errors

class LLMAPIError(SUJBOT2Error):
    """
    Error calling LLM API.

    Raised when Claude API calls fail due to rate limits, token limits,
    or network issues.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        """
        Initialize LLMAPIError.

        Args:
            message: Error description
            status_code: HTTP status code if available
            details: Optional additional context
        """
        if details is None:
            details = {}
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details)


class RateLimitError(LLMAPIError):
    """
    Rate limit exceeded.

    Raised when Claude API rate limits are exceeded.
    """

    def __init__(self, retry_after: Optional[int] = None):
        """
        Initialize RateLimitError.

        Args:
            retry_after: Seconds to wait before retrying
        """
        message = "Claude API rate limit exceeded"
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
            message += f" (retry after {retry_after}s)"
        super().__init__(message, status_code=429, details=details)


class TokenLimitError(LLMAPIError):
    """
    Token limit exceeded.

    Raised when request exceeds model's token limit.
    """

    def __init__(self, tokens_used: int, token_limit: int):
        """
        Initialize TokenLimitError.

        Args:
            tokens_used: Number of tokens in request
            token_limit: Model's token limit
        """
        message = f"Token limit exceeded: {tokens_used} tokens (limit: {token_limit})"
        details = {
            "tokens_used": tokens_used,
            "token_limit": token_limit
        }
        super().__init__(message, status_code=400, details=details)


# Batch processing errors

class BatchProcessingError(SUJBOT2Error):
    """
    Error during batch processing.

    Raised when batch compliance check fails.
    """

    def __init__(self, message: str, batch_id: Optional[str] = None, failed_count: Optional[int] = None):
        """
        Initialize BatchProcessingError.

        Args:
            message: Error description
            batch_id: Batch identifier
            failed_count: Number of failed requests
        """
        details = {}
        if batch_id:
            details["batch_id"] = batch_id
        if failed_count:
            details["failed_count"] = failed_count
        super().__init__(message, details)


# Validation errors

class ValidationError(SUJBOT2Error):
    """
    Input validation error.

    Raised when user input fails validation checks.
    """

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """
        Initialize ValidationError.

        Args:
            message: Error description
            field: The invalid field name
            value: The invalid value
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, details)


# Export utility functions

def wrap_exception(original_exception: Exception, new_exception_class: type, message: str) -> SUJBOT2Error:
    """
    Wrap a standard Python exception in a SUJBOT2Error subclass.

    Args:
        original_exception: The original exception
        new_exception_class: SUJBOT2Error subclass to wrap with
        message: New error message

    Returns:
        Wrapped exception with original as cause

    Example:
        try:
            file = open("missing.txt")
        except FileNotFoundError as e:
            raise wrap_exception(e, DocumentNotFoundError, "Document missing")
    """
    wrapped = new_exception_class(message, details={"original_error": str(original_exception)})
    wrapped.__cause__ = original_exception
    return wrapped


__all__ = [
    # Base
    "SUJBOT2Error",

    # Document
    "DocumentNotFoundError",
    "DocumentParsingError",
    "UnsupportedDocumentTypeError",

    # Indexing
    "IndexingError",
    "ChunkingError",
    "EmbeddingError",
    "IndexStorageError",

    # Retrieval
    "RetrievalError",
    "VectorSearchError",
    "BM25SearchError",
    "RerankingError",

    # Compliance
    "ComplianceCheckError",
    "ProvisionMappingError",
    "GapAnalysisError",

    # Query
    "QueryProcessingError",
    "QueryDecompositionError",

    # Knowledge Graph
    "KnowledgeGraphError",
    "GraphBuildError",
    "ReferenceLinkingError",
    "SemanticLinkingError",

    # Configuration
    "ConfigurationError",
    "APIKeyError",

    # Models
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",

    # LLM API
    "LLMAPIError",
    "RateLimitError",
    "TokenLimitError",

    # Batch
    "BatchProcessingError",

    # Validation
    "ValidationError",

    # Utilities
    "wrap_exception",
]
