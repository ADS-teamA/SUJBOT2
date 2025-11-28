"""
Custom exception hierarchy for SUJBOT2.

Provides typed exceptions for better error handling and debugging.
Use these instead of generic Exception where possible.

Exception Hierarchy:
    SujbotError (base)
    ├── ExtractionError → PDFExtractionError, ChunkedExtractionError, JSONRepairError
    ├── ValidationError → ConfigurationError, SchemaValidationError
    ├── ProviderError → APIKeyError, RateLimitError, ProviderTimeoutError, ModelNotFoundError
    ├── ToolExecutionError → ToolNotFoundError, ToolValidationError, ToolTimeoutError
    ├── StorageError → DatabaseConnectionError, VectorStoreError, GraphStoreError
    ├── AgentError → AgentInitializationError, AgentExecutionError, OrchestratorError,
    │                ToolHallucinationError, AgentTimeoutError, MaxIterationsError
    ├── EvaluationError → JudgeError, TrajectoryError, FeedbackSubmissionError
    └── RetrievalError → EmbeddingError, SearchError

Usage:
    from src.exceptions import (
        ExtractionError,
        ValidationError,
        ToolExecutionError,
        ProviderError,
        ToolHallucinationError,
        EvaluationError,
    )

    try:
        result = extract_document(file_path)
    except ExtractionError as e:
        logger.error(f"Extraction failed: {e}")
        # Handle extraction-specific recovery
    except ValidationError as e:
        logger.warning(f"Validation issue: {e}")
        # Handle validation-specific recovery
"""

from typing import Any, Dict, Optional


class SujbotError(Exception):
    """Base exception for all SUJBOT2 errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {type(self.cause).__name__}: {self.cause})"
        return self.message


# =============================================================================
# Extraction Errors
# =============================================================================

class ExtractionError(SujbotError):
    """Error during document extraction (PDF parsing, OCR, etc.)."""
    pass


class PDFExtractionError(ExtractionError):
    """Error extracting from PDF file."""
    pass


class ChunkedExtractionError(ExtractionError):
    """Error during chunked extraction of large documents."""
    pass


class JSONRepairError(ExtractionError):
    """Error repairing malformed JSON from extraction."""
    pass


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(SujbotError):
    """Error validating input data or configuration."""
    pass


class ConfigurationError(ValidationError):
    """Error in configuration (missing keys, invalid values)."""
    pass


class SchemaValidationError(ValidationError):
    """Error validating data against schema."""
    pass


# =============================================================================
# Provider Errors (LLM API)
# =============================================================================

class ProviderError(SujbotError):
    """Error from LLM provider (API error, rate limit, etc.)."""
    pass


class APIKeyError(ProviderError):
    """Missing or invalid API key."""
    pass


class RateLimitError(ProviderError):
    """API rate limit exceeded."""
    pass


class ProviderTimeoutError(ProviderError):
    """LLM provider request timed out."""
    pass


class ModelNotFoundError(ProviderError):
    """Requested model not found or not available."""
    pass


# =============================================================================
# Tool Execution Errors
# =============================================================================

class ToolExecutionError(SujbotError):
    """Error executing a tool."""
    pass


class ToolNotFoundError(ToolExecutionError):
    """Requested tool not found in registry."""
    pass


class ToolValidationError(ToolExecutionError):
    """Tool input validation failed."""
    pass


class ToolTimeoutError(ToolExecutionError):
    """Tool execution timed out."""
    pass


# =============================================================================
# Storage Errors
# =============================================================================

class StorageError(SujbotError):
    """Error in storage layer (database, vector store)."""
    pass


class DatabaseConnectionError(StorageError):
    """Failed to connect to database."""
    pass


class VectorStoreError(StorageError):
    """Error in vector store operations."""
    pass


class GraphStoreError(StorageError):
    """Error in knowledge graph operations."""
    pass


# =============================================================================
# Agent Errors
# =============================================================================

class AgentError(SujbotError):
    """Error in agent execution."""
    pass


class AgentInitializationError(AgentError):
    """Error initializing agent."""
    pass


class AgentExecutionError(AgentError):
    """Error during agent execution."""
    pass


class OrchestratorError(AgentError):
    """Error in orchestrator routing or synthesis."""
    pass


class ToolHallucinationError(AgentError):
    """Agent called a tool that doesn't exist (hallucination)."""
    pass


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""
    pass


class MaxIterationsError(AgentError):
    """Agent exceeded maximum iterations without completing."""
    pass


# =============================================================================
# Evaluation Errors
# =============================================================================

class EvaluationError(SujbotError):
    """Error in evaluation pipeline."""
    pass


class JudgeError(EvaluationError):
    """Error in LLM-as-Judge evaluation."""
    pass


class TrajectoryError(EvaluationError):
    """Error capturing or analyzing trajectory."""
    pass


class FeedbackSubmissionError(EvaluationError):
    """Error submitting feedback to LangSmith."""
    pass


# =============================================================================
# Retrieval Errors
# =============================================================================

class RetrievalError(SujbotError):
    """Error in retrieval pipeline."""
    pass


class EmbeddingError(RetrievalError):
    """Error generating embeddings."""
    pass


class SearchError(RetrievalError):
    """Error performing search."""
    pass


# =============================================================================
# Helper functions
# =============================================================================

def wrap_exception(
    exception: Exception,
    target_type: type = SujbotError,
    message: Optional[str] = None
) -> SujbotError:
    """
    Wrap a generic exception in a typed SUJBOT exception.

    Args:
        exception: Original exception
        target_type: Type of SujbotError to create
        message: Optional custom message (defaults to str(exception))

    Returns:
        Typed SujbotError with original exception as cause
    """
    msg = message or str(exception)
    return target_type(
        message=msg,
        details={"original_type": type(exception).__name__},
        cause=exception
    )


def is_recoverable(exception: Exception) -> bool:
    """
    Check if an exception is recoverable (can continue with degraded functionality).

    Unrecoverable exceptions:
    - KeyboardInterrupt, SystemExit
    - MemoryError, RecursionError
    - ConfigurationError (fix config first)
    - APIKeyError (fix credentials first)

    Returns:
        True if execution can continue with fallback
    """
    unrecoverable_types = (
        KeyboardInterrupt,
        SystemExit,
        MemoryError,
        RecursionError,
        ConfigurationError,
        APIKeyError,
    )
    return not isinstance(exception, unrecoverable_types)
