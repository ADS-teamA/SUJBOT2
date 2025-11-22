"""
Utility modules for MY_SUJBOT pipeline.

This package provides shared utilities for:
- API client management
- Batch API processing
- Error sanitization (security)
- Retry logic with exponential backoff
- Persistence (save/load)
- Statistics tracking
- Model registry
- Metadata structures
"""

# Security utilities (CRITICAL - use everywhere)
from .security import sanitize_error, mask_api_key

# API client factory
from .api_clients import APIClientFactory

# Retry decorator
from .retry import retry_with_exponential_backoff

# Batch API client
from .batch_api import BatchAPIClient, BatchRequest

# Persistence utilities
from .persistence import PersistenceManager, VectorStoreLoader

# Statistics utilities
from .statistics import OperationStats, compute_hit_rate

# Model registry
from .model_registry import ModelRegistry

# Metadata structures
from .metadata import ChunkMetadata

# FAISS utilities
from .faiss_utils import reconstruct_all_vectors, get_index_stats, validate_index

__all__ = [
    # Security
    "sanitize_error",
    "mask_api_key",
    # API clients
    "APIClientFactory",
    # Retry
    "retry_with_exponential_backoff",
    # Batch API
    "BatchAPIClient",
    "BatchRequest",
    # Persistence
    "PersistenceManager",
    "VectorStoreLoader",
    # Statistics
    "OperationStats",
    "compute_hit_rate",
    # Model registry
    "ModelRegistry",
    # Metadata
    "ChunkMetadata",
    # FAISS utilities
    "reconstruct_all_vectors",
    "get_index_stats",
    "validate_index",
]
