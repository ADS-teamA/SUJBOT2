"""
LlamaIndex-based indexing pipeline with state management.

This package provides a wrapper around the existing IndexingPipeline
that adds Redis-backed state persistence, document deduplication,
and entity labeling via Gemini 2.5 Flash.

Components:
    - SujbotIngestionPipeline: Main wrapper with caching and partial re-indexing
    - GeminiEntityLabeler: TransformComponent for entity extraction
"""

from src.indexing.llamaindex_wrapper import SujbotIngestionPipeline

__all__ = ["SujbotIngestionPipeline"]
