"""
Unstructured.io-based document extraction module.

This module provides advanced document structure extraction using Unstructured.io,
with specialized support for legal and technical documents.
"""

# Configuration (centralized)
from .config import ExtractionConfig, RAGConfig

# Unstructured extractor with hierarchy detection and summaries
from .unstructured_extractor import UnstructuredExtractor
from .summary_generator import SummaryGenerator

# PHASE 3: Multi-layer chunking with SAC
from .multi_layer_chunker import MultiLayerChunker, Chunk, ChunkMetadata

# PHASE 4: Embedding & FAISS indexing
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig
from .faiss_vector_store import FAISSVectorStore
from .indexing_pipeline import IndexingPipeline, IndexingConfig

__all__ = [
    # Configuration
    "ExtractionConfig",
    "RAGConfig",
    # Extraction
    "UnstructuredExtractor",
    "SummaryGenerator",
    # Chunking
    "MultiLayerChunker",
    "Chunk",
    "ChunkMetadata",
    # Embedding & Indexing
    "EmbeddingGenerator",
    "EmbeddingConfig",
    "FAISSVectorStore",
    "IndexingPipeline",
    "IndexingConfig",
]
