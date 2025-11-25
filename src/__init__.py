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

# Gemini extractor (optional - requires GOOGLE_API_KEY)
try:
    from .gemini_extractor import GeminiExtractor, get_extractor
except ImportError:
    GeminiExtractor = None  # type: ignore
    get_extractor = None  # type: ignore

# PHASE 3: Multi-layer chunking with SAC
from .multi_layer_chunker import MultiLayerChunker, Chunk, ChunkMetadata

# PHASE 4: Embedding & indexing
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig
from .indexing_pipeline import IndexingPipeline, IndexingConfig

# Optional: FAISS vector store (not required for PostgreSQL backend)
try:
    from .faiss_vector_store import FAISSVectorStore
except ImportError:
    FAISSVectorStore = None  # type: ignore

__all__ = [
    # Configuration
    "ExtractionConfig",
    "RAGConfig",
    # Extraction
    "UnstructuredExtractor",
    "SummaryGenerator",
    "GeminiExtractor",
    "get_extractor",
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
