"""RAG (Retrieval Augmented Generation) components for legal document analysis."""

# Document processing
from .document_reader import LegalDocumentReader
from .chunker import (
    LegalChunk,
    ChunkingConfig,
    HierarchicalLegalChunker,
    LawCodeChunker,
)

# Embeddings and indexing
from .embeddings import LegalEmbedder, EmbeddingConfig
from .indexing import MultiDocumentVectorStore, VectorStoreConfig

# Retrieval
from .hybrid_retriever import HybridRetriever
from .cross_doc_retrieval import ComparativeRetriever
from .reranker import CrossEncoderReranker

# Analysis
from .knowledge_graph import LegalKnowledgeGraph
from .advanced_compliance_pipeline import AdvancedCompliancePipeline, ComplianceReport, ComplianceIssue

# Configuration
from .config import Config

# Exceptions
from .exceptions import (
    DocumentProcessingError,
    IndexingError,
    RetrievalError,
    ConfigurationError
)

__all__ = [
    # Document processing
    "LegalDocumentReader",
    "LegalChunk",
    "ChunkingConfig",
    "HierarchicalLegalChunker",
    "LawCodeChunker",
    # Embeddings
    "LegalEmbedder",
    "EmbeddingConfig",
    # Indexing
    "MultiDocumentVectorStore",
    "VectorStoreConfig",
    # Retrieval
    "HybridRetriever",
    "ComparativeRetriever",
    "CrossEncoderReranker",
    # Analysis
    "LegalKnowledgeGraph",
    "AdvancedCompliancePipeline",
    "ComplianceReport",
    "ComplianceIssue",
    # Config
    "Config",
    # Exceptions
    "DocumentProcessingError",
    "IndexingError",
    "RetrievalError",
    "ConfigurationError",
]
