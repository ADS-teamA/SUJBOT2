"""
Abstract Vector Store Adapter Interface

Defines the contract that all vector store implementations must follow.
This enables zero-code changes in RAG tools when switching backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np


class VectorStoreAdapter(ABC):
    """
    Abstract interface for vector storage backends.

    All RAG tools depend on this interface, allowing seamless backend switching
    between FAISS (current) and PostgreSQL (new) without code changes.

    Key Design Principles:
    - Interface mirrors existing FAISSVectorStore API (minimal migration friction)
    - Supports 3-layer hierarchical search (document → section → chunk)
    - Enables hybrid search (dense + sparse retrieval)
    - Provides metadata access for tools that directly access vectors
    """

    @abstractmethod
    def hierarchical_search(
        self,
        query_embedding: np.ndarray,
        k_layer3: int = 6,
        use_doc_filtering: bool = True,
        similarity_threshold_offset: float = 0.25,
        query_text: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Primary search method - hierarchical 3-layer retrieval.

        Strategy:
        1. Search Layer 1 (document-level) → find top document
        2. Filter Layer 3 (chunk-level) by document_id
        3. Return top-k chunks with similarity > threshold

        Args:
            query_embedding: Query vector (3072 dimensions)
            k_layer3: Number of chunks to retrieve (default: 6)
            use_doc_filtering: Filter by top document from Layer 1 (default: True)
            similarity_threshold_offset: Keep results within threshold of top score (default: 0.25)
            query_text: Optional query text for hybrid search (dense + sparse)

        Returns:
            {
                'layer1': [{chunk_id, document_id, content, score, ...}, ...],  # 1 result
                'layer2': [{...}, ...],  # N results (section-level)
                'layer3': [{...}, ...]   # k_layer3 results (chunk-level, PRIMARY)
            }

        Example:
            >>> results = adapter.hierarchical_search(
            ...     query_embedding=np.array([0.1, 0.2, ..., 0.9]),  # 3072-dim
            ...     k_layer3=6,
            ...     query_text="GDPR Article 17"
            ... )
            >>> print(results['layer3'][0]['content'])
        """
        pass

    @abstractmethod
    def search_layer3(
        self,
        query_embedding: np.ndarray,
        k: int = 6,
        document_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Direct Layer 3 search (chunk-level, no hierarchical filtering).

        Used by tools that need direct chunk retrieval without document filtering.

        Args:
            query_embedding: Query vector (3072 dimensions)
            k: Number of results to retrieve
            document_filter: Optional document_id to filter results
            similarity_threshold: Optional minimum similarity score

        Returns:
            [
                {
                    'chunk_id': 'doc1:sec1:0',
                    'document_id': 'doc1',
                    'content': 'Chunk text...',
                    'score': 0.85,
                    'section_id': 'sec1',
                    'hierarchical_path': 'doc1 > section1 > subsection1',
                    ...metadata...
                },
                ...
            ]

        Example:
            >>> chunks = adapter.search_layer3(
            ...     query_embedding=emb,
            ...     k=10,
            ...     document_filter="BZ_VR1.pdf",
            ...     similarity_threshold=0.7
            ... )
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            {
                'documents': int,              # Unique document count
                'total_vectors': int,          # Total vectors across all layers
                'layer1_count': int,           # Document-level vectors
                'layer2_count': int,           # Section-level vectors
                'layer3_count': int,           # Chunk-level vectors (primary)
                'dimensions': int,             # Embedding dimensions (3072)
                'backend': str                 # 'faiss' or 'postgresql'
            }

        Example:
            >>> stats = adapter.get_stats()
            >>> print(f"Indexed {stats['documents']} documents, {stats['total_vectors']} vectors")
        """
        pass

    # ============================================================================
    # Properties: Metadata Access (Backward Compatibility)
    # ============================================================================
    # Some tools access metadata_layer1/2/3 directly instead of searching.
    # We provide these as properties for compatibility.

    @property
    @abstractmethod
    def metadata_layer1(self) -> List[Dict]:
        """
        Layer 1 metadata (document-level).

        Returns:
            List of metadata dicts for all document-level chunks.

        Note:
            This materializes all metadata in memory (expensive for large datasets).
            Tools should prefer search methods when possible.
        """
        pass

    @property
    @abstractmethod
    def metadata_layer2(self) -> List[Dict]:
        """
        Layer 2 metadata (section-level).

        Returns:
            List of metadata dicts for all section-level chunks.
        """
        pass

    @property
    @abstractmethod
    def metadata_layer3(self) -> List[Dict]:
        """
        Layer 3 metadata (chunk-level, PRIMARY).

        Returns:
            List of metadata dicts for all chunk-level chunks.
        """
        pass

    # ============================================================================
    # Backward Compatibility: FAISS-specific Properties
    # ============================================================================

    @property
    def faiss_store(self) -> Optional["VectorStoreAdapter"]:
        """
        Backward compatibility property for tools checking `hasattr(store, 'faiss_store')`.

        FAISSVectorStoreAdapter returns self (the wrapped FAISS store).
        PostgresVectorStoreAdapter returns None.

        Returns:
            Self for FAISS adapter, None for PostgreSQL adapter.

        Example:
            >>> if hasattr(adapter, 'faiss_store') and adapter.faiss_store:
            ...     print("Using FAISS backend")
            ... else:
            ...     print("Using PostgreSQL backend")
        """
        return None

    @property
    def bm25_store(self) -> Optional[Any]:
        """
        Backward compatibility for BM25 hybrid search.

        FAISSVectorStoreAdapter returns BM25Store if available.
        PostgresVectorStoreAdapter returns None (BM25 integrated in PostgreSQL).

        Returns:
            BM25Store for FAISS adapter, None for PostgreSQL adapter.
        """
        return None

    # ============================================================================
    # Optional Methods (Not Required for All Adapters)
    # ============================================================================

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        layer: int = 3,
    ) -> None:
        """
        Add new chunks to the vector store (optional, for indexing).

        Args:
            chunks: List of chunk metadata dicts
            embeddings: Embedding matrix (N x 3072)
            layer: Target layer (1, 2, or 3)

        Note:
            Not all adapters support incremental indexing.
            FAISS does, PostgreSQL should.

        Raises:
            NotImplementedError: If adapter doesn't support adding chunks.
        """
        raise NotImplementedError("This adapter does not support add_chunks()")

    def merge(self, other_store_path: str) -> None:
        """
        Merge another vector store into this one (optional, for FAISS).

        Args:
            other_store_path: Path to vector store to merge

        Note:
            Primarily for FAISS multi-document indexing.
            PostgreSQL uses batch inserts instead.

        Raises:
            NotImplementedError: If adapter doesn't support merging.
        """
        raise NotImplementedError("This adapter does not support merge()")

    def save(self, path: str) -> None:
        """
        Save vector store to disk (optional, for FAISS).

        Args:
            path: Directory path to save vector store

        Note:
            FAISS saves to disk (binary files).
            PostgreSQL persists automatically (database).

        Raises:
            NotImplementedError: If adapter doesn't support saving.
        """
        raise NotImplementedError("This adapter does not support save()")

    @classmethod
    def load(cls, path: str) -> "VectorStoreAdapter":
        """
        Load vector store from disk (optional, for FAISS).

        Args:
            path: Directory path containing vector store

        Returns:
            VectorStoreAdapter instance

        Note:
            Use `load_vector_store_adapter()` factory function instead
            for backend-agnostic loading.

        Raises:
            NotImplementedError: If adapter doesn't support loading.
        """
        raise NotImplementedError("Use load_vector_store_adapter() instead")
