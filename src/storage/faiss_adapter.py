"""
FAISS Vector Store Adapter

Wraps existing FAISSVectorStore with zero changes to the underlying implementation.
Pure delegation pattern - all methods proxy to the wrapped store.
"""

from typing import List, Dict, Optional, Any
import numpy as np

from .vector_store_adapter import VectorStoreAdapter


class FAISSVectorStoreAdapter(VectorStoreAdapter):
    """
    Adapter wrapping existing FAISS implementation.

    Design: Pure delegation - no logic added, just interface compliance.
    Benefits: Zero changes to FAISSVectorStore, preserves all existing behavior.
    """

    def __init__(self, faiss_store, bm25_store=None):
        """
        Initialize FAISS adapter.

        Args:
            faiss_store: Existing FAISSVectorStore instance
            bm25_store: Optional BM25Store for hybrid search
        """
        self._faiss_store = faiss_store
        self._bm25_store = bm25_store

        # Create HybridVectorStore if BM25 available
        if bm25_store:
            from src.hybrid_search import HybridVectorStore

            self._hybrid_store = HybridVectorStore(faiss_store, bm25_store)
        else:
            self._hybrid_store = None

    def hierarchical_search(
        self,
        query_embedding: np.ndarray,
        k_layer3: int = 6,
        use_doc_filtering: bool = True,
        similarity_threshold_offset: float = 0.25,
        query_text: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """Delegate to FAISS or Hybrid store."""
        # Use hybrid search if query text provided and BM25 available
        if self._hybrid_store and query_text:
            return self._hybrid_store.hierarchical_search(
                query_text=query_text,
                query_embedding=query_embedding,
                k_layer3=k_layer3,
                use_doc_filtering=use_doc_filtering,
                similarity_threshold_offset=similarity_threshold_offset,
            )
        else:
            # Pure FAISS search
            return self._faiss_store.hierarchical_search(
                query_embedding=query_embedding,
                k_layer3=k_layer3,
                use_doc_filtering=use_doc_filtering,
                similarity_threshold_offset=similarity_threshold_offset,
            )

    def search_layer1(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        document_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Delegate to FAISSVectorStore."""
        return self._faiss_store.search_layer1(
            query_embedding=query_embedding,
            k=k,
            document_filter=document_filter,
            similarity_threshold=similarity_threshold,
        )

    def search_layer2(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        document_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Delegate to FAISSVectorStore."""
        return self._faiss_store.search_layer2(
            query_embedding=query_embedding,
            k=k,
            document_filter=document_filter,
            similarity_threshold=similarity_threshold,
        )

    def search_layer3(
        self,
        query_embedding: np.ndarray,
        k: int = 6,
        document_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Delegate to FAISSVectorStore."""
        return self._faiss_store.search_layer3(
            query_embedding=query_embedding,
            k=k,
            document_filter=document_filter,
            similarity_threshold=similarity_threshold,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Delegate to FAISSVectorStore."""
        stats = self._faiss_store.get_stats()
        stats["backend"] = "faiss"
        return stats

    # ============================================================================
    # Properties - Delegate to Wrapped Store
    # ============================================================================

    @property
    def metadata_layer1(self) -> List[Dict]:
        """Delegate to FAISSVectorStore."""
        return self._faiss_store.metadata_layer1

    @property
    def metadata_layer2(self) -> List[Dict]:
        """Delegate to FAISSVectorStore."""
        return self._faiss_store.metadata_layer2

    @property
    def metadata_layer3(self) -> List[Dict]:
        """Delegate to FAISSVectorStore."""
        return self._faiss_store.metadata_layer3

    @property
    def faiss_store(self):
        """Return wrapped FAISS store for backward compatibility."""
        return self._faiss_store

    @property
    def bm25_store(self):
        """Return BM25 store for backward compatibility."""
        return self._bm25_store

    # ============================================================================
    # Optional Methods - Delegate to FAISS
    # ============================================================================

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        layer: int = 3,
    ) -> None:
        """Delegate to FAISSVectorStore."""
        self._faiss_store.add_chunks(chunks=chunks, embeddings=embeddings, layer=layer)

    def merge(self, other_store_path: str) -> None:
        """Delegate to FAISSVectorStore."""
        self._faiss_store.merge(other_store_path)

    def save(self, path: str) -> None:
        """Delegate to FAISSVectorStore."""
        self._faiss_store.save(path)

    @classmethod
    def load(cls, path: str) -> "FAISSVectorStoreAdapter":
        """Load FAISS store and wrap in adapter."""
        from src.faiss_vector_store import FAISSVectorStore
        from src.hybrid_search import BM25Store
        from pathlib import Path

        faiss_store = FAISSVectorStore.load(Path(path))

        # Try to load BM25 store
        try:
            bm25_store = BM25Store.load(Path(path))
        except FileNotFoundError:
            bm25_store = None

        return cls(faiss_store=faiss_store, bm25_store=bm25_store)
