"""
Storage Abstraction Layer for SUJBOT2

Provides unified interface for vector stores and knowledge graphs,
enabling zero-code changes when switching between FAISS and PostgreSQL.

Key Components:
- VectorStoreAdapter: Abstract interface for vector stores
- FAISSVectorStoreAdapter: Wraps existing FAISS implementation
- PostgresVectorStoreAdapter: PostgreSQL + pgvector implementation

Usage:
    from src.storage import create_vector_store_adapter

    # Create adapter based on config
    adapter = create_vector_store_adapter(
        backend="postgresql",  # or "faiss"
        connection_string="postgresql://..."
    )

    # All RAG tools use the same interface
    results = adapter.hierarchical_search(query_emb, k=6)
"""

from .vector_store_adapter import VectorStoreAdapter
from .faiss_adapter import FAISSVectorStoreAdapter
from .postgres_adapter import PostgresVectorStoreAdapter

__all__ = [
    "VectorStoreAdapter",
    "FAISSVectorStoreAdapter",
    "PostgresVectorStoreAdapter",
    "create_vector_store_adapter",
    "load_vector_store_adapter",
]


def create_vector_store_adapter(backend: str, **kwargs) -> VectorStoreAdapter:
    """
    Factory function to create vector store adapter based on backend type.

    Args:
        backend: Backend type ("faiss" or "postgresql")
        **kwargs: Backend-specific arguments
            For FAISS:
                - faiss_store: FAISSVectorStore instance
                - bm25_store: Optional BM25Store instance
            For PostgreSQL:
                - connection_string: PostgreSQL connection string
                - pool_size: Connection pool size (default: 20)
                - dimensions: Embedding dimensions (default: 3072)

    Returns:
        VectorStoreAdapter instance

    Raises:
        ValueError: If backend is unknown
        ConnectionError: If PostgreSQL connection fails (with FAISS fallback)

    Example:
        >>> # FAISS backend
        >>> adapter = create_vector_store_adapter(
        ...     backend="faiss",
        ...     faiss_store=faiss_store,
        ...     bm25_store=bm25_store
        ... )
        >>>
        >>> # PostgreSQL backend
        >>> adapter = create_vector_store_adapter(
        ...     backend="postgresql",
        ...     connection_string="postgresql://user:pass@host:5432/db",
        ...     pool_size=20
        ... )
    """
    if backend == "faiss":
        faiss_store = kwargs.get("faiss_store")
        if not faiss_store:
            raise ValueError("FAISS backend requires 'faiss_store' argument")

        bm25_store = kwargs.get("bm25_store")
        return FAISSVectorStoreAdapter(faiss_store=faiss_store, bm25_store=bm25_store)

    elif backend == "postgresql":
        connection_string = kwargs.get("connection_string")
        if not connection_string:
            raise ValueError("PostgreSQL backend requires 'connection_string' argument")

        pool_size = kwargs.get("pool_size", 20)
        dimensions = kwargs.get("dimensions", 3072)

        try:
            return PostgresVectorStoreAdapter(
                connection_string=connection_string,
                pool_size=pool_size,
                dimensions=dimensions,
            )
        except Exception as e:
            # Graceful degradation: Fall back to FAISS if PostgreSQL unavailable
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"PostgreSQL connection failed: {e}. "
                f"Falling back to FAISS if available."
            )

            # Try to load FAISS as fallback
            faiss_store = kwargs.get("faiss_store")
            if faiss_store:
                logger.warning("Using FAISS fallback")
                return FAISSVectorStoreAdapter(
                    faiss_store=faiss_store, bm25_store=kwargs.get("bm25_store")
                )
            else:
                raise ConnectionError(
                    f"PostgreSQL connection failed and no FAISS fallback available: {e}"
                ) from e

    else:
        raise ValueError(
            f"Unknown backend: {backend}. " f"Supported: 'faiss', 'postgresql'"
        )


async def load_vector_store_adapter(backend: str, path: str = None, **kwargs) -> VectorStoreAdapter:
    """
    Load vector store adapter from persistent storage.

    Args:
        backend: Backend type ("faiss" or "postgresql")
        path: Path to vector store directory (for FAISS) or connection string (for PostgreSQL)
        **kwargs: Additional backend-specific arguments

    Returns:
        VectorStoreAdapter instance

    Example:
        >>> # Load FAISS from disk
        >>> adapter = load_vector_store_adapter(
        ...     backend="faiss",
        ...     path="vector_db/"
        ... )
        >>>
        >>> # Load PostgreSQL (no path needed, uses connection string)
        >>> adapter = load_vector_store_adapter(
        ...     backend="postgresql",
        ...     connection_string="postgresql://..."
        ... )
    """
    if backend == "faiss":
        from src.faiss_vector_store import FAISSVectorStore
        from src.hybrid_search import BM25Store
        from pathlib import Path

        if not path:
            raise ValueError("FAISS backend requires 'path' argument")

        # Load FAISS store
        faiss_store = FAISSVectorStore.load(Path(path))

        # Load BM25 store (optional)
        bm25_store = None
        try:
            bm25_store = BM25Store.load(Path(path))
        except FileNotFoundError:
            pass

        return FAISSVectorStoreAdapter(faiss_store=faiss_store, bm25_store=bm25_store)

    elif backend == "postgresql":
        connection_string = kwargs.get("connection_string", path)
        if not connection_string:
            raise ValueError("PostgreSQL backend requires 'connection_string' argument")

        pool_size = kwargs.get("pool_size", 20)
        dimensions = kwargs.get("dimensions", 3072)

        adapter = PostgresVectorStoreAdapter(
            connection_string=connection_string,
            pool_size=pool_size,
            dimensions=dimensions,
        )
        await adapter.initialize()
        return adapter

    else:
        raise ValueError(f"Unknown backend: {backend}")
