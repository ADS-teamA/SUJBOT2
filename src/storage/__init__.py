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

# Optional PostgreSQL adapter (requires asyncpg)
try:
    from .postgres_adapter import PostgresVectorStoreAdapter
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    PostgresVectorStoreAdapter = None

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
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "PostgreSQL backend requires 'asyncpg' package. "
                "Install with: pip install asyncpg"
            )

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
        from src.hybrid_search_multilang import MultiLangBM25Store
        from pathlib import Path
        import logging
        import json

        logger = logging.getLogger(__name__)

        if not path:
            raise ValueError("FAISS backend requires 'path' argument")

        # Load FAISS store
        faiss_store = FAISSVectorStore.load(Path(path))

        # Load BM25 store (optional) - support both MultiLang (v3.0) and regular (v2.0) formats
        bm25_store = None
        try:
            # Check format version to determine which loader to use
            config_path = Path(path) / "bm25_store_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                format_version = config.get("format_version", "2.0")

                if format_version == "3.0":
                    # MultiLang format (Czech+English bilingual)
                    logger.info("Loading BM25 store with MultiLangBM25Store (format 3.0)")
                    bm25_store = MultiLangBM25Store.load(Path(path))
                else:
                    # Regular format (single language)
                    logger.info(f"Loading BM25 store with BM25Store (format {format_version})")
                    bm25_store = BM25Store.load(Path(path))
            else:
                # No config file - try regular BM25Store (backward compatibility)
                logger.info("No BM25 config found, trying BM25Store.load()")
                bm25_store = BM25Store.load(Path(path))

        except FileNotFoundError as e:
            logger.warning(f"BM25 store not found: {e}")
        except Exception as e:
            logger.error(f"Failed to load BM25 store: {e}", exc_info=True)

        return FAISSVectorStoreAdapter(faiss_store=faiss_store, bm25_store=bm25_store)

    elif backend == "postgresql":
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "PostgreSQL backend requires 'asyncpg' package. "
                "Install with: pip install asyncpg"
            )

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
