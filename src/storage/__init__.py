"""
Storage Abstraction Layer for SUJBOT2

Provides unified interface for PostgreSQL pgvector storage.

Key Components:
- VectorStoreAdapter: Abstract interface for vector stores
- PostgresVectorStoreAdapter: PostgreSQL + pgvector implementation

Usage:
    from src.storage import create_vector_store_adapter

    # Create adapter
    adapter = create_vector_store_adapter(
        backend="postgresql",
        connection_string="postgresql://..."
    )

    # All RAG tools use the same interface
    results = adapter.search_layer3(query_emb, k=6)
"""

from .vector_store_adapter import VectorStoreAdapter

# PostgreSQL adapter (requires asyncpg)
try:
    from .postgres_adapter import PostgresVectorStoreAdapter, MetadataFilter
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    PostgresVectorStoreAdapter = None
    MetadataFilter = None

__all__ = [
    "VectorStoreAdapter",
    "PostgresVectorStoreAdapter",
    "MetadataFilter",
    "create_vector_store_adapter",
    "load_vector_store_adapter",
]


def create_vector_store_adapter(backend: str = "postgresql", **kwargs) -> VectorStoreAdapter:
    """
    Factory function to create vector store adapter.

    Args:
        backend: Backend type (only "postgresql" supported)
        **kwargs: Backend-specific arguments
            - connection_string: PostgreSQL connection string
            - pool_size: Connection pool size (default: 20)
            - dimensions: Embedding dimensions (default: 4096 for Qwen3-Embedding-8B)

    Returns:
        VectorStoreAdapter instance

    Raises:
        ValueError: If backend is unknown or required args missing
        ImportError: If asyncpg not installed

    Example:
        >>> adapter = create_vector_store_adapter(
        ...     backend="postgresql",
        ...     connection_string="postgresql://user:pass@host:5432/db",
        ...     pool_size=20
        ... )
    """
    if backend == "postgresql":
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "PostgreSQL backend requires 'asyncpg' package. "
                "Install with: pip install asyncpg"
            )

        connection_string = kwargs.get("connection_string")
        if not connection_string:
            raise ValueError("PostgreSQL backend requires 'connection_string' argument")

        pool_size = kwargs.get("pool_size", 20)
        dimensions = kwargs.get("dimensions", 4096)  # Qwen3-Embedding-8B

        return PostgresVectorStoreAdapter(
            connection_string=connection_string,
            pool_size=pool_size,
            dimensions=dimensions,
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. Only 'postgresql' is supported."
        )


async def load_vector_store_adapter(backend: str = "postgresql", path: str = None, **kwargs) -> VectorStoreAdapter:
    """
    Load vector store adapter from persistent storage.

    Args:
        backend: Backend type (only "postgresql" supported)
        path: Unused for PostgreSQL (kept for interface compatibility)
        **kwargs: Additional backend-specific arguments
            - connection_string: PostgreSQL connection string
            - pool_size: Connection pool size (default: 20)
            - dimensions: Embedding dimensions (default: 4096)

    Returns:
        VectorStoreAdapter instance

    Example:
        >>> adapter = await load_vector_store_adapter(
        ...     backend="postgresql",
        ...     connection_string="postgresql://..."
        ... )
    """
    if backend == "postgresql":
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "PostgreSQL backend requires 'asyncpg' package. "
                "Install with: pip install asyncpg"
            )

        connection_string = kwargs.get("connection_string", path)
        if not connection_string:
            raise ValueError("PostgreSQL backend requires 'connection_string' argument")

        pool_size = kwargs.get("pool_size", 20)
        dimensions = kwargs.get("dimensions", 4096)  # Qwen3-Embedding-8B

        adapter = PostgresVectorStoreAdapter(
            connection_string=connection_string,
            pool_size=pool_size,
            dimensions=dimensions,
        )
        await adapter.initialize()
        return adapter

    else:
        raise ValueError(f"Unknown backend: {backend}. Only 'postgresql' is supported.")
