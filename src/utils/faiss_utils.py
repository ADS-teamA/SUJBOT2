"""
FAISS Utilities - Shared helper functions for FAISS operations.

Provides common utilities for working with FAISS indexes:
- Vector reconstruction from indexes
- Bulk operations
- Index introspection

Single source of truth for FAISS helper functions used across scripts.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def reconstruct_all_vectors(index, dim: int) -> np.ndarray:
    """
    Reconstruct all vectors from a FAISS IndexFlatIP.

    Falls back to per-row reconstruction if bulk method is unavailable.
    This is a common operation needed for:
    - Clustering analysis
    - Visualization (UMAP, t-SNE)
    - Export/analysis workflows

    Args:
        index: FAISS index (IndexFlatIP or compatible)
        dim: Vector dimensionality

    Returns:
        np.ndarray: All vectors from index, shape (n_vectors, dim)
                   Returns empty array if index is empty
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "FAISS required for vector reconstruction. "
            "Install with: uv pip install faiss-cpu (or faiss-gpu)"
        )

    n = index.ntotal
    if n == 0:
        logger.warning("FAISS index is empty, returning zero array")
        return np.zeros((0, dim), dtype=np.float32)

    # Prefer fast bulk reconstruction if available
    if hasattr(index, "reconstruct_n"):
        try:
            vectors = np.zeros((n, dim), dtype=np.float32)
            index.reconstruct_n(0, n, vectors)
            logger.debug(f"Reconstructed {n} vectors using bulk method (reconstruct_n)")
            return vectors
        except Exception as e:
            logger.warning(f"Bulk reconstruction failed: {e}, falling back to per-row")

    # Fallback: reconstruct row by row
    logger.debug(f"Using per-row reconstruction for {n} vectors")
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        try:
            vectors[i] = index.reconstruct(i)
        except Exception as e:
            logger.error(f"Failed to reconstruct vector {i}: {e}")
            # Leave as zeros if reconstruction fails
            continue

    logger.debug(f"Reconstructed {n} vectors using per-row method")
    return vectors


def get_index_stats(index) -> dict:
    """
    Get statistics about a FAISS index.

    Args:
        index: FAISS index

    Returns:
        dict: Index statistics including:
            - total_vectors: Number of vectors in index
            - dimensions: Vector dimensionality
            - index_type: Type of FAISS index
            - is_trained: Whether index is trained (for IVF indexes)
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS required. Install with: uv pip install faiss-cpu")

    stats = {
        "total_vectors": index.ntotal,
        "dimensions": index.d,
        "index_type": type(index).__name__,
    }

    # Check if index requires training (IVF indexes)
    if hasattr(index, "is_trained"):
        stats["is_trained"] = index.is_trained

    return stats


def validate_index(index, expected_dim: int = None) -> bool:
    """
    Validate FAISS index integrity.

    Args:
        index: FAISS index to validate
        expected_dim: Expected vector dimensionality (optional)

    Returns:
        bool: True if index is valid

    Raises:
        ValueError: If index is invalid
    """
    if index is None:
        raise ValueError("Index is None")

    if index.ntotal < 0:
        raise ValueError(f"Invalid index size: {index.ntotal}")

    if expected_dim is not None and index.d != expected_dim:
        raise ValueError(
            f"Dimension mismatch: index has {index.d}D, expected {expected_dim}D"
        )

    # For trainable indexes, check training status
    if hasattr(index, "is_trained") and not index.is_trained:
        raise ValueError("Index is not trained")

    return True
