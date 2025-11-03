"""
Document duplicate detection using semantic similarity.

Detects semantically similar documents before indexing to prevent duplicates
in BM25, FAISS, and knowledge graph stores.

Features:
- Fast first-page text extraction (PyMuPDF, 50-200ms)
- Embedding-based similarity (cosine distance)
- Lazy loading (embedder, vector store)
- 98% similarity threshold (configurable)
- Embedding cache (in-memory)

Usage:
    from src.duplicate_detector import DuplicateDetector, DuplicateDetectionConfig

    config = DuplicateDetectionConfig(threshold=0.98)
    detector = DuplicateDetector(config, vector_store_path="vector_db")

    # Check if document is duplicate
    is_duplicate, similarity, match_doc_id = detector.check_duplicate(
        file_path="data/new_doc.pdf"
    )

    if is_duplicate:
        print(f"Duplicate of {match_doc_id} (similarity: {similarity:.1%})")
"""

import logging
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DuplicateDetectionConfig:
    """Configuration for duplicate detection."""

    enabled: bool = True
    similarity_threshold: float = 0.98  # 98% cosine similarity
    sample_pages: int = 1  # Number of pages to sample (1 = first page only)
    cache_size: int = 1000  # Max cached embeddings

    def validate(self):
        """Validate configuration."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be 0-1, got {self.similarity_threshold}"
            )
        if self.sample_pages < 1:
            raise ValueError(f"sample_pages must be >= 1, got {self.sample_pages}")
        if self.cache_size < 0:
            raise ValueError(f"cache_size must be >= 0, got {self.cache_size}")


class DuplicateDetector:
    """
    Document duplicate detector using semantic similarity.

    Compares new documents against existing documents using:
    1. Fast text extraction (first page via PyMuPDF)
    2. Embedding generation (same model as indexing)
    3. Cosine similarity against existing document embeddings
    4. Configurable threshold (default 98%)

    Performance:
    - Text extraction: 50-200ms per document
    - Embedding generation: ~100ms per document
    - Similarity search: O(n) over existing documents (typically <10ms)

    Example:
        >>> config = DuplicateDetectionConfig(threshold=0.98)
        >>> detector = DuplicateDetector(config, "output/vector_store")
        >>> is_dup, sim, doc_id = detector.check_duplicate("data/new.pdf")
        >>> if is_dup:
        ...     print(f"Duplicate of {doc_id} ({sim:.1%})")
    """

    def __init__(
        self,
        config: DuplicateDetectionConfig,
        vector_store_path: Optional[str] = None,
    ):
        """
        Initialize duplicate detector.

        Args:
            config: Duplicate detection configuration
            vector_store_path: Path to existing vector store (for lazy loading)
        """
        self.config = config
        self.config.validate()

        self.vector_store_path = vector_store_path

        # Lazy-loaded components
        self._embedder = None
        self._vector_store = None

        # In-memory cache: file_hash -> (embedding, document_id)
        self._embedding_cache: Dict[str, Tuple[np.ndarray, str]] = {}

        logger.info(
            f"DuplicateDetector initialized: "
            f"threshold={config.similarity_threshold:.1%}, "
            f"sample_pages={config.sample_pages}"
        )

    def check_duplicate(
        self,
        file_path: str,
        document_id: Optional[str] = None,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check if document is a duplicate.

        Args:
            file_path: Path to document to check
            document_id: Optional document ID (for logging)

        Returns:
            Tuple of (is_duplicate, similarity_score, matching_document_id)

        Example:
            >>> is_dup, sim, match = detector.check_duplicate("data/doc.pdf")
            >>> if is_dup:
            ...     print(f"Found duplicate: {match} ({sim:.1%})")
        """
        if not self.config.enabled:
            return False, 0.0, None

        # Step 1: Extract text sample
        text = self._extract_text_sample(file_path)
        if not text or len(text.strip()) < 100:
            logger.warning(f"Document too short for duplicate detection: {file_path}")
            return False, 0.0, None

        # Step 2: Compute file hash (for caching)
        file_hash = self._compute_file_hash(file_path)

        # Step 3: Check cache
        if file_hash in self._embedding_cache:
            embedding, _ = self._embedding_cache[file_hash]
            logger.debug(f"Using cached embedding for {file_path}")
        else:
            # Generate embedding
            embedder = self._get_embedder()
            embedding = embedder.embed_texts([text])[0]

            # Cache embedding
            self._cache_embedding(file_hash, embedding, document_id or file_path)

        # Step 4: Search for similar documents
        is_duplicate, similarity, match_doc_id = self._find_similar_document(
            embedding, document_id
        )

        if is_duplicate:
            logger.warning(
                f"Duplicate detected: {file_path} matches {match_doc_id} "
                f"({similarity:.1%} similarity)"
            )
        else:
            logger.info(f"No duplicate found for {file_path}")

        return is_duplicate, similarity, match_doc_id

    def _extract_text_sample(self, file_path: str) -> str:
        """
        Extract text sample from document (first page via PyMuPDF).

        Fast extraction using PyMuPDF - typically 50-200ms.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")

        try:
            doc = fitz.open(file_path)

            # Extract first N pages
            text_parts = []
            for page_num in range(min(self.config.sample_pages, len(doc))):
                page = doc[page_num]
                text_parts.append(page.get_text())

            doc.close()

            text = "\n".join(text_parts)
            logger.debug(
                f"Extracted {len(text)} chars from first {self.config.sample_pages} page(s)"
            )

            return text

        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return ""

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for caching."""
        hash_obj = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read in chunks for memory efficiency
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def _get_embedder(self):
        """Lazy load embedding generator."""
        if self._embedder is None:
            from .embedding_generator import EmbeddingGenerator
            from .config import EmbeddingConfig

            config = EmbeddingConfig.from_env()
            self._embedder = EmbeddingGenerator(config)

            logger.info("Loaded EmbeddingGenerator for duplicate detection")

        return self._embedder

    def _get_vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None and self.vector_store_path:
            from .hybrid_search import HybridVectorStore

            self._vector_store = HybridVectorStore.load(self.vector_store_path)
            logger.info(f"Loaded vector store from {self.vector_store_path}")

        return self._vector_store

    def _cache_embedding(self, file_hash: str, embedding: np.ndarray, document_id: str):
        """Cache embedding with LRU-style eviction."""
        # Simple cache size management
        if len(self._embedding_cache) >= self.config.cache_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]

        self._embedding_cache[file_hash] = (embedding, document_id)

    def _find_similar_document(
        self,
        embedding: np.ndarray,
        exclude_doc_id: Optional[str] = None,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Find most similar document in vector store.

        Returns:
            (is_duplicate, similarity, matching_document_id)
        """
        vector_store = self._get_vector_store()
        if vector_store is None:
            logger.debug("No vector store available for duplicate check")
            return False, 0.0, None

        # Search in Layer 1 (document-level embeddings)
        # Use k=5 to check multiple candidates
        # Access underlying FAISS store from HybridVectorStore
        faiss_store = getattr(vector_store, 'faiss_store', vector_store)
        results = faiss_store.search_layer1(
            query_embedding=embedding,
            k=5,
        )

        if not results:
            return False, 0.0, None

        # Find best match (excluding current document)
        best_match = None
        best_similarity = 0.0

        for result in results:
            # FAISS returns flat structure with document_id at top level
            doc_id = result.get("document_id")
            similarity = result.get("score", 0.0)

            # Skip if it's the same document
            if doc_id == exclude_doc_id:
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = doc_id

        # Check threshold
        is_duplicate = best_similarity >= self.config.similarity_threshold

        return is_duplicate, best_similarity, best_match

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "enabled": self.config.enabled,
            "threshold": self.config.similarity_threshold,
            "cache_size": len(self._embedding_cache),
            "cache_limit": self.config.cache_size,
            "embedder_loaded": self._embedder is not None,
            "vector_store_loaded": self._vector_store is not None,
        }
