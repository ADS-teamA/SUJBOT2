"""
Legal Document Embedder - BGE-M3 Model
Generates contextualized embeddings for legal chunks with hierarchical context
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field

from .device_utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class LegalChunk:
    """A chunk of legal document optimized for retrieval"""

    # Identity
    chunk_id: str
    chunk_index: Optional[int] = None

    # Content
    content: str = ""

    # Document context
    document_id: str = ""
    document_type: str = ""  # 'law_code' | 'contract' | 'regulation'

    # Legal structure
    hierarchy_path: str = ""  # "Část II > Hlava III > §89"
    legal_reference: str = ""  # "§89" or "Článek 5.2"
    structural_level: str = ""  # 'paragraph' | 'article' | 'subsection' | etc.

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_citation(self) -> str:
        """Get properly formatted citation"""
        if self.document_type == 'law_code':
            law_name = self.metadata.get('law_citation', '')
            if law_name:
                return f"{law_name}, {self.legal_reference}"
            return self.legal_reference
        else:
            return self.hierarchy_path or self.legal_reference


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""

    # Model settings
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"  # auto | cuda | mps | cpu (auto = cuda > mps > cpu)
    batch_size: int = 32
    max_sequence_length: int = 8192
    normalize: bool = True

    # Contextualization
    add_hierarchical_context: bool = True
    context_format: str = "{legal_ref} | {hierarchy} | {content}"

    # Performance
    show_progress_bar: bool = True


class LegalEmbedder:
    """Generate contextualized embeddings for legal chunks using BGE-M3"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the legal embedder

        Args:
            config: Embedding configuration (uses defaults if None)
        """
        self.config = config or EmbeddingConfig()
        self.model: Optional[SentenceTransformer] = None
        self.device = self._get_device()
        self._initialize_model()

    def _get_device(self) -> str:
        """
        Determine the best available device with automatic fallback.

        Priority: CUDA → MPS → CPU

        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        return get_device(self.config.device)

    def _initialize_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.device
            )

            # Set max sequence length if supported
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.config.max_sequence_length

            logger.info(
                f"Embedding model loaded successfully on device: {self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise ModelLoadError(f"Could not load model {self.config.model_name}: {e}")

    async def embed_chunks(
        self,
        chunks: List[LegalChunk],
        add_context: Optional[bool] = None,
        progress_callback: Optional[callable] = None
    ) -> np.ndarray:
        """
        Generate embeddings for chunks with optional context

        Args:
            chunks: List of legal chunks
            add_context: Whether to add hierarchical context (uses config default if None)
            progress_callback: Optional callback(current, total) for embedding progress

        Returns:
            Normalized embedding matrix (N x 1024)
        """
        if not chunks:
            return np.array([])

        # Use config default if not specified
        if add_context is None:
            add_context = self.config.add_hierarchical_context

        # Prepare texts
        if add_context:
            texts = [self._contextualize(chunk) for chunk in chunks]
        else:
            texts = [chunk.content for chunk in chunks]

        # Batch encode
        try:
            embeddings = await asyncio.to_thread(
                self._encode_texts,
                texts,
                progress_callback
            )
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")

    def _encode_texts(self, texts: List[str], progress_callback: Optional[callable] = None) -> np.ndarray:
        """Encode texts using the model (runs in thread)"""
        # If we have a progress callback, we need to manually handle batch processing
        # to call the callback after each batch
        if progress_callback and len(texts) > 0:
            from tqdm import tqdm
            import numpy as np

            batch_size = self.config.batch_size
            total_batches = (len(texts) + batch_size - 1) // batch_size
            all_embeddings = []

            # Process in batches with progress callback
            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx:batch_idx + batch_size]

                # Encode this batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    normalize_embeddings=self.config.normalize,
                    show_progress_bar=False,  # Disable tqdm, use our callback
                    device=self.device,
                    convert_to_numpy=True
                )

                all_embeddings.append(batch_embeddings)

                # Call progress callback
                current_batch = (batch_idx // batch_size) + 1
                progress_callback(current_batch, total_batches)

            # Stack all batches
            return np.vstack(all_embeddings) if all_embeddings else np.array([])
        else:
            # Original behavior when no callback
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=self.config.show_progress_bar,
                device=self.device,
                convert_to_numpy=True
            )
            return embeddings

    def _contextualize(self, chunk: LegalChunk) -> str:
        """
        Add hierarchical context to chunk content

        Format: "{legal_reference} | {hierarchy_path} | {content}"

        Example:
            "§89 odst. 2 | Část II > Hlava III > §89 | Dodavatel odpovídá za vady..."
        """
        context_parts = []

        # Add legal reference
        if chunk.legal_reference:
            context_parts.append(chunk.legal_reference)

        # Add hierarchy path
        if chunk.hierarchy_path and chunk.hierarchy_path != chunk.legal_reference:
            context_parts.append(chunk.hierarchy_path)

        # Add content
        context_parts.append(chunk.content)

        return " | ".join(context_parts)

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query

        Args:
            query: Search query string

        Returns:
            Normalized query embedding (1 x 1024)
        """
        try:
            # Create a temporary chunk for the query
            query_chunk = LegalChunk(
                chunk_id="query",
                content=query
            )

            # Embed without context
            embeddings = await self.embed_chunks(
                [query_chunk],
                add_context=False
            )

            return embeddings[0]

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise EmbeddingError(f"Query embedding failed: {e}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        if self.model is None:
            return 1024  # BGE-M3 default
        return self.model.get_sentence_embedding_dimension()


class EmbeddingCache:
    """Cache embeddings for frequently accessed chunks"""

    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: List[str] = []

    async def get_or_compute(
        self,
        chunk_id: str,
        chunk: LegalChunk,
        embedder: LegalEmbedder
    ) -> np.ndarray:
        """
        Get from cache or compute embedding

        Args:
            chunk_id: Unique chunk identifier
            chunk: Legal chunk to embed
            embedder: Embedder instance

        Returns:
            Embedding vector
        """
        if chunk_id in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(chunk_id)
            self.access_order.append(chunk_id)
            return self.cache[chunk_id]

        # Compute embedding
        embeddings = await embedder.embed_chunks([chunk])
        embedding = embeddings[0]

        # Store with LRU eviction
        if len(self.cache) >= self.max_size:
            # Remove oldest (least recently used)
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[chunk_id] = embedding
        self.access_order.append(chunk_id)

        return embedding

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_order.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "usage_percent": int(len(self.cache) / self.max_size * 100)
        }


# Custom Exceptions

class EmbeddingError(Exception):
    """Base exception for embedding errors"""
    pass


class ModelLoadError(EmbeddingError):
    """Failed to load embedding model"""
    pass


# Utility functions

async def embed_chunks_batched(
    chunks: List[LegalChunk],
    embedder: LegalEmbedder,
    batch_size: int = 32
) -> np.ndarray:
    """
    Embed chunks in batches for better GPU utilization

    Args:
        chunks: List of legal chunks
        embedder: Embedder instance
        batch_size: Number of chunks per batch

    Returns:
        Stacked embedding matrix
    """
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = await embedder.embed_chunks(batch)
        all_embeddings.append(embeddings)

    if not all_embeddings:
        return np.array([])

    return np.vstack(all_embeddings)
