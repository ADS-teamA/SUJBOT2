"""
PHASE 4: Embedding Generation Module

Based on research:
- LegalBench-RAG: text-embedding-3-large baseline (3072 dims)
- MLEB 2025: Kanon 2 #1 (86% NDCG@10), Voyage 3 Large #2 (85.7%)
- Multilingual: BGE-M3 (100+ languages, 8192 tokens)

Supports:
- kanon-2 (Voyage AI, 1024 dims, MLEB #1, default)
- voyage-3-large (Voyage AI, 1024 dims, MLEB #2)
- voyage-law-2 (Voyage AI, 1024 dims, legal-optimized)
- text-embedding-3-large (OpenAI, 3072 dims)
- BAAI/bge-m3 (HuggingFace, 1024 dims, Czech support)

Implementation:
- Batch embedding for efficiency
- Layer-specific embedding (use 'content' field with SAC for Layer 3)
- Normalization for cosine similarity (FAISS IndexFlatIP)
"""

import logging
import os
import hashlib
from typing import List, Dict, Optional, Union
from collections import OrderedDict
import numpy as np

try:
    from .cost_tracker import get_global_tracker
    from .config import EmbeddingConfig  # Import unified config from src.config
    from .utils.security import sanitize_error
except ImportError:
    from cost_tracker import get_global_tracker
    from config import EmbeddingConfig  # Import unified config from src.config
    from utils.security import sanitize_error

# Re-export for backward compatibility (will be removed in v2.0)
# This allows existing code importing from embedding_generator to continue working
__all__ = ["EmbeddingGenerator", "EmbeddingConfig"]

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for multi-layer chunks.

    Supports:
    - Voyage AI: kanon-2 (default, #1 MLEB), voyage-3-large, voyage-law-2
    - OpenAI: text-embedding-3-large (3072 dims)
    - HuggingFace: BAAI/bge-m3 (1024 dims, multilingual)

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024)
    - MLEB 2025 benchmark
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding generator.

        Args:
            config: EmbeddingConfig instance (defaults to kanon-2)
        """
        self.config = config or EmbeddingConfig()
        self.model_name = self.config.model
        self.batch_size = self.config.batch_size

        # Initialize embedding cache (similar_query_cache infrastructure)
        self._cache_enabled = self.config.cache_enabled
        self._cache_max_size = self.config.cache_max_size
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_errors = 0  # Track cache failures

        # Initialize cost tracker
        self.tracker = get_global_tracker()

        logger.info(f"Initializing EmbeddingGenerator with model: {self.model_name}")
        if self._cache_enabled:
            logger.info(f"Embedding cache enabled: max_size={self._cache_max_size}")

        # Initialize model based on type
        if "deepinfra" in self.model_name.lower() or "qwen" in self.model_name.lower():
            self._init_deepinfra_model()
        elif "voyage" in self.model_name.lower() or "kanon" in self.model_name.lower():
            self._init_voyage_model()
        elif self.model_name.startswith("text-embedding"):
            self._init_openai_model()
        elif "bge" in self.model_name.lower():
            self._init_bge_model()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        logger.info(f"EmbeddingGenerator initialized: {self.dimensions} dimensions")

    def _init_voyage_model(self):
        """Initialize Voyage AI embedding model (Kanon 2, Voyage 3, etc.)."""
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai package required for Voyage models. "
                "Install with: uv pip install voyageai"
            )

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "VOYAGE_API_KEY environment variable required for Voyage models. "
                "Get your key at: https://www.voyageai.com/"
            )

        self.client = voyageai.Client(api_key=api_key)
        self.model_type = "voyage"

        # All Voyage models use 1024 dimensions
        self.dimensions = 1024

        logger.info(f"Voyage AI model initialized: {self.model_name} ({self.dimensions}D)")

    def _init_deepinfra_model(self):
        """Initialize DeepInfra embedding model (Qwen3-Embedding-8B)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for DeepInfra. "
                "Install with: uv pip install openai"
            )

        api_key = os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY environment variable required for DeepInfra models. "
                "Get your key at: https://deepinfra.com/"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
            timeout=60,
            max_retries=3,
        )
        self.model_type = "deepinfra"

        # Qwen3-Embedding-8B uses 4096 dimensions
        self.dimensions = 4096

        # Mask API key for logging
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        logger.info(f"DeepInfra model initialized: {self.model_name} ({self.dimensions}D, key={masked_key})")

    def _init_openai_model(self):
        """Initialize OpenAI embedding model."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required for text-embedding models. "
                "Install with: uv pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        self.client = OpenAI(api_key=api_key)
        self.model_type = "openai"

        # Dimensions for OpenAI models
        if self.model_name == "text-embedding-3-large":
            self.dimensions = 3072
        elif self.model_name == "text-embedding-3-small":
            self.dimensions = 1536
        else:
            self.dimensions = 1536  # Default

        logger.info(f"OpenAI model initialized: {self.model_name} ({self.dimensions}D)")

    def _init_bge_model(self):
        """Initialize BGE embedding model (HuggingFace)."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers required for BGE models. "
                "Install with: uv pip install sentence-transformers torch"
            )

        logger.info(f"Loading BGE model: {self.model_name}")

        # Map model names
        model_map = {
            "bge-m3": "BAAI/bge-m3",
            "bge-m3": "BAAI/bge-m3",
            "bge-large": "BAAI/bge-large-en-v1.5",
        }

        hf_model_name = model_map.get(self.model_name, self.model_name)

        # Detect device (MPS for M1/M2 Mac, CUDA for GPU, CPU fallback)
        if torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            logger.info("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA GPU acceleration")
        else:
            device = "cpu"
            logger.info("Using CPU (no GPU acceleration)")

        self.model = SentenceTransformer(hf_model_name, device=device)
        self.model_type = "sentence_transformer"
        self.dimensions = self.model.get_sentence_embedding_dimension()

        logger.info(f"BGE model loaded: {hf_model_name} ({self.dimensions}D) on {device}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts with caching support.

        Cache key is generated from joined text strings (hash-based).
        NOTE: Cache only matches EXACT text lists. Semantically similar
        queries with different wording will NOT hit cache.
        Cache effectiveness depends on query repetition patterns.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings, shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([])

        # Check cache if enabled
        cache_key = None  # Initialize to avoid UnboundLocalError
        if self._cache_enabled:
            try:
                cache_key = self._generate_cache_key(texts)
                if cache_key in self._embedding_cache:
                    # Move to end (access tracking for potential LRU)
                    self._embedding_cache.move_to_end(cache_key)
                    cached_embedding = self._embedding_cache[cache_key]

                    # Validate cached data (assumes 2D array: [num_texts, dimensions])
                    if not isinstance(cached_embedding, np.ndarray):
                        logger.error(f"Invalid cache entry type: {type(cached_embedding)}")
                        self._embedding_cache.pop(cache_key)
                        # Treat as miss (validation failed)
                    elif cached_embedding.ndim != 2:
                        logger.error(
                            f"Cache dimension error: expected 2D array, "
                            f"got {cached_embedding.ndim}D"
                        )
                        self._embedding_cache.pop(cache_key)
                        # Treat as miss (validation failed)
                    elif cached_embedding.shape[1] != self.dimensions:
                        logger.error(
                            f"Cache dimension mismatch: expected {self.dimensions}, "
                            f"got {cached_embedding.shape[1]}"
                        )
                        self._embedding_cache.pop(cache_key)
                        # Treat as miss (validation failed)
                    elif cached_embedding.shape[0] != len(texts):
                        logger.error(
                            f"Cache count mismatch: expected {len(texts)} texts, "
                            f"got {cached_embedding.shape[0]} embeddings. Possible hash collision!"
                        )
                        self._embedding_cache.pop(cache_key)
                        # Treat as miss (validation failed)
                    else:
                        # Valid cache hit - only increment now
                        self._cache_hits += 1
                        logger.debug(
                            f"Cache HIT: {len(texts)} texts "
                            f"(hit_rate: {self._get_cache_hit_rate():.1%})"
                        )
                        return cached_embedding

                # If we get here, it's a miss (not in cache OR validation failed)
                self._cache_misses += 1
                logger.debug(f"Cache MISS: {len(texts)} texts")
            except Exception as e:
                logger.error(
                    f"Cache lookup failed, falling back to embedding: {sanitize_error(e)}",
                    exc_info=True,
                )
                self._cache_misses += 1
                self._cache_errors += 1

                # Disable cache after too many errors
                if self._cache_errors > 10:
                    logger.error(
                        f"❌ Cache disabled due to repeated errors ({self._cache_errors} failures). "
                        f"Performance degraded. Check logs for details."
                    )
                    self._cache_enabled = False
                # Continue to embedding below

        logger.info(f"Embedding {len(texts)} texts...")

        if self.model_type == "deepinfra":
            embeddings = self._embed_deepinfra(texts)
        elif self.model_type == "voyage":
            embeddings = self._embed_voyage(texts)
        elif self.model_type == "openai":
            embeddings = self._embed_openai(texts)
        elif self.model_type == "sentence_transformer":
            embeddings = self._embed_sentence_transformer(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Normalize for cosine similarity if configured
        if self.config.normalize:
            embeddings = self._normalize_embeddings(embeddings)

        # Store in cache if enabled and cache_key is available
        if self._cache_enabled and cache_key is not None:
            try:
                self._add_to_cache(cache_key, embeddings)
            except MemoryError as e:
                logger.warning(
                    f"Cache storage failed due to memory: {sanitize_error(e)}. Clearing cache and disabling."
                )
                self._embedding_cache.clear()  # Free memory
                self._cache_enabled = False  # Disable cache on memory error
                self._cache_errors += 1
            except Exception as e:
                logger.error(
                    f"Failed to store embeddings in cache: {sanitize_error(e)}", exc_info=True
                )
                self._cache_errors += 1

                # Disable cache after too many storage errors
                if self._cache_errors > 10:
                    logger.error(
                        f"❌ Cache disabled due to storage errors ({self._cache_errors} failures). "
                        f"Performance degraded."
                    )
                    self._cache_enabled = False
                # Continue execution - cache failure shouldn't break embedding

        logger.info(f"Embeddings generated: shape {embeddings.shape}")
        return embeddings

    def _generate_cache_key(self, texts: List[str]) -> str:
        """
        Generate cache key from text list (SHA256 hash).

        WARNING: Uses "|" as separator. In rare cases, different text lists
        can produce identical keys (e.g., ["a|b", "c"] and ["a", "b|c"]).
        The shape[0] validation in cache lookup detects such collisions.
        """
        combined = "|".join(texts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _add_to_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """
        Add embeddings to cache with FIFO eviction.

        NOTE: Despite OrderedDict with move_to_end for access tracking,
        eviction uses FIFO (removes oldest insertion, not least recently used).
        Cache hits call move_to_end() but eviction ignores this ordering.
        """
        # Remove oldest insertion if at capacity (FIFO eviction)
        if len(self._embedding_cache) >= self._cache_max_size:
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[cache_key] = embeddings

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring."""
        total_requests = self._cache_hits + self._cache_misses
        error_rate = self._cache_errors / total_requests if total_requests > 0 else 0
        return {
            "enabled": self._cache_enabled,
            "max_size": self._cache_max_size,
            "current_size": len(self._embedding_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "errors": self._cache_errors,
            "hit_rate": self._get_cache_hit_rate(),
            "error_rate": error_rate,
        }

    def _embed_voyage(self, texts: List[str]) -> np.ndarray:
        """Embed texts using Voyage AI API."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            logger.debug(
                f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}"
            )

            result = self.client.embed(
                texts=batch, model=self.model_name, input_type="document"  # For indexing/storage
            )

            # Track cost (Voyage API returns total_tokens)
            if hasattr(result, "total_tokens"):
                self.tracker.track_embedding(
                    provider="voyage",
                    model=self.model_name,
                    tokens=result.total_tokens,
                    operation="embedding",
                )

            batch_embeddings = result.embeddings
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_deepinfra(self, texts: List[str]) -> np.ndarray:
        """Embed texts using DeepInfra API (OpenAI-compatible)."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            logger.debug(
                f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}"
            )

            response = self.client.embeddings.create(
                model=self.model_name, input=batch, encoding_format="float"
            )

            # Track cost (DeepInfra uses same format as OpenAI)
            self.tracker.track_embedding(
                provider="deepinfra",
                model=self.model_name,
                tokens=response.usage.total_tokens,
                operation="embedding",
            )

            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            logger.debug(
                f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}"
            )

            response = self.client.embeddings.create(
                model=self.model_name, input=batch, encoding_format="float"
            )

            # Track cost
            self.tracker.track_embedding(
                provider="openai",
                model=self.model_name,
                tokens=response.usage.total_tokens,
                operation="embedding",
            )

            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Embed texts using sentence-transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We handle normalization separately
        )

        # No cost tracking needed - local model (FREE)
        # BGE-M3 runs locally and has $0.00 cost

        return embeddings.astype(np.float32)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def embed_chunks(self, chunks: List, layer: int) -> np.ndarray:
        """
        Embed chunks from a specific layer.

        CRITICAL: For Layer 3, uses 'content' field (with SAC).
                  For Layers 1-2, also uses 'content' field.

        Args:
            chunks: List of Chunk objects
            layer: Layer number (1, 2, or 3)

        Returns:
            numpy array of embeddings
        """
        if not chunks:
            return np.array([])

        logger.info(f"Embedding Layer {layer}: {len(chunks)} chunks")

        # Extract texts with breadcrumb path prefix
        # Format: [section_path > section_title]\n\n{chunk.content}
        # This improves retrieval by adding hierarchical location context
        texts = []
        for chunk in chunks:
            # Build breadcrumb from metadata
            breadcrumb_parts = []
            if hasattr(chunk, "metadata") and chunk.metadata:
                if chunk.metadata.section_path:
                    breadcrumb_parts.append(chunk.metadata.section_path)
                if (
                    chunk.metadata.section_title
                    and chunk.metadata.section_title != chunk.metadata.section_path
                ):
                    breadcrumb_parts.append(chunk.metadata.section_title)

            # Construct embedding text: [breadcrumb]\n\ncontent
            if breadcrumb_parts:
                breadcrumb = " > ".join(breadcrumb_parts)
                text = f"[{breadcrumb}]\n\n{chunk.content}"
            else:
                text = chunk.content

            texts.append(text)

        # Find valid (non-empty) texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            logger.warning(f"Layer {layer}: No valid texts to embed")
            # Return zero vectors for all chunks to maintain 1:1 mapping
            return np.zeros((len(chunks), self.dimensions), dtype=np.float32)

        # Generate embeddings only for valid texts
        valid_embeddings = self.embed_texts(valid_texts)

        # Create full embedding array with zeros for empty chunks
        # This maintains 1:1 mapping between chunks and embeddings
        embeddings = np.zeros((len(chunks), self.dimensions), dtype=np.float32)
        for idx, valid_idx in enumerate(valid_indices):
            embeddings[valid_idx] = valid_embeddings[idx]

        if len(valid_indices) < len(chunks):
            logger.warning(
                f"Layer {layer}: {len(chunks) - len(valid_indices)} empty chunks "
                f"(filled with zero vectors)"
            )

        logger.info(
            f"Layer {layer} embeddings generated: {len(embeddings)} vectors, {self.dimensions}D"
        )

        return embeddings

    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict:
        """Get statistics about embeddings."""
        if embeddings.size == 0:
            return {}

        return {
            "count": len(embeddings),
            "dimensions": embeddings.shape[1] if embeddings.ndim > 1 else 0,
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "memory_mb": float(embeddings.nbytes / (1024 * 1024)),
        }


# Example usage
if __name__ == "__main__":
    from config import ExtractionConfig
    from multi_layer_chunker import MultiLayerChunker
    from docling_extractor_v2 import DoclingExtractorV2

    # Extract and chunk document
    config = ExtractionConfig(enable_smart_hierarchy=True, generate_summaries=True)

    extractor = DoclingExtractorV2(config)
    result = extractor.extract("document.pdf")

    chunker = MultiLayerChunker(chunk_size=500, enable_sac=True)
    chunks = chunker.chunk_document(result)

    # Generate embeddings
    embedding_config = EmbeddingConfig(
        model="text-embedding-3-large", batch_size=100, normalize=True
    )

    embedder = EmbeddingGenerator(embedding_config)

    # Embed all layers
    layer1_embeddings = embedder.embed_chunks(chunks["layer1"], layer=1)
    layer2_embeddings = embedder.embed_chunks(chunks["layer2"], layer=2)
    layer3_embeddings = embedder.embed_chunks(chunks["layer3"], layer=3)

    print(f"Layer 1: {layer1_embeddings.shape}")
    print(f"Layer 2: {layer2_embeddings.shape}")
    print(f"Layer 3: {layer3_embeddings.shape}")
