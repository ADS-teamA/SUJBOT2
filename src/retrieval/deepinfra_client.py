"""
DeepInfra API Client

Single client for both embedding and LLM inference via DeepInfra.
Uses OpenAI-compatible API format.

Models:
- Embedding: Qwen/Qwen3-Embedding-8B (4096 dimensions)
- LLM: Qwen/Qwen2.5-7B-Instruct
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Module-level embedding cache (survives across client instances)
# Key: MD5 hash of text, Value: embedding vector
_embedding_cache: Dict[str, np.ndarray] = {}
_EMBEDDING_CACHE_MAX_SIZE = 1000  # ~4GB memory max (1000 * 4096 * 4 bytes)


def _get_text_hash(text: str) -> str:
    """Generate hash key for text (for cache lookup)."""
    return hashlib.md5(text.encode()).hexdigest()


@dataclass
class DeepInfraConfig:
    """Configuration for DeepInfra API."""

    api_key: Optional[str] = None  # Defaults to DEEPINFRA_API_KEY env var
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_url: str = "https://api.deepinfra.com/v1/openai"
    timeout: int = 60
    max_retries: int = 3
    embedding_dimensions: int = 4096  # Qwen3-Embedding-8B output
    embedding_batch_size: int = 32  # Batch size for embedding API calls


class DeepInfraClient:
    """
    DeepInfra API client for embedding and LLM.

    Uses OpenAI-compatible API format with custom base_url.

    Example:
        >>> client = DeepInfraClient()
        >>> embeddings = client.embed_texts(["Hello world"])
        >>> response = client.generate("What is 2+2?")
    """

    def __init__(self, config: Optional[DeepInfraConfig] = None):
        """
        Initialize DeepInfra client.

        Args:
            config: Configuration (uses defaults + env vars if not provided)
        """
        self.config = config or DeepInfraConfig()

        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY not found. "
                "Set it in .env file or pass via DeepInfraConfig(api_key=...)"
            )

        # Initialize OpenAI client with DeepInfra base URL
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for DeepInfra. "
                "Install with: pip install openai"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

        # Mask API key for logging
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        logger.info(
            f"DeepInfra client initialized "
            f"(embedding={self.config.embedding_model}, "
            f"llm={self.config.llm_model}, "
            f"key={masked_key})"
        )

    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed texts using DeepInfra embedding model with caching.

        Uses in-memory cache to avoid redundant API calls for repeated texts.

        Args:
            texts: List of texts to embed
            normalize: L2 normalize embeddings (default True for cosine similarity)

        Returns:
            np.ndarray of shape (len(texts), embedding_dimensions)
        """
        if not texts:
            return np.array([])

        # Check cache and identify texts needing embedding
        text_hashes = [_get_text_hash(t) for t in texts]
        cached_indices = []
        uncached_texts = []
        uncached_indices = []

        for i, (text, hash_key) in enumerate(zip(texts, text_hashes)):
            if hash_key in _embedding_cache:
                cached_indices.append(i)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        cache_hits = len(cached_indices)
        cache_misses = len(uncached_texts)

        if cache_hits > 0:
            logger.info(f"Embedding cache: {cache_hits} hits, {cache_misses} misses")

        # Embed only uncached texts
        new_embeddings = []
        if uncached_texts:
            batch_size = self.config.embedding_batch_size

            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i : i + batch_size]

                try:
                    response = self.client.embeddings.create(
                        model=self.config.embedding_model,
                        input=batch,
                        encoding_format="float",
                    )

                    batch_embeddings = [item.embedding for item in response.data]
                    new_embeddings.extend(batch_embeddings)

                    logger.debug(
                        f"Embedded batch {i // batch_size + 1}: "
                        f"{len(batch)} texts, {response.usage.total_tokens} tokens"
                    )

                except Exception as e:
                    logger.error(f"Embedding failed for batch {i // batch_size + 1}: {e}")
                    raise

        # Store new embeddings in cache (before normalization for reuse)
        for i, emb in enumerate(new_embeddings):
            text_idx = uncached_indices[i]
            hash_key = text_hashes[text_idx]

            # Evict oldest if cache full
            if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX_SIZE:
                oldest_key = next(iter(_embedding_cache))
                del _embedding_cache[oldest_key]

            _embedding_cache[hash_key] = np.array(emb, dtype=np.float32)

        # Reconstruct results in original order
        all_embeddings = [None] * len(texts)

        # Fill cached embeddings
        for i in cached_indices:
            all_embeddings[i] = _embedding_cache[text_hashes[i]]

        # Fill new embeddings
        for i, emb in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = np.array(emb, dtype=np.float32)

        # Convert to numpy array
        embeddings = np.array(all_embeddings, dtype=np.float32)

        # L2 normalize for cosine similarity
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms > 0, norms, 1)

        logger.info(f"Embedded {len(texts)} texts → shape {embeddings.shape}")
        return embeddings

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text completion using DeepInfra LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            generated_text = response.choices[0].message.content

            logger.debug(
                f"Generated {response.usage.completion_tokens} tokens "
                f"(input: {response.usage.prompt_tokens})"
            )

            return generated_text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    @property
    def embedding_dimensions(self) -> int:
        """Return embedding dimensionality."""
        return self.config.embedding_dimensions
