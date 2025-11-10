"""
Model registry for MY_SUJBOT pipeline.

Provides centralized model name resolution across:
- config.py (MODEL_ALIASES)
- reranker.py (RERANKER_MODELS)

Features:
- Single source of truth for model aliases
- Separate registries for LLM, embedding, and reranker models
- Easy addition of new models
- Validation helpers

Usage:
    from src.utils import ModelRegistry

    # Resolve model alias
    model = ModelRegistry.resolve_llm("haiku")
    # Returns: "claude-haiku-4-5-20251001"

    # Check if model is local
    is_local = ModelRegistry.is_local_embedding("bge-m3")
    # Returns: True
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all model aliases.

    Maintains separate registries for:
    - LLM models (Claude, OpenAI)
    - Embedding models (OpenAI, Voyage, HuggingFace)
    - Reranker models (MS MARCO, BGE)
    """

    # ========================================================================
    # LLM MODELS
    # ========================================================================

    LLM_MODELS: Dict[str, str] = {
        # Claude models (Anthropic)
        "haiku": "claude-haiku-4-5-20251001",
        "claude-haiku": "claude-haiku-4-5-20251001",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-5-20250929",
        "claude-sonnet": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
        "opus": "claude-opus-4",
        "claude-opus": "claude-opus-4",
        # OpenAI models
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        # GPT-5 models (2025)
        "gpt-5": "gpt-5",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
        # O-series reasoning models
        "o1": "o1",
        "o1-mini": "o1-mini",
        "o3-mini": "o3-mini",
        # Google Gemini models (2025)
        "gemini": "gemini-2.5-flash",  # Default - agentic use (250 RPD free)
        "gemini-flash": "gemini-2.5-flash",  # Best for agents (10 RPM, 250 RPD)
        "gemini-pro": "gemini-2.5-pro",  # Best reasoning (5 RPM, 100 RPD)
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",  # High volume (15 RPM, 1000 RPD)
        # Local models
        "saul-7b": "Equall/Saul-7B-Instruct-v1",
        "saul": "Equall/Saul-7B-Instruct-v1",
    }

    # ========================================================================
    # EMBEDDING MODELS
    # ========================================================================

    EMBEDDING_MODELS: Dict[str, str] = {
        # OpenAI embeddings
        "text-embedding-3-large": "text-embedding-3-large",
        "text-embedding-3-small": "text-embedding-3-small",
        "text-embedding-ada-002": "text-embedding-ada-002",
        # Voyage AI embeddings
        "voyage-3-large": "voyage-3-large",
        "voyage-3": "voyage-3",
        "voyage-3-lite": "voyage-3-lite",
        "voyage-law-2": "voyage-law-2",
        # HuggingFace embeddings (local)
        "bge-m3": "BAAI/bge-m3",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-small": "BAAI/bge-small-en-v1.5",
        # Aliases
        "default": "BAAI/bge-m3",  # Default to local model (free)
    }

    # ========================================================================
    # RERANKER MODELS
    # ========================================================================

    RERANKER_MODELS: Dict[str, str] = {
        # MS MARCO models (cross-encoders)
        "ms-marco-mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-base": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        # BAAI rerankers
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
        # Aliases
        "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "accurate": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "sota": "BAAI/bge-reranker-large",
    }

    # ========================================================================
    # RESOLUTION METHODS
    # ========================================================================

    @classmethod
    def resolve_llm(cls, alias: str) -> str:
        """
        Resolve LLM model alias to full model name.

        Args:
            alias: Model alias (e.g., "haiku", "sonnet")

        Returns:
            Full model name (e.g., "claude-haiku-4-5-20251001")

        Example:
            >>> ModelRegistry.resolve_llm("haiku")
            'claude-haiku-4-5-20251001'

            >>> ModelRegistry.resolve_llm("gpt-4o-mini")
            'gpt-4o-mini'

            >>> ModelRegistry.resolve_llm("unknown-model")
            'unknown-model'  # Returns as-is if not found
        """
        resolved = cls.LLM_MODELS.get(alias, alias)

        if resolved != alias:
            logger.debug(f"Resolved LLM alias: {alias} → {resolved}")

        return resolved

    @classmethod
    def resolve_embedding(cls, alias: str) -> str:
        """
        Resolve embedding model alias to full model name.

        Args:
            alias: Model alias (e.g., "bge-m3", "voyage-3-large")

        Returns:
            Full model name (e.g., "BAAI/bge-m3")

        Example:
            >>> ModelRegistry.resolve_embedding("bge-m3")
            'BAAI/bge-m3'

            >>> ModelRegistry.resolve_embedding("text-embedding-3-large")
            'text-embedding-3-large'
        """
        resolved = cls.EMBEDDING_MODELS.get(alias, alias)

        if resolved != alias:
            logger.debug(f"Resolved embedding alias: {alias} → {resolved}")

        return resolved

    @classmethod
    def resolve_reranker(cls, alias: str) -> str:
        """
        Resolve reranker model alias to full model name.

        Args:
            alias: Model alias (e.g., "default", "fast", "sota")

        Returns:
            Full model name (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")

        Example:
            >>> ModelRegistry.resolve_reranker("fast")
            'cross-encoder/ms-marco-MiniLM-L-6-v2'

            >>> ModelRegistry.resolve_reranker("sota")
            'BAAI/bge-reranker-large'
        """
        resolved = cls.RERANKER_MODELS.get(alias, alias)

        if resolved != alias:
            logger.debug(f"Resolved reranker alias: {alias} → {resolved}")

        return resolved

    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================

    @classmethod
    def is_local_embedding(cls, model: str) -> bool:
        """
        Check if embedding model is local (HuggingFace).

        Args:
            model: Model name or alias

        Returns:
            True if model is local (no API key required)

        Example:
            >>> ModelRegistry.is_local_embedding("bge-m3")
            True

            >>> ModelRegistry.is_local_embedding("text-embedding-3-large")
            False
        """
        # Resolve alias first
        resolved = cls.resolve_embedding(model)

        # Check if model starts with HuggingFace patterns
        local_patterns = ["BAAI/", "sentence-transformers/", "intfloat/"]
        return any(resolved.startswith(pattern) for pattern in local_patterns)

    @classmethod
    def get_provider(cls, model: str, model_type: str = "llm") -> str:
        """
        Get provider name for a model.

        Args:
            model: Model name or alias
            model_type: Type of model ("llm", "embedding", "reranker")

        Returns:
            Provider name ("anthropic", "openai", "voyage", "huggingface")

        Example:
            >>> ModelRegistry.get_provider("haiku", "llm")
            'anthropic'

            >>> ModelRegistry.get_provider("bge-m3", "embedding")
            'huggingface'
        """
        # Resolve alias
        if model_type == "llm":
            resolved = cls.resolve_llm(model)
        elif model_type == "embedding":
            resolved = cls.resolve_embedding(model)
        elif model_type == "reranker":
            resolved = cls.resolve_reranker(model)
        else:
            resolved = model

        # Detect provider by model name pattern
        if (
            "claude" in resolved.lower()
            or "haiku" in resolved.lower()
            or "sonnet" in resolved.lower()
        ):
            return "anthropic"
        elif "gpt-" in resolved.lower() or "o1" in resolved.lower() or "o3" in resolved.lower():
            return "openai"
        elif "gemini" in resolved.lower():
            return "google"
        elif "voyage" in resolved.lower():
            return "voyage"
        elif "text-embedding" in resolved.lower():
            return "openai"
        elif "BAAI/" in resolved or "intfloat/" in resolved or "sentence-transformers/" in resolved:
            return "huggingface"
        else:
            # Cannot auto-detect provider - user must specify explicitly
            raise ValueError(
                f"Unable to auto-detect provider for model '{resolved}' (type: {model_type}).\n"
                f"Supported providers: anthropic, openai, google, voyage, huggingface\n"
                f"Please set '{model_type}_provider' explicitly in config.json"
            )

    @classmethod
    def list_models(cls, model_type: str = "all") -> Dict[str, str]:
        """
        List all models of given type.

        Args:
            model_type: Type of models to list ("llm", "embedding", "reranker", "all")

        Returns:
            Dictionary of alias -> full_name

        Example:
            >>> models = ModelRegistry.list_models("llm")
            >>> for alias, name in models.items():
            >>>     print(f"{alias} → {name}")
        """
        if model_type == "llm":
            return cls.LLM_MODELS.copy()
        elif model_type == "embedding":
            return cls.EMBEDDING_MODELS.copy()
        elif model_type == "reranker":
            return cls.RERANKER_MODELS.copy()
        elif model_type == "all":
            return {
                "llm": cls.LLM_MODELS.copy(),
                "embedding": cls.EMBEDDING_MODELS.copy(),
                "reranker": cls.RERANKER_MODELS.copy(),
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


# Example usage
if __name__ == "__main__":
    print("=== Model Registry Examples ===\n")

    # Example 1: Resolve LLM aliases
    print("1. LLM model resolution...")
    llm_aliases = ["haiku", "sonnet", "gpt-4o-mini"]
    for alias in llm_aliases:
        resolved = ModelRegistry.resolve_llm(alias)
        print(f"   {alias:20s} → {resolved}")

    # Example 2: Resolve embedding aliases
    print("\n2. Embedding model resolution...")
    embed_aliases = ["bge-m3", "text-embedding-3-large", "voyage-3-large"]
    for alias in embed_aliases:
        resolved = ModelRegistry.resolve_embedding(alias)
        is_local = ModelRegistry.is_local_embedding(alias)
        local_str = " (local)" if is_local else " (cloud)"
        print(f"   {alias:25s} → {resolved:40s} {local_str}")

    # Example 3: Resolve reranker aliases
    print("\n3. Reranker model resolution...")
    reranker_aliases = ["fast", "accurate", "sota"]
    for alias in reranker_aliases:
        resolved = ModelRegistry.resolve_reranker(alias)
        print(f"   {alias:15s} → {resolved}")

    # Example 4: Get provider
    print("\n4. Provider detection...")
    models = [
        ("haiku", "llm"),
        ("gpt-4o-mini", "llm"),
        ("bge-m3", "embedding"),
        ("text-embedding-3-large", "embedding"),
    ]
    for model, model_type in models:
        provider = ModelRegistry.get_provider(model, model_type)
        print(f"   {model:25s} ({model_type:10s}) → {provider}")

    # Example 5: List all models
    print("\n5. List all embedding models...")
    embeddings = ModelRegistry.list_models("embedding")
    print(f"   Total: {len(embeddings)} models")

    print("\n=== All examples completed ===")
