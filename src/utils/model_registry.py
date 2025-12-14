"""
Model registry for SUJBOT2 pipeline.

SSOT: All model aliases are now loaded from config.json (model_registry section).
This class provides helper methods for model resolution with backward-compatible
fallbacks if config.json doesn't have the model_registry section.

Features:
- Single source of truth for model aliases (config.json)
- Lazy loading - config loaded on first access
- Backward compatible - falls back to built-in defaults if config missing
- Separate registries for LLM, embedding, and reranker models
- Embedding dimensions lookup

Usage:
    from src.utils.model_registry import ModelRegistry

    # Resolve model alias
    model = ModelRegistry.resolve_llm("haiku")
    # Returns: "claude-haiku-4-5-20251001"

    # Get embedding dimensions
    dims = ModelRegistry.get_embedding_dimensions("Qwen/Qwen3-Embedding-8B")
    # Returns: 4096

    # Check if model is local
    is_local = ModelRegistry.is_local_embedding("bge-m3")
    # Returns: True
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Global cache for config-loaded models (lazy initialization)
# ============================================================================

_config_loaded = False
_LLM_MODELS: Dict[str, str] = {}
_EMBEDDING_MODELS: Dict[str, str] = {}
_RERANKER_MODELS: Dict[str, str] = {}
_EMBEDDING_DIMENSIONS: Dict[str, int] = {}


def _load_builtin_defaults():
    """
    Load built-in defaults (backward compatibility).

    These are used when config.json doesn't have model_registry section.
    """
    global _LLM_MODELS, _EMBEDDING_MODELS, _RERANKER_MODELS, _EMBEDDING_DIMENSIONS

    _LLM_MODELS = {
        # Claude models (Anthropic)
        "haiku": "claude-haiku-4-5-20251001",
        "claude-haiku": "claude-haiku-4-5-20251001",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-5-20250929",
        "claude-sonnet": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
        "opus": "claude-opus-4-5-20251101",
        "claude-opus": "claude-opus-4-5-20251101",
        "claude-opus-4-5": "claude-opus-4-5-20251101",
        # OpenAI models
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        # O-series reasoning models
        "o1": "o1",
        "o1-mini": "o1-mini",
        "o3-mini": "o3-mini",
        # DeepInfra models (Qwen)
        "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        # Google Gemini models (2025)
        "gemini": "gemini-2.5-flash",
        "gemini-flash": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        # Local models
        "saul-7b": "Equall/Saul-7B-Instruct-v1",
        "saul": "Equall/Saul-7B-Instruct-v1",
    }

    _EMBEDDING_MODELS = {
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
        # DeepInfra embeddings
        "qwen3-embedding": "Qwen/Qwen3-Embedding-8B",
        # Aliases
        "default": "BAAI/bge-m3",
    }

    _RERANKER_MODELS = {
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

    _EMBEDDING_DIMENSIONS = {
        # DeepInfra / Qwen
        "Qwen/Qwen3-Embedding-8B": 4096,
        # OpenAI
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
        # Voyage AI
        "voyage-3-large": 1024,
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        # HuggingFace / BGE
        "BAAI/bge-m3": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
    }


def _ensure_config_loaded():
    """
    Load model definitions from config.json on first access.

    This function is called automatically before any registry access.
    Falls back to built-in defaults if config.json doesn't have model_registry.
    """
    global _config_loaded, _LLM_MODELS, _EMBEDDING_MODELS, _RERANKER_MODELS, _EMBEDDING_DIMENSIONS

    if _config_loaded:
        return

    try:
        from src.config import get_config
        config = get_config()

        if config.model_registry is not None:
            # Load from config.json (SSOT)
            _LLM_MODELS = dict(config.model_registry.llm_models)
            _EMBEDDING_MODELS = dict(config.model_registry.embedding_models)
            _RERANKER_MODELS = dict(config.model_registry.reranker_models)
            _EMBEDDING_DIMENSIONS = dict(config.model_registry.embedding_dimensions)

            logger.info(
                f"ModelRegistry loaded from config.json: "
                f"{len(_LLM_MODELS)} LLM, {len(_EMBEDDING_MODELS)} embedding, "
                f"{len(_RERANKER_MODELS)} reranker models"
            )
        else:
            # Fallback to built-in defaults (backward compatibility)
            logger.warning(
                "model_registry section not found in config.json, using built-in defaults. "
                "Consider adding model_registry to config.json for SSOT."
            )
            _load_builtin_defaults()

        _config_loaded = True

    except ImportError as e:
        # Config module not available - expected during testing
        logger.info(f"Config module not available: {e}. Using built-in model registry.")
        _load_builtin_defaults()
        _config_loaded = True

    except (KeyError, AttributeError, TypeError) as e:
        # Config schema mismatch - log with traceback for debugging
        logger.warning(
            f"Model registry config schema error: {e}. Using built-in defaults.",
            exc_info=True,
        )
        _load_builtin_defaults()
        _config_loaded = True


def reload_registry():
    """
    Force reload of model registry from config.

    Use this after modifying config.json at runtime (rare).
    """
    global _config_loaded
    _config_loaded = False
    _ensure_config_loaded()


class ModelRegistry:
    """
    Central registry for all model aliases - reads from config.json.

    Maintains separate registries for:
    - LLM models (Claude, OpenAI, Gemini, Qwen)
    - Embedding models (OpenAI, Voyage, HuggingFace)
    - Reranker models (MS MARCO, BGE)
    - Embedding dimensions per model

    SSOT: All data comes from config.json model_registry section.
    Built-in fallbacks used only if config section is missing.
    """

    # ========================================================================
    # PROPERTY ACCESSORS (load config on first access)
    # ========================================================================

    @classmethod
    def _get_llm_models(cls) -> Dict[str, str]:
        """Get LLM models dict (internal)."""
        _ensure_config_loaded()
        return _LLM_MODELS

    @classmethod
    def _get_embedding_models(cls) -> Dict[str, str]:
        """Get embedding models dict (internal)."""
        _ensure_config_loaded()
        return _EMBEDDING_MODELS

    @classmethod
    def _get_reranker_models(cls) -> Dict[str, str]:
        """Get reranker models dict (internal)."""
        _ensure_config_loaded()
        return _RERANKER_MODELS

    @classmethod
    def _get_embedding_dimensions(cls) -> Dict[str, int]:
        """Get embedding dimensions dict (internal)."""
        _ensure_config_loaded()
        return _EMBEDDING_DIMENSIONS

    # ========================================================================
    # BACKWARD COMPATIBILITY: Class-level dict access
    # ========================================================================
    # These are kept for code that accesses ModelRegistry.LLM_MODELS directly

    LLM_MODELS = property(lambda self: ModelRegistry._get_llm_models())
    EMBEDDING_MODELS = property(lambda self: ModelRegistry._get_embedding_models())
    RERANKER_MODELS = property(lambda self: ModelRegistry._get_reranker_models())

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
            Returns alias as-is if not found in registry.

        Example:
            >>> ModelRegistry.resolve_llm("haiku")
            'claude-haiku-4-5-20251001'

            >>> ModelRegistry.resolve_llm("gpt-4o-mini")
            'gpt-4o-mini'

            >>> ModelRegistry.resolve_llm("unknown-model")
            'unknown-model'  # Returns as-is if not found
        """
        models = cls._get_llm_models()
        resolved = models.get(alias, alias)

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
        models = cls._get_embedding_models()
        resolved = models.get(alias, alias)

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
        models = cls._get_reranker_models()
        resolved = models.get(alias, alias)

        if resolved != alias:
            logger.debug(f"Resolved reranker alias: {alias} → {resolved}")

        return resolved

    @classmethod
    def get_embedding_dimensions(cls, model: str) -> int:
        """
        Get embedding dimensions for a model.

        Args:
            model: Model name or alias

        Returns:
            Embedding dimensions (default: 4096 for Qwen3-Embedding-8B)

        Example:
            >>> ModelRegistry.get_embedding_dimensions("Qwen/Qwen3-Embedding-8B")
            4096

            >>> ModelRegistry.get_embedding_dimensions("bge-m3")
            1024
        """
        dims = cls._get_embedding_dimensions()

        # First try direct lookup
        if model in dims:
            return dims[model]

        # Try resolving alias first
        resolved = cls.resolve_embedding(model)
        if resolved in dims:
            return dims[resolved]

        # Default to 4096 (Qwen3-Embedding-8B - current system default)
        logger.warning(
            f"Embedding dimensions not found for '{model}', defaulting to 4096 (Qwen3)"
        )
        return 4096

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
            Provider name ("anthropic", "openai", "google", "voyage", "huggingface", "deepinfra")

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
        resolved_lower = resolved.lower()

        if any(kw in resolved_lower for kw in ["claude", "haiku", "sonnet", "opus"]):
            return "anthropic"
        elif any(kw in resolved_lower for kw in ["gpt-", "o1", "o3", "text-embedding"]):
            return "openai"
        elif "gemini" in resolved_lower:
            return "google"
        elif "voyage" in resolved_lower:
            return "voyage"
        elif any(pattern in resolved for pattern in ["BAAI/", "intfloat/", "sentence-transformers/", "cross-encoder/"]):
            return "huggingface"
        elif any(kw in resolved for kw in ["Qwen/", "meta-llama/", "qwen"]):
            return "deepinfra"
        else:
            # Cannot auto-detect provider - user must specify explicitly
            raise ValueError(
                f"Unable to auto-detect provider for model '{resolved}' (type: {model_type}).\n"
                f"Supported providers: anthropic, openai, google, voyage, huggingface, deepinfra\n"
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
            return cls._get_llm_models().copy()
        elif model_type == "embedding":
            return cls._get_embedding_models().copy()
        elif model_type == "reranker":
            return cls._get_reranker_models().copy()
        elif model_type == "all":
            return {
                "llm": cls._get_llm_models().copy(),
                "embedding": cls._get_embedding_models().copy(),
                "reranker": cls._get_reranker_models().copy(),
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
        dims = ModelRegistry.get_embedding_dimensions(alias)
        is_local = ModelRegistry.is_local_embedding(alias)
        local_str = " (local)" if is_local else " (cloud)"
        print(f"   {alias:25s} → {resolved:40s} dims={dims}{local_str}")

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
