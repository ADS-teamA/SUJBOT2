"""
Provider factory for creating appropriate provider instance.

Automatically detects provider from model name and creates correct implementation.
"""

import logging
import os
from typing import Optional

from .anthropic_provider import AnthropicProvider
from .base import BaseProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .deepinfra_provider import DeepInfraProvider

logger = logging.getLogger(__name__)


def create_provider(
    model: str,
    api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
) -> BaseProvider:
    """
    Create appropriate provider based on model name.

    Automatically detects provider from model name and loads API key from
    environment if not provided.

    Args:
        model: Model name or alias (e.g., "claude-haiku-4-5", "gpt-5-mini", "gemini-flash")
        api_key: API key for the provider (deprecated, use provider-specific keys)
        anthropic_api_key: Anthropic API key (optional, defaults to ANTHROPIC_API_KEY env var)
        openai_api_key: OpenAI API key (optional, defaults to OPENAI_API_KEY env var)
        google_api_key: Google API key (optional, defaults to GOOGLE_API_KEY env var)

    Returns:
        Provider instance (AnthropicProvider, OpenAIProvider, or GeminiProvider)

    Raises:
        ValueError: If provider cannot be determined or API key is missing

    Examples:
        >>> # Using model registry aliases
        >>> provider = create_provider("haiku")  # Uses ANTHROPIC_API_KEY from env
        >>> provider = create_provider("gpt-5-mini")  # Uses OPENAI_API_KEY from env
        >>>
        >>> # Using full model names
        >>> provider = create_provider("claude-haiku-4-5-20251001")
        >>> provider = create_provider("gpt-5-nano")
        >>>
        >>> # Providing API keys explicitly
        >>> provider = create_provider("haiku", anthropic_api_key="sk-ant-...")
        >>> provider = create_provider("gpt-5-mini", openai_api_key="sk-...")
    """
    # Import here to avoid circular dependency
    try:
        from ...utils.model_registry import ModelRegistry
    except ImportError:
        # Fallback if model_registry not available
        logger.warning("ModelRegistry not available, using direct model name")
        resolved_model = model
        provider_name = _detect_provider_from_model(model)
    else:
        # Resolve model alias
        resolved_model = ModelRegistry.resolve_llm(model)
        provider_name = ModelRegistry.get_provider(resolved_model, "llm")

    logger.info(f"Creating provider: model={resolved_model}, provider={provider_name}")

    # Create provider instance
    if provider_name == "anthropic":
        # Get API key (priority: explicit param > env var)
        key = anthropic_api_key or api_key or os.getenv("ANTHROPIC_API_KEY")

        if not key:
            raise ValueError(
                "Anthropic API key required for Claude models.\n"
                "Set ANTHROPIC_API_KEY environment variable or pass anthropic_api_key parameter.\n"
                "Example: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        return AnthropicProvider(api_key=key, model=resolved_model)

    elif provider_name == "openai":
        # Get API key (priority: explicit param > env var)
        key = openai_api_key or api_key or os.getenv("OPENAI_API_KEY")

        if not key:
            raise ValueError(
                "OpenAI API key required for GPT models.\n"
                "Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.\n"
                "Example: export OPENAI_API_KEY=sk-..."
            )

        return OpenAIProvider(api_key=key, model=resolved_model)

    elif provider_name == "google":
        # Get API key (priority: explicit param > env var)
        key = google_api_key or api_key or os.getenv("GOOGLE_API_KEY")

        if not key:
            raise ValueError(
                "Google API key required for Gemini models.\n"
                "Set GOOGLE_API_KEY environment variable or pass google_api_key parameter.\n"
                "Example: export GOOGLE_API_KEY=AIza..."
            )

        return GeminiProvider(api_key=key, model=resolved_model)

    elif provider_name == "deepinfra":
        # Get API key from env var
        key = api_key or os.getenv("DEEPINFRA_API_KEY")

        if not key:
            raise ValueError(
                "DeepInfra API key required for Qwen models.\n"
                "Set DEEPINFRA_API_KEY environment variable or pass api_key parameter.\n"
                "Example: export DEEPINFRA_API_KEY=..."
            )

        return DeepInfraProvider(api_key=key, model=resolved_model)

    else:
        raise ValueError(
            f"Unsupported provider: {provider_name} for model: {model}\n"
            f"Supported providers: anthropic (Claude), openai (GPT-5), google (Gemini), deepinfra (Qwen)"
        )


def _detect_provider_from_model(model: str) -> str:
    """
    Fallback provider detection from model name (when ModelRegistry unavailable).

    Args:
        model: Model name

    Returns:
        Provider name ("anthropic" or "openai")

    Raises:
        ValueError: If provider cannot be determined
    """
    model_lower = model.lower()

    # Anthropic patterns
    if any(pattern in model_lower for pattern in ["claude", "haiku", "sonnet", "opus"]):
        return "anthropic"

    # OpenAI patterns
    if any(pattern in model_lower for pattern in ["gpt-", "o1", "o3"]):
        return "openai"

    # Google Gemini patterns
    if "gemini" in model_lower:
        return "google"

    # DeepInfra patterns (Qwen models)
    if "qwen" in model_lower or model_lower.startswith("qwen/"):
        return "deepinfra"

    raise ValueError(
        f"Cannot determine provider for model: {model}\n"
        f"Model name should contain: 'claude', 'haiku', 'sonnet', 'opus' (Anthropic), "
        f"'gpt-', 'o1', 'o3' (OpenAI), 'gemini' (Google), or 'qwen' (DeepInfra)"
    )
