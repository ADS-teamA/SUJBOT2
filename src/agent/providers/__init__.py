"""
Provider abstraction for multi-LLM support.

Supports:
- Anthropic Claude (Haiku, Sonnet, Opus) with prompt caching
- OpenAI GPT-5 (Mini, Nano) with function calling
- Google Gemini (2.5 Flash, Pro) with context caching

Usage:
    from src.agent.providers import create_provider

    provider = create_provider(model="gemini-2.5-flash", google_api_key=api_key)
    response = provider.create_message(messages, tools, system, ...)
"""

from .base import BaseProvider, ProviderResponse
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .factory import create_provider

# Optional: GeminiProvider (requires compatible google-generativeai version)
try:
    from .gemini_provider import GeminiProvider
    _GEMINI_AVAILABLE = True
except (ImportError, AttributeError) as e:
    import logging
    logging.getLogger(__name__).warning(f"GeminiProvider not available: {e}")

    # Create a placeholder class that raises helpful error on instantiation
    class GeminiProvider:  # type: ignore
        """Placeholder when GeminiProvider is unavailable."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GeminiProvider is not available due to SDK incompatibility. "
                "Install compatible google-generativeai version or use a different provider."
            )

    _GEMINI_AVAILABLE = False

__all__ = [
    "BaseProvider",
    "ProviderResponse",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "create_provider",
    "_GEMINI_AVAILABLE",  # Export availability flag
]
