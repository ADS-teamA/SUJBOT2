"""
Anthropic Claude provider implementation.

PRAGMATIC NOTES:
- Reuses existing agent_core.py logic (proven, battle-tested)
- Native Anthropic tool format (no translation needed)
- Full support for prompt caching (90% cost savings)
- Native streaming support

Features:
- ✅ Prompt caching (cache_control markers)
- ✅ Streaming responses
- ✅ Tool use (native format)
- ✅ Structured system prompts
"""

import logging
from typing import Any, Dict, Iterator, List

import anthropic

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude provider with native prompt caching support.

    Supports all Claude models:
    - Haiku 4.5 (fast, cost-effective)
    - Sonnet 4.5 (balanced)
    - Opus 4 (most capable)
    """

    def __init__(self, api_key: str, model: str):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (sk-ant-...)
            model: Model name (e.g., "claude-haiku-4-5-20251001")

        Raises:
            ValueError: If API key or model is invalid
        """
        if not api_key or not api_key.startswith("sk-ant-"):
            raise ValueError("Invalid Anthropic API key format (should start with sk-ant-)")

        # Validate model name before creating client
        if not any(pattern in model.lower() for pattern in ["claude", "haiku", "sonnet"]):
            raise ValueError(f"Invalid Claude model: {model}")

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        logger.info(f"AnthropicProvider initialized: model={model}")

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ProviderResponse:
        """
        Create message synchronously (non-streaming).

        Uses native Anthropic format (no translation needed).

        Args:
            messages: Conversation history (Anthropic format)
            tools: Tool definitions (Anthropic format)
            system: System prompt (structured list or string)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional Anthropic parameters

        Returns:
            ProviderResponse with content, usage, etc.
        """
        # Build kwargs dynamically - only include tools if provided
        create_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
            **kwargs,
        }

        # Only add tools parameter if tools are provided (not None and not empty)
        if tools:
            create_kwargs["tools"] = tools

        response = self._client.messages.create(**create_kwargs)

        return ProviderResponse(
            content=[
                block.model_dump() for block in response.content
            ],  # Convert Pydantic models to dicts
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            },
            model=response.model,
        )

    def stream_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Iterator[Any]:
        """
        Create message with streaming.

        PRAGMATIC NOTE:
        Returns native Anthropic stream object. AgentCore knows how to handle it
        (existing streaming logic). This avoids complex abstraction layers.

        Args:
            messages: Conversation history (Anthropic format)
            tools: Tool definitions (Anthropic format)
            system: System prompt (structured list or string)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional Anthropic parameters

        Returns:
            Anthropic MessageStream context manager

        Example:
            >>> stream = provider.stream_message(...)
            >>> with stream as anthropic_stream:
            >>>     for event in anthropic_stream:
            >>>         if event.type == "content_block_delta":
            >>>             print(event.delta.text)
        """
        # Build kwargs dynamically - only include tools if provided
        stream_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
            **kwargs,
        }

        # Only add tools parameter if tools are provided (not None and not empty)
        if tools:
            stream_kwargs["tools"] = tools

        return self._client.messages.stream(**stream_kwargs)

    def supports_feature(self, feature: str) -> bool:
        """
        Check if Anthropic supports a feature.

        Supported features:
        - prompt_caching: ✅ Yes (90% cost savings)
        - streaming: ✅ Yes
        - tool_use: ✅ Yes
        - structured_system: ✅ Yes (native format)

        Args:
            feature: Feature name

        Returns:
            True if supported
        """
        supported_features = {
            "prompt_caching",
            "streaming",
            "tool_use",
            "structured_system",
        }
        return feature in supported_features

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model

    def set_model(self, model: str) -> None:
        """
        Change model.

        Args:
            model: New model name (must be Claude model)

        Raises:
            ValueError: If model is not a Claude model
        """
        if (
            "claude" not in model.lower()
            and "haiku" not in model.lower()
            and "sonnet" not in model.lower()
            and "opus" not in model.lower()
        ):
            raise ValueError(f"Invalid Claude model: {model}")

        old_model = self.model
        self.model = model
        logger.info(f"Model changed: {old_model} → {model}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "anthropic"
