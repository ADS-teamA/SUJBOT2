"""
OpenAI GPT provider implementation.

PRAGMATIC DESIGN:
- Translates Anthropic format → OpenAI format
- No prompt caching (graceful degradation)
- Tool schema translation via ToolSchemaTranslator
- Streaming support (different format than Anthropic)

Features:
- ❌ Prompt caching (not supported by OpenAI)
- ✅ Streaming responses
- ✅ Tool use (function calling)
- ❌ Structured system (uses simple string)
"""

import json
import logging
from typing import Any, Dict, Iterator, List

import openai
from langsmith.wrappers import wrap_openai

from .base import BaseProvider, ProviderResponse
from .tool_translator import ToolSchemaTranslator

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI GPT provider with format translation.

    Supports OpenAI models:
    - gpt-4o (most capable)
    - gpt-4o-mini (balanced, cost-effective)
    - o-series (reasoning models: o1, o3, o4)
    """

    def __init__(self, api_key: str, model: str):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (sk-...)
            model: Model name (e.g., "gpt-4o-mini")

        Raises:
            ValueError: If API key is invalid
        """
        if not api_key or not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
            raise ValueError("Invalid OpenAI API key format (should start with sk- or sk-proj-)")

        self._client = wrap_openai(openai.OpenAI(api_key=api_key))
        self.model = model
        self._translator = ToolSchemaTranslator()

        logger.info(f"OpenAIProvider initialized: model={model}")

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

        Translates Anthropic format → OpenAI format → back to Anthropic.

        Args:
            messages: Conversation history (Anthropic format)
            tools: Tool definitions (Anthropic format)
            system: System prompt (structured list or string)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional OpenAI parameters

        Returns:
            ProviderResponse (Anthropic format)
        """
        # Translate formats
        openai_messages = self._convert_messages_to_openai(messages, system)
        openai_tools = self._translator.to_openai(tools) if tools else None

        # FIX: Build kwargs dynamically - only include tools if provided (same pattern as Anthropic)
        # Before: Always included "tools": openai_tools (could be None, causing BadRequestError)
        api_params = {
            "model": self.model,
            "messages": openai_messages,
            **kwargs,
        }

        # Only add tools parameter if tools are provided (not None and not empty)
        if openai_tools:
            api_params["tools"] = openai_tools

        # O-series reasoning models only support default temperature (1.0)
        if not self._requires_default_temperature():
            api_params["temperature"] = temperature

        # Use correct parameter name based on model
        if self._uses_max_completion_tokens():
            api_params["max_completion_tokens"] = max_tokens
        else:
            api_params["max_tokens"] = max_tokens

        # Call OpenAI API
        response = self._client.chat.completions.create(**api_params)

        # Convert back to Anthropic format
        content = self._convert_response_to_anthropic(response)

        return ProviderResponse(
            content=content,
            stop_reason=response.choices[0].finish_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "cache_read_tokens": 0,  # OpenAI doesn't support caching
                "cache_creation_tokens": 0,
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
        Returns OpenAI stream object. AgentCore type-checks provider
        and handles OpenAI streaming appropriately.

        Args:
            messages: Conversation history (Anthropic format)
            tools: Tool definitions (Anthropic format)
            system: System prompt (structured list or string)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional OpenAI parameters

        Returns:
            OpenAI stream iterator
        """
        openai_messages = self._convert_messages_to_openai(messages, system)
        openai_tools = self._translator.to_openai(tools) if tools else None

        # Prepare API parameters
        api_params = {
            "model": self.model,
            "messages": openai_messages,
            "tools": openai_tools,
            "stream": True,
            **kwargs,
        }

        # O-series reasoning models only support default temperature (1.0)
        if not self._requires_default_temperature():
            api_params["temperature"] = temperature

        # Use correct parameter name based on model
        if self._uses_max_completion_tokens():
            api_params["max_completion_tokens"] = max_tokens
        else:
            api_params["max_tokens"] = max_tokens

        return self._client.chat.completions.create(**api_params)

    def _convert_messages_to_openai(
        self, messages: List[Dict[str, Any]], system: List[Dict[str, Any]] | str
    ) -> List[Dict[str, Any]]:
        """
        Convert Anthropic messages → OpenAI format.

        Handles:
        - System prompt (structured → string)
        - Tool results (different format)
        - Tool calls (different format)
        - Cache control removal

        Args:
            messages: Anthropic conversation history
            system: System prompt (structured or string)

        Returns:
            OpenAI messages array
        """
        openai_msgs = []

        # Add system message (OpenAI uses simple string)
        if isinstance(system, list):
            # Extract text from structured system blocks
            system_text = " ".join(block["text"] for block in system if block.get("type") == "text")
        else:
            system_text = system

        openai_msgs.append({"role": "system", "content": system_text})

        # Convert conversation history
        for msg in messages:
            if msg["role"] == "user":
                # Handle tool results
                content = msg["content"]

                if isinstance(content, list):
                    # Multiple content blocks (tool results)
                    for block in content:
                        if block.get("type") == "tool_result":
                            # OpenAI tool result format
                            openai_msgs.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block["tool_use_id"],
                                    "content": block["content"],
                                }
                            )
                        elif block.get("type") == "text":
                            openai_msgs.append({"role": "user", "content": block["text"]})
                else:
                    # Simple text message
                    openai_msgs.append({"role": "user", "content": content})

            elif msg["role"] == "assistant":
                # Handle tool calls
                content = msg["content"]

                if isinstance(content, list):
                    text_parts = []
                    tool_calls = []

                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            # OpenAI tool call format
                            tool_calls.append(
                                {
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": json.dumps(block["input"]),
                                    },
                                }
                            )

                    msg_dict = {"role": "assistant"}

                    if text_parts:
                        msg_dict["content"] = " ".join(text_parts)
                    else:
                        # OpenAI requires content field even if empty
                        msg_dict["content"] = None

                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls

                    openai_msgs.append(msg_dict)
                else:
                    # Simple text message
                    openai_msgs.append({"role": "assistant", "content": content})

        return openai_msgs

    def _convert_response_to_anthropic(self, response) -> List[Dict[str, Any]]:
        """
        Convert OpenAI response → Anthropic content format.

        Args:
            response: OpenAI ChatCompletion response

        Returns:
            Anthropic content blocks

        Raises:
            ValueError: If response structure is invalid
        """
        if not response.choices:
            logger.error(f"OpenAI response has no choices: {response}")
            raise ValueError("Invalid OpenAI response: no choices returned")

        choice = response.choices[0]
        message = choice.message
        content = []

        # Text content
        if message.content:
            content.append({"type": "text", "text": message.content})

        # Tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    if not tool_call.function:
                        logger.warning(f"Tool call missing function: {tool_call}")
                        continue

                    parsed_args = json.loads(tool_call.function.arguments)

                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": parsed_args,
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse tool arguments: {e}. "
                        f"Tool: {tool_call.function.name}, Args: {tool_call.function.arguments[:200]}"
                    )
                    # Skip malformed tool call
                    continue

        return content

    def supports_feature(self, feature: str) -> bool:
        """
        Check if OpenAI supports a feature.

        Supported features:
        - prompt_caching: ❌ No (OpenAI doesn't support native caching)
        - streaming: ✅ Yes (via stream_message method)
        - tool_use: ✅ Yes (function calling)
        - structured_system: ❌ No (uses simple string)

        Args:
            feature: Feature name

        Returns:
            True if supported
        """
        supported_features = {
            "tool_use",
            "streaming",
        }
        return feature in supported_features

    def _uses_max_completion_tokens(self) -> bool:
        """
        Check if model uses max_completion_tokens instead of max_tokens.

        O-series reasoning models (o1, o3, o4) use max_completion_tokens.
        Older models (gpt-4, gpt-3.5) use max_tokens.

        Returns:
            True if should use max_completion_tokens
        """
        model_lower = self.model.lower()
        # O-series reasoning models
        return any(
            pattern in model_lower for pattern in ["o1", "o3", "o4"]
        )

    def _requires_default_temperature(self) -> bool:
        """
        Check if model requires default temperature (1.0).

        O-series reasoning models (o1, o3, o4) only support temperature=1.
        Older models support custom temperature values.

        Returns:
            True if should omit temperature parameter
        """
        model_lower = self.model.lower()
        # O-series reasoning models
        return any(
            pattern in model_lower for pattern in ["o1", "o3", "o4"]
        )

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model

    def set_model(self, model: str) -> None:
        """
        Change model.

        Args:
            model: New model name (must be GPT model)

        Raises:
            ValueError: If model is not a GPT model
        """
        if "gpt-" not in model.lower() and "o1" not in model.lower() and "o3" not in model.lower():
            raise ValueError(f"Invalid OpenAI model: {model}")

        old_model = self.model
        self.model = model
        logger.info(f"Model changed: {old_model} → {model}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"
