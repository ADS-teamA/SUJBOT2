"""
DeepInfra LLM Provider using OpenAI-compatible API.

Supports Llama and Qwen models via DeepInfra's OpenAI-compatible endpoint.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Iterator, FrozenSet

from openai import OpenAI
from langsmith.wrappers import wrap_openai

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)

# Supported models for validation
SUPPORTED_MODELS: FrozenSet[str] = frozenset({
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
})


class DeepInfraProvider(BaseProvider):
    """
    DeepInfra provider for open-source LLMs.

    Uses OpenAI-compatible API format (https://api.deepinfra.com/v1/openai).
    Supports:
    - meta-llama/Meta-Llama-3.1-70B-Instruct (recommended for agents)
    - meta-llama/Meta-Llama-3.1-8B-Instruct (lighter model)
    - Qwen/Qwen2.5-72B-Instruct
    - Qwen/Qwen2.5-7B-Instruct
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"):
        """
        Initialize DeepInfra provider.

        Args:
            api_key: DeepInfra API key (defaults to DEEPINFRA_API_KEY env var)
            model: Model identifier (default: meta-llama/Meta-Llama-3.1-70B-Instruct)

        Raises:
            ValueError: If API key is missing or model is not supported
        """
        self.api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY required. "
                "Set in .env file or pass to constructor."
            )

        # Validate model
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported models: {', '.join(sorted(SUPPORTED_MODELS))}"
            )

        self.model = model

        # Initialize OpenAI client with DeepInfra base URL (wrapped for LangSmith tracing)
        self.client = wrap_openai(OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepinfra.com/v1/openai",
            timeout=60.0,
            max_retries=3
        ))

        logger.info(f"DeepInfraProvider initialized: {model}")

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        **kwargs
    ) -> ProviderResponse:
        """
        Create chat completion (non-streaming).

        Args:
            messages: Conversation messages in Anthropic format
            tools: Tool definitions in Anthropic format
            system: System prompt (string or list of dicts)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            ProviderResponse with content, usage, etc.
        """
        # Format messages for OpenAI API
        formatted_messages = self._format_messages(messages, system)

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {})
                    }
                }
                for tool in tools
            ]

        # Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=openai_tools,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"DeepInfra API error: {e}")
            raise

        # Convert to Anthropic-compatible format
        return self._convert_response(response)

    def stream_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        **kwargs
    ) -> Iterator:
        """
        Stream chat completion.

        Args:
            messages: Conversation messages in Anthropic format
            tools: Tool definitions in Anthropic format
            system: System prompt (string or list of dicts)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Iterator of streaming response chunks
        """
        formatted_messages = self._format_messages(messages, system)

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {})
                    }
                }
                for tool in tools
            ]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            tools=openai_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )

        return stream

    def _format_messages(self, messages: List[Dict], system: Any) -> List[Dict]:
        """
        Convert Anthropic message format to OpenAI format.

        Args:
            messages: Anthropic-format messages
            system: System prompt (string or list of dicts)

        Returns:
            OpenAI-format messages
        """
        formatted = []

        # Add system message
        if system:
            if isinstance(system, str):
                formatted.append({"role": "system", "content": system})
            elif isinstance(system, list):
                # Extract text from Anthropic structured format
                system_text = "".join(
                    block.get("text", "")
                    for block in system
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if system_text:
                    formatted.append({"role": "system", "content": system_text})

        # Convert messages
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                formatted.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Extract text blocks
                text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if text:
                    formatted.append({"role": role, "content": text})

        return formatted

    def _convert_response(self, response) -> ProviderResponse:
        """
        Convert OpenAI response to Anthropic ProviderResponse format.

        Args:
            response: OpenAI API response

        Returns:
            ProviderResponse in Anthropic format

        Raises:
            ValueError: If response has no choices
        """
        if not response.choices:
            raise ValueError("Empty response from DeepInfra API - no choices returned")

        choice = response.choices[0]
        message = choice.message

        # Build content blocks
        content_blocks = []

        if message.content:
            content_blocks.append({
                "type": "text",
                "text": message.content
            })

        # Convert tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments)
                })

        # Map finish reason
        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens"
        }

        return ProviderResponse(
            content=content_blocks,
            stop_reason=stop_reason_map.get(choice.finish_reason, "end_turn"),
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "cache_read_tokens": 0,  # DeepInfra doesn't support caching
                "cache_creation_tokens": 0
            },
            model=self.model
        )

    def supports_feature(self, feature: str) -> bool:
        """
        Check feature support.

        Args:
            feature: Feature name ("streaming", "tool_use", "prompt_caching", etc.)

        Returns:
            True if feature is supported
        """
        supported_features = {
            "streaming": True,
            "tool_use": True,
            "prompt_caching": False,  # DeepInfra doesn't support Anthropic-style caching
            "structured_system": False
        }
        return supported_features.get(feature, False)

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model

    def set_model(self, model: str) -> None:
        """
        Set new model.

        Args:
            model: Model identifier

        Raises:
            ValueError: If model is not supported
        """
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported models: {', '.join(sorted(SUPPORTED_MODELS))}"
            )
        self.model = model
        logger.info(f"Switched to model: {model}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "deepinfra"
