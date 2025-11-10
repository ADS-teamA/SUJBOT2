"""
Google Gemini provider for agent system.

Implements BaseProvider interface using Google's genai SDK.
Handles tool schema translation and context caching.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """
    Google Gemini provider implementation.

    Features:
    - Tool use via function calling
    - Context caching (90% cost savings on cache hits)
    - Streaming support
    - Tool schema translation (Anthropic → Gemini format)
    - Response translation (Gemini → Anthropic format)
    """

    def __init__(self, api_key: str, model: str):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key
            model: Gemini model name (e.g., "gemini-2.5-flash")

        Raises:
            ValueError: If api_key or model is invalid
        """
        # Validate API key
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is required")

        if not api_key.startswith("AIza"):
            logger.warning("Unusual API key format (expected AIza prefix)")

        # Validate model name
        if "gemini" not in model.lower():
            raise ValueError(
                f"Invalid Gemini model: {model}\n"
                f"Expected model name containing 'gemini' (e.g., 'gemini-2.5-flash')"
            )

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self._cache: Optional[types.CachedContent] = None
        logger.info(f"Initialized Gemini provider: model={model}")

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
        Create message (non-streaming).

        Args:
            messages: Conversation history (Anthropic format)
            tools: Tool definitions (Anthropic format)
            system: System prompt (structured or string)
            max_tokens: Maximum output tokens
            temperature: Sampling temperature

        Returns:
            ProviderResponse with translated content
        """
        # Convert to Gemini format
        gemini_messages = self._convert_messages_to_gemini(messages)
        gemini_tools = self._convert_tools_to_gemini(tools) if tools else None

        # Extract system instruction
        system_instruction = self._extract_system_instruction(system) if system else None

        # Build config
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Add system instruction if provided (CRITICAL for tool calling)
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        # Add tools if provided
        if gemini_tools:
            config_params["tools"] = gemini_tools

        # Use cached content if available
        if self._cache:
            config_params["cached_content"] = self._cache.name
            logger.debug(f"Using cached content: {self._cache.name}")

        config = types.GenerateContentConfig(**config_params)

        # Call Gemini API
        try:
            # DEBUG: Log what we're sending
            logger.debug(f"Sending to Gemini: {len(gemini_messages)} messages, tools={'yes' if gemini_tools else 'no'}, system={'yes' if system_instruction else 'no'}")

            response = self._client.models.generate_content(
                model=self.model,
                contents=gemini_messages,
                config=config,
            )

            # DEBUG: Log what we received
            if response.candidates:
                candidate = response.candidates[0]
                parts_info = []
                for part in candidate.content.parts:
                    if part.text:
                        parts_info.append(f"text({len(part.text)} chars)")
                    elif part.function_call:
                        parts_info.append(f"function_call({part.function_call.name})")
                logger.debug(f"Gemini response: {len(response.candidates)} candidates, parts=[{', '.join(parts_info)}]")
            else:
                logger.warning("Gemini returned NO candidates!")

            # Convert to Anthropic format
            return self._convert_response_to_anthropic(response)

        except google_exceptions.Unauthenticated as e:
            logger.error(f"Gemini API authentication failed: {e}")
            raise ValueError("Invalid Google API key. Please check GOOGLE_API_KEY in .env") from e

        except google_exceptions.NotFound as e:
            logger.error(f"Gemini model not found: {self.model}")
            raise ValueError(f"Model '{self.model}' not available. Check model name.") from e

        except google_exceptions.ResourceExhausted as e:
            logger.error(f"Gemini quota exceeded: {e}")
            raise ValueError("Gemini API quota exceeded. Try again later or upgrade plan.") from e

        except google_exceptions.InvalidArgument as e:
            logger.error(f"Gemini rejected request (content policy?): {e}")
            raise ValueError(f"Gemini rejected request: {e}") from e

        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Failed to parse Gemini response: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Gemini response format: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected Gemini API error: {e}", exc_info=True)
            raise

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

        PRAGMATIC NOTE: Returns Gemini stream object directly.
        AgentCore handles provider-specific streaming.

        Args:
            messages: Conversation history (Anthropic format)
            tools: Tool definitions (Anthropic format)
            system: System prompt
            max_tokens: Maximum output tokens
            temperature: Sampling temperature

        Yields:
            Gemini stream chunks
        """
        # Convert to Gemini format
        gemini_messages = self._convert_messages_to_gemini(messages)
        gemini_tools = self._convert_tools_to_gemini(tools) if tools else None

        # Extract system instruction
        system_instruction = self._extract_system_instruction(system) if system else None

        # Build config
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Add system instruction if provided (CRITICAL for tool calling)
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        # Add tools if provided
        if gemini_tools:
            config_params["tools"] = gemini_tools

        # Use cached content if available
        if self._cache:
            config_params["cached_content"] = self._cache.name

        config = types.GenerateContentConfig(**config_params)

        # Stream from Gemini API
        try:
            stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=gemini_messages,
                config=config,
            )

            return stream

        except google_exceptions.Unauthenticated as e:
            logger.error(f"Gemini API authentication failed: {e}")
            raise ValueError("Invalid Google API key. Please check GOOGLE_API_KEY in .env") from e

        except google_exceptions.NotFound as e:
            logger.error(f"Gemini model not found: {self.model}")
            raise ValueError(f"Model '{self.model}' not available. Check model name.") from e

        except google_exceptions.ResourceExhausted as e:
            logger.error(f"Gemini quota exceeded: {e}")
            raise ValueError("Gemini API quota exceeded. Try again later or upgrade plan.") from e

        except google_exceptions.InvalidArgument as e:
            logger.error(f"Gemini rejected request (content policy?): {e}")
            raise ValueError(f"Gemini rejected request: {e}") from e

        except google_exceptions.DeadlineExceeded as e:
            logger.error(f"Gemini streaming timeout: {e}")
            raise TimeoutError("Gemini stream timed out. Try a shorter query.") from e

        except Exception as e:
            logger.error(f"Unexpected Gemini streaming error: {e}", exc_info=True)
            raise

    def supports_feature(self, feature: str) -> bool:
        """
        Check if provider supports a feature.

        Supported features:
        - "prompt_caching": Yes (context caching)
        - "streaming": Yes
        - "tool_use": Yes (function calling)
        - "structured_system": No (uses string only)

        Args:
            feature: Feature name

        Returns:
            True if supported
        """
        supported = {
            "prompt_caching": True,  # Context caching
            "streaming": True,
            "tool_use": True,  # Function calling
            "structured_system": False,  # Gemini uses string system instruction
        }
        return supported.get(feature, False)

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model

    def set_model(self, model: str) -> None:
        """
        Change model.

        Args:
            model: New Gemini model name
        """
        self.model = model
        # Clear cache when switching models
        self._cache = None
        logger.info(f"Switched to model: {model}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "google"

    # ========================================================================
    # CONTEXT CACHING
    # ========================================================================

    def create_cache(
        self,
        system_instruction: str,
        tools: List[Dict[str, Any]],
        ttl_seconds: int = 300,
    ) -> None:
        """
        Create context cache for system prompt and tools.

        Cache requirements (Gemini):
        - Minimum 1024 tokens (2.5 Flash) or 4096 tokens (2.5 Pro)
        - Default TTL: 5 minutes (300s)

        Args:
            system_instruction: System prompt
            tools: Tool definitions (Anthropic format)
            ttl_seconds: Time-to-live in seconds
        """
        try:
            gemini_tools = self._convert_tools_to_gemini(tools) if tools else []

            # Create cached content
            cache = self._client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    display_name=f"agent_cache_{self.model}",
                    system_instruction=system_instruction,
                    tools=gemini_tools,
                    ttl=f"{ttl_seconds}s",
                ),
            )

            self._cache = cache
            logger.info(
                f"Created context cache: {cache.name} (TTL={ttl_seconds}s, "
                f"tokens={cache.usage_metadata.total_token_count if hasattr(cache, 'usage_metadata') else 'unknown'})"
            )

        except Exception as e:
            logger.warning(f"Failed to create cache: {e}. Continuing without caching.")
            self._cache = None

    def clear_cache(self) -> None:
        """Clear cached content."""
        if self._cache:
            try:
                self._client.caches.delete(name=self._cache.name)
                logger.info(f"Deleted cache: {self._cache.name}")
            except Exception as e:
                logger.warning(f"Failed to delete cache: {e}")
            finally:
                self._cache = None

    # ========================================================================
    # FORMAT CONVERSION
    # ========================================================================

    def _convert_messages_to_gemini(
        self, messages: List[Dict[str, Any]]
    ) -> List[types.Content]:
        """
        Convert Anthropic message format to Gemini format.

        Anthropic format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": [...]}]

        Gemini format:
            [Content(role="user", parts=[Part(text="...")]), ...]

        Args:
            messages: Anthropic-format messages

        Returns:
            Gemini Content objects
        """
        gemini_messages = []

        # Track tool_use_id -> function_name mapping for tool results
        # Gemini requires FunctionResponse.name to match the original function name
        tool_id_to_name = {}

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Convert role (anthropic → gemini)
            if role == "assistant":
                gemini_role = "model"
            else:
                gemini_role = "user"

            # Convert content blocks
            parts = []
            if isinstance(content, str):
                parts.append(types.Part(text=content))
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        parts.append(types.Part(text=block["text"]))
                    elif block.get("type") == "tool_use":
                        # Gemini function call format
                        # Track the mapping for later tool_result conversion
                        tool_id = block.get("id")
                        tool_name = block["name"]
                        if tool_id:
                            tool_id_to_name[tool_id] = tool_name

                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=tool_name,
                                    args=block.get("input", {}),
                                )
                            )
                        )
                    elif block.get("type") == "tool_result":
                        # Gemini function response format
                        # CRITICAL: Use function name (not tool_use_id) to match FunctionCall
                        tool_use_id = block["tool_use_id"]
                        function_name = tool_id_to_name.get(tool_use_id)

                        if not function_name:
                            # Fallback: extract name from ID (toolu_FUNCNAME -> FUNCNAME)
                            function_name = tool_use_id.replace("toolu_", "")
                            logger.warning(
                                f"Could not find function name for tool_use_id={tool_use_id}, "
                                f"using fallback: {function_name}"
                            )

                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=function_name,  # Use function name (not ID)
                                    response={"result": block.get("content", "")},
                                )
                            )
                        )

            gemini_messages.append(types.Content(role=gemini_role, parts=parts))

        return gemini_messages

    def _convert_tools_to_gemini(self, tools: List[Dict[str, Any]]) -> List[types.Tool]:
        """
        Convert Anthropic tool format to Gemini function declarations.

        Anthropic format:
            {"name": "search", "description": "...", "input_schema": {...}}

        Gemini format:
            Tool(function_declarations=[FunctionDeclaration(...)])

        Args:
            tools: Anthropic tool definitions

        Returns:
            Gemini Tool object
        """
        function_declarations = []

        for tool in tools:
            # Convert input_schema → parameters
            input_schema = tool.get("input_schema", {})
            parameters = {
                "type": input_schema.get("type", "object"),
                "properties": input_schema.get("properties", {}),
                "required": input_schema.get("required", []),
            }

            func_decl = types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=parameters,
            )
            function_declarations.append(func_decl)

        return [types.Tool(function_declarations=function_declarations)]

    def _convert_response_to_anthropic(
        self, response: types.GenerateContentResponse
    ) -> ProviderResponse:
        """
        Convert Gemini response to Anthropic ProviderResponse format.

        Args:
            response: Gemini API response

        Returns:
            ProviderResponse (Anthropic format)
        """
        # Extract content blocks
        content_blocks = []

        if response.candidates:
            candidate = response.candidates[0]

            for part in candidate.content.parts:
                if part.text:
                    # Text block
                    content_blocks.append({"type": "text", "text": part.text})
                elif part.function_call:
                    # Tool use block
                    content_blocks.append({
                        "type": "tool_use",
                        "id": f"toolu_{part.function_call.name}",  # Generate ID
                        "name": part.function_call.name,
                        "input": dict(part.function_call.args),
                    })

            # Determine stop reason
            # CRITICAL: Check if response contains tool_use blocks first
            has_tool_use = any(block.get("type") == "tool_use" for block in content_blocks)

            if has_tool_use:
                # Gemini called a function - must continue loop for tool execution
                stop_reason = "tool_use"
            else:
                # Normal finish reasons
                finish_reason = candidate.finish_reason
                if finish_reason == types.FinishReason.STOP:
                    stop_reason = "end_turn"
                elif finish_reason == types.FinishReason.MAX_TOKENS:
                    stop_reason = "max_tokens"
                else:
                    stop_reason = "end_turn"  # Default

        else:
            content_blocks = [{"type": "text", "text": ""}]
            stop_reason = "end_turn"

        # Extract usage (handle None values carefully)
        usage_metadata = response.usage_metadata

        # Extract tokens with None-safety
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0

        if usage_metadata:
            # Input tokens
            input_tokens = (
                usage_metadata.prompt_token_count
                if usage_metadata.prompt_token_count is not None
                else 0
            )

            # Output tokens
            output_tokens = (
                usage_metadata.candidates_token_count
                if usage_metadata.candidates_token_count is not None
                else 0
            )

            # Cache read tokens (can be None even if attribute exists)
            if hasattr(usage_metadata, "cached_content_token_count"):
                cached_count = usage_metadata.cached_content_token_count
                cache_read_tokens = cached_count if cached_count is not None else 0

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": 0,  # Gemini doesn't report separately
            "cache_read_tokens": cache_read_tokens,
        }

        return ProviderResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            usage=usage,
            model=self.model,
        )

    def _extract_system_instruction(self, system: List[Dict[str, Any]] | str) -> str:
        """
        Extract system instruction from structured or string format.

        Args:
            system: System prompt (Anthropic format)

        Returns:
            System instruction string
        """
        if isinstance(system, str):
            return system
        elif isinstance(system, list):
            # Extract text from structured blocks
            parts = []
            for block in system:
                if block.get("type") == "text":
                    parts.append(block["text"])
            return "\n".join(parts)
        else:
            return ""
