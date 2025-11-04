"""
Agent Core - Claude SDK Orchestration

Main agent loop with:
- Claude SDK integration
- Tool execution
- Conversation management
- Streaming responses
"""

import json
import logging
from typing import Any, Dict, Generator, List, Optional

import anthropic
import numpy as np

from .config import AgentConfig
from .providers import create_provider, AnthropicProvider, GeminiProvider, OpenAIProvider
from .tools.base import ToolResult
from .tools.registry import get_registry

try:
    from ..cost_tracker import get_global_tracker
    from ..utils.security import sanitize_error
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from cost_tracker import get_global_tracker
    from utils.security import sanitize_error

logger = logging.getLogger(__name__)

# Configuration constants
MAX_HISTORY_MESSAGES = 50  # Keep last 50 messages to maintain conversation context
MAX_QUERY_LENGTH = 10000  # Maximum characters in a single query

# ANSI color codes for terminal output
COLOR_BLUE = "\033[1;34m"  # Bold blue for tool calls and debug
COLOR_RESET = "\033[0m"  # Reset color


class AgentCore:
    """
    Core agent orchestration using Claude SDK.

    Manages:
    - Claude SDK client
    - Conversation history
    - Tool execution loop
    - Streaming responses
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize agent core.

        Args:
            config: AgentConfig instance
        """
        self.config = config

        # Initialize cost tracker
        self.tracker = get_global_tracker()

        if config.debug_mode:
            logger.debug("Initializing AgentCore...")
            logger.debug(f"Model: {config.model}")
            logger.debug(f"Max tokens: {config.max_tokens}")
            logger.debug(f"Temperature: {config.temperature}")
            logger.debug(f"Prompt caching: {config.enable_prompt_caching}")

        # Validate config
        config.validate()

        # Initialize provider (Anthropic, OpenAI, or Google)
        if config.debug_mode:
            logger.debug("Initializing provider...")
        self.provider = create_provider(
            model=config.model,
            anthropic_api_key=config.anthropic_api_key,
            openai_api_key=config.openai_api_key,
            google_api_key=config.google_api_key,
        )

        # Log provider info
        logger.info(
            f"Provider initialized: {self.provider.get_provider_name()} / {self.provider.get_model_name()}"
        )

        # Check feature support and log warnings
        if config.enable_prompt_caching and not self.provider.supports_feature("prompt_caching"):
            logger.warning(
                f"Prompt caching requested but not supported by {self.provider.get_model_name()}. "
                f"Continuing without caching (costs will be higher)."
            )

        # Get tool registry
        if config.debug_mode:
            logger.debug("Getting tool registry...")
        self.registry = get_registry()

        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_call_history: List[Dict[str, Any]] = []

        logger.info(
            f"AgentCore initialized: model={config.model}, "
            f"tools={len(self.registry)}, streaming={config.cli_config.enable_streaming}, "
            f"caching={config.enable_prompt_caching}"
        )

        if config.debug_mode:
            tool_list = list(self.registry._tool_classes.keys())
            logger.debug(f"Available tools: {tool_list}")

        # Flag to track if initialized
        self._initialized_with_documents = False

    def _clean_summary_text(self, text: str) -> str:
        """
        Clean summary text for conversation history.

        Removes:
        - HTML entities (&lt;, &gt;, etc.)
        - Markdown formatting (##, **, etc.)
        - Extra whitespace and newlines
        """
        import html
        import re

        # Unescape HTML entities
        text = html.unescape(text)

        # Remove markdown headers (## Header)
        text = re.sub(r"#+\s+", "", text)

        # Remove markdown bold/italic (**text**, *text*)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)

        # Replace multiple newlines with space
        text = re.sub(r"\n+", " ", text)

        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def initialize_with_documents(self):
        """
        Initialize conversation by providing available documents.

        Adds document list to conversation history so agent knows what documents
        are available for search. Tool definitions are provided separately via
        the API 'tools' parameter and cached for efficiency.

        Note: Tool list is NOT included here to avoid duplication - Claude receives
        tool definitions directly via API parameter with prompt caching.
        """
        if self._initialized_with_documents:
            return  # Already initialized

        try:
            # Get document list tool
            doc_list_tool = self.registry.get_tool("get_document_list")
            if not doc_list_tool:
                logger.warning("get_document_list tool not available for initialization")
                return

            # Execute document list tool
            doc_result = doc_list_tool.execute()

            if not doc_result.success or not doc_result.data:
                logger.warning("Failed to get document list for initialization")
                return

            documents = doc_result.data.get("documents", [])
            count = doc_result.data.get("count", 0)

            if count == 0:
                logger.info("No documents available for initialization")
                return

            # Build document list message (ONLY documents, NOT tools)
            # Tools are provided via API 'tools' parameter and cached separately
            doc_list_text = f"Available documents in the system ({count}):\n\n"
            for doc in documents:
                doc_id = doc.get("id", "Unknown")
                summary = doc.get("summary", "No summary")

                # Clean summary text
                summary = self._clean_summary_text(summary)

                # Truncate to first sentence or 150 chars
                if len(summary) > 150:
                    # Try to cut at sentence boundary
                    sentence_end = summary.find(". ", 0, 150)
                    if sentence_end > 50:  # Found reasonable sentence
                        summary = summary[: sentence_end + 1]
                    else:
                        summary = summary[:150] + "..."

                doc_list_text += f"- {doc_id}: {summary}\n"

            # Add footer explaining initialization
            init_message = doc_list_text
            init_message += "\n(These are the documents available in the system. Use your tools to search and analyze them.)"

            # Add as first message in conversation history (Anthropic format: content as list)
            self.conversation_history.append({
                "role": "user",
                "content": [{"type": "text", "text": init_message}]
            })

            # Add simple acknowledgment from assistant (Anthropic format: content as list)
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I understand. I have access to these documents and will use the appropriate tools to search and analyze them.",
                        }
                    ],
                }
            )

            self._initialized_with_documents = True
            logger.info(f"Initialized conversation with {count} documents")

        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Document initialization failed - file access issue: {e}")
            # Expected - vector store may be empty or permissions issue
        except KeyError as e:
            logger.error(f"Invalid metadata structure during initialization: {e}")
            # This indicates a data corruption issue - log as error
        except Exception as e:
            logger.error(
                f"Unexpected error initializing documents: {sanitize_error(e)}", exc_info=True
            )
            # Log as error for unexpected issues, but don't crash - continue without initialization

    def reset_conversation(self):
        """Reset conversation history and reinitialize with documents."""
        self.conversation_history = []
        self.tool_call_history = []
        self._initialized_with_documents = False
        logger.info("Conversation reset")

        # Reinitialize with documents and tools list
        self.initialize_with_documents()
        logger.info("Agent reinitialized with documents after reset")

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "message_count": len(self.conversation_history),
            "tool_calls": len(self.tool_call_history),
            "tools_used": list(set(t["tool_name"] for t in self.tool_call_history)),
        }

    def get_latest_rag_confidence(self) -> Optional[Dict[str, Any]]:
        """
        Get RAG confidence from the most recent tool call (if available).

        Returns:
            Dict with confidence data, or None if no confidence available
        """
        # Search backwards through tool call history for most recent confidence
        for tool_call in reversed(self.tool_call_history):
            if "rag_confidence" in tool_call:
                return tool_call["rag_confidence"]
        return None

    def _prune_tool_results(self, keep_last_n: int = 3):
        """
        Remove tool calls, intermediate messages, and results from old messages.
        Keep only user questions and final assistant answers.

        This prevents quadratic cost growth by removing the bulk of tokens
        (tool results and intermediate reasoning) from older messages.

        Strategy:
        - Keep last N messages intact (with all tool calls/results)
        - For older messages: Remove entire tool exchanges (assistant tool_use + user tool_result)
        - Keep only substantial assistant responses (final answers with text > 50 chars)
        - Preserve all user question text

        Args:
            keep_last_n: Number of recent messages to keep with full tool context
        """
        if len(self.conversation_history) <= keep_last_n:
            return  # Nothing to prune

        # Calculate cutoff index (messages before this are pruned)
        cutoff_idx = len(self.conversation_history) - keep_last_n
        tokens_saved_estimate = 0
        removed_tool_use_ids = set()

        # Pass 1: Identify and prune assistant messages with tools
        # Remove tool_use blocks and track IDs for corresponding tool_result removal
        for i in range(cutoff_idx):
            msg = self.conversation_history[i]

            # Skip non-list content (system notices)
            if not isinstance(msg.get("content"), list):
                continue

            # Prune assistant messages
            if msg["role"] == "assistant":
                text_blocks = []
                tool_blocks = []
                total_text_length = 0

                for block in msg["content"]:
                    if block.get("type") == "text":
                        text_blocks.append(block)
                        total_text_length += len(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_blocks.append(block)
                        # Track removed tool_use ID
                        removed_tool_use_ids.add(block.get("id"))

                if tool_blocks:
                    # Estimate tokens saved
                    tokens_saved_estimate += len(tool_blocks) * 50

                    # If this is an intermediate message (minimal text), remove text too
                    if total_text_length < 50:  # Threshold for "substantial" text
                        # Replace with minimal placeholder
                        msg["content"] = [
                            {"type": "text", "text": "[Intermediate reasoning removed]"}
                        ]
                        tokens_saved_estimate += total_text_length // 4  # Estimate tokens
                    else:
                        # Keep substantial text, remove only tool_use blocks
                        if not text_blocks:
                            text_blocks = [{"type": "text", "text": "[Tool execution removed]"}]
                        msg["content"] = text_blocks

        # Pass 2: Remove corresponding tool_result blocks from ALL user messages
        # (must scan entire history because tool_result may be after cutoff)
        for i in range(len(self.conversation_history)):
            msg = self.conversation_history[i]

            # Skip non-list content
            if not isinstance(msg.get("content"), list):
                continue

            # Prune user messages: remove tool_result blocks
            if msg["role"] == "user":
                text_blocks = []
                tool_result_blocks = []

                for block in msg["content"]:
                    if block.get("type") == "text":
                        text_blocks.append(block)
                    elif block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id")
                        if tool_use_id in removed_tool_use_ids:
                            # This tool_result corresponds to removed tool_use
                            tool_result_blocks.append(block)
                        else:
                            # Keep tool_result (corresponding tool_use still exists)
                            text_blocks.append(block)

                if tool_result_blocks:
                    # Estimate tokens saved (rough: 500 tokens per tool_result)
                    tokens_saved_estimate += len(tool_result_blocks) * 500
                    # If no text blocks remain after removing tool_results, add placeholder
                    if not text_blocks:
                        text_blocks = [{"type": "text", "text": "[Tool results removed]"}]
                    msg["content"] = text_blocks

        # Clear the removed_tool_use_ids set to prevent memory leak
        removed_tool_use_ids.clear()

        if tokens_saved_estimate > 0:
            logger.info(
                f"Tool pruning: Removed tool exchanges and intermediate messages from {cutoff_idx} messages "
                f"(~{tokens_saved_estimate:,} tokens saved, keeping last {keep_last_n} messages intact)"
            )

    def _trim_history(self):
        """
        Trim conversation history to prevent unbounded memory growth.

        IMPORTANT IMPLICATIONS:
        - User IS notified when history is trimmed (transparency)
        - Claude loses access to earlier conversation context
        - May break multi-turn reasoning across >15 message exchanges
        - Tool results in trimmed messages are lost permanently
        - Trimming happens BEFORE Claude API call

        NOTES:
        - Each "message" may contain multiple tool results
        - Actual context size varies significantly per message
        - Keeps the last MAX_HISTORY_MESSAGES messages
        - Adds notification message to conversation explaining trimming
        """
        if len(self.conversation_history) > MAX_HISTORY_MESSAGES:
            old_len = len(self.conversation_history)
            messages_removed = old_len - MAX_HISTORY_MESSAGES
            self.conversation_history = self.conversation_history[-MAX_HISTORY_MESSAGES:]

            logger.warning(
                f"Conversation history trimmed: {old_len} → {MAX_HISTORY_MESSAGES} messages "
                f"({messages_removed} messages removed)"
            )

            # CRITICAL: Notify user about trimming
            # Add a system message explaining what happened
            system_notice = {
                "role": "assistant",
                "content": f"[Note: Conversation history was trimmed. I can only access the last {MAX_HISTORY_MESSAGES} messages. Earlier context ({messages_removed} messages) is no longer available.]",
            }
            # Insert notice at beginning of trimmed history so Claude sees it
            self.conversation_history.insert(0, system_notice)

    def _prepare_system_prompt_with_cache(self) -> List[Dict[str, Any]] | str:
        """
        Prepare system prompt with cache control for Anthropic prompt caching.

        For Anthropic: Returns structured format with cache_control
        For OpenAI: Returns simple string

        Returns:
            List of system prompt blocks (Anthropic) or string (OpenAI)
        """
        # Check if provider supports structured system prompts
        if self.provider.supports_feature("structured_system"):
            # Anthropic format
            system_block = {"type": "text", "text": self.config.system_prompt}

            # Add cache control if enabled and supported
            if self.config.enable_prompt_caching and self.provider.supports_feature(
                "prompt_caching"
            ):
                system_block["cache_control"] = {"type": "ephemeral"}

            return [system_block]
        else:
            # OpenAI format (simple string)
            return self.config.system_prompt

    def _prepare_tools_with_cache(self, tools: List[Dict]) -> List[Dict]:
        """
        Add cache control to tools array (on last tool) for Anthropic prompt caching.

        Caching strategy:
        - Cache control on last tool in array
        - Reduces API costs by 90% for tool definitions
        - TTL: 5 minutes (Anthropic default)
        - Only applied if provider supports prompt_caching

        Args:
            tools: List of tool definitions

        Returns:
            Tools with cache control added (if enabled and supported)
        """
        if not self.config.enable_prompt_caching or not tools:
            return tools

        # Only add cache control if provider supports it
        if not self.provider.supports_feature("prompt_caching"):
            return tools

        # Deep copy to avoid modifying original
        import copy

        cached_tools = copy.deepcopy(tools)

        # Add cache control to last tool
        cached_tools[-1]["cache_control"] = {"type": "ephemeral"}

        return cached_tools

    def _add_cache_control_to_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add cache control to specific messages for Anthropic prompt caching.

        Caching strategy:
        - Cache document initialization message (first 2 messages)
        - Allows cached context across conversations

        Args:
            messages: Conversation history

        Returns:
            Messages with cache control added (if enabled)
        """
        if not self.config.enable_prompt_caching or len(messages) < 2:
            return messages

        # Deep copy to avoid modifying original
        import copy

        cached_messages = copy.deepcopy(messages)

        # Add cache control to assistant's acknowledgment (2nd message after init)
        # This caches the document list and tool list initialization
        if len(cached_messages) >= 2 and cached_messages[1]["role"] == "assistant":
            # Convert content to structured format if it's a string
            if isinstance(cached_messages[1]["content"], str):
                cached_messages[1]["content"] = [
                    {
                        "type": "text",
                        "text": cached_messages[1]["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(cached_messages[1]["content"], list):
                # If already structured, add cache_control to last block
                if cached_messages[1]["content"]:
                    if isinstance(cached_messages[1]["content"][-1], dict):
                        cached_messages[1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        return cached_messages

    def process_message(
        self, user_message: str, stream: bool = None
    ) -> Generator[str, None, None] | str:
        """
        Process user message with full agent loop.

        Flow:
        1. Validate query length
        2. Add user message to history
        3. Trim history if needed
        4. Call Claude with tools
        5. Execute tools if requested
        6. Get final answer
        7. Stream or return response

        Args:
            user_message: User's question/request
            stream: Enable streaming (default from config)

        Returns:
            Generator of text chunks (if streaming) or full text

        Raises:
            ValueError: If query is too long
        """
        # Validate query length (prevent DoS via huge queries)
        if len(user_message) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long ({len(user_message)} chars). "
                f"Maximum length is {MAX_QUERY_LENGTH} characters."
            )

        # Add user message to history (Anthropic format: content as list)
        self.conversation_history.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        })

        # Prune tool results from old messages (keep only Q&A text)
        self._prune_tool_results(keep_last_n=self.config.context_management_keep)

        # Trim history to prevent unbounded growth
        self._trim_history()

        # Determine streaming
        if stream is None:
            stream = self.config.cli_config.enable_streaming

        # Get Claude SDK tools
        tools = self.registry.get_claude_sdk_tools()

        logger.info(
            f"Processing message (streaming={stream}, tools={len(tools)}): "
            f"{user_message[:100]}..."
        )

        if stream:
            return self._process_streaming(tools)
        else:
            return self._process_non_streaming(tools)

    def _process_streaming(self, tools: List[Dict]) -> Generator[str, None, None]:
        """
        Process message with streaming.

        Yields text chunks as they arrive from Claude.
        Handles API errors gracefully to avoid incomplete responses.
        """
        try:
            # Import anthropic for error handling
            import anthropic

            # Import openai for error handling
            import openai

            # Prepare cached system prompt and tools (static)
            system_prompt = self._prepare_system_prompt_with_cache()
            cached_tools = self._prepare_tools_with_cache(tools)

            while True:
                # Update cached messages with latest conversation history (includes tool results)
                cached_messages = self._add_cache_control_to_messages(self.conversation_history)

                # Use provider abstraction for streaming (with fallback for OpenAI verification issues)
                try:
                    stream = self.provider.stream_message(
                        messages=cached_messages,
                        tools=cached_tools,
                        system=system_prompt,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                except openai.BadRequestError as e:
                    # Check if this is the organization verification error for streaming
                    # Use error code if available, otherwise fall back to message checking
                    error_message = str(e)
                    error_code = getattr(e, "code", None)

                    # Only fallback for known streaming verification errors
                    is_verification_error = error_code == "organization_not_verified" or (
                        "organization" in error_message.lower()
                        and "verified" in error_message.lower()
                        and "stream" in error_message.lower()
                    )

                    if is_verification_error:
                        logger.warning(
                            f"OpenAI organization not verified for streaming. Falling back to non-streaming mode. "
                            f"Error: {error_message[:200]}"
                        )
                        yield "[⚠️  Your OpenAI organization needs verification to use streaming. Using non-streaming mode.]\n\n"

                        # Fallback to non-streaming for this request
                        result = self._process_non_streaming(tools)
                        yield result
                        return
                    else:
                        # Log the actual error before re-raising
                        logger.error(
                            f"OpenAI API error during streaming initialization: {error_message[:200]}",
                            exc_info=True,
                        )
                        # Re-raise if it's a different error
                        raise

                # Handle provider-specific streaming (PRAGMATIC: type-check instead of abstraction)
                if isinstance(self.provider, AnthropicProvider):
                    # Anthropic streaming (native format)
                    with stream as anthropic_stream:
                        # Collect assistant message
                        assistant_message = {"role": "assistant", "content": []}
                        tool_uses = []

                        # Stream text and collect tool uses
                        for event in anthropic_stream:
                            if event.type == "content_block_start":
                                if event.content_block.type == "text":
                                    assistant_message["content"].append(
                                        {"type": "text", "text": ""}
                                    )
                                elif event.content_block.type == "tool_use":
                                    # Stream tool call notification immediately
                                    tool_name = event.content_block.name
                                    # Always send marker for web interface inline display
                                    yield f"\n{COLOR_BLUE}[Using {tool_name}...]{COLOR_RESET}\n"

                            elif event.type == "content_block_delta":
                                if event.delta.type == "text_delta":
                                    # Stream text to user
                                    yield event.delta.text

                                    # Add to message content
                                    for block in assistant_message["content"]:
                                        if block["type"] == "text":
                                            block["text"] += event.delta.text
                                            break

                            elif event.type == "content_block_stop":
                                if (
                                    hasattr(event, "content_block")
                                    and event.content_block.type == "tool_use"
                                ):
                                    tool_uses.append(event.content_block)
                                    assistant_message["content"].append(
                                        {
                                            "type": "tool_use",
                                            "id": event.content_block.id,
                                            "name": event.content_block.name,
                                            "input": event.content_block.input,
                                        }
                                    )

                        # Get final message
                        final_message = anthropic_stream.get_final_message()

                        # Track cost (including cache statistics if available)
                        cache_creation = getattr(
                            final_message.usage, "cache_creation_input_tokens", 0
                        )
                        cache_read = getattr(final_message.usage, "cache_read_input_tokens", 0)

                        self.tracker.track_llm(
                            provider="anthropic",
                            model=self.config.model,
                            input_tokens=final_message.usage.input_tokens,
                            output_tokens=final_message.usage.output_tokens,
                            operation="agent",
                            cache_creation_tokens=cache_creation,
                            cache_read_tokens=cache_read,
                        )

                        # Log cache hit info if debug mode
                        if self.config.debug_mode and (cache_creation > 0 or cache_read > 0):
                            logger.debug(
                                f"Cache usage: {cache_read} tokens read, {cache_creation} tokens created"
                            )

                        # Add assistant message to history
                        if assistant_message["content"]:
                            self.conversation_history.append(assistant_message)

                elif isinstance(self.provider, GeminiProvider):
                    # Gemini streaming (similar to OpenAI)
                    assistant_message = {"role": "assistant", "content": []}
                    tool_uses = []
                    full_text = ""
                    announced_tools = set()  # Track which tools we've announced

                    # Track usage
                    input_tokens = 0
                    output_tokens = 0
                    cache_read_tokens = 0

                    # Iterate Gemini stream
                    for chunk in stream:
                        # Gemini stream format: chunks with candidates
                        if not hasattr(chunk, "candidates") or not chunk.candidates:
                            logger.warning(f"Gemini chunk missing candidates: {chunk}")
                            # Check for safety blocks
                            if hasattr(chunk, "prompt_feedback") and chunk.prompt_feedback:
                                logger.error(
                                    f"Gemini content blocked: {chunk.prompt_feedback.block_reason}"
                                )
                                yield "\n\n⚠️ [Content blocked by Gemini safety filters]\n\n"
                            continue

                        candidate = chunk.candidates[0]
                        if not hasattr(candidate, "content") or not candidate.content.parts:
                            logger.warning(
                                f"Gemini candidate missing content: candidate={candidate}"
                            )
                            continue

                        for part in candidate.content.parts:
                            # Stream text content
                            if hasattr(part, "text") and part.text:
                                yield part.text
                                full_text += part.text

                            # Collect tool calls
                            elif hasattr(part, "function_call") and part.function_call:
                                # Validate function_call has required attributes
                                if (
                                    not hasattr(part.function_call, "name")
                                    or not part.function_call.name
                                ):
                                    logger.error(
                                        f"Gemini returned malformed function_call missing 'name': {part.function_call}"
                                    )
                                    yield "\n[⚠️  Warning: Gemini returned malformed tool call - skipping]\n"
                                    continue

                                # Stream tool call notification immediately (once per tool)
                                tool_name = part.function_call.name
                                if tool_name not in announced_tools:
                                    announced_tools.add(tool_name)
                                    # Always send marker for web interface inline display
                                    yield f"\n{COLOR_BLUE}[Using {tool_name}...]{COLOR_RESET}\n"

                                # Extract arguments with error handling (consistent with OpenAI path)
                                tool_input = {}
                                if (
                                    hasattr(part.function_call, "args")
                                    and part.function_call.args is not None
                                ):
                                    try:
                                        tool_input = dict(part.function_call.args)
                                    except (TypeError, ValueError) as e:
                                        logger.error(
                                            f"Failed to convert tool arguments for {part.function_call.name}: {e}. "
                                            f"Args type: {type(part.function_call.args)}"
                                        )
                                        tool_input = {}
                                        yield f"\n[⚠️  Warning: Malformed tool arguments from API - tool may fail]\n"

                                # Create dynamic ToolUse class instance (consistent with OpenAI path at line 841)
                                # Required for attribute access in tool execution loop (line 902: tool_use.name, tool_use.input)
                                # Using dict would cause AttributeError during tool execution
                                tool_uses.append(
                                    type(
                                        "ToolUse",
                                        (),
                                        {
                                            "id": f"toolu_{part.function_call.name}",
                                            "name": part.function_call.name,
                                            "input": tool_input,
                                        },
                                    )()
                                )

                        # Extract usage from chunk if available
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            input_tokens = chunk.usage_metadata.prompt_token_count or 0
                            output_tokens = chunk.usage_metadata.candidates_token_count or 0
                            if hasattr(chunk.usage_metadata, "cached_content_token_count"):
                                cache_read_tokens = (
                                    chunk.usage_metadata.cached_content_token_count or 0
                                )

                    # Build assistant message content
                    if full_text:
                        assistant_message["content"].append({"type": "text", "text": full_text})

                    # Add tool uses to message (with validation)
                    for tool_use in tool_uses:
                        # Validate tool_use has required attributes
                        if (
                            not hasattr(tool_use, "id")
                            or not hasattr(tool_use, "name")
                            or not hasattr(tool_use, "input")
                        ):
                            logger.error(
                                f"Malformed tool_use object - skipping. Attributes: {dir(tool_use)}"
                            )
                            continue

                        assistant_message["content"].append(
                            {
                                "type": "tool_use",
                                "id": tool_use.id,
                                "name": tool_use.name,
                                "input": tool_use.input,
                            }
                        )

                    # Track cost (with cache support)
                    self.tracker.track_llm(
                        provider="google",
                        model=self.config.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        operation="agent",
                        cache_creation_tokens=0,  # Gemini doesn't report separately
                        cache_read_tokens=cache_read_tokens,
                    )

                    # Add assistant message to history
                    if assistant_message["content"]:
                        self.conversation_history.append(assistant_message)

                    # Set final_message for compatibility
                    class FinalMessage:
                        def __init__(self):
                            self.stop_reason = "tool_calls" if tool_uses else "stop"

                    final_message = FinalMessage()

                elif isinstance(self.provider, OpenAIProvider):
                    # OpenAI streaming (different format)
                    assistant_message = {"role": "assistant", "content": []}
                    tool_uses = []
                    full_text = ""
                    tool_calls_buffer = {}
                    announced_tools = set()  # Track which tools we've announced

                    # Track usage
                    input_tokens = 0
                    output_tokens = 0

                    for chunk in stream:
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta

                        # Stream text content
                        if delta.content:
                            yield delta.content
                            full_text += delta.content

                        # Collect tool calls
                        if delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                idx = tool_call.index
                                if idx not in tool_calls_buffer:
                                    tool_calls_buffer[idx] = {
                                        "id": tool_call.id or "",
                                        "name": "",
                                        "arguments": "",
                                    }

                                if tool_call.function:
                                    if tool_call.function.name:
                                        tool_name = tool_call.function.name
                                        tool_calls_buffer[idx]["name"] = tool_name

                                        # Stream tool call notification immediately (once per tool)
                                        if tool_name and tool_name not in announced_tools:
                                            announced_tools.add(tool_name)
                                            # Always send marker for web interface inline display
                                            yield f"\n{COLOR_BLUE}[Using {tool_name}...]{COLOR_RESET}\n"

                                    if tool_call.function.arguments:
                                        tool_calls_buffer[idx][
                                            "arguments"
                                        ] += tool_call.function.arguments

                    # Build assistant message content
                    if full_text:
                        assistant_message["content"].append({"type": "text", "text": full_text})

                    # Convert tool calls to Anthropic format
                    for tool_call_data in tool_calls_buffer.values():
                        if tool_call_data["name"]:
                            import json

                            # Parse tool arguments with error handling
                            try:
                                parsed_input = json.loads(tool_call_data["arguments"])
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Failed to parse tool arguments for {tool_call_data['name']}: {e}. "
                                    f"Raw arguments: {tool_call_data['arguments'][:200]}"
                                )
                                # Use empty dict as fallback (tool will fail gracefully)
                                parsed_input = {}
                                # Notify user
                                yield f"\n[⚠️  Warning: Malformed tool call from API - tool may fail]\n"

                            tool_uses.append(
                                type(
                                    "ToolUse",
                                    (),
                                    {
                                        "id": tool_call_data["id"],
                                        "name": tool_call_data["name"],
                                        "input": parsed_input,
                                    },
                                )()
                            )
                            assistant_message["content"].append(
                                {
                                    "type": "tool_use",
                                    "id": tool_call_data["id"],
                                    "name": tool_call_data["name"],
                                    "input": parsed_input,
                                }
                            )

                    # Estimate tokens (OpenAI doesn't provide usage in streaming)
                    # Rough estimate: 4 chars per token
                    input_tokens = sum(len(str(m)) for m in cached_messages) // 4
                    output_tokens = (len(full_text) + sum(len(str(t)) for t in tool_uses)) // 4

                    # Track cost
                    self.tracker.track_llm(
                        provider="openai",
                        model=self.config.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        operation="agent",
                        cache_creation_tokens=0,  # OpenAI doesn't support caching
                        cache_read_tokens=0,
                    )

                    # Add assistant message to history
                    if assistant_message["content"]:
                        self.conversation_history.append(assistant_message)

                    # Set final_message for compatibility
                    class FinalMessage:
                        def __init__(self):
                            self.stop_reason = "tool_calls" if tool_uses else "stop"

                    final_message = FinalMessage()

                # Check stop reason (outside of with block but inside while)
                stop_reason = final_message.stop_reason

                # Normalize stop reasons (OpenAI uses different names)
                if stop_reason in ["end_turn", "stop", "length", "max_tokens"]:
                    # Done - no tool calls
                    break

                elif stop_reason in ["tool_use", "tool_calls"]:
                    # Execute tools
                    yield "\n\n"  # Newline before tool execution

                    tool_results = []
                    for tool_use in tool_uses:
                        tool_name = tool_use.name
                        tool_input = tool_use.input

                        # Execute tool (notification already streamed at content_block_start)
                        logger.info(f"Executing tool: {tool_name} with input {tool_input}")

                        result = self.registry.execute_tool(tool_name, **tool_input)

                        # Estimate cost of tool result (tokens added to context)
                        # Tool results are sent back to Claude in the next API call as input tokens
                        estimated_cost = 0.0
                        if result.estimated_tokens > 0:
                            # Get input cost per 1M tokens for current model
                            try:
                                from ..cost_tracker import PRICING

                                model_pricing = PRICING.get("anthropic", {}).get(self.config.model)
                                if model_pricing:
                                    input_price_per_1m = model_pricing.get("input", 0.0)
                                    estimated_cost = (
                                        result.estimated_tokens / 1_000_000
                                    ) * input_price_per_1m
                            except (KeyError, ValueError, ImportError) as e:
                                logger.error(
                                    f"Failed to calculate tool result cost for {tool_name}: {e}"
                                )

                        # Log tool execution with cost estimate
                        logger.info(
                            f"Tool '{tool_name}' result: ~{result.estimated_tokens} tokens, "
                            f"~${estimated_cost:.6f} cost estimate"
                        )

                        # Display RAG confidence if available (for search tool)
                        if result.metadata and "rag_confidence" in result.metadata:
                            confidence = result.metadata.get("rag_confidence", {})

                            # Validate confidence dict structure
                            if not isinstance(confidence, dict):
                                logger.error(
                                    f"Invalid confidence data type: {type(confidence)}. "
                                    f"Expected dict from RAGConfidenceScore.to_dict()."
                                )
                                confidence = {}

                            # Extract with defaults
                            conf_score = confidence.get("overall_confidence", None)
                            conf_interp = confidence.get("interpretation", "Unknown")
                            should_review = confidence.get("should_flag_for_review", False)

                            # Validate confidence score is numeric and finite
                            if conf_score is None or (
                                isinstance(conf_score, float)
                                and (np.isnan(conf_score) or np.isinf(conf_score))
                            ):
                                logger.error(
                                    f"Invalid confidence score: {conf_score}. Using 0.0 (unknown confidence)."
                                )
                                conf_score = 0.0

                            # Validate interpretation is string
                            if not isinstance(conf_interp, str):
                                logger.warning(
                                    f"Invalid interpretation type: {type(conf_interp)}. Using 'Unknown'."
                                )
                                conf_interp = "Unknown"

                            # Show confidence in blue (tool notification color)
                            # Always send for web interface display
                            emoji = "⚠️" if should_review else "✓"
                            try:
                                yield f"{COLOR_BLUE}[{emoji} RAG Confidence: {conf_interp} ({conf_score:.2f})]{COLOR_RESET}\n"
                            except (ValueError, TypeError) as e:
                                logger.error(f"Failed to format confidence display: {e}")
                                yield f"{COLOR_BLUE}[⚠️ RAG Confidence: {conf_interp} (unavailable)]{COLOR_RESET}\n"

                        # Check for tool failure and alert user
                        if not result.success:
                            logger.error(
                                f"Tool '{tool_name}' failed: {result.error}",
                                extra={"tool_input": tool_input, "metadata": result.metadata},
                            )
                            # Alert user in streaming mode (always send for web interface)
                            yield f"{COLOR_BLUE}[⚠️  Tool '{tool_name}' failed: {result.error}]{COLOR_RESET}\n"

                        # Track in history (including RAG confidence if available)
                        tool_call_record = {
                            "tool_name": tool_name,
                            "input": tool_input,
                            "success": result.success,
                            "execution_time_ms": result.execution_time_ms,
                            "estimated_tokens": result.estimated_tokens,
                            "estimated_cost": estimated_cost,
                        }

                        # Add RAG confidence if available
                        if result.metadata and "rag_confidence" in result.metadata:
                            tool_call_record["rag_confidence"] = result.metadata["rag_confidence"]

                        self.tool_call_history.append(tool_call_record)

                        # Format tool result for Claude
                        tool_result_content = self._format_tool_result(result)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": tool_result_content,
                            }
                        )

                    # Add tool results to conversation
                    self.conversation_history.append({"role": "user", "content": tool_results})

                    # Continue loop to get final answer
                    yield "\n"  # Newline after tool execution

                else:
                    logger.warning(f"Unexpected stop reason: {final_message.stop_reason}")
                    break

        except anthropic.APITimeoutError as e:
            logger.error(f"API timeout during streaming: {e}")
            yield "\n\n[⚠️  API timeout - response incomplete. Please try again.]\n"
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit hit: {e}")
            yield "\n\n[⚠️  Rate limit exceeded - please wait a moment and try again.]\n"
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {sanitize_error(e)}")
            yield f"\n\n[❌ API Error: {sanitize_error(e)}]\n"
        except Exception as e:
            logger.error(f"Streaming failed: {sanitize_error(e)}", exc_info=True)
            yield f"\n\n[❌ Unexpected error: {type(e).__name__}: {sanitize_error(e)}]\n"

    def _process_non_streaming(self, tools: List[Dict]) -> str:
        """
        Process message without streaming (synchronous).

        Returns complete response after tool execution.
        """
        full_response_text = ""

        # Prepare cached system prompt and tools (static)
        system_prompt = self._prepare_system_prompt_with_cache()
        cached_tools = self._prepare_tools_with_cache(tools)

        try:
            while True:
                # Update cached messages with latest conversation history (includes tool results)
                cached_messages = self._add_cache_control_to_messages(self.conversation_history)

                # Use provider abstraction
                response = self.provider.create_message(
                    messages=cached_messages,
                    tools=cached_tools,
                    system=system_prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

                # Track cost (ProviderResponse has usage dict)
                self.tracker.track_llm(
                    provider=self.provider.get_provider_name(),
                    model=self.config.model,
                    input_tokens=response.usage["input_tokens"],
                    output_tokens=response.usage["output_tokens"],
                    operation="agent",
                    cache_creation_tokens=response.usage.get("cache_creation_tokens", 0),
                    cache_read_tokens=response.usage.get("cache_read_tokens", 0),
                )

                # Log cache hit info if debug mode
                cache_read = response.usage.get("cache_read_tokens", 0)
                cache_creation = response.usage.get("cache_creation_tokens", 0)
                if self.config.debug_mode and (cache_creation > 0 or cache_read > 0):
                    logger.debug(
                        f"Cache usage: {cache_read} tokens read, {cache_creation} tokens created"
                    )

                # Add assistant message to history
                self.conversation_history.append({"role": "assistant", "content": response.content})

                # Extract text
                for block in response.content:
                    if block.get("type") == "text":
                        full_response_text += block["text"]

                # Check stop reason (normalize across providers)
                stop_reason = response.stop_reason
                if stop_reason in ["end_turn", "stop", "length", "max_tokens"]:
                    break

                elif stop_reason in ["tool_use", "tool_calls"]:
                    # Execute tools
                    tool_results = []

                    for block in response.content:
                        if block.get("type") == "tool_use":
                            tool_name = block["name"]
                            tool_input = block["input"]

                            logger.info(f"Executing tool: {tool_name}")

                            result = self.registry.execute_tool(tool_name, **tool_input)

                            # Estimate cost of tool result (tokens added to context)
                            # Tool results are sent back to Claude in the next API call as input tokens
                            estimated_cost = 0.0
                            if result.estimated_tokens > 0:
                                # Get input cost per 1M tokens for current model
                                try:
                                    from ..cost_tracker import PRICING

                                    model_pricing = PRICING.get("anthropic", {}).get(
                                        self.config.model
                                    )
                                    if model_pricing:
                                        input_price_per_1m = model_pricing.get("input", 0.0)
                                        estimated_cost = (
                                            result.estimated_tokens / 1_000_000
                                        ) * input_price_per_1m
                                except (KeyError, ValueError, ImportError) as e:
                                    logger.error(
                                        f"Failed to calculate tool result cost for {tool_name}: {e}"
                                    )

                            # Log tool execution with cost estimate
                            logger.info(
                                f"Tool '{tool_name}' result: ~{result.estimated_tokens} tokens, "
                                f"~${estimated_cost:.6f} cost estimate"
                            )

                            # Log RAG confidence if available (for search tool)
                            if result.metadata and "rag_confidence" in result.metadata:
                                confidence = result.metadata["rag_confidence"]
                                conf_score = confidence.get("overall_confidence", 0.0)
                                conf_interp = confidence.get("interpretation", "Unknown")
                                logger.info(f"RAG Confidence: {conf_interp} ({conf_score:.3f})")

                            # Check for tool failure and log error
                            if not result.success:
                                error_msg = f"⚠️  Tool '{tool_name}' failed: {result.error}"
                                logger.error(
                                    f"Tool '{tool_name}' failed: {result.error}",
                                    extra={"tool_input": tool_input, "metadata": result.metadata},
                                )
                                # Show error to user in non-streaming mode
                                full_response_text += f"\n[{error_msg}]\n"

                            # Track in history (including RAG confidence if available)
                            tool_call_record = {
                                "tool_name": tool_name,
                                "input": tool_input,
                                "success": result.success,
                                "execution_time_ms": result.execution_time_ms,
                                "estimated_tokens": result.estimated_tokens,
                                "estimated_cost": estimated_cost,
                            }

                            # Add RAG confidence if available
                            if result.metadata and "rag_confidence" in result.metadata:
                                tool_call_record["rag_confidence"] = result.metadata[
                                    "rag_confidence"
                                ]

                            self.tool_call_history.append(tool_call_record)

                            # Format tool result
                            tool_result_content = self._format_tool_result(result)

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block["id"],
                                    "content": tool_result_content,
                                }
                            )

                    # Add tool results to conversation
                    self.conversation_history.append({"role": "user", "content": tool_results})

                else:
                    logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                    break

            return full_response_text

        except anthropic.APITimeoutError as e:
            logger.error(f"API timeout in non-streaming mode: {e}")
            return "[⚠️  API timeout - response incomplete. Please try again.]"
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit hit: {e}")
            return "[⚠️  Rate limit exceeded - please wait a moment and try again.]"
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {sanitize_error(e)}")
            return f"[❌ API Error: {sanitize_error(e)}]"
        except Exception as e:
            # Catch both Anthropic and OpenAI errors
            import openai

            if isinstance(e, (openai.APITimeoutError, openai.RateLimitError, openai.APIError)):
                logger.error(f"OpenAI API error: {sanitize_error(e)}")
                return f"[❌ API Error: {sanitize_error(e)}]"

            logger.error(f"Non-streaming processing failed: {sanitize_error(e)}", exc_info=True)
            return f"[❌ Unexpected error: {type(e).__name__}: {sanitize_error(e)}]"

    def _format_tool_result(self, result: ToolResult) -> str:
        """
        Format ToolResult for Claude SDK.

        Args:
            result: ToolResult from tool execution

        Returns:
            Formatted string for Claude
        """
        if not result.success:
            return json.dumps(
                {"error": result.error, "metadata": result.metadata}, indent=2, ensure_ascii=False
            )

        # Format successful result
        formatted = {"data": result.data, "metadata": result.metadata}

        if result.citations:
            formatted["citations"] = result.citations

        # Add RAG confidence summary if available (for search tool)
        if result.metadata and "rag_confidence" in result.metadata:
            confidence = result.metadata["rag_confidence"]
            formatted["rag_confidence_summary"] = {
                "confidence": confidence.get("overall_confidence", 0.0),
                "interpretation": confidence.get("interpretation", "Unknown"),
                "should_review": confidence.get("should_flag_for_review", False),
            }

        return json.dumps(formatted, indent=2, ensure_ascii=False)
