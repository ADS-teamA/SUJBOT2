"""
Agent Adapter - Wraps existing AgentCore without modifying src/.

This adapter:
1. Imports AgentCore from src/agent/agent_core.py
2. Handles SSE event formatting
3. Tracks cost per message
4. Provides clean interface for FastAPI
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

from src.agent.agent_core import AgentCore
from src.agent.config import AgentConfig
from src.agent.providers import create_provider
from src.agent.tools.registry import get_registry
from src.context_assembly import CitationFormat, ContextAssembler
from src.cost_tracker import get_global_tracker, reset_global_tracker
from src.embedding_generator import EmbeddingGenerator
from src.hybrid_search import HybridVectorStore
from src.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class AgentAdapter:
    """
    Adapter wrapping AgentCore for web frontend.

    Responsibilities:
    - Initialize AgentCore with config
    - Convert streaming events to SSE format
    - Track costs per request
    - Provide model switching capability
    """

    def __init__(self, vector_store_path: Optional[Path] = None, model: Optional[str] = None):
        """
        Initialize agent adapter.

        Args:
            vector_store_path: Path to vector store (default: ../vector_db from backend/)
            model: Model to use (default: from config)
        """
        # Load config from environment
        config_overrides = {}

        # Set default vector_store_path relative to project root (parent of backend/)
        if vector_store_path:
            config_overrides["vector_store_path"] = vector_store_path
        else:
            # Backend runs from backend/, so vector_db is in parent directory
            project_root = Path(__file__).parent.parent
            config_overrides["vector_store_path"] = project_root / "vector_db"

        if model:
            config_overrides["model"] = model

        self.config = AgentConfig.from_env(**config_overrides)

        # Validate config
        try:
            self.config.validate()
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise

        # Initialize pipeline components (same as CLI)
        logger.info("Loading vector store...")
        vector_store = HybridVectorStore.load(str(self.config.vector_store_path))

        logger.info("Initializing embedder...")
        embedder = EmbeddingGenerator()

        # Track degraded components (instance variable for health endpoint)
        self.degraded_components = []

        # Initialize reranker (optional, lazy load)
        reranker = None
        if self.config.tool_config.enable_reranking:
            if not self.config.tool_config.lazy_load_reranker:
                logger.info("Loading reranker...")
                try:
                    reranker = CrossEncoderReranker(
                        model_name=self.config.tool_config.reranker_model
                    )
                except (ImportError, ModuleNotFoundError) as e:
                    logger.error(
                        f"Reranker dependencies missing: {e}. "
                        f"Install with: uv pip install sentence-transformers"
                    )
                    self.config.tool_config.enable_reranking = False
                    self.degraded_components.append({
                        "component": "reranker",
                        "error": f"Missing dependencies: {e}",
                        "severity": "high",
                        "user_message": "Search quality reduced - reranker unavailable",
                        "action_required": "Install with: uv pip install sentence-transformers"
                    })
                except (FileNotFoundError, ValueError) as e:
                    logger.error(
                        f"Reranker configuration error: {e}. "
                        f"Check model name '{self.config.tool_config.reranker_model}' in config."
                    )
                    self.config.tool_config.enable_reranking = False
                    self.degraded_components.append({
                        "component": "reranker",
                        "error": f"Configuration error: {e}",
                        "severity": "high",
                        "user_message": "Search quality reduced - reranker misconfigured",
                        "action_required": f"Check reranker model name '{self.config.tool_config.reranker_model}' in config"
                    })
                except RuntimeError as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        logger.warning(
                            f"GPU unavailable for reranker: {e}. This is expected on CPU-only systems."
                        )
                        self.config.tool_config.enable_reranking = False
                        self.degraded_components.append({
                            "component": "reranker",
                            "error": "GPU unavailable (CPU-only mode)",
                            "severity": "medium",
                            "user_message": "Running on CPU only - reranker disabled for performance"
                        })
                    else:
                        logger.critical(f"Unexpected runtime error loading reranker: {e}", exc_info=True)
                        self.config.tool_config.enable_reranking = False
                        self.degraded_components.append({
                            "component": "reranker",
                            "error": f"Runtime error: {e}",
                            "severity": "critical",
                            "user_message": "Search quality reduced - reranker failed to initialize"
                        })
                except Exception as e:
                    # Catch-all for truly unexpected errors
                    logger.critical(
                        f"Unexpected error loading reranker: {type(e).__name__}: {e}",
                        exc_info=True
                    )
                    self.config.tool_config.enable_reranking = False
                    self.degraded_components.append({
                        "component": "reranker",
                        "error": f"Unexpected error: {type(e).__name__}",
                        "severity": "critical",
                        "user_message": "Search quality reduced - reranker unavailable"
                    })
            else:
                logger.info("Reranker set to lazy load")

        # Load knowledge graph (optional) - UNIFIED WITH CLI
        knowledge_graph = None
        graph_retriever = None
        if self.config.enable_knowledge_graph and self.config.knowledge_graph_path:
            logger.info("Loading knowledge graph...")
            try:
                from src.agent.graph_loader import load_knowledge_graph

                # Use shared loader (handles Neo4j/JSON with intelligent fallback)
                knowledge_graph, graph_retriever = load_knowledge_graph(
                    kg_path=self.config.knowledge_graph_path,
                    vector_store=vector_store,
                    user_print=None  # WebApp doesn't print to console
                )

            except RuntimeError as e:
                # Explicit Neo4j config errors (fail-fast when KG_BACKEND=neo4j)
                logger.error(f"Knowledge graph backend misconfigured: {e}")
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"Neo4j misconfiguration: {e}",
                    "severity": "critical",
                    "action_required": "Fix Neo4j config in .env or change KG_BACKEND=simple"
                })
            except (ImportError, ModuleNotFoundError) as e:
                logger.error(
                    f"Knowledge graph dependencies missing: {e}. "
                    f"Install with: uv pip install networkx neo4j"
                )
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"Missing dependencies: {e}",
                    "severity": "high"
                })
            except FileNotFoundError as e:
                logger.warning(
                    f"Knowledge graph file not found: {e}. "
                    f"Run indexing with ENABLE_KNOWLEDGE_GRAPH=true to generate it: "
                    f"uv run python run_pipeline.py data/your_docs/"
                )
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"File not found - run indexing to generate",
                    "severity": "medium"
                })
            except (ValueError, KeyError, TypeError) as e:
                logger.error(
                    f"Knowledge graph file corrupted or invalid format: {e}. "
                    f"Re-run indexing pipeline to regenerate."
                )
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"Invalid file format: {e}"
                })
            except Exception as e:
                # Catch-all for truly unexpected errors
                logger.critical(
                    f"Unexpected error loading knowledge graph: {type(e).__name__}: {e}",
                    exc_info=True
                )
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"Unexpected error: {type(e).__name__}"
                })

        # Warn if running in degraded mode with detailed component info
        if self.degraded_components:
            component_names = [d["component"] for d in self.degraded_components]

            # Log detailed degradation info
            logger.warning("=" * 60)
            logger.warning("⚠️  WARNING: RUNNING IN DEGRADED MODE")
            logger.warning("=" * 60)

            for component in self.degraded_components:
                severity = component.get("severity", "medium")
                symbol = "🔴" if severity == "critical" else "🟡" if severity == "high" else "🟢"

                logger.warning(f"{symbol} {component['component']}: {component['error']}")

                if "user_message" in component:
                    logger.warning(f"   → {component['user_message']}")

                if "action_required" in component:
                    logger.warning(f"   → Action: {component['action_required']}")

            logger.warning("=" * 60)
            logger.warning(f"Degraded components: {', '.join(component_names)}")
            logger.warning("Some agent tools may be unavailable or produce lower-quality results.")

        # Initialize context assembler
        logger.info("Initializing context assembler...")
        citation_format_map = {
            "inline": CitationFormat.INLINE,
            "detailed": CitationFormat.DETAILED,
            "footnote": CitationFormat.FOOTNOTE,
        }
        context_assembler = ContextAssembler(
            citation_format=citation_format_map.get(
                self.config.cli_config.citation_format, CitationFormat.INLINE
            )
        )

        # Initialize tools (CRITICAL - must happen before AgentCore)
        logger.info("Initializing tools...")
        registry = get_registry()
        registry.initialize_tools(
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            graph_retriever=graph_retriever,
            knowledge_graph=knowledge_graph,
            context_assembler=context_assembler,
            config=self.config.tool_config,
        )
        logger.info(f"✅ {len(registry)} tools initialized")

        # Initialize AgentCore (reuses existing implementation)
        self.agent = AgentCore(self.config)

        # Initialize with document list
        self.agent.initialize_with_documents()

        logger.info(
            f"AgentAdapter initialized: model={self.config.model}, "
            f"vector_store={self.config.vector_store_path}, tools={len(registry)}"
        )

    async def stream_response(
        self,
        query: str,
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent response as SSE-compatible events.

        Yields SSE events in format:
        {
            "event": "text_delta" | "tool_call" | "tool_calls_summary" | "cost_update" | "done" | "error",
            "data": {...}
        }

        Event types:
        - text_delta: Streaming text chunks from agent response
        - tool_call: Tool invocation detected (streamed immediately when Claude decides to use tool)
        - tool_calls_summary: Summary of all tool calls with results (sent after response completes)
        - cost_update: Token usage and cost information
        - done: Stream completed successfully
        - error: Error occurred during streaming

        Args:
            query: User query
            conversation_id: Optional conversation ID for context

        Yields:
            Dict containing event type and data
        """
        # Reset cost tracker for this request
        reset_global_tracker()
        tracker = get_global_tracker()

        try:
            # Stream from AgentCore using process_message
            # Note: process_message is synchronous generator, but we can iterate
            # in async context with periodic yields to event loop

            # Get streaming generator from AgentCore
            text_stream = self.agent.process_message(query, stream=True)

            # Stream text chunks (including markers for inline tool display)
            # DO NOT send separate tool_call events during streaming
            # Reasons:
            # 1. Tool IDs are not known yet (Anthropic assigns them after execution)
            # 2. Frontend expects markers in text for inline rendering
            # 3. Sending both causes duplicate display (nahoře + inline)

            for chunk in text_stream:
                # Strip ANSI color codes only
                clean_chunk = re.sub(r'\033\[[0-9;]+m', '', chunk)

                if clean_chunk:  # Only send non-empty chunks
                    # Send ALL text including [Using tool...] markers
                    # Frontend will parse markers and render tools inline
                    yield {
                        "event": "text_delta",
                        "data": {
                            "content": clean_chunk
                        }
                    }
                    # Yield control to event loop (allows concurrent requests)
                    await asyncio.sleep(0)

            # Extract tool calls from conversation history
            # Tool calls are stored in assistant message content blocks (Anthropic format)
            # Tool results are in subsequent user messages with metadata
            #
            # Three-pass extraction is required because:
            # 1. Tool calls (tool_use) are in assistant messages with role="assistant"
            # 2. Tool results (tool_result) are in user messages with role="user"
            # 3. They must be joined by tool_use_id to create complete tool call objects
            #
            # Performance optimization: Scan last 10 messages only (not entire history)
            # Rationale: In tool-heavy conversations, each turn can generate 2-5 messages:
            #   - 1 assistant message (with tool_use blocks)
            #   - 1-4 user messages (one tool_result per tool called)
            # Therefore, 10 messages covers the most recent 2-4 turns, which is sufficient
            # because tool_use/tool_result pairs are always in adjacent messages.
            # For no-tool conversations: 10 messages = 5 complete turns.
            tool_calls_info = []
            tool_results_map = {}  # tool_use_id -> result metadata

            if self.agent.conversation_history:
                # Pass 1: Collect tool_use blocks (from assistant messages)
                for message in self.agent.conversation_history[-10:]:
                    if message.get("role") == "assistant" and "content" in message:
                        content = message["content"]

                        # Validate content is a list (defensive programming)
                        if not isinstance(content, list):
                            logger.error(
                                f"Invalid message content format: expected list, got {type(content).__name__}. "
                                f"Message role={message.get('role')}, content preview={str(content)[:100]}"
                            )
                            continue

                        for content_block in content:
                            # Validate content_block is a dict
                            if not isinstance(content_block, dict):
                                logger.warning(f"Skipping non-dict content block: {type(content_block).__name__}")
                                continue

                            if content_block.get("type") == "tool_use":
                                # Validate required fields exist
                                if "id" not in content_block or "name" not in content_block:
                                    logger.error(
                                        f"tool_use block missing required fields. "
                                        f"Has id={('id' in content_block)}, name={('name' in content_block)}"
                                    )
                                    continue

                                tool_calls_info.append({
                                    "id": content_block.get("id", ""),
                                    "name": content_block.get("name", ""),
                                    "input": content_block.get("input", {}),
                                })

                # Pass 2: Collect tool_result blocks with metadata (from user messages)
                for message in self.agent.conversation_history[-10:]:
                    if message.get("role") == "user" and "content" in message:
                        content = message["content"]

                        # Validate content is a list
                        if not isinstance(content, list):
                            logger.error(
                                f"Invalid message content format: expected list, got {type(content).__name__}. "
                                f"Message role={message.get('role')}"
                            )
                            continue

                        for content_block in content:
                            # Validate content_block is a dict
                            if not isinstance(content_block, dict):
                                logger.warning(f"Skipping non-dict content block: {type(content_block).__name__}")
                                continue

                            if content_block.get("type") == "tool_result":
                                tool_use_id = content_block.get("tool_use_id")
                                if tool_use_id:
                                    # Note: _metadata was removed from agent_core.py (API compliance fix)
                                    # We can no longer access execution_time_ms, success, error from here
                                    # This will be handled differently in future versions
                                    tool_results_map[tool_use_id] = {
                                        "result": content_block.get("content"),
                                        "metadata": {},  # Empty - no longer available from API responses
                                    }

                # Pass 3: Merge tool_use and tool_result data by tool_use_id
                for tool_call in tool_calls_info:
                    tool_id = tool_call["id"]
                    if tool_id in tool_results_map:
                        result_data = tool_results_map[tool_id]
                        tool_call["result"] = result_data.get("result")

                        # Metadata is no longer in API responses (removed for API compliance)
                        # Will be enriched from tool_call_history in Pass 4
                        tool_call["executionTimeMs"] = 0  # Placeholder, will be updated in Pass 4
                        tool_call["success"] = True  # Default assumption
                        tool_call["error"] = None
                        tool_call["explicitParams"] = []

                # Pass 4: Enrich with execution metadata from tool_call_history
                # tool_call_history contains execution_time_ms, success, error, etc.
                # Match by tool name and use counters to handle multiple calls to same tool
                if self.agent.tool_call_history:
                    # Build mapping: tool_name -> list of execution records (in order)
                    tool_history_by_name = {}
                    for record in self.agent.tool_call_history:
                        tool_name = record.get("tool_name", "")
                        if tool_name not in tool_history_by_name:
                            tool_history_by_name[tool_name] = []
                        tool_history_by_name[tool_name].append(record)

                    # Track how many times we've matched each tool name (for multiple calls)
                    tool_name_counters = {}

                    # Enrich tool_calls_info with execution metadata
                    for tool_call in tool_calls_info:
                        tool_name = tool_call.get("name", "")

                        # Get current index for this tool name
                        current_index = tool_name_counters.get(tool_name, 0)
                        tool_name_counters[tool_name] = current_index + 1

                        # Find matching execution record
                        if tool_name in tool_history_by_name:
                            history_records = tool_history_by_name[tool_name]

                            # Use most recent N calls (in reverse order to match recent conversation)
                            # We want the Nth-most-recent call for this tool
                            if current_index < len(history_records):
                                # Match from the end (most recent first)
                                record_index = len(history_records) - 1 - current_index
                                if record_index >= 0:
                                    record = history_records[record_index]

                                    # Enrich with execution metadata
                                    tool_call["executionTimeMs"] = record.get("execution_time_ms", 0)
                                    tool_call["success"] = record.get("success", True)
                                    # Only include error if present
                                    if "error" in record:
                                        tool_call["error"] = record.get("error")
                                    # Include RAG confidence if available
                                    if "rag_confidence" in record:
                                        tool_call["ragConfidence"] = record.get("rag_confidence")

                    logger.info(f"Enriched {len(tool_calls_info)} tool calls with execution metadata from tool_call_history")

            # Send tool_calls_summary with all tool results
            # Frontend uses this for inline rendering (matches markers in text)
            # No need for individual tool_result events - summary has everything
            if tool_calls_info:
                logger.info(f"Extracted {len(tool_calls_info)} tool calls from conversation history")
                yield {
                    "event": "tool_calls_summary",
                    "data": {
                        "tool_calls": tool_calls_info,
                        "count": len(tool_calls_info)
                    }
                }
            else:
                logger.debug("No tool calls found in conversation history")

            # Send final cost update
            cost_summary = tracker.get_session_cost_summary()
            yield {
                "event": "cost_update",
                "data": {
                    "summary": cost_summary,
                    "total_cost": tracker.get_total_cost(),
                    "input_tokens": tracker._total_input_tokens,
                    "output_tokens": tracker._total_output_tokens,
                    "cached_tokens": tracker._total_cache_read_tokens
                }
            }

            # Signal completion
            yield {
                "event": "done",
                "data": {}
            }

        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": {
                    "error": str(e),
                    "type": type(e).__name__
                }
            }

    def get_available_models(self) -> list[Dict[str, Any]]:
        """
        Get list of available models (same as CLI).

        Returns:
            List of model info dicts matching CLI's _list_available_models()
        """
        return [
            # Anthropic Claude models (from CLI)
            {
                "id": "claude-haiku-4-5-20251001",
                "name": "Claude Haiku 4.5",
                "provider": "anthropic",
                "description": "Fast & cost-effective (✅ caching)"
            },
            {
                "id": "claude-sonnet-4-5-20250929",
                "name": "Claude Sonnet 4.5",
                "provider": "anthropic",
                "description": "Balanced performance (✅ caching)"
            },
            # Google Gemini models
            {
                "id": "gemini-2.5-flash",
                "name": "Gemini 2.5 Flash",
                "provider": "google",
                "description": "Fast & agentic (✅ caching, 250/day free)"
            },
            {
                "id": "gemini-2.5-pro",
                "name": "Gemini 2.5 Pro",
                "provider": "google",
                "description": "Best reasoning (✅ caching, 100/day free)"
            },
            {
                "id": "gemini-2.5-flash-lite",
                "name": "Gemini 2.5 Flash Lite",
                "provider": "google",
                "description": "High volume (✅ caching, 1000/day free)"
            },
            # OpenAI GPT-5 models (from CLI)
            {
                "id": "gpt-5-nano",
                "name": "GPT-5 Nano",
                "provider": "openai",
                "description": "Ultra-fast, minimal cost (❌ no caching)"
            },
            {
                "id": "gpt-5-mini",
                "name": "GPT-5 Mini",
                "provider": "openai",
                "description": "Balanced & affordable (❌ no caching)"
            },
            {
                "id": "gpt-5",
                "name": "GPT-5",
                "provider": "openai",
                "description": "Most capable (❌ no caching)"
            }
        ]

    def switch_model(self, model: str) -> None:
        """
        Switch to a different model (UNIFIED WITH CLI).

        Preserves conversation history and document list.
        System prompt, tools, and history are sent on every API call automatically.

        Args:
            model: Model identifier
        """
        old_model = self.config.model

        # Update config
        self.config.model = model

        # Create new provider (same logic as CLI)
        new_provider = create_provider(
            model=model,
            anthropic_api_key=self.config.anthropic_api_key,
            openai_api_key=self.config.openai_api_key,
            google_api_key=self.config.google_api_key,
        )

        # Switch provider on existing agent (preserves conversation_history)
        self.agent.provider = new_provider

        # Auto-adjust streaming based on provider support (same as CLI)
        streaming_supported = new_provider.supports_feature('streaming')
        old_streaming = self.config.cli_config.enable_streaming
        self.config.cli_config.enable_streaming = streaming_supported

        logger.info(
            f"Model switched: {old_model} → {model} "
            f"(provider: {new_provider.get_provider_name()}, "
            f"streaming: {old_streaming} → {streaming_supported}, "
            f"history preserved: {len(self.agent.conversation_history)} messages)"
        )

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get agent health status.

        Returns:
            Health status dict with degraded component warnings
        """
        try:
            # Check if agent is properly initialized
            if not self.agent:
                return {
                    "status": "error",
                    "message": "Agent not initialized",
                    "details": {},
                    "degraded_components": []
                }

            # Check vector store
            vector_store_exists = self.config.vector_store_path.exists()

            # Check API keys
            has_anthropic_key = bool(self.config.anthropic_api_key)
            has_openai_key = bool(self.config.openai_api_key)
            has_google_key = bool(self.config.google_api_key)

            if not vector_store_exists:
                return {
                    "status": "error",
                    "message": "Vector store not found",
                    "details": {
                        "vector_store_path": str(self.config.vector_store_path)
                    },
                    "degraded_components": []
                }

            if not has_anthropic_key and not has_openai_key and not has_google_key:
                return {
                    "status": "error",
                    "message": "No API keys configured",
                    "details": {},
                    "degraded_components": []
                }

            # Determine overall status based on degraded components
            status = "degraded" if self.degraded_components else "healthy"
            message = "Agent ready" if status == "healthy" else "Agent running in degraded mode"

            return {
                "status": status,
                "message": message,
                "details": {
                    "model": self.config.model,
                    "vector_store": str(self.config.vector_store_path),
                    "has_anthropic_key": has_anthropic_key,
                    "has_openai_key": has_openai_key,
                    "has_google_key": has_google_key
                },
                "degraded_components": self.degraded_components  # Expose to UI
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "details": {}
            }
