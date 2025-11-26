"""
Agent Adapter - Wraps Multi-Agent Runner for web frontend.

This adapter:
1. Initializes MultiAgentRunner from src/multi_agent/runner.py
2. Handles SSE event formatting for workflow execution
3. Tracks cost per message
4. Provides clean interface for FastAPI
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

from src.multi_agent.runner import MultiAgentRunner
from src.agent.config import AgentConfig
from src.cost_tracker import get_global_tracker, reset_global_tracker

logger = logging.getLogger(__name__)


class AgentAdapter:
    """
    Adapter wrapping MultiAgentRunner for web frontend.

    Responsibilities:
    - Initialize multi-agent system with config
    - Convert workflow events to SSE format
    - Track costs per request
    - Provide model switching capability
    """

    def __init__(self, vector_store_path: Optional[Path] = None, model: Optional[str] = None):
        """
        Initialize agent adapter with multi-agent system.

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

        # Load full configuration
        self.config = AgentConfig.from_env(**config_overrides)

        # Validate config
        try:
            self.config.validate()
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise

        # Load multi-agent configuration
        # Check if config.json has multi_agent section
        config_path = project_root / "config.json"
        multi_agent_config = {}

        if config_path.exists():
            try:
                with open(config_path) as f:
                    full_config = json.load(f)
                    multi_agent_config = full_config.get("multi_agent", {})
            except Exception as e:
                logger.warning(f"Could not load multi_agent config from config.json: {e}")

        # If no multi_agent config, load from extension file
        if not multi_agent_config:
            extension_path = project_root / "config_multi_agent_extension.json"
            if extension_path.exists():
                try:
                    with open(extension_path) as f:
                        extension_config = json.load(f)
                        multi_agent_config = extension_config.get("multi_agent", {})
                except Exception as e:
                    logger.error(f"Could not load config_multi_agent_extension.json: {e}")
                    raise

        if not multi_agent_config:
            raise ValueError(
                "Multi-agent configuration not found. "
                "Add 'multi_agent' section to config.json or ensure config_multi_agent_extension.json exists."
            )

        # Build runner configuration
        runner_config = {
            "api_keys": {
                "anthropic_api_key": self.config.anthropic_api_key,
                "openai_api_key": self.config.openai_api_key,
                "google_api_key": self.config.google_api_key,
                "deepinfra_api_key": os.getenv("DEEPINFRA_API_KEY"),
            },
            "vector_store_path": str(self.config.vector_store_path),
            "models": full_config.get("models", {}),  # Add models section for embedding config
            "storage": full_config.get("storage", {}),  # Add storage section for backend selection
            "agent_tools": full_config.get("agent_tools", {}),  # Add agent_tools for reranking config
            "knowledge_graph": full_config.get("knowledge_graph", {}),  # Add knowledge_graph config
            "neo4j": full_config.get("neo4j", {}),  # Add neo4j connection config
            "multi_agent": multi_agent_config,
        }

        # Initialize multi-agent runner
        logger.info("Initializing multi-agent system...")
        self.runner = MultiAgentRunner(runner_config)

        # Track degraded components for health endpoint
        self.degraded_components = []

        # HITL: In-memory storage for pending clarifications
        # TODO: In production, use Redis or database for multi-instance deployments
        self._pending_clarifications: Dict[str, Dict[str, Any]] = {}

        # Store current model
        self.current_model = model or multi_agent_config.get("orchestrator", {}).get("model", "claude-sonnet-4-5-20250929")

        logger.info(
            f"AgentAdapter initialized with multi-agent system: "
            f"model={self.current_model}, vector_store={self.config.vector_store_path}"
        )

    def _apply_variant_overrides(self, variant: str, multi_agent_config: dict) -> dict:
        """
        Apply variant-specific model overrides to multi-agent configuration.

        Args:
            variant: Agent variant ('premium' or 'local')
            multi_agent_config: Original multi_agent config from config.json

        Returns:
            Modified config with variant models applied
        """
        import copy

        # Map variants to models
        variant_models = {
            "premium": "claude-haiku-4-5",
            "local": "meta-llama/Meta-Llama-3.1-70B-Instruct"
        }

        model = variant_models.get(variant)
        if not model:
            logger.warning(f"Unknown variant '{variant}', using config defaults")
            return multi_agent_config

        # Deep copy to avoid modifying original
        config = copy.deepcopy(multi_agent_config)

        # Override orchestrator model
        if "orchestrator" in config:
            config["orchestrator"]["model"] = model
            logger.debug(f"Orchestrator model: {model}")

        # Override all agent models
        if "agents" in config:
            for agent_name in config["agents"]:
                config["agents"][agent_name]["model"] = model
                logger.debug(f"Agent {agent_name} model: {model}")

        logger.info(f"Applied variant '{variant}' - all agents using {model}")
        return config

    async def initialize(self) -> bool:
        """
        Initialize the multi-agent runner.

        Returns:
            True if successful
        """
        try:
            success = await self.runner.initialize()
            if not success:
                logger.error("Multi-agent system initialization failed")
                self.degraded_components.append({
                    "component": "multi_agent_system",
                    "error": "Initialization failed",
                    "severity": "critical",
                    "user_message": "Multi-agent system failed to start",
                })
                return False

            logger.info("Multi-agent system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize multi-agent system: {e}", exc_info=True)
            self.degraded_components.append({
                "component": "multi_agent_system",
                "error": str(e),
                "severity": "critical",
                "user_message": f"System initialization error: {str(e)}",
            })
            return False

    async def stream_response(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent response as SSE-compatible events.

        Note: Multi-agent system executes workflow asynchronously and returns
        final result. We emit progress updates during execution.

        Yields SSE events in format:
        {
            "event": "progress" | "agent_start" | "agent_complete" | "tool_call" | "text_delta" | "done" | "error",
            "data": {...}
        }

        Event types:
        - progress: Workflow progress updates (e.g., "Running extractor agent...")
        - agent_start: Agent execution started
        - agent_complete: Agent execution completed
        - text_delta: Final answer text (streamed after workflow completes)
        - done: Stream completed successfully
        - error: Error occurred during execution

        Note: Cost tracking is done internally but not shown to users

        Args:
            query: User query
            conversation_id: Optional conversation ID for context (not used in multi-agent)
            user_id: User ID for loading agent variant preference

        Yields:
            Dict containing event type and data
        """
        # Reset cost tracker for this request
        reset_global_tracker()
        tracker = get_global_tracker()

        # Load user's variant and create runner with variant-specific config
        variant = "premium"  # default
        runner_to_use = self.runner  # default to existing runner

        if user_id:
            try:
                from backend.routes.auth import get_auth_queries
                queries = get_auth_queries()
                variant = await queries.get_agent_variant(user_id)
                logger.info(f"User {user_id} variant: {variant}")

                # Only create new runner if variant is not premium (to avoid overhead)
                if variant != "premium":
                    # Load config and apply variant overrides
                    project_root = Path(__file__).parent.parent
                    config_path = project_root / "config.json"

                    with open(config_path) as f:
                        full_config = json.load(f)
                        multi_agent_config = full_config.get("multi_agent", {})

                    # Apply variant overrides
                    multi_agent_config = self._apply_variant_overrides(variant, multi_agent_config)

                    # Build runner config with variant models
                    runner_config = {
                        "api_keys": {
                            "anthropic_api_key": self.config.anthropic_api_key,
                            "openai_api_key": self.config.openai_api_key,
                            "google_api_key": self.config.google_api_key,
                            "deepinfra_api_key": os.getenv("DEEPINFRA_API_KEY"),
                        },
                        "vector_store_path": str(self.config.vector_store_path),
                        "models": full_config.get("models", {}),
                        "storage": full_config.get("storage", {}),
                        "agent_tools": full_config.get("agent_tools", {}),
                        "knowledge_graph": full_config.get("knowledge_graph", {}),
                        "neo4j": full_config.get("neo4j", {}),
                        "multi_agent": multi_agent_config,
                    }

                    # Create fresh runner with variant config
                    new_runner = MultiAgentRunner(runner_config)
                    await new_runner.initialize()
                    runner_to_use = new_runner  # Only assign after successful init
                    logger.info(f"Created fresh runner with variant '{variant}'")

            except Exception as e:
                logger.warning(f"Failed to load/apply variant for user {user_id}: {e}")
                # Continue with default runner (runner_to_use unchanged)
                runner_to_use = self.runner

        try:
            # Emit start event
            yield {
                "event": "progress",
                "data": {
                    "message": "Initializing multi-agent workflow...",
                    "stage": "init"
                }
            }
            await asyncio.sleep(0)

            # Execute multi-agent workflow with progress streaming
            logger.info("Starting multi-agent query execution with streaming...")

            # Stream progress events from runner (use variant-specific runner if applicable)
            result = None
            async for event in runner_to_use.run_query(query, stream_progress=True):
                if event.get("type") == "agent_start":
                    # Agent start event (new format from runner.py)
                    agent_name = event.get("agent", "unknown")

                    # Map agent name to user-friendly role
                    agent_roles = {
                        "extractor": "Searching documents",
                        "classifier": "Classifying query",
                        "compliance": "Checking compliance",
                        "risk_verifier": "Verifying risks",
                        "citation_auditor": "Auditing citations",
                        "gap_synthesizer": "Synthesizing information",
                        "report_generator": "Generating report"
                    }
                    message = agent_roles.get(agent_name, f"Running {agent_name}")

                    yield {
                        "event": "agent_start",
                        "data": {
                            "agent": agent_name,
                            "message": message
                        }
                    }
                    await asyncio.sleep(0)

                elif event.get("type") == "progress":
                    # Legacy progress event (workflow init/complete)
                    yield {
                        "event": "progress",
                        "data": event
                    }
                    await asyncio.sleep(0)

                elif event.get("type") == "tool_call":
                    # Tool call event (running/completed/failed)
                    yield {
                        "event": "tool_call",
                        "data": {
                            "agent": event.get("agent"),
                            "tool": event.get("tool"),
                            "status": event.get("status"),
                            "timestamp": event.get("timestamp")
                        }
                    }
                    await asyncio.sleep(0)

                elif event.get("type") == "final":
                    # Final result
                    result = event
                    break

            if not result:
                raise RuntimeError("Workflow produced no result")

            # Check if clarification is needed (HITL)
            if result.get("clarification_needed", False):
                logger.info("HITL: Clarification needed, emitting event...")

                thread_id = result.get("thread_id")

                # Store clarification state for resume
                self._pending_clarifications[thread_id] = {
                    "original_query": result.get("original_query"),
                    "complexity_score": result.get("complexity_score"),
                    "query_type": result.get("query_type"),
                    "questions": result.get("questions", []),
                    "quality_metrics": result.get("quality_metrics", {}),
                    # Need to store agent_sequence for resume
                    "agent_sequence": result.get("agent_sequence", []),
                }

                yield {
                    "event": "clarification_needed",
                    "data": {
                        "thread_id": thread_id,
                        "questions": result.get("questions", []),
                        "quality_metrics": result.get("quality_metrics", {}),
                        "original_query": result.get("original_query"),
                        "complexity_score": result.get("complexity_score"),
                        "query_type": result.get("query_type"),
                    }
                }

                # Don't emit "done" - waiting for user response
                return

            # Check if execution was successful
            if not result.get("success", False):
                error_msg = result.get("final_answer", "Unknown error")
                yield {
                    "event": "error",
                    "data": {
                        "error": error_msg,
                        "type": "ExecutionError",
                        "errors": result.get("errors", [])
                    }
                }
                return

            # Extract workflow metadata
            complexity_score = result.get("complexity_score", 0)
            query_type = result.get("query_type", "unknown")
            agent_sequence = result.get("agent_sequence", [])

            # Emit workflow summary
            yield {
                "event": "progress",
                "data": {
                    "message": f"Workflow executed: {', '.join(agent_sequence)}",
                    "stage": "complete",
                    "complexity_score": complexity_score,
                    "query_type": query_type,
                    "agents_used": len(agent_sequence)
                }
            }
            await asyncio.sleep(0)

            # Extract final answer
            final_answer = result.get("final_answer") or "No answer generated"

            # Stream final answer as text chunks (simulate streaming for UX consistency)
            # Split into paragraphs for progressive display
            paragraphs = final_answer.split('\n\n')

            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    # Add back paragraph separator except for last paragraph
                    chunk = paragraph + ('\n\n' if i < len(paragraphs) - 1 else '')

                    yield {
                        "event": "text_delta",
                        "data": {
                            "content": chunk
                        }
                    }
                    # Small delay for UX (simulate natural response streaming)
                    await asyncio.sleep(0.05)

            # Extract tool calls if available (from tool_executions in state)
            tool_executions = result.get("tool_executions", [])
            if tool_executions:
                # Convert to frontend-compatible format
                tool_calls_info = []
                for execution in tool_executions:
                    tool_calls_info.append({
                        "id": execution.get("tool_id", ""),
                        "name": execution.get("tool_name", ""),
                        "input": execution.get("inputs", {}),
                        "result": execution.get("output", {}),
                        "executionTimeMs": execution.get("execution_time_ms", 0),
                        "success": execution.get("success", True),
                        "error": execution.get("error"),
                        "agent": execution.get("agent_name", ""),
                    })

                yield {
                    "event": "tool_calls_summary",
                    "data": {
                        "tool_calls": tool_calls_info,
                        "count": len(tool_calls_info)
                    }
                }

            # Emit cost summary event with per-agent breakdown
            tracker = get_global_tracker()
            total_cost_usd = tracker.get_total_cost()
            agent_breakdown = tracker.get_agent_breakdown()

            # Build agent breakdown array for frontend with defensive access
            agent_costs = []
            for agent_name, stats in agent_breakdown.items():
                try:
                    agent_costs.append({
                        "agent": agent_name,
                        "cost": stats.get("cost", 0.0),
                        "input_tokens": stats.get("input_tokens", 0),
                        "output_tokens": stats.get("output_tokens", 0),
                        "cache_read_tokens": stats.get("cache_read_tokens", 0),
                        "cache_creation_tokens": stats.get("cache_creation_tokens", 0),
                        "call_count": stats.get("call_count", 0),
                        "response_time_ms": stats.get("response_time_ms", 0.0)
                    })
                except Exception as e:
                    logger.error(f"Failed to format cost for agent {agent_name}: {e}", exc_info=True)
                    # Continue with remaining agents - partial cost data is better than none

            # Sort by cost descending
            agent_costs.sort(key=lambda x: x["cost"], reverse=True)

            yield {
                "event": "cost_summary",
                "data": {
                    "total_cost": total_cost_usd,
                    "agent_breakdown": agent_costs,
                    "total_input_tokens": tracker.total_input_tokens,
                    "total_output_tokens": tracker.total_output_tokens,
                    "cache_stats": tracker.get_cache_stats()
                }
            }
            await asyncio.sleep(0)

            # Signal completion
            yield {
                "event": "done",
                "data": {
                    "agent_sequence": agent_sequence,
                    "complexity_score": complexity_score,
                    "query_type": query_type,
                }
            }

        except Exception as e:
            # Capture execution context for debugging
            context = {
                "query": query[:200] if query else "N/A",
                "conversation_id": conversation_id,
                "agent_sequence": agent_sequence if 'agent_sequence' in locals() else [],
                "last_agent": agent_sequence[-1] if 'agent_sequence' in locals() and agent_sequence else None,
                "error_phase": "multi_agent_execution"
            }

            logger.error(
                f"Error during multi-agent execution: {type(e).__name__}: {e}",
                exc_info=True,
                extra=context
            )

            yield {
                "event": "error",
                "data": {
                    "error": str(e),
                    "type": type(e).__name__,
                    "context": context
                }
            }

    def get_available_models(self) -> list[Dict[str, Any]]:
        """
        Get list of available models.

        Returns:
            List of model info dicts
        """
        return [
            # Anthropic Claude models
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
            # OpenAI GPT models
            {
                "id": "gpt-4o-mini",
                "name": "GPT-4o Mini",
                "provider": "openai",
                "description": "Fast and affordable (❌ no caching)"
            },
        ]

    def switch_model(self, model: str) -> None:
        """
        Switch to a different model.

        Note: In multi-agent system, model switching affects orchestrator.
        Individual agents use their configured models.

        Args:
            model: Model identifier
        """
        old_model = self.current_model
        self.current_model = model

        # Update orchestrator model in runner config
        if hasattr(self.runner, 'multi_agent_config'):
            if 'orchestrator' in self.runner.multi_agent_config:
                self.runner.multi_agent_config['orchestrator']['model'] = model

        logger.info(
            f"Model switched: {old_model} → {model} "
            f"(Note: Individual agents use their configured models)"
        )

    async def resume_clarification(
        self, thread_id: str, user_response: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Resume interrupted workflow with user clarification.

        Args:
            thread_id: Thread ID from clarification_needed event
            user_response: User's free-form clarification text

        Yields:
            SSE events (same format as stream_response)
        """
        # Reset cost tracker
        reset_global_tracker()
        tracker = get_global_tracker()

        try:
            # Retrieve pending clarification
            if thread_id not in self._pending_clarifications:
                yield {
                    "event": "error",
                    "data": {
                        "error": f"No pending clarification found for thread {thread_id}",
                        "type": "NotFoundError",
                    }
                }
                return

            clarification_data = self._pending_clarifications[thread_id]

            # Emit resume event
            yield {
                "event": "progress",
                "data": {
                    "message": "Resuming workflow with clarification...",
                    "stage": "resume",
                }
            }
            await asyncio.sleep(0)

            # Build original state from clarification data
            original_state = {
                "query": clarification_data["original_query"],
                "complexity_score": clarification_data["complexity_score"],
                "query_type": clarification_data["query_type"],
                "agent_sequence": clarification_data["agent_sequence"],
                "execution_phase": "agent_execution",
                "agent_outputs": {},
                "tool_executions": [],
                "documents": [],
                "citations": [],
                "total_cost_cents": 0.0,
                "errors": [],
            }

            # Resume workflow
            result = await self.runner.resume_with_clarification(
                thread_id=thread_id,
                user_response=user_response,
                original_state=original_state,
            )

            # Clean up pending clarification
            del self._pending_clarifications[thread_id]

            # Check if successful
            if not result.get("success", False):
                error_msg = result.get("final_answer", "Resume failed")
                yield {
                    "event": "error",
                    "data": {
                        "error": error_msg,
                        "type": "ResumeError",
                        "errors": result.get("errors", []),
                    }
                }
                return

            # Extract workflow metadata
            complexity_score = result.get("complexity_score", 0)
            query_type = result.get("query_type", "unknown")
            agent_sequence = result.get("agent_sequence", [])

            # Emit workflow summary
            yield {
                "event": "progress",
                "data": {
                    "message": f"Workflow completed: {', '.join(agent_sequence)}",
                    "stage": "complete",
                    "complexity_score": complexity_score,
                    "query_type": query_type,
                    "agents_used": len(agent_sequence),
                    "enriched_query": result.get("enriched_query"),
                }
            }
            await asyncio.sleep(0)

            # Stream final answer (same as stream_response)
            final_answer = result.get("final_answer") or "No answer generated"
            paragraphs = final_answer.split("\n\n")

            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    chunk = paragraph + ("\n\n" if i < len(paragraphs) - 1 else "")
                    yield {
                        "event": "text_delta",
                        "data": {"content": chunk},
                    }
                    await asyncio.sleep(0.05)

            # Emit cost summary event with per-agent breakdown
            tracker = get_global_tracker()
            total_cost_usd = tracker.get_total_cost()
            agent_breakdown = tracker.get_agent_breakdown()

            # Build agent breakdown array for frontend with defensive access
            agent_costs = []
            for agent_name, stats in agent_breakdown.items():
                try:
                    agent_costs.append({
                        "agent": agent_name,
                        "cost": stats.get("cost", 0.0),
                        "input_tokens": stats.get("input_tokens", 0),
                        "output_tokens": stats.get("output_tokens", 0),
                        "cache_read_tokens": stats.get("cache_read_tokens", 0),
                        "cache_creation_tokens": stats.get("cache_creation_tokens", 0),
                        "call_count": stats.get("call_count", 0),
                        "response_time_ms": stats.get("response_time_ms", 0.0)
                    })
                except Exception as e:
                    logger.error(f"Failed to format cost for agent {agent_name}: {e}", exc_info=True)
                    # Continue with remaining agents - partial cost data is better than none

            # Sort by cost descending
            agent_costs.sort(key=lambda x: x["cost"], reverse=True)

            yield {
                "event": "cost_summary",
                "data": {
                    "total_cost": total_cost_usd,
                    "agent_breakdown": agent_costs,
                    "total_input_tokens": tracker.total_input_tokens,
                    "total_output_tokens": tracker.total_output_tokens,
                    "cache_stats": tracker.get_cache_stats()
                }
            }
            await asyncio.sleep(0)

            # Emit done
            yield {"event": "done", "data": {}}

        except Exception as e:
            # Capture execution context for debugging
            context = {
                "query": query[:200] if query else "N/A",
                "conversation_id": conversation_id,
                "error_phase": "simple_mode_execution"
            }

            logger.error(
                f"Resume error: {type(e).__name__}: {e}",
                exc_info=True,
                extra=context
            )

            yield {
                "event": "error",
                "data": {
                    "error": str(e),
                    "type": type(e).__name__,
                    "context": context
                },
            }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get agent health status.

        Returns:
            Health status dict with degraded component warnings
        """
        try:
            # Check if runner is properly initialized
            if not self.runner:
                return {
                    "status": "error",
                    "message": "Multi-agent runner not initialized",
                    "details": {},
                    "degraded_components": []
                }

            # Check vector store (skip for PostgreSQL backend)
            # Read storage backend from environment variable (set in docker-compose.yml)
            import os
            storage_backend = os.getenv("STORAGE_BACKEND", "faiss")

            if storage_backend == "faiss":
                vector_store_exists = self.config.vector_store_path.exists()
                if not vector_store_exists:
                    return {
                        "status": "error",
                        "message": "Vector store not found",
                        "details": {
                            "vector_store_path": str(self.config.vector_store_path)
                        },
                        "degraded_components": []
                    }

            # Check API keys
            has_anthropic_key = bool(self.config.anthropic_api_key)
            has_openai_key = bool(self.config.openai_api_key)
            has_google_key = bool(self.config.google_api_key)

            if not has_anthropic_key and not has_openai_key and not has_google_key:
                return {
                    "status": "error",
                    "message": "No API keys configured",
                    "details": {},
                    "degraded_components": []
                }

            # Determine overall status based on degraded components
            status = "degraded" if self.degraded_components else "healthy"
            message = "Multi-agent system ready" if status == "healthy" else "Multi-agent system running in degraded mode"

            return {
                "status": status,
                "message": message,
                "details": {
                    "model": self.current_model,
                    "vector_store": str(self.config.vector_store_path),
                    "has_anthropic_key": has_anthropic_key,
                    "has_openai_key": has_openai_key,
                    "has_google_key": has_google_key,
                    "agents_registered": len(self.runner.agent_registry._agent_instances) if hasattr(self.runner, 'agent_registry') and self.runner.agent_registry else 0,
                },
                "degraded_components": self.degraded_components
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "details": {},
                "degraded_components": []
            }
