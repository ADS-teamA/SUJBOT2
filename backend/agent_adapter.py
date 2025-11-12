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
            },
            "vector_store_path": str(self.config.vector_store_path),
            "models": full_config.get("models", {}),  # Add models section for embedding config
            "storage": full_config.get("storage", {}),  # Add storage section for backend selection
            "agent_tools": full_config.get("agent_tools", {}),  # Add agent_tools for reranking config
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
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent response as SSE-compatible events.

        Note: Multi-agent system executes workflow asynchronously and returns
        final result. We emit progress updates during execution.

        Yields SSE events in format:
        {
            "event": "progress" | "agent_start" | "agent_complete" | "tool_call" | "text_delta" | "cost_update" | "done" | "error",
            "data": {...}
        }

        Event types:
        - progress: Workflow progress updates (e.g., "Running extractor agent...")
        - agent_start: Agent execution started
        - agent_complete: Agent execution completed
        - text_delta: Final answer text (streamed after workflow completes)
        - cost_update: Token usage and cost information
        - done: Stream completed successfully
        - error: Error occurred during execution

        Args:
            query: User query
            conversation_id: Optional conversation ID for context (not used in multi-agent)

        Yields:
            Dict containing event type and data
        """
        # Reset cost tracker for this request
        reset_global_tracker()
        tracker = get_global_tracker()

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

            # Stream progress events from runner
            result = None
            async for event in self.runner.run_query(query, stream_progress=True):
                if event.get("type") == "progress":
                    # Agent progress event
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

            # Send final cost update (use CostTracker for accurate model-specific pricing)
            tracker = get_global_tracker()
            total_cost_usd = tracker.get_total_cost()
            total_cost_cents = total_cost_usd * 100.0

            yield {
                "event": "cost_update",
                "data": {
                    "summary": {
                        "total_cost_cents": total_cost_cents,
                        "total_cost_usd": total_cost_usd,
                    },
                    "total_cost": total_cost_usd,
                    "complexity_score": complexity_score,
                    "agents_used": len(agent_sequence),
                }
            }

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
            logger.error(f"Error during multi-agent execution: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": {
                    "error": str(e),
                    "type": type(e).__name__
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

            # Emit cost update (use CostTracker for accurate model-specific pricing)
            cost_summary = tracker.get_summary()
            total_cost_usd = tracker.get_total_cost()

            yield {
                "event": "cost_update",
                "data": {
                    "summary": cost_summary,
                    "total_cost": total_cost_usd,
                },
            }

            # Emit done
            yield {"event": "done", "data": {}}

        except Exception as e:
            logger.error(f"Resume error: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": {"error": str(e), "type": type(e).__name__},
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
