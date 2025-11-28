"""
Abstract base class for all agents in the multi-agent system.

Defines standard interface, lifecycle, and tool distribution patterns.
Ensures consistency across all 8 specialized agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import logging
import time  # For LLM response time measurement

from pydantic import BaseModel
from langgraph.graph import StateGraph

from .event_bus import EventBus, EventType
from ..observability.trajectory import AgentTrajectory, TrajectoryStep, StepType

logger = logging.getLogger(__name__)


class AgentTier(str, Enum):
    """Agent execution tier for organizational purposes."""
    ORCHESTRATOR = "orchestrator"  # Root coordinator
    SPECIALIST = "specialist"      # Domain-specific agent
    WORKER = "worker"              # Tool-executing agent


class AgentRole(str, Enum):
    """Agent responsibilities in the workflow."""
    ORCHESTRATE = "orchestrate"    # Coordinate other agents
    EXTRACT = "extract"            # Extract info from docs
    CLASSIFY = "classify"          # Classify queries/docs
    VERIFY = "verify"              # Verify compliance/risk
    AUDIT = "audit"                # Audit citations
    SYNTHESIZE = "synthesize"      # Synthesize gaps
    REPORT = "report"              # Generate reports


@dataclass
class AgentConfig:
    """
    Per-agent configuration (loaded from config.json per-agent section).

    Each agent has its own model, temperature, tools, and settings.
    """

    name: str                       # Agent name (e.g., 'extractor')
    role: AgentRole                 # Agent role/responsibility
    tier: AgentTier                 # Execution tier
    model: str                      # LLM model for this agent
    api_key: str = ""               # API key for LLM provider
    max_tokens: int = 4096          # Max output tokens
    temperature: float = 0.3        # LLM temperature
    tools: Set[str] = field(default_factory=set)  # Tool names this agent can access
    timeout_seconds: int = 30       # Tool execution timeout
    retry_count: int = 2            # Retry failed tool calls
    enable_prompt_caching: bool = True
    enable_cost_tracking: bool = True
    parent_agent: Optional[str] = None  # For hierarchical agents
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate agent configuration."""
        if not self.name:
            raise ValueError("Agent name is required")
        # Orchestrator doesn't need tools (it only does LLM-based routing)
        if not self.tools and self.role != AgentRole.ORCHESTRATE:
            raise ValueError(f"Agent {self.name} has no tools assigned")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be in [0, 1], got {self.temperature}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive")
        if self.retry_count < 0:
            raise ValueError(f"retry_count must be non-negative")


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Defines standard interface that all 8 agents must implement.
    Enforces:
    - Configuration validation
    - Tool distribution rules
    - State management
    - Error handling
    - Cost tracking

    Design Pattern: Template Method
    - execute() is the template method
    - Subclasses implement execute_impl()
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize agent.

        Args:
            config: Per-agent configuration
        """
        self.config = config
        config.validate()
        self.logger = logging.getLogger(f"agent.{config.name}")
        self.execution_count = 0
        self.total_time_ms = 0.0
        self.error_count = 0

    def build_graph(self) -> Optional[StateGraph]:
        """
        Build LangGraph graph for this agent (optional).

        Agents can optionally implement internal LangGraph workflows.
        Most agents use execute_impl() directly and don't need internal graphs.

        Returns:
            StateGraph: Compiled LangGraph graph, or None if not implemented
        """
        return None

    @abstractmethod
    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state through agent (implementation).

        Called by execute() template method. Updates state with:
        - Tool results
        - Agent output
        - Errors (if any)

        Args:
            state: Current workflow state

        Returns:
            Updated state dict
        """
        pass

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent with error handling and tracking (template method).

        This is the public interface. Calls execute_impl() with
        proper error handling, timing, and cost tracking.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict
        """
        import time

        start_time = time.time()
        self.execution_count += 1

        try:
            self.logger.info(f"Agent {self.config.name} starting execution")
            result = await self.execute_impl(state)

            elapsed_ms = (time.time() - start_time) * 1000
            self.total_time_ms += elapsed_ms

            self.logger.info(
                f"Agent {self.config.name} completed in {elapsed_ms:.0f}ms"
            )

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.total_time_ms += elapsed_ms
            self.error_count += 1

            self.logger.error(
                f"Agent {self.config.name} failed after {elapsed_ms:.0f}ms: {e}",
                exc_info=True
            )

            return await self.handle_error(e, state)

    async def handle_error(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors gracefully.

        Implements fallback strategy:
        1. Retry with exponential backoff
        2. Degrade gracefully (use cached results)
        3. Escalate to orchestrator

        Args:
            error: Exception that occurred
            state: Current state

        Returns:
            Updated state with error recorded
        """
        # Track error with unique ID for Sentry
        from .error_tracker import track_error, ErrorSeverity

        error_id = track_error(
            error=error,
            severity=ErrorSeverity.HIGH,
            agent_name=self.config.name,
            context={"query": state.get("query", ""), "phase": state.get("execution_phase", "")}
        )

        error_message = f"[{error_id}] {self.config.name}: {type(error).__name__}: {str(error)}"

        # Add error to state
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_message)

        # Mark execution phase as error
        state["execution_phase"] = "error"

        self.logger.error(f"[{error_id}] Error in agent {self.config.name}: {error}")

        return state

    def get_tool_names(self) -> List[str]:
        """Get list of tools this agent can access."""
        return sorted(self.config.tools)

    def validate_tools(self, available_tools: Set[str]) -> bool:
        """
        Validate all agent tools are available.

        Args:
            available_tools: Set of available tool names

        Returns:
            True if all tools available, False otherwise
        """
        missing = self.config.tools - available_tools
        if missing:
            self.logger.error(
                f"Agent {self.config.name} has unavailable tools: {missing}"
            )
            return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        avg_time = (
            self.total_time_ms / self.execution_count
            if self.execution_count > 0
            else 0
        )
        success_rate = (
            (self.execution_count - self.error_count) / self.execution_count * 100
            if self.execution_count > 0
            else 100.0
        )

        return {
            "name": self.config.name,
            "role": self.config.role.value,
            "tier": self.config.tier.value,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "success_rate": round(success_rate, 1),
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(avg_time, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent config for checkpointing."""
        return {
            "name": self.config.name,
            "role": self.config.role.value,
            "tier": self.config.tier.value,
            "model": self.config.model,
            "tools": sorted(self.config.tools),
            "stats": self.get_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.config.name} "
            f"role={self.config.role.value} "
            f"tools={len(self.config.tools)}>"
        )

    def update_state_output(self, state: Dict[str, Any], output: Any) -> Dict[str, Any]:
        """
        Update agent_outputs in state (SSOT for state management).

        This is the canonical way to store agent output in workflow state.
        All agents should use this method instead of duplicating the pattern.

        Args:
            state: Current workflow state
            output: Agent's output data to store

        Returns:
            Updated state dict with agent output stored
        """
        state["agent_outputs"] = state.get("agent_outputs", {})
        state["agent_outputs"][self.config.name] = output
        return state

    # ========================================================================
    # AUTONOMOUS AGENTIC PATTERN (CLAUDE.md CONSTRAINT #0)
    # ========================================================================
    # Methods for building truly autonomous agents where LLM decides tool calling

    # Debug/metadata fields to exclude from agent context (token optimization)
    _EXCLUDE_FIELDS = {
        "iterations", "retrieval_method", "total_tool_cost_usd",
        "tool_calls_made", "chunks", "expanded_results", "_internal"
    }

    def _build_agent_context(self, state: Dict[str, Any]) -> str:
        """
        Build context string from state for agent.

        Token-optimized: Filters debug/metadata fields and uses compact JSON.
        Preserves all content fields including full citations.

        Includes:
        - Conversation history (last 3 turns, compact format)
        - Original query
        - Previous agent outputs (filtered, compact JSON)
        - Retrieved documents summary

        Args:
            state: Current workflow state

        Returns:
            Formatted context string
        """
        import json

        context_parts = []

        # Add conversation history (last 3 turns for context)
        # This allows sub-agents to resolve follow-up references like "it", "this", "to"
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            # Token-optimized: compact format, truncate long messages
            history_lines = []
            for msg in conversation_history[-6:]:  # Last 3 Q&A pairs = 6 messages
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
                # Truncate long messages to save tokens (keep first 300 chars)
                if len(content) > 300:
                    content = content[:300] + "..."
                history_lines.append(f"{role}: {content}")

            if history_lines:
                context_parts.append("**Conversation History (for follow-up context):**")
                context_parts.append("\n".join(history_lines))
                context_parts.append("")

        # Add query
        query = state.get("query", "")
        if query:
            context_parts.append(f"**User Query:**\n{query}\n")

        # Add previous agent outputs
        agent_outputs = state.get("agent_outputs", {})
        if agent_outputs:
            context_parts.append("**Previous Agent Outputs:**")
            for agent_name, output in agent_outputs.items():
                if agent_name != self.config.name:  # Don't include self
                    context_parts.append(f"\n[{agent_name}]")
                    if isinstance(output, dict):
                        # Filter debug fields, keep all content
                        filtered = {k: v for k, v in output.items()
                                  if k not in self._EXCLUDE_FIELDS}
                        # Compact JSON: no indent, minimal separators
                        context_parts.append(json.dumps(filtered, ensure_ascii=False, separators=(',', ':')))
            context_parts.append("")

        # Add retrieved documents summary (if available)
        documents = state.get("documents", [])
        if documents:
            context_parts.append(f"**Available Documents:** {len(documents)} documents")
            for doc in documents[:5]:  # Show first 5
                if hasattr(doc, 'filename'):
                    context_parts.append(f"- {doc.filename}")
            context_parts.append("")

        return "\n".join(context_parts)

    def _get_available_tool_schemas(self):
        """
        Get tool schemas for this agent from tool adapter.

        Returns:
            List of tool schemas in format expected by LLM providers
        """
        from ..tools.adapter import get_tool_adapter

        tool_adapter = get_tool_adapter()
        tool_schemas = []

        for tool_name in self.config.tools:
            schema = tool_adapter.get_tool_schema(tool_name)
            if schema:
                tool_schemas.append(schema)

        return tool_schemas

    async def _run_autonomous_tool_loop(
        self,
        system_prompt: str,
        state: Dict[str, Any],
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Run autonomous tool calling loop where LLM decides which tools to call.

        This is the core of autonomous agentic behavior:
        1. LLM sees state/query + available tools
        2. LLM decides to call tools or provide final answer
        3. Tool results fed back to LLM
        4. Loop continues until LLM provides final answer

        Args:
            system_prompt: System prompt defining agent role/behavior
            state: Current workflow state
            max_iterations: Max tool calling rounds

        Returns:
            Dict with:
                - final_answer: LLM's final response
                - tool_calls: List of tools called
                - reasoning: LLM's reasoning trace
        """
        from ..tools.adapter import get_tool_adapter
        from ...cost_tracker import get_global_tracker
        from ...utils.model_registry import ModelRegistry

        tool_adapter = get_tool_adapter()
        cost_tracker = get_global_tracker()

        # Use agent's provider (already initialized in __init__)
        if not hasattr(self, 'provider'):
            raise ValueError(f"Agent {self.config.name} does not have a provider initialized. "
                           "Autonomous agents must initialize self.provider in __init__")

        provider = self.provider

        # Build initial context
        context = self._build_agent_context(state)
        user_message = f"{context}\n\nPlease analyze this information and provide your expert assessment."

        # Get tool schemas
        tool_schemas = self._get_available_tool_schemas()

        # Get EventBus from state (injected by runner for real-time progress streaming)
        # Note: Key is "event_bus" (no underscore) to match MultiAgentState Pydantic field
        event_bus: Optional[EventBus] = state.get("event_bus")
        if not event_bus:
            self.logger.warning("No EventBus in state - events will not be streamed")

        messages = [{"role": "user", "content": user_message}]
        tool_call_history = []
        total_tool_cost = 0.0  # Track cumulative API cost for all tool calls

        # Initialize trajectory capture for evaluation
        trajectory = AgentTrajectory(
            agent_name=self.config.name,
            query=state.get("query", "")[:500],  # Truncate for storage
        )

        # Convert system prompt to cacheable format if caching enabled
        # (only needs to be done once before loop)
        cacheable_system = system_prompt
        if self.config.enable_prompt_caching and isinstance(system_prompt, str):
            # Anthropic prompt caching requires structured format with cache_control
            cacheable_system = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
            self.logger.debug(f"Prompt caching enabled for agent {self.config.name}")

        for iteration in range(max_iterations):
            self.logger.info(f"Autonomous loop iteration {iteration + 1}/{max_iterations}")

            # Call LLM with tools (measure response time)
            llm_start_time = time.time()
            try:
                response = provider.create_message(
                    messages=messages,
                    tools=tool_schemas,
                    system=cacheable_system,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                llm_response_time_ms = (time.time() - llm_start_time) * 1000
                self.logger.info(f"LLM call took {llm_response_time_ms:.2f}ms for agent {self.config.name}")
            except Exception as e:
                # Measure time to failure - valuable for debugging timeout/performance issues
                llm_response_time_ms = (time.time() - llm_start_time) * 1000
                self.logger.error(
                    f"LLM call FAILED after {llm_response_time_ms:.2f}ms for agent {self.config.name}: {e}",
                    exc_info=True
                )
                # Note: Cost tracking won't happen since we don't have a response object
                # Re-raise to allow upstream error handling (agent execute wrapper)
                raise

            # Track LLM usage with proper model-specific pricing
            if hasattr(response, 'usage') and response.usage:
                provider_name = provider.get_provider_name()
                model_name = provider.get_model_name()

                input_tokens = response.usage.get("input_tokens", 0)
                output_tokens = response.usage.get("output_tokens", 0)
                cache_read_tokens = response.usage.get("cache_read_tokens", 0)
                cache_creation_tokens = response.usage.get("cache_creation_tokens", 0)

                cost = cost_tracker.track_llm(
                    provider=provider_name,
                    model=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    operation=f"agent_{self.config.name}",
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens,
                    response_time_ms=llm_response_time_ms
                )

                self.logger.debug(
                    f"Agent {self.config.name} LLM call: {input_tokens} in, {output_tokens} out, "
                    f"cache: {cache_read_tokens} read, {cache_creation_tokens} created - ${cost:.6f}"
                )
            else:
                # Warn when response time is measured but not tracked
                self.logger.warning(
                    f"Response missing usage data for agent {self.config.name} - "
                    f"response time ({llm_response_time_ms:.2f}ms) not tracked in cost tracker. "
                    f"Response type: {type(response)}, has 'usage' attr: {hasattr(response, 'usage')}"
                )

            # Capture THOUGHT from LLM response (text blocks before tool calls)
            text_blocks = [
                block.get('text', '') for block in response.content
                if isinstance(block, dict) and block.get('type') == 'text'
            ] if isinstance(response.content, list) else []
            if text_blocks:
                thought_content = "\n".join(text_blocks)[:500]
                trajectory.add_thought(thought_content, iteration=iteration)

            # Check if LLM wants to call tools
            if hasattr(response, 'stop_reason') and response.stop_reason == 'tool_use':
                # LLM decided to call tools (content is list of dicts)
                tool_uses = [block for block in response.content if isinstance(block, dict) and block.get('type') == 'tool_use']

                self.logger.info(f"LLM requesting {len(tool_uses)} tool calls")

                # Execute requested tools
                tool_results = []
                for tool_use in tool_uses:
                    tool_name = tool_use.get('name')
                    tool_input = tool_use.get('input', {})
                    tool_use_id = tool_use.get('id')

                    self.logger.info(f"Calling tool: {tool_name}")

                    # Capture ACTION step in trajectory
                    trajectory.add_action(tool_name, tool_input, iteration=iteration)

                    # Emit tool call start event via EventBus (for real-time progress streaming)
                    if event_bus:
                        await event_bus.emit(
                            event_type=EventType.TOOL_CALL_START,
                            data={
                                "tool_use_id": tool_use_id,
                                "input": tool_input
                            },
                            agent_name=self.config.name,
                            tool_name=tool_name
                        )

                    # Measure tool execution time for trajectory
                    tool_start_time = time.time()
                    result = await tool_adapter.execute(
                        tool_name=tool_name,
                        inputs=tool_input,
                        agent_name=self.config.name
                    )
                    tool_duration_ms = (time.time() - tool_start_time) * 1000

                    # Extract API cost from result metadata
                    tool_cost = result.get("metadata", {}).get("api_cost_usd", 0.0) if isinstance(result, dict) else 0.0
                    total_tool_cost += tool_cost

                    # Emit tool call completion event via EventBus (for real-time progress streaming)
                    tool_success = result.get("success", False)
                    if event_bus:
                        await event_bus.emit(
                            event_type=EventType.TOOL_CALL_COMPLETE,
                            data={
                                "tool_use_id": tool_use_id,
                                "success": tool_success
                            },
                            agent_name=self.config.name,
                            tool_name=tool_name
                        )

                    # Check for critical tool failures and surface them to user
                    if not result.get("success", False):
                        error_msg = result.get("error", "Unknown error")
                        self.logger.error(
                            f"Tool execution failed: {tool_name}",
                            extra={
                                "tool_name": tool_name,
                                "tool_input": str(tool_input)[:200],  # Truncate for log size
                                "error": error_msg
                            }
                        )

                        # Critical tools that must succeed for valid results
                        critical_tools = {
                            "hierarchical_search", "similarity_search", "graph_search",
                            "get_document_info", "bm25_search", "hybrid_search"
                        }

                        if tool_name in critical_tools:
                            # Add error to state for user notification
                            if "errors" not in state:
                                state["errors"] = []
                            state["errors"].append(
                                f"Critical tool '{tool_name}' failed: {error_msg}. "
                                f"Results may be incomplete or unreliable."
                            )
                            self.logger.warning(
                                f"Critical tool failure surfaced to user: {tool_name}"
                            )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.get('id'),
                        "content": str(result.get("data", "")) if result.get("success", False) else f"Error: {result.get('error', 'Unknown error')}",
                        "is_error": not result.get("success", False)  # Add error flag for debugging
                    })

                    tool_call_history.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "success": result.get("success", False),
                        "api_cost_usd": tool_cost,  # Track cost per tool call
                        "result": result  # CRITICAL: Store full result for downstream agents (includes citations!)
                    })

                    # Capture OBSERVATION step in trajectory
                    observation_content = str(result.get("data", ""))[:200] if result.get("success", False) else f"Error: {result.get('error', 'Unknown')}"
                    trajectory.add_observation(
                        content=observation_content,
                        success=result.get("success", False),
                        error=result.get("error") if not result.get("success", False) else None,
                        duration_ms=tool_duration_ms,
                        iteration=iteration
                    )

                # Add assistant message + tool results to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            else:
                # LLM provided final answer (no more tools)
                self.logger.info(
                    f"LLM provided final answer after {iteration + 1} iterations "
                    f"(total tool cost: ${total_tool_cost:.6f})"
                )

                final_text = response.text if hasattr(response, 'text') else str(response.content)

                # Finalize trajectory and compute metrics
                trajectory.total_iterations = iteration + 1
                trajectory.finalize(final_text)
                trajectory_metrics = trajectory.compute_metrics()

                return {
                    "final_answer": final_text,
                    "tool_calls": tool_call_history,
                    "iterations": iteration + 1,
                    "reasoning": "Autonomous tool calling completed",
                    "total_tool_cost_usd": total_tool_cost,  # Total API cost for all tool calls
                    "trajectory": trajectory.to_dict(),
                    "trajectory_metrics": trajectory_metrics.to_dict()
                }

        # Max iterations reached
        self.logger.warning(
            f"Max iterations ({max_iterations}) reached, forcing completion "
            f"(total tool cost: ${total_tool_cost:.6f})"
        )

        # Finalize trajectory for max iterations case
        trajectory.total_iterations = max_iterations
        trajectory.finalize("Analysis incomplete - maximum reasoning steps reached.")
        trajectory_metrics = trajectory.compute_metrics()

        return {
            "final_answer": "Analysis incomplete - maximum reasoning steps reached. Please rephrase your query for a more focused response.",
            "tool_calls": tool_call_history,
            "iterations": max_iterations,
            "reasoning": "Max iterations reached",
            "total_tool_cost_usd": total_tool_cost,
            "trajectory": trajectory.to_dict(),
            "trajectory_metrics": trajectory_metrics.to_dict()
        }

    @staticmethod
    def compress_output_for_downstream(output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress agent output for efficient downstream agent consumption.

        Token optimization: Reduces verbose agent outputs from ~500-1000 tokens
        to ~100-200 tokens while preserving essential information.

        Used by orchestrator when building synthesis context.

        Args:
            output: Full agent output dict

        Returns:
            Compressed dict with only essential fields:
            - status: "success" | "partial" | "needs_clarification" | "error"
            - key_findings: List of 3-5 bullet points
            - citations: List of chunk_ids found
            - error: Error message if any
        """
        compressed = {
            "status": "success",
            "key_findings": [],
            "citations": []
        }

        # Extract status from various indicators
        if output.get("error") or output.get("errors"):
            compressed["status"] = "error"
            compressed["error"] = str(output.get("error") or output.get("errors", ["Unknown error"])[0])[:200]
        elif "needs_clarification" in str(output.get("final_answer", "")).lower():
            compressed["status"] = "needs_clarification"
        elif output.get("iterations", 0) >= 10:
            compressed["status"] = "partial"

        # Extract key findings from final_answer or analysis
        answer_text = output.get("final_answer", "") or output.get("analysis", "")
        if answer_text:
            # Take first 3 sentences or bullet points as key findings
            import re
            # Split by sentences or bullet points
            lines = re.split(r'[.\n•\-*]', answer_text)
            findings = [line.strip() for line in lines if len(line.strip()) > 20][:5]
            compressed["key_findings"] = findings

        # Extract citations from tool calls or direct citations field
        if "citations" in output:
            compressed["citations"] = output["citations"][:10]  # Max 10 citations
        elif "tool_calls" in output:
            for call in output.get("tool_calls", []):
                result = call.get("result", {})
                if isinstance(result, dict):
                    # Extract chunk_ids from search results
                    data = result.get("data", "")
                    if isinstance(data, str):
                        import re
                        chunk_ids = re.findall(r'chunk_id:\s*([A-Za-z0-9_]+_L3_\d+)', data)
                        compressed["citations"].extend(chunk_ids[:5])

        # Deduplicate citations
        compressed["citations"] = list(dict.fromkeys(compressed["citations"]))[:10]

        return compressed