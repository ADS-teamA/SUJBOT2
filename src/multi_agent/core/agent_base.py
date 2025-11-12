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

from pydantic import BaseModel
from langgraph.graph import StateGraph

from .event_bus import EventBus, EventType

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

    # ========================================================================
    # AUTONOMOUS AGENTIC PATTERN (CLAUDE.md CONSTRAINT #0)
    # ========================================================================
    # Methods for building truly autonomous agents where LLM decides tool calling

    def _build_agent_context(self, state: Dict[str, Any]) -> str:
        """
        Build context string from state for agent.

        Includes:
        - Original query
        - Previous agent outputs
        - Retrieved documents/chunks

        Args:
            state: Current workflow state

        Returns:
            Formatted context string
        """
        context_parts = []

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
                    # Summarize output (don't dump full chunks)
                    if isinstance(output, dict):
                        summary = {k: v for k, v in output.items() if k not in ['chunks', 'expanded_results']}
                        context_parts.append(str(summary)[:500])
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

        tool_adapter = get_tool_adapter()

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
        event_bus: Optional[EventBus] = state.get("_event_bus")
        if not event_bus:
            self.logger.warning("No EventBus in state - events will not be streamed")

        messages = [{"role": "user", "content": user_message}]
        tool_call_history = []
        total_tool_cost = 0.0  # Track cumulative API cost for all tool calls

        for iteration in range(max_iterations):
            self.logger.info(f"Autonomous loop iteration {iteration + 1}/{max_iterations}")

            # Call LLM with tools
            response = provider.create_message(
                messages=messages,
                tools=tool_schemas,
                system=system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

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

                    result = await tool_adapter.execute(
                        tool_name=tool_name,
                        inputs=tool_input,
                        agent_name=self.config.name
                    )

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
                        "api_cost_usd": tool_cost  # Track cost per tool call
                    })

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

                return {
                    "final_answer": final_text,
                    "tool_calls": tool_call_history,
                    "iterations": iteration + 1,
                    "reasoning": "Autonomous tool calling completed",
                    "total_tool_cost_usd": total_tool_cost  # Total API cost for all tool calls
                }

        # Max iterations reached
        self.logger.warning(
            f"Max iterations ({max_iterations}) reached, forcing completion "
            f"(total tool cost: ${total_tool_cost:.6f})"
        )

        return {
            "final_answer": "Analysis incomplete - maximum reasoning steps reached. Please rephrase your query for a more focused response.",
            "tool_calls": tool_call_history,
            "iterations": max_iterations,
            "reasoning": "Max iterations reached",
            "total_tool_cost_usd": total_tool_cost
        }
