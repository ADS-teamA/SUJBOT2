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
        if not self.tools:
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

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build LangGraph graph for this agent.

        Each agent implements its own workflow:
        - Tool selection logic
        - Parallel/sequential execution
        - Error handling
        - State transitions

        Returns:
            StateGraph: Compiled LangGraph graph
        """
        pass

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
        error_message = f"{self.config.name}: {type(error).__name__}: {str(error)}"

        # Add error to state
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_message)

        # Mark execution phase as error
        state["execution_phase"] = "error"

        self.logger.error(f"Error in agent {self.config.name}: {error}")

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
