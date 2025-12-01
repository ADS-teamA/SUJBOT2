"""
Agent Metrics Aggregator.

Collects and aggregates metrics across all agents in a workflow:
- Per-agent performance (latency, token usage, success rate)
- Cross-agent patterns (handoff efficiency, bottlenecks)
- Workflow-level statistics

Usage:
    aggregator = AgentMetricsAggregator()

    # Record agent execution
    aggregator.record_agent_execution(
        agent_name="extractor",
        duration_ms=1500,
        tokens_used=2000,
        success=True,
    )

    # Get aggregated stats
    stats = aggregator.get_workflow_stats()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentExecution:
    """Record of a single agent execution."""

    agent_name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class AgentStats:
    """Aggregated statistics for a single agent."""

    agent_name: str
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0

    def avg_duration_ms(self) -> float:
        """Calculate average execution duration."""
        if self.execution_count == 0:
            return 0.0
        return self.total_duration_ms / self.execution_count

    def avg_tokens(self) -> float:
        """Calculate average tokens per execution."""
        if self.execution_count == 0:
            return 0.0
        return self.total_tokens / self.execution_count

    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.execution_count == 0:
            return 1.0
        return self.success_count / self.execution_count

    def tool_success_rate(self) -> float:
        """Calculate tool call success rate."""
        if self.total_tool_calls == 0:
            return 1.0
        return self.successful_tool_calls / self.total_tool_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate(), 3),
            "avg_duration_ms": round(self.avg_duration_ms(), 2),
            "avg_tokens": round(self.avg_tokens(), 1),
            "total_tool_calls": self.total_tool_calls,
            "tool_success_rate": round(self.tool_success_rate(), 3),
        }


@dataclass
class WorkflowStats:
    """Aggregated statistics for entire workflow."""

    workflow_name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    total_duration_ms: float = 0.0
    agent_count: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    agent_stats: Dict[str, AgentStats] = field(default_factory=dict)
    execution_sequence: List[str] = field(default_factory=list)
    bottleneck_agent: Optional[str] = None  # Agent with highest avg latency

    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_name": self.workflow_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "agent_count": self.agent_count,
            "total_executions": self.total_executions,
            "success_rate": round(self.success_rate(), 3),
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "bottleneck_agent": self.bottleneck_agent,
            "execution_sequence": self.execution_sequence,
            "agent_stats": {
                name: stats.to_dict()
                for name, stats in self.agent_stats.items()
            },
        }

    def to_feedback_dict(self) -> Dict[str, float]:
        """
        Convert to LangSmith feedback format.

        Returns:
            Dict mapping feedback keys to scores (0.0 to 1.0)
        """
        return {
            "workflow_success_rate": self.success_rate(),
            "workflow_efficiency": self._calculate_efficiency(),
        }

    def _calculate_efficiency(self) -> float:
        """
        Calculate workflow efficiency score.

        Based on:
        - Success rate
        - Agent utilization (less re-executions = better)
        - Tool success rate
        """
        if self.total_executions == 0:
            return 1.0

        # Factor 1: Success rate
        success_factor = self.success_rate()

        # Factor 2: Agent efficiency (fewer executions per unique agent = better)
        unique_agents = len(self.agent_stats)
        if unique_agents > 0:
            efficiency_factor = min(1.0, unique_agents / self.total_executions)
        else:
            efficiency_factor = 1.0

        # Weighted average
        return 0.7 * success_factor + 0.3 * efficiency_factor


class AgentMetricsAggregator:
    """
    Aggregates metrics across all agents in a workflow.

    Provides workflow-level insights and identifies bottlenecks.
    """

    def __init__(self, workflow_name: str = "default"):
        """
        Initialize aggregator.

        Args:
            workflow_name: Name of the workflow being tracked
        """
        self.workflow_name = workflow_name
        self.started_at = datetime.now()
        self.ended_at: Optional[datetime] = None

        self.executions: List[AgentExecution] = []
        self.agent_stats: Dict[str, AgentStats] = {}
        self.execution_sequence: List[str] = []

        logger.debug(f"AgentMetricsAggregator initialized for workflow: {workflow_name}")

    def record_agent_execution(
        self,
        agent_name: str,
        duration_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tool_calls: int = 0,
        successful_tool_calls: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> AgentExecution:
        """
        Record an agent execution.

        Args:
            agent_name: Name of agent
            duration_ms: Execution duration in milliseconds
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            tool_calls: Number of tool calls made
            successful_tool_calls: Number of successful tool calls
            success: Whether execution was successful
            error: Error message if failed

        Returns:
            AgentExecution record
        """
        total_tokens = prompt_tokens + completion_tokens
        ended_at = datetime.now()
        started_at = datetime.fromtimestamp(
            ended_at.timestamp() - duration_ms / 1000
        )

        execution = AgentExecution(
            agent_name=agent_name,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            successful_tool_calls=successful_tool_calls,
            success=success,
            error=error,
        )

        self.executions.append(execution)
        self.execution_sequence.append(agent_name)

        # Update agent stats
        self._update_agent_stats(execution)

        logger.debug(
            f"Recorded execution: {agent_name}, "
            f"duration={duration_ms:.0f}ms, success={success}"
        )

        return execution

    def _update_agent_stats(self, execution: AgentExecution) -> None:
        """Update per-agent statistics."""
        name = execution.agent_name

        if name not in self.agent_stats:
            self.agent_stats[name] = AgentStats(agent_name=name)

        stats = self.agent_stats[name]
        stats.execution_count += 1
        stats.total_duration_ms += execution.duration_ms
        stats.total_tokens += execution.total_tokens
        stats.total_tool_calls += execution.tool_calls
        stats.successful_tool_calls += execution.successful_tool_calls

        if execution.success:
            stats.success_count += 1
        else:
            stats.failure_count += 1

    def finalize(self) -> None:
        """Mark workflow as complete."""
        self.ended_at = datetime.now()

    def get_workflow_stats(self) -> WorkflowStats:
        """
        Get aggregated workflow statistics.

        Returns:
            WorkflowStats with all metrics
        """
        # Calculate total duration
        if self.ended_at:
            total_duration_ms = (
                self.ended_at - self.started_at
            ).total_seconds() * 1000
        elif self.executions:
            total_duration_ms = sum(e.duration_ms for e in self.executions)
        else:
            total_duration_ms = 0.0

        # Find bottleneck (agent with highest avg duration)
        bottleneck_agent = None
        max_avg_duration = 0.0
        for name, stats in self.agent_stats.items():
            avg_duration = stats.avg_duration_ms()
            if avg_duration > max_avg_duration:
                max_avg_duration = avg_duration
                bottleneck_agent = name

        return WorkflowStats(
            workflow_name=self.workflow_name,
            started_at=self.started_at,
            ended_at=self.ended_at,
            total_duration_ms=total_duration_ms,
            agent_count=len(self.agent_stats),
            total_executions=len(self.executions),
            successful_executions=sum(1 for e in self.executions if e.success),
            total_tokens=sum(e.total_tokens for e in self.executions),
            total_tool_calls=sum(e.tool_calls for e in self.executions),
            agent_stats=self.agent_stats.copy(),
            execution_sequence=self.execution_sequence.copy(),
            bottleneck_agent=bottleneck_agent,
        )

    def get_agent_stats(self, agent_name: str) -> Optional[AgentStats]:
        """
        Get statistics for a specific agent.

        Args:
            agent_name: Name of agent

        Returns:
            AgentStats or None if agent not found
        """
        return self.agent_stats.get(agent_name)

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """
        Analyze bottlenecks in the workflow.

        Returns:
            Dict with bottleneck analysis
        """
        if not self.agent_stats:
            return {"bottleneck": None, "recommendation": "No agent executions recorded"}

        # Sort agents by average duration
        sorted_agents = sorted(
            self.agent_stats.items(),
            key=lambda x: x[1].avg_duration_ms(),
            reverse=True,
        )

        bottleneck = sorted_agents[0] if sorted_agents else None

        analysis = {
            "bottleneck_agent": bottleneck[0] if bottleneck else None,
            "bottleneck_avg_duration_ms": bottleneck[1].avg_duration_ms() if bottleneck else 0,
            "agent_ranking": [
                {
                    "agent": name,
                    "avg_duration_ms": stats.avg_duration_ms(),
                    "execution_count": stats.execution_count,
                }
                for name, stats in sorted_agents
            ],
        }

        # Generate recommendation
        if bottleneck and bottleneck[1].avg_duration_ms() > 5000:
            analysis["recommendation"] = (
                f"Agent '{bottleneck[0]}' is a bottleneck with "
                f"{bottleneck[1].avg_duration_ms():.0f}ms avg duration. "
                "Consider optimizing tool calls or reducing context size."
            )
        else:
            analysis["recommendation"] = "No significant bottlenecks detected."

        return analysis

    def reset(self) -> None:
        """Reset all metrics (for testing or between runs)."""
        self.started_at = datetime.now()
        self.ended_at = None
        self.executions.clear()
        self.agent_stats.clear()
        self.execution_sequence.clear()


def create_metrics_aggregator(workflow_name: str = "default") -> AgentMetricsAggregator:
    """
    Create a new metrics aggregator.

    Args:
        workflow_name: Name of workflow to track

    Returns:
        AgentMetricsAggregator instance
    """
    return AgentMetricsAggregator(workflow_name)
