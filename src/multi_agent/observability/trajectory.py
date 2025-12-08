"""
Trajectory Evaluation System for Multi-Agent Workflows.

Captures and analyzes agent decision sequences (Thought → Action → Observation)
for debugging, evaluation, and performance optimization.

Based on:
- LangChain/LangSmith trajectory analysis patterns
- ReAct (Reasoning + Acting) paradigm

Usage:
    # Create trajectory at start of agent execution
    trajectory = AgentTrajectory(agent_name="extractor", query="...")

    # Capture steps during execution
    trajectory.add_thought("Analyzing query for entity extraction...")
    trajectory.add_action("search", {"query": "...", "k": 5})
    trajectory.add_observation(result, success=True)

    # Finalize and compute metrics
    trajectory.finalize("Final answer here")
    metrics = trajectory.compute_metrics()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Type of step in agent trajectory."""

    THOUGHT = "thought"           # LLM reasoning/planning
    ACTION = "action"             # Tool call initiated
    OBSERVATION = "observation"   # Tool result received
    FINAL_ANSWER = "final_answer" # Agent's final response


@dataclass
class TrajectoryStep:
    """Single step in agent trajectory."""

    step_type: StepType
    timestamp: datetime
    agent_name: str
    content: str                              # LLM text or tool result summary

    # For ACTION steps
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None

    # For OBSERVATION steps
    success: bool = True
    error: Optional[str] = None

    # Timing
    duration_ms: float = 0.0
    iteration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "step_type": self.step_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "content": self.content[:500] if self.content else "",  # Truncate for storage
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "iteration": self.iteration,
        }


@dataclass
class AgentTrajectory:
    """Complete trajectory for a single agent execution."""

    agent_name: str
    query: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_iterations: int = 0
    final_answer: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    # Computed during execution
    tool_call_count: int = 0
    unique_tools_used: Set[str] = field(default_factory=set)
    failed_tool_calls: int = 0
    error_recovery_count: int = 0  # Tool failed then succeeded later

    # Internal state
    _last_failed_tool: Optional[str] = field(default=None, repr=False)

    def add_thought(self, content: str, iteration: int = 0) -> TrajectoryStep:
        """
        Add a THOUGHT step (LLM reasoning).

        Args:
            content: The LLM's reasoning text
            iteration: Current iteration number

        Returns:
            The created TrajectoryStep
        """
        step = TrajectoryStep(
            step_type=StepType.THOUGHT,
            timestamp=datetime.now(),
            agent_name=self.agent_name,
            content=content,
            iteration=iteration,
        )
        self.steps.append(step)
        return step

    def add_action(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        iteration: int = 0,
    ) -> TrajectoryStep:
        """
        Add an ACTION step (tool call initiated).

        Args:
            tool_name: Name of tool being called
            tool_input: Input arguments to tool
            iteration: Current iteration number

        Returns:
            The created TrajectoryStep
        """
        step = TrajectoryStep(
            step_type=StepType.ACTION,
            timestamp=datetime.now(),
            agent_name=self.agent_name,
            content=f"Calling tool: {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input,
            iteration=iteration,
        )
        self.steps.append(step)
        self.tool_call_count += 1
        self.unique_tools_used.add(tool_name)
        return step

    def add_observation(
        self,
        content: str,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
        iteration: int = 0,
    ) -> TrajectoryStep:
        """
        Add an OBSERVATION step (tool result).

        Args:
            content: Result summary or error message
            success: Whether tool execution succeeded
            error: Error message if failed
            duration_ms: Tool execution time
            iteration: Current iteration number

        Returns:
            The created TrajectoryStep
        """
        step = TrajectoryStep(
            step_type=StepType.OBSERVATION,
            timestamp=datetime.now(),
            agent_name=self.agent_name,
            content=content,
            success=success,
            error=error,
            duration_ms=duration_ms,
            iteration=iteration,
        )
        self.steps.append(step)

        if not success:
            self.failed_tool_calls += 1
            # Track for error recovery detection
            if len(self.steps) >= 2:
                prev_action = self.steps[-2]
                if prev_action.step_type == StepType.ACTION:
                    self._last_failed_tool = prev_action.tool_name
        elif self._last_failed_tool:
            # Previous tool failed but this one succeeded → recovery
            self.error_recovery_count += 1
            self._last_failed_tool = None

        return step

    def finalize(self, final_answer: Optional[str] = None) -> None:
        """
        Finalize the trajectory after agent execution completes.

        Args:
            final_answer: The agent's final answer
        """
        self.ended_at = datetime.now()
        self.final_answer = final_answer

        if final_answer:
            step = TrajectoryStep(
                step_type=StepType.FINAL_ANSWER,
                timestamp=self.ended_at,
                agent_name=self.agent_name,
                content=final_answer[:1000],  # Truncate for storage
                iteration=self.total_iterations,
            )
            self.steps.append(step)

    def compute_metrics(self) -> "TrajectoryMetrics":
        """
        Compute metrics from trajectory.

        Returns:
            TrajectoryMetrics with computed values
        """
        action_count = sum(1 for s in self.steps if s.step_type == StepType.ACTION)
        observation_count = sum(1 for s in self.steps if s.step_type == StepType.OBSERVATION)
        thought_count = sum(1 for s in self.steps if s.step_type == StepType.THOUGHT)

        # Calculate durations
        total_duration_ms = 0.0
        if self.started_at and self.ended_at:
            total_duration_ms = (self.ended_at - self.started_at).total_seconds() * 1000

        observation_durations = [s.duration_ms for s in self.steps if s.step_type == StepType.OBSERVATION]
        avg_step_duration_ms = sum(observation_durations) / max(len(observation_durations), 1)

        # Tool success rate
        successful_tools = observation_count - self.failed_tool_calls
        tool_success_rate = successful_tools / max(observation_count, 1)

        # Tool repetition rate (same tool called multiple times)
        tool_counts: Dict[str, int] = {}
        for s in self.steps:
            if s.step_type == StepType.ACTION and s.tool_name:
                tool_counts[s.tool_name] = tool_counts.get(s.tool_name, 0) + 1
        repeated_calls = sum(c - 1 for c in tool_counts.values() if c > 1)
        tool_repetition_rate = repeated_calls / max(action_count, 1)

        # Efficiency score: 1.0 = optimal, lower = more steps than needed
        # Heuristic: optimal is ~3 steps per iteration (thought, action, observation)
        expected_steps = self.total_iterations * 3 if self.total_iterations > 0 else 3
        actual_steps = len(self.steps)
        efficiency_score = min(1.0, expected_steps / max(actual_steps, 1))

        # Error recovery rate
        error_recovery_rate = self.error_recovery_count / max(self.failed_tool_calls, 1)

        return TrajectoryMetrics(
            total_steps=len(self.steps),
            action_count=action_count,
            observation_count=observation_count,
            thought_count=thought_count,
            tool_success_rate=tool_success_rate,
            tool_repetition_rate=tool_repetition_rate,
            avg_step_duration_ms=avg_step_duration_ms,
            total_duration_ms=total_duration_ms,
            efficiency_score=efficiency_score,
            error_recovery_rate=error_recovery_rate,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "query": self.query[:500] if self.query else "",
            "steps": [s.to_dict() for s in self.steps],
            "total_iterations": self.total_iterations,
            "final_answer": self.final_answer[:1000] if self.final_answer else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "tool_call_count": self.tool_call_count,
            "unique_tools_used": list(self.unique_tools_used),
            "failed_tool_calls": self.failed_tool_calls,
            "error_recovery_count": self.error_recovery_count,
        }


@dataclass
class TrajectoryMetrics:
    """Aggregated metrics from trajectory analysis."""

    total_steps: int
    action_count: int
    observation_count: int
    thought_count: int

    tool_success_rate: float      # successful_tools / total_tools
    tool_repetition_rate: float   # repeated calls to same tool
    avg_step_duration_ms: float
    total_duration_ms: float

    efficiency_score: float       # 1.0 / (steps_to_answer / optimal_steps)
    error_recovery_rate: float    # recoveries / failures

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_steps": self.total_steps,
            "action_count": self.action_count,
            "observation_count": self.observation_count,
            "thought_count": self.thought_count,
            "tool_success_rate": round(self.tool_success_rate, 3),
            "tool_repetition_rate": round(self.tool_repetition_rate, 3),
            "avg_step_duration_ms": round(self.avg_step_duration_ms, 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "efficiency_score": round(self.efficiency_score, 3),
            "error_recovery_rate": round(self.error_recovery_rate, 3),
        }

    def to_langsmith_feedback(self) -> Dict[str, float]:
        """
        Convert metrics to LangSmith feedback format.

        Returns:
            Dict mapping feedback keys to scores (0.0 to 1.0)
        """
        return {
            "trajectory_tool_success_rate": self.tool_success_rate,
            "trajectory_efficiency": self.efficiency_score,
            "trajectory_error_recovery": self.error_recovery_rate,
        }


class TrajectoryCollector:
    """
    Collects trajectories across multiple agents in a workflow.

    Useful for analyzing entire multi-agent conversations.
    """

    def __init__(self, workflow_name: str):
        """
        Initialize collector.

        Args:
            workflow_name: Name of the workflow being traced
        """
        self.workflow_name = workflow_name
        self.trajectories: List[AgentTrajectory] = []
        self.started_at = datetime.now()
        self.ended_at: Optional[datetime] = None

    def add_trajectory(self, trajectory: AgentTrajectory) -> None:
        """Add a completed agent trajectory."""
        self.trajectories.append(trajectory)

    def finalize(self) -> None:
        """Mark collection as complete."""
        self.ended_at = datetime.now()

    def get_workflow_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics across all trajectories.

        Returns:
            Dict with workflow-level metrics
        """
        if not self.trajectories:
            return {"error": "No trajectories collected"}

        # Compute metrics for each trajectory
        all_metrics = [t.compute_metrics() for t in self.trajectories]

        # Aggregate
        total_steps = sum(m.total_steps for m in all_metrics)
        total_duration_ms = sum(m.total_duration_ms for m in all_metrics)
        avg_tool_success = sum(m.tool_success_rate for m in all_metrics) / len(all_metrics)
        avg_efficiency = sum(m.efficiency_score for m in all_metrics) / len(all_metrics)

        # Per-agent breakdown
        agent_metrics = {
            t.agent_name: t.compute_metrics().to_dict()
            for t in self.trajectories
        }

        return {
            "workflow_name": self.workflow_name,
            "agent_count": len(self.trajectories),
            "total_steps": total_steps,
            "total_duration_ms": round(total_duration_ms, 2),
            "avg_tool_success_rate": round(avg_tool_success, 3),
            "avg_efficiency_score": round(avg_efficiency, 3),
            "agent_metrics": agent_metrics,
        }
