"""Observability and monitoring for multi-agent framework.

Integrates with LangSmith for:
- Workflow tracing
- Agent execution tracking
- Cost monitoring
- Performance profiling
- Error tracking
- Evaluation and feedback

Components:
- LangSmithIntegration: Core LangSmith setup and feedback submission
- AgentTrajectory: Captures Thought → Action → Observation sequences
- LLMJudge: LLM-as-Judge for evaluating agent outputs
- LangSmithFeedback: Coordinates feedback submission from all sources
- AgentMetricsAggregator: Aggregates metrics across agents
"""

from .langsmith_integration import LangSmithIntegration, setup_langsmith
from .trajectory import (
    AgentTrajectory,
    TrajectoryStep,
    TrajectoryMetrics,
    TrajectoryCollector,
    StepType,
)
from .llm_judge import LLMJudge, EvaluationResult, JudgeScore, create_llm_judge
from .langsmith_feedback import LangSmithFeedback, create_feedback_coordinator
from .agent_metrics import (
    AgentMetricsAggregator,
    AgentExecution,
    AgentStats,
    WorkflowStats,
    create_metrics_aggregator,
)

__all__ = [
    # LangSmith integration
    "LangSmithIntegration",
    "setup_langsmith",
    # Trajectory
    "AgentTrajectory",
    "TrajectoryStep",
    "TrajectoryMetrics",
    "TrajectoryCollector",
    "StepType",
    # LLM Judge
    "LLMJudge",
    "EvaluationResult",
    "JudgeScore",
    "create_llm_judge",
    # Feedback
    "LangSmithFeedback",
    "create_feedback_coordinator",
    # Agent Metrics
    "AgentMetricsAggregator",
    "AgentExecution",
    "AgentStats",
    "WorkflowStats",
    "create_metrics_aggregator",
]
