"""
Comprehensive tests for the evaluation infrastructure.

Tests:
1. Trajectory capture and metrics
2. Tool usage metrics and hallucination detection
3. LLM Judge evaluation
4. LangSmith feedback coordination
5. Agent metrics aggregation
6. Exception types
7. Integration tests
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# =============================================================================
# Trajectory Tests
# =============================================================================


class TestTrajectory:
    """Tests for trajectory capture and metrics."""

    def test_trajectory_creation(self):
        """Test basic trajectory creation."""
        from src.multi_agent.observability import AgentTrajectory, StepType

        trajectory = AgentTrajectory(agent_name="extractor", query="What is compliance?")

        assert trajectory.agent_name == "extractor"
        assert trajectory.query == "What is compliance?"
        assert len(trajectory.steps) == 0
        assert trajectory.tool_call_count == 0

    def test_trajectory_add_thought(self):
        """Test adding thought steps."""
        from src.multi_agent.observability import AgentTrajectory, StepType

        trajectory = AgentTrajectory(agent_name="extractor", query="test")
        step = trajectory.add_thought("Analyzing query for key terms...", iteration=1)

        assert step.step_type == StepType.THOUGHT
        assert step.agent_name == "extractor"
        assert "Analyzing" in step.content
        assert step.iteration == 1
        assert len(trajectory.steps) == 1

    def test_trajectory_add_action(self):
        """Test adding action steps."""
        from src.multi_agent.observability import AgentTrajectory, StepType

        trajectory = AgentTrajectory(agent_name="extractor", query="test")
        step = trajectory.add_action(
            tool_name="search",
            tool_input={"query": "compliance requirements", "k": 5},
            iteration=1,
        )

        assert step.step_type == StepType.ACTION
        assert step.tool_name == "search"
        assert step.tool_input["query"] == "compliance requirements"
        assert trajectory.tool_call_count == 1
        assert "search" in trajectory.unique_tools_used

    def test_trajectory_add_observation(self):
        """Test adding observation steps."""
        from src.multi_agent.observability import AgentTrajectory, StepType

        trajectory = AgentTrajectory(agent_name="extractor", query="test")
        trajectory.add_action("search", {"query": "test"}, iteration=1)
        step = trajectory.add_observation(
            content="Found 5 relevant documents",
            success=True,
            duration_ms=150.5,
            iteration=1,
        )

        assert step.step_type == StepType.OBSERVATION
        assert step.success is True
        assert step.duration_ms == 150.5
        assert trajectory.failed_tool_calls == 0

    def test_trajectory_failed_observation(self):
        """Test recording failed tool calls."""
        from src.multi_agent.observability import AgentTrajectory

        trajectory = AgentTrajectory(agent_name="extractor", query="test")
        trajectory.add_action("search", {"query": "test"}, iteration=1)
        trajectory.add_observation(
            content="Connection timeout",
            success=False,
            error="TimeoutError",
            iteration=1,
        )

        assert trajectory.failed_tool_calls == 1

    def test_trajectory_error_recovery_detection(self):
        """Test error recovery detection."""
        from src.multi_agent.observability import AgentTrajectory

        trajectory = AgentTrajectory(agent_name="extractor", query="test")

        # First attempt fails
        trajectory.add_action("search", {"query": "test"}, iteration=1)
        trajectory.add_observation("Error", success=False, error="Timeout", iteration=1)

        # Second attempt succeeds
        trajectory.add_action("search", {"query": "test"}, iteration=2)
        trajectory.add_observation("Success", success=True, iteration=2)

        assert trajectory.error_recovery_count == 1

    def test_trajectory_finalize(self):
        """Test trajectory finalization."""
        from src.multi_agent.observability import AgentTrajectory, StepType

        trajectory = AgentTrajectory(agent_name="extractor", query="test")
        trajectory.add_thought("Thinking...", iteration=1)
        trajectory.add_action("search", {"query": "test"}, iteration=1)
        trajectory.add_observation("Found results", success=True, iteration=1)
        trajectory.finalize("Here is the final answer based on the documents.")

        assert trajectory.final_answer is not None
        assert trajectory.ended_at is not None
        # Final answer step should be added
        assert any(s.step_type == StepType.FINAL_ANSWER for s in trajectory.steps)

    def test_trajectory_compute_metrics(self):
        """Test trajectory metrics computation."""
        from src.multi_agent.observability import AgentTrajectory

        trajectory = AgentTrajectory(agent_name="extractor", query="test")
        trajectory.total_iterations = 2

        # Iteration 1
        trajectory.add_thought("Thinking...", iteration=1)
        trajectory.add_action("search", {"query": "test"}, iteration=1)
        trajectory.add_observation("Results", success=True, duration_ms=100, iteration=1)

        # Iteration 2
        trajectory.add_thought("Analyzing...", iteration=2)
        trajectory.add_action("analyze", {"data": "..."}, iteration=2)
        trajectory.add_observation("Analysis done", success=True, duration_ms=200, iteration=2)

        trajectory.finalize("Final answer")
        metrics = trajectory.compute_metrics()

        assert metrics.total_steps == 7  # 2 thoughts + 2 actions + 2 observations + 1 final
        assert metrics.action_count == 2
        assert metrics.observation_count == 2
        assert metrics.thought_count == 2
        assert metrics.tool_success_rate == 1.0
        assert metrics.avg_step_duration_ms == 150.0  # (100 + 200) / 2

    def test_trajectory_to_dict(self):
        """Test trajectory serialization."""
        from src.multi_agent.observability import AgentTrajectory

        trajectory = AgentTrajectory(agent_name="extractor", query="test query")
        trajectory.add_thought("Thinking...", iteration=1)
        trajectory.finalize("Answer")

        data = trajectory.to_dict()

        assert data["agent_name"] == "extractor"
        assert data["query"] == "test query"
        assert len(data["steps"]) == 2
        assert data["final_answer"] == "Answer"

    def test_trajectory_metrics_to_langsmith_feedback(self):
        """Test metrics conversion to LangSmith feedback format."""
        from src.multi_agent.observability import AgentTrajectory

        trajectory = AgentTrajectory(agent_name="extractor", query="test")
        trajectory.add_action("search", {"query": "test"}, iteration=1)
        trajectory.add_observation("Results", success=True, iteration=1)
        trajectory.finalize("Answer")

        metrics = trajectory.compute_metrics()
        feedback = metrics.to_langsmith_feedback()

        assert "trajectory_tool_success_rate" in feedback
        assert "trajectory_efficiency" in feedback
        assert "trajectory_error_recovery" in feedback
        assert 0.0 <= feedback["trajectory_tool_success_rate"] <= 1.0


class TestTrajectoryCollector:
    """Tests for trajectory collector."""

    def test_collector_add_trajectories(self):
        """Test adding multiple trajectories to collector."""
        from src.multi_agent.observability import AgentTrajectory, TrajectoryCollector

        collector = TrajectoryCollector(workflow_name="test_workflow")

        # Add extractor trajectory
        t1 = AgentTrajectory(agent_name="extractor", query="test")
        t1.add_action("search", {}, iteration=1)
        t1.add_observation("OK", success=True, duration_ms=100, iteration=1)
        t1.finalize("Result 1")
        collector.add_trajectory(t1)

        # Add classifier trajectory
        t2 = AgentTrajectory(agent_name="classifier", query="test")
        t2.add_action("classify", {}, iteration=1)
        t2.add_observation("OK", success=True, duration_ms=50, iteration=1)
        t2.finalize("Result 2")
        collector.add_trajectory(t2)

        assert len(collector.trajectories) == 2

    def test_collector_workflow_metrics(self):
        """Test workflow-level metrics aggregation."""
        from src.multi_agent.observability import AgentTrajectory, TrajectoryCollector

        collector = TrajectoryCollector(workflow_name="test_workflow")

        t1 = AgentTrajectory(agent_name="extractor", query="test")
        t1.add_action("search", {}, iteration=1)
        t1.add_observation("OK", success=True, duration_ms=100, iteration=1)
        t1.finalize("Result")
        collector.add_trajectory(t1)

        collector.finalize()
        metrics = collector.get_workflow_metrics()

        assert metrics["workflow_name"] == "test_workflow"
        assert metrics["agent_count"] == 1
        assert "agent_metrics" in metrics
        assert "extractor" in metrics["agent_metrics"]


# =============================================================================
# Tool Usage Metrics Tests
# =============================================================================


class TestToolUsageMetrics:
    """Tests for tool usage metrics and hallucination tracking."""

    def test_tool_metrics_creation(self):
        """Test basic metrics creation."""
        from src.multi_agent.core.state import ToolUsageMetrics

        metrics = ToolUsageMetrics()

        assert metrics.total_calls == 0
        assert metrics.hallucination_rate() == 0.0
        assert metrics.success_rate() == 1.0

    def test_tool_metrics_record_execution(self):
        """Test recording tool executions."""
        from src.multi_agent.core.state import ToolUsageMetrics, ToolExecution

        metrics = ToolUsageMetrics()

        execution = ToolExecution(
            tool_name="search",
            agent_name="extractor",
            timestamp=datetime.now(),
            duration_ms=150.0,
            input_tokens=50,
            output_tokens=100,
            success=True,
            error=None,
            result_summary="Found 5 results",
            was_hallucinated=False,
            validation_error=None,
        )

        metrics.record_execution(execution)

        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.hallucination_rate() == 0.0

    def test_tool_metrics_hallucination_tracking(self):
        """Test hallucination tracking."""
        from src.multi_agent.core.state import ToolUsageMetrics, ToolExecution

        metrics = ToolUsageMetrics()

        # Record hallucinated call
        hallucinated = ToolExecution(
            tool_name="nonexistent_tool",
            agent_name="extractor",
            timestamp=datetime.now(),
            duration_ms=10.0,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error="Tool not found",
            result_summary="Error",
            was_hallucinated=True,
            validation_error=None,
        )
        metrics.record_execution(hallucinated)

        # Record successful call
        success = ToolExecution(
            tool_name="search",
            agent_name="extractor",
            timestamp=datetime.now(),
            duration_ms=100.0,
            input_tokens=50,
            output_tokens=100,
            success=True,
            error=None,
            result_summary="OK",
            was_hallucinated=False,
            validation_error=None,
        )
        metrics.record_execution(success)

        assert metrics.total_calls == 2
        assert metrics.hallucinated_calls == 1
        assert metrics.hallucination_rate() == 0.5

    def test_tool_metrics_validation_errors(self):
        """Test validation error tracking."""
        from src.multi_agent.core.state import ToolUsageMetrics, ToolExecution

        metrics = ToolUsageMetrics()

        validation_error = ToolExecution(
            tool_name="search",
            agent_name="extractor",
            timestamp=datetime.now(),
            duration_ms=5.0,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error="Invalid input",
            result_summary="Error",
            was_hallucinated=False,
            validation_error="query: field required",
        )
        metrics.record_execution(validation_error)

        assert metrics.validation_errors == 1
        assert metrics.error_rate() == 1.0

    def test_tool_metrics_per_tool_stats(self):
        """Test per-tool statistics."""
        from src.multi_agent.core.state import ToolUsageMetrics, ToolExecution

        metrics = ToolUsageMetrics()

        # Multiple calls to same tool
        for i in range(3):
            execution = ToolExecution(
                tool_name="search",
                agent_name="extractor",
                timestamp=datetime.now(),
                duration_ms=100.0 + i * 10,
                input_tokens=50,
                output_tokens=100,
                success=True,
                error=None,
                result_summary="OK",
                was_hallucinated=False,
                validation_error=None,
            )
            metrics.record_execution(execution)

        assert "search" in metrics.tool_stats
        assert metrics.tool_stats["search"].call_count == 3
        assert metrics.tool_stats["search"].success_rate() == 1.0

    def test_tool_metrics_to_feedback_dict(self):
        """Test conversion to LangSmith feedback format."""
        from src.multi_agent.core.state import ToolUsageMetrics, ToolExecution

        metrics = ToolUsageMetrics()

        execution = ToolExecution(
            tool_name="search",
            agent_name="extractor",
            timestamp=datetime.now(),
            duration_ms=100.0,
            input_tokens=50,
            output_tokens=100,
            success=True,
            error=None,
            result_summary="OK",
            was_hallucinated=False,
            validation_error=None,
        )
        metrics.record_execution(execution)

        feedback = metrics.to_feedback_dict()

        assert "tool_success_rate" in feedback
        assert "tool_hallucination_rate" in feedback
        assert "tool_error_rate" in feedback


# =============================================================================
# Tool Adapter Hallucination Detection Tests
# =============================================================================


class TestToolAdapterHallucinationDetection:
    """Tests for hallucination detection in tool adapter."""

    def test_hallucination_detection(self):
        """Test that calling non-existent tool is detected as hallucination."""
        import asyncio
        from src.multi_agent.tools.adapter import ToolAdapter, ToolErrorType

        adapter = ToolAdapter()

        async def run():
            return await adapter.execute(
                tool_name="completely_fake_tool_xyz",
                inputs={"query": "test"},
                agent_name="test_agent",
            )

        result = asyncio.run(run())

        assert result["success"] is False
        assert result["error_type"] == ToolErrorType.HALLUCINATION
        assert result["metadata"]["was_hallucinated"] is True
        assert "does not exist" in result["error"]

    def test_hallucination_metrics_updated(self):
        """Test that hallucination updates metrics."""
        import asyncio
        from src.multi_agent.tools.adapter import ToolAdapter

        adapter = ToolAdapter()
        adapter.clear_history()

        async def run():
            await adapter.execute(
                tool_name="fake_tool",
                inputs={},
                agent_name="test_agent",
            )

        asyncio.run(run())

        metrics = adapter.get_usage_metrics()
        assert metrics.hallucinated_calls == 1
        assert metrics.hallucination_rate() == 1.0

    def test_validation_error_detection(self):
        """Test that invalid inputs are detected."""
        import asyncio
        from src.multi_agent.tools.adapter import ToolAdapter, ToolErrorType

        adapter = ToolAdapter()

        # 'search' tool exists but may require specific inputs
        # We'll test with a known tool that should exist
        available_tools = adapter.get_available_tools()

        if "search" in available_tools:
            async def run():
                # Try with invalid input type
                return await adapter.execute(
                    tool_name="search",
                    inputs={"invalid_param": "should_fail"},
                    agent_name="test_agent",
                )

            result = asyncio.run(run())

            # If validation fails, check error type
            if not result["success"]:
                assert result["error_type"] in [
                    ToolErrorType.VALIDATION,
                    ToolErrorType.EXECUTION,
                ]

    def test_execution_stats_include_hallucinations(self):
        """Test that execution stats include hallucination info."""
        import asyncio
        from src.multi_agent.tools.adapter import ToolAdapter

        adapter = ToolAdapter()
        adapter.clear_history()

        async def run():
            # Create some hallucinations
            await adapter.execute("fake1", {}, "agent1")
            await adapter.execute("fake2", {}, "agent2")

        asyncio.run(run())

        stats = adapter.get_execution_stats()

        assert stats["hallucinated"] == 2
        assert stats["hallucination_rate"] == 1.0


# =============================================================================
# LLM Judge Tests
# =============================================================================


class TestLLMJudge:
    """Tests for LLM-as-Judge evaluation."""

    def test_judge_creation_disabled(self):
        """Test judge creation when disabled."""
        from src.multi_agent.observability import create_llm_judge

        config = {"enable_llm_judge": False}
        judge = create_llm_judge(config)

        assert judge is None

    def test_judge_creation_enabled(self):
        """Test judge creation when enabled."""
        from src.multi_agent.observability import LLMJudge

        config = {
            "enable_llm_judge": True,
            "llm_judge_model": "claude-haiku-4-5",
            "llm_judge_criteria": ["relevance", "coherence"],
        }
        judge = LLMJudge(config)

        assert judge.enabled is True
        assert judge.model == "claude-haiku-4-5"
        assert "relevance" in judge.default_criteria

    def test_evaluation_result_to_feedback(self):
        """Test EvaluationResult conversion to feedback."""
        from src.multi_agent.observability.llm_judge import (
            EvaluationResult,
            JudgeScore,
            EvaluationCriteria,
        )

        result = EvaluationResult(
            scores={
                EvaluationCriteria.RELEVANCE: JudgeScore(
                    criterion=EvaluationCriteria.RELEVANCE,
                    score=0.9,
                    reasoning="Very relevant",
                ),
                EvaluationCriteria.COHERENCE: JudgeScore(
                    criterion=EvaluationCriteria.COHERENCE,
                    score=0.8,
                    reasoning="Well structured",
                ),
            },
            overall_score=0.85,
            model_used="claude-haiku-4-5",
        )

        feedback = result.to_feedback_dict()

        assert "judge_relevance" in feedback
        assert "judge_coherence" in feedback
        assert "judge_overall" in feedback
        assert feedback["judge_overall"] == 0.85

    def test_evaluation_result_to_dict(self):
        """Test EvaluationResult serialization."""
        from src.multi_agent.observability.llm_judge import (
            EvaluationResult,
            JudgeScore,
            EvaluationCriteria,
        )

        result = EvaluationResult(
            scores={
                EvaluationCriteria.RELEVANCE: JudgeScore(
                    criterion=EvaluationCriteria.RELEVANCE,
                    score=0.9,
                    reasoning="Good",
                ),
            },
            overall_score=0.9,
            model_used="claude-haiku-4-5",
            evaluation_duration_ms=150.0,
        )

        data = result.to_dict()

        assert "scores" in data
        assert "relevance" in data["scores"]
        assert data["overall_score"] == 0.9
        assert data["model_used"] == "claude-haiku-4-5"


# =============================================================================
# Agent Metrics Aggregator Tests
# =============================================================================


class TestAgentMetricsAggregator:
    """Tests for agent metrics aggregation."""

    def test_aggregator_creation(self):
        """Test aggregator creation."""
        from src.multi_agent.observability import create_metrics_aggregator

        aggregator = create_metrics_aggregator("test_workflow")

        assert aggregator.workflow_name == "test_workflow"
        assert len(aggregator.executions) == 0

    def test_record_agent_execution(self):
        """Test recording agent execution."""
        from src.multi_agent.observability import AgentMetricsAggregator

        aggregator = AgentMetricsAggregator("test")

        execution = aggregator.record_agent_execution(
            agent_name="extractor",
            duration_ms=1500,
            prompt_tokens=1000,
            completion_tokens=500,
            tool_calls=3,
            successful_tool_calls=3,
            success=True,
        )

        assert execution.agent_name == "extractor"
        assert execution.duration_ms == 1500
        assert execution.total_tokens == 1500
        assert len(aggregator.executions) == 1

    def test_agent_stats_update(self):
        """Test per-agent statistics update."""
        from src.multi_agent.observability import AgentMetricsAggregator

        aggregator = AgentMetricsAggregator("test")

        # Multiple executions for same agent
        aggregator.record_agent_execution("extractor", 1000)
        aggregator.record_agent_execution("extractor", 2000)
        aggregator.record_agent_execution("extractor", 1500, success=False, error="Timeout")

        stats = aggregator.get_agent_stats("extractor")

        assert stats.execution_count == 3
        assert stats.success_count == 2
        assert stats.failure_count == 1
        assert stats.avg_duration_ms() == 1500.0

    def test_workflow_stats(self):
        """Test workflow-level statistics."""
        from src.multi_agent.observability import AgentMetricsAggregator

        aggregator = AgentMetricsAggregator("test_workflow")

        aggregator.record_agent_execution("extractor", 1000, prompt_tokens=500)
        aggregator.record_agent_execution("classifier", 500, prompt_tokens=300)
        aggregator.record_agent_execution("synthesizer", 2000, prompt_tokens=1000)

        aggregator.finalize()
        stats = aggregator.get_workflow_stats()

        assert stats.workflow_name == "test_workflow"
        assert stats.agent_count == 3
        assert stats.total_executions == 3
        assert stats.successful_executions == 3
        assert stats.total_tokens == 1800

    def test_bottleneck_detection(self):
        """Test bottleneck agent detection."""
        from src.multi_agent.observability import AgentMetricsAggregator

        aggregator = AgentMetricsAggregator("test")

        aggregator.record_agent_execution("fast_agent", 100)
        aggregator.record_agent_execution("slow_agent", 5000)
        aggregator.record_agent_execution("medium_agent", 500)

        stats = aggregator.get_workflow_stats()

        assert stats.bottleneck_agent == "slow_agent"

    def test_bottleneck_analysis(self):
        """Test detailed bottleneck analysis."""
        from src.multi_agent.observability import AgentMetricsAggregator

        aggregator = AgentMetricsAggregator("test")

        aggregator.record_agent_execution("fast", 100)
        aggregator.record_agent_execution("slow", 6000)

        analysis = aggregator.get_bottleneck_analysis()

        assert analysis["bottleneck_agent"] == "slow"
        assert "recommendation" in analysis
        assert len(analysis["agent_ranking"]) == 2

    def test_workflow_stats_to_feedback(self):
        """Test workflow stats conversion to feedback."""
        from src.multi_agent.observability import AgentMetricsAggregator

        aggregator = AgentMetricsAggregator("test")
        aggregator.record_agent_execution("extractor", 1000)
        aggregator.finalize()

        stats = aggregator.get_workflow_stats()
        feedback = stats.to_feedback_dict()

        assert "workflow_success_rate" in feedback
        assert "workflow_efficiency" in feedback


# =============================================================================
# LangSmith Feedback Tests
# =============================================================================


class TestLangSmithFeedback:
    """Tests for LangSmith feedback coordination."""

    def test_feedback_coordinator_creation(self):
        """Test feedback coordinator creation."""
        from src.multi_agent.observability import create_feedback_coordinator

        # With disabled LangSmith
        coordinator = create_feedback_coordinator(None)
        assert coordinator is None

    def test_feedback_coordinator_with_mock_langsmith(self):
        """Test feedback coordinator with mocked LangSmith."""
        from src.multi_agent.observability import LangSmithFeedback

        mock_langsmith = Mock()
        mock_langsmith.is_enabled.return_value = True
        mock_langsmith.send_multiple_feedback.return_value = 3

        coordinator = LangSmithFeedback(mock_langsmith)

        assert coordinator.langsmith == mock_langsmith

    def test_submit_trajectory_metrics(self):
        """Test submitting trajectory metrics."""
        from src.multi_agent.observability import LangSmithFeedback, AgentTrajectory

        mock_langsmith = Mock()
        mock_langsmith.is_enabled.return_value = True
        mock_langsmith.send_multiple_feedback.return_value = 3

        coordinator = LangSmithFeedback(mock_langsmith)

        # Create trajectory with metrics
        trajectory = AgentTrajectory(agent_name="test", query="test")
        trajectory.add_action("search", {}, iteration=1)
        trajectory.add_observation("OK", success=True, iteration=1)
        trajectory.finalize("Answer")

        metrics = trajectory.compute_metrics()
        count = coordinator.submit_trajectory_metrics("run-123", metrics)

        assert count == 3
        mock_langsmith.send_multiple_feedback.assert_called_once()

    def test_submit_all_metrics(self):
        """Test submitting all metrics at once."""
        from src.multi_agent.observability import LangSmithFeedback, AgentTrajectory
        from src.multi_agent.core.state import ToolUsageMetrics

        mock_langsmith = Mock()
        mock_langsmith.is_enabled.return_value = True
        mock_langsmith.send_multiple_feedback.return_value = 3

        coordinator = LangSmithFeedback(mock_langsmith)

        trajectory = AgentTrajectory(agent_name="test", query="test")
        trajectory.finalize("Answer")

        tool_metrics = ToolUsageMetrics()

        total = coordinator.submit_all(
            run_id="run-123",
            trajectory_metrics=trajectory.compute_metrics(),
            tool_metrics=tool_metrics,
            custom_metrics={"custom_score": 0.9},
        )

        # Should call send_multiple_feedback multiple times
        assert mock_langsmith.send_multiple_feedback.call_count >= 3


# =============================================================================
# Exception Types Tests
# =============================================================================


class TestExceptionTypes:
    """Tests for new exception types."""

    def test_tool_hallucination_error(self):
        """Test ToolHallucinationError."""
        from src.exceptions import ToolHallucinationError

        error = ToolHallucinationError(
            message="Agent called non-existent tool 'fake_tool'",
            details={"tool_name": "fake_tool", "agent": "extractor"},
        )

        assert "fake_tool" in str(error)
        assert error.details["tool_name"] == "fake_tool"

    def test_evaluation_error(self):
        """Test EvaluationError and subclasses."""
        from src.exceptions import EvaluationError, JudgeError, TrajectoryError

        eval_error = EvaluationError("Evaluation failed")
        judge_error = JudgeError("Judge model unavailable")
        trajectory_error = TrajectoryError("Trajectory capture failed")

        assert isinstance(judge_error, EvaluationError)
        assert isinstance(trajectory_error, EvaluationError)

    def test_agent_timeout_error(self):
        """Test AgentTimeoutError."""
        from src.exceptions import AgentTimeoutError

        error = AgentTimeoutError(
            message="Agent 'extractor' timed out after 30s",
            details={"agent": "extractor", "timeout_seconds": 30},
        )

        assert "timed out" in str(error)

    def test_max_iterations_error(self):
        """Test MaxIterationsError."""
        from src.exceptions import MaxIterationsError

        error = MaxIterationsError(
            message="Agent exceeded 10 iterations",
            details={"max_iterations": 10, "agent": "extractor"},
        )

        assert "10" in str(error)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the evaluation infrastructure."""

    def test_full_trajectory_workflow(self):
        """Test complete trajectory capture workflow."""
        from src.multi_agent.observability import (
            AgentTrajectory,
            TrajectoryCollector,
        )

        collector = TrajectoryCollector(workflow_name="compliance_check")

        # Simulate extractor agent
        extractor_traj = AgentTrajectory(agent_name="extractor", query="What are the compliance requirements?")
        extractor_traj.add_thought("Analyzing query for compliance-related terms...", iteration=1)
        extractor_traj.add_action("search", {"query": "compliance requirements", "k": 10}, iteration=1)
        extractor_traj.add_observation("Found 10 relevant documents", success=True, duration_ms=250, iteration=1)
        extractor_traj.total_iterations = 1
        extractor_traj.finalize("Extracted compliance requirements from documents")
        collector.add_trajectory(extractor_traj)

        # Simulate classifier agent
        classifier_traj = AgentTrajectory(agent_name="classifier", query="What are the compliance requirements?")
        classifier_traj.add_thought("Classifying extracted requirements...", iteration=1)
        classifier_traj.add_action("classify", {"requirements": "..."}, iteration=1)
        classifier_traj.add_observation("Classified 15 requirements", success=True, duration_ms=150, iteration=1)
        classifier_traj.total_iterations = 1
        classifier_traj.finalize("Classified requirements by category")
        collector.add_trajectory(classifier_traj)

        collector.finalize()
        workflow_metrics = collector.get_workflow_metrics()

        assert workflow_metrics["agent_count"] == 2
        assert "extractor" in workflow_metrics["agent_metrics"]
        assert "classifier" in workflow_metrics["agent_metrics"]

    def test_tool_metrics_integration(self):
        """Test tool metrics integration with adapter."""
        import asyncio
        from src.multi_agent.tools.adapter import ToolAdapter

        adapter = ToolAdapter()
        adapter.clear_history()

        async def run_test():
            # Simulate some tool calls (including hallucinations)
            await adapter.execute("fake_tool_1", {}, "agent1")  # Hallucination
            await adapter.execute("fake_tool_2", {}, "agent2")  # Hallucination

            return adapter.get_usage_metrics()

        metrics = asyncio.run(run_test())

        assert metrics.total_calls == 2
        assert metrics.hallucinated_calls == 2
        assert metrics.hallucination_rate() == 1.0

        # Check feedback format
        feedback = metrics.to_feedback_dict()
        assert feedback["tool_hallucination_rate"] == 1.0

    def test_agent_metrics_integration(self):
        """Test agent metrics integration."""
        from src.multi_agent.observability import AgentMetricsAggregator

        aggregator = AgentMetricsAggregator("integration_test")

        # Simulate workflow execution
        agents = [
            ("orchestrator", 200, 100, 50, 0),
            ("extractor", 1500, 1000, 500, 5),
            ("classifier", 800, 600, 300, 2),
            ("synthesizer", 2000, 1500, 800, 3),
        ]

        for name, duration, prompt, completion, tools in agents:
            aggregator.record_agent_execution(
                agent_name=name,
                duration_ms=duration,
                prompt_tokens=prompt,
                completion_tokens=completion,
                tool_calls=tools,
                successful_tool_calls=tools,
            )

        aggregator.finalize()
        stats = aggregator.get_workflow_stats()

        assert stats.agent_count == 4
        assert stats.total_executions == 4
        assert stats.success_rate() == 1.0
        assert stats.bottleneck_agent == "synthesizer"  # Highest duration


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
