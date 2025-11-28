"""
LangSmith Feedback Submission.

Coordinates feedback submission from multiple evaluation sources:
- Trajectory metrics (tool success rate, efficiency)
- Tool usage metrics (hallucination rate)
- LLM Judge scores (relevance, groundedness, coherence)

Usage:
    feedback = LangSmithFeedback(langsmith_integration)

    # Submit all metrics for a run
    feedback.submit_all(
        run_id="...",
        trajectory_metrics=trajectory.compute_metrics(),
        tool_metrics=adapter.get_usage_metrics(),
        judge_result=judge_result,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .trajectory import TrajectoryMetrics
    from .llm_judge import EvaluationResult
    from ..core.state import ToolUsageMetrics

logger = logging.getLogger(__name__)


@dataclass
class FeedbackBatch:
    """Batch of feedback items to submit."""

    run_id: str
    feedbacks: Dict[str, float]  # key -> score
    comments: Dict[str, str]     # key -> comment
    source: str                  # Source of feedback (trajectory, judge, etc.)


class LangSmithFeedback:
    """
    Coordinates feedback submission to LangSmith.

    Collects metrics from multiple sources and submits them
    as feedback to LangSmith runs.
    """

    def __init__(self, langsmith_integration):
        """
        Initialize feedback coordinator.

        Args:
            langsmith_integration: LangSmithIntegration instance
        """
        self.langsmith = langsmith_integration
        self._pending_batches: List[FeedbackBatch] = []

        logger.info("LangSmithFeedback initialized")

    def submit_trajectory_metrics(
        self,
        run_id: str,
        trajectory_metrics: "TrajectoryMetrics",
    ) -> int:
        """
        Submit trajectory metrics as feedback.

        Args:
            run_id: LangSmith run ID
            trajectory_metrics: TrajectoryMetrics from trajectory.compute_metrics()

        Returns:
            Number of feedback items submitted
        """
        if not self.langsmith or not self.langsmith.is_enabled():
            logger.debug("LangSmith not enabled, skipping trajectory feedback")
            return 0

        feedbacks = trajectory_metrics.to_langsmith_feedback()
        comments = {
            "trajectory_tool_success_rate": f"Tool success: {trajectory_metrics.tool_success_rate:.1%}",
            "trajectory_efficiency": f"Steps: {trajectory_metrics.total_steps}, Duration: {trajectory_metrics.total_duration_ms:.0f}ms",
            "trajectory_error_recovery": f"Recovered from {int(trajectory_metrics.error_recovery_rate * trajectory_metrics.total_steps)} errors",
        }

        return self.langsmith.send_multiple_feedback(
            feedbacks=feedbacks,
            run_id=run_id,
            comments=comments,
        )

    def submit_tool_metrics(
        self,
        run_id: str,
        tool_metrics: "ToolUsageMetrics",
    ) -> int:
        """
        Submit tool usage metrics as feedback.

        Args:
            run_id: LangSmith run ID
            tool_metrics: ToolUsageMetrics from adapter.get_usage_metrics()

        Returns:
            Number of feedback items submitted
        """
        if not self.langsmith or not self.langsmith.is_enabled():
            logger.debug("LangSmith not enabled, skipping tool feedback")
            return 0

        feedbacks = tool_metrics.to_feedback_dict()
        comments = {
            "tool_success_rate": f"{tool_metrics.successful_calls}/{tool_metrics.total_calls} successful",
            "tool_hallucination_rate": f"{tool_metrics.hallucinated_calls} hallucinated calls",
            "tool_error_rate": f"{tool_metrics.failed_calls} failures, {tool_metrics.validation_errors} validation errors",
        }

        return self.langsmith.send_multiple_feedback(
            feedbacks=feedbacks,
            run_id=run_id,
            comments=comments,
        )

    def submit_judge_scores(
        self,
        run_id: str,
        judge_result: "EvaluationResult",
    ) -> int:
        """
        Submit LLM Judge scores as feedback.

        Args:
            run_id: LangSmith run ID
            judge_result: EvaluationResult from LLMJudge.evaluate()

        Returns:
            Number of feedback items submitted
        """
        if not self.langsmith or not self.langsmith.is_enabled():
            logger.debug("LangSmith not enabled, skipping judge feedback")
            return 0

        feedbacks = judge_result.to_feedback_dict()
        comments = {}

        for criterion, score in judge_result.scores.items():
            if score.reasoning:
                comments[f"judge_{criterion.value}"] = score.reasoning[:200]

        return self.langsmith.send_multiple_feedback(
            feedbacks=feedbacks,
            run_id=run_id,
            comments=comments,
        )

    def submit_all(
        self,
        run_id: str,
        trajectory_metrics: Optional["TrajectoryMetrics"] = None,
        tool_metrics: Optional["ToolUsageMetrics"] = None,
        judge_result: Optional["EvaluationResult"] = None,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Submit all available metrics as feedback.

        Args:
            run_id: LangSmith run ID
            trajectory_metrics: Optional trajectory metrics
            tool_metrics: Optional tool usage metrics
            judge_result: Optional LLM judge result
            custom_metrics: Optional custom metrics dict

        Returns:
            Total number of feedback items submitted
        """
        if not self.langsmith or not self.langsmith.is_enabled():
            logger.debug("LangSmith not enabled, skipping all feedback")
            return 0

        total_submitted = 0

        # Submit trajectory metrics
        if trajectory_metrics:
            try:
                count = self.submit_trajectory_metrics(run_id, trajectory_metrics)
                total_submitted += count
                logger.debug(f"Submitted {count} trajectory metrics")
            except Exception as e:
                logger.error(f"Failed to submit trajectory metrics: {e}")

        # Submit tool metrics
        if tool_metrics:
            try:
                count = self.submit_tool_metrics(run_id, tool_metrics)
                total_submitted += count
                logger.debug(f"Submitted {count} tool metrics")
            except Exception as e:
                logger.error(f"Failed to submit tool metrics: {e}")

        # Submit judge scores
        if judge_result:
            try:
                count = self.submit_judge_scores(run_id, judge_result)
                total_submitted += count
                logger.debug(f"Submitted {count} judge scores")
            except Exception as e:
                logger.error(f"Failed to submit judge scores: {e}")

        # Submit custom metrics
        if custom_metrics:
            try:
                count = self.langsmith.send_multiple_feedback(
                    feedbacks=custom_metrics,
                    run_id=run_id,
                )
                total_submitted += count
                logger.debug(f"Submitted {count} custom metrics")
            except Exception as e:
                logger.error(f"Failed to submit custom metrics: {e}")

        logger.info(
            f"LangSmith feedback submitted: {total_submitted} items for run {run_id[:8]}..."
        )

        return total_submitted

    def submit_user_feedback(
        self,
        run_id: str,
        rating: int,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Submit user feedback (thumbs up/down or rating).

        Args:
            run_id: LangSmith run ID
            rating: User rating (1-5 or 0/1 for thumbs)
            comment: Optional user comment

        Returns:
            True if submitted successfully
        """
        if not self.langsmith or not self.langsmith.is_enabled():
            return False

        # Normalize rating to 0-1 scale
        if rating <= 1:
            score = float(rating)
        else:
            score = (rating - 1) / 4.0  # Convert 1-5 to 0-1

        return self.langsmith.send_feedback(
            key="user_rating",
            score=score,
            comment=comment,
            run_id=run_id,
        )


def create_feedback_coordinator(
    langsmith_integration
) -> Optional[LangSmithFeedback]:
    """
    Create feedback coordinator from LangSmith integration.

    Args:
        langsmith_integration: LangSmithIntegration instance

    Returns:
        LangSmithFeedback instance or None if LangSmith not enabled
    """
    if langsmith_integration is None or not langsmith_integration.is_enabled():
        logger.info("LangSmith not enabled, feedback coordinator not created")
        return None

    return LangSmithFeedback(langsmith_integration)
