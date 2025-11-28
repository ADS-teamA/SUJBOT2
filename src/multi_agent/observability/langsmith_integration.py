"""
LangSmith Integration - Full observability for multi-agent workflows.

Provides:
1. Workflow tracing with LangSmith
2. Agent execution tracking
3. Cost monitoring
4. Performance profiling
5. Error tracking and debugging
6. Feedback submission for evaluation metrics
"""

import logging
import os
import uuid
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Lazy import for langsmith client
_langsmith_client = None


class LangSmithIntegration:
    """
    LangSmith integration for multi-agent observability.

    Configures LangSmith tracing for all agent executions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LangSmith integration.

        Args:
            config: LangSmith configuration dict
        """
        self.config = config

        # Extract configuration
        self.enabled = config.get("enabled", False)
        # Priority: env var > config (for security - don't store API key in config.json)
        self.api_key = os.getenv("LANGSMITH_API_KEY") or config.get("api_key")
        self.project_name = os.getenv("LANGSMITH_PROJECT_NAME") or config.get("project_name", "sujbot2-multi-agent")
        # EU endpoint support - use eu.api.smith.langchain.com for EU workspaces
        self.endpoint = os.getenv("LANGSMITH_ENDPOINT") or config.get("endpoint", "https://api.smith.langchain.com")
        self.trace_logging_level = config.get("trace_logging_level", "INFO")
        self.sample_rate = config.get("sample_rate", 1.0)

        # Tracing state
        self._tracing_enabled = False
        self._client = None
        self._current_run_id: Optional[str] = None

        logger.info(
            f"LangSmithIntegration initialized: "
            f"enabled={self.enabled}, project={self.project_name}"
        )

    def setup(self) -> bool:
        """
        Set up LangSmith tracing.

        Returns:
            True if setup successful
        """
        if not self.enabled:
            logger.info("LangSmith tracing disabled in configuration")
            return False

        if not self.api_key:
            logger.warning(
                "LangSmith API key not provided, tracing disabled. "
                "Set LANGSMITH_API_KEY environment variable to enable."
            )
            return False

        try:
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint

            # Set sampling rate (if < 1.0)
            if self.sample_rate < 1.0:
                os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = str(self.sample_rate)

            self._tracing_enabled = True

            # Initialize LangSmith client for feedback submission
            try:
                from langsmith import Client
                self._client = Client()
                logger.info("LangSmith client initialized for feedback submission")
            except ImportError:
                logger.warning(
                    "langsmith package not installed, feedback submission disabled. "
                    "Install with: pip install langsmith"
                )
            except Exception as e:
                logger.warning(f"LangSmith client initialization failed: {e}")

            logger.info(
                f"LangSmith tracing enabled: project={self.project_name}, "
                f"endpoint={self.endpoint}, sample_rate={self.sample_rate}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to set up LangSmith: {e}", exc_info=True)
            return False

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._tracing_enabled

    @contextmanager
    def trace_workflow(self, workflow_name: str, metadata: Optional[Dict] = None):
        """
        Context manager for tracing a workflow with run ID capture.

        Args:
            workflow_name: Name of workflow
            metadata: Optional metadata dict

        Yields:
            str: The run ID for this workflow trace
        """
        if not self._tracing_enabled:
            yield None
            return

        # Generate unique run ID for this workflow
        run_id = str(uuid.uuid4())
        self._current_run_id = run_id

        # Set run ID in environment for LangChain/LangGraph to pick up
        os.environ["LANGCHAIN_RUN_ID"] = run_id

        try:
            logger.debug(f"Starting trace: {workflow_name} (run_id={run_id[:8]}...)")
            yield run_id
            logger.debug(f"Completed trace: {workflow_name}")
        except Exception as e:
            logger.error(f"Workflow trace error: {e}", exc_info=True)
            raise
        finally:
            # Clean up run ID from environment
            os.environ.pop("LANGCHAIN_RUN_ID", None)

    def get_current_run_id(self) -> Optional[str]:
        """
        Get the current workflow run ID.

        Returns:
            Current run ID or None if not in a trace context
        """
        return self._current_run_id

    def send_feedback(
        self,
        key: str,
        score: Optional[float] = None,
        comment: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> bool:
        """
        Send feedback to LangSmith for a run.

        Args:
            key: Feedback key (e.g., "relevance", "correctness", "user_rating")
            score: Numeric score (0.0 to 1.0)
            comment: Optional comment explaining the score
            run_id: Run ID to attach feedback to (uses current if not provided)

        Returns:
            True if feedback sent successfully, False otherwise
        """
        if not self._client:
            logger.warning("LangSmith client not initialized, cannot send feedback")
            return False

        target_run_id = run_id or self._current_run_id
        if not target_run_id:
            logger.warning("No run ID available for feedback")
            return False

        try:
            self._client.create_feedback(
                run_id=target_run_id,
                key=key,
                score=score,
                comment=comment,
            )
            logger.info(
                f"Feedback sent: run={target_run_id[:8]}..., "
                f"key={key}, score={score}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send feedback: {e}")
            return False

    def send_multiple_feedback(
        self,
        feedbacks: Dict[str, float],
        run_id: Optional[str] = None,
        comments: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Send multiple feedback scores to LangSmith.

        Args:
            feedbacks: Dict mapping feedback keys to scores
            run_id: Run ID to attach feedback to
            comments: Optional dict mapping keys to comments

        Returns:
            Number of feedback items successfully sent
        """
        if not self._client:
            logger.warning("LangSmith client not initialized")
            return 0

        target_run_id = run_id or self._current_run_id
        if not target_run_id:
            logger.warning("No run ID available")
            return 0

        comments = comments or {}
        sent_count = 0

        for key, score in feedbacks.items():
            try:
                self._client.create_feedback(
                    run_id=target_run_id,
                    key=key,
                    score=score,
                    comment=comments.get(key),
                )
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send feedback for {key}: {e}")

        logger.info(f"Sent {sent_count}/{len(feedbacks)} feedback items")
        return sent_count

    def disable(self) -> None:
        """Disable LangSmith tracing."""
        if self._tracing_enabled:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
            os.environ.pop("LANGCHAIN_PROJECT", None)
            os.environ.pop("LANGCHAIN_TRACING_SAMPLING_RATE", None)

            self._tracing_enabled = False

            logger.info("LangSmith tracing disabled")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get LangSmith integration statistics.

        Returns:
            Dict with stats
        """
        return {
            "enabled": self._tracing_enabled,
            "project_name": self.project_name,
            "sample_rate": self.sample_rate,
            "trace_logging_level": self.trace_logging_level,
        }


def setup_langsmith(config: Dict[str, Any]) -> Optional[LangSmithIntegration]:
    """
    Set up LangSmith integration from configuration.

    Args:
        config: Multi-agent config dict

    Returns:
        LangSmithIntegration instance or None if disabled
    """
    langsmith_config = config.get("langsmith", {})

    try:
        integration = LangSmithIntegration(langsmith_config)

        if integration.setup():
            logger.info("LangSmith integration set up successfully")
            return integration
        else:
            logger.info("LangSmith integration not enabled")
            return None

    except Exception as e:
        logger.error(f"Failed to set up LangSmith integration: {e}", exc_info=True)
        return None
