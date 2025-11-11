"""
LangSmith Integration - Full observability for multi-agent workflows.

Provides:
1. Workflow tracing with LangSmith
2. Agent execution tracking
3. Cost monitoring
4. Performance profiling
5. Error tracking and debugging
"""

import logging
import os
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


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
        self.api_key = config.get("api_key")
        self.project_name = config.get("project_name", "sujbot2-multi-agent")
        self.trace_logging_level = config.get("trace_logging_level", "INFO")
        self.sample_rate = config.get("sample_rate", 1.0)

        # Tracing state
        self._tracing_enabled = False

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

            # Set sampling rate (if < 1.0)
            if self.sample_rate < 1.0:
                os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = str(self.sample_rate)

            self._tracing_enabled = True

            logger.info(
                f"LangSmith tracing enabled: project={self.project_name}, "
                f"sample_rate={self.sample_rate}"
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
        Context manager for tracing a workflow.

        Args:
            workflow_name: Name of workflow
            metadata: Optional metadata dict

        Yields:
            None
        """
        if not self._tracing_enabled:
            yield
            return

        try:
            # LangSmith will automatically trace all LangChain/LangGraph operations
            logger.debug(f"Starting trace for workflow: {workflow_name}")

            yield

            logger.debug(f"Completed trace for workflow: {workflow_name}")

        except Exception as e:
            logger.error(f"Workflow trace error: {e}", exc_info=True)
            raise

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
