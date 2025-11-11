"""
Error ID Tracking System - Generate unique error IDs for Sentry integration.

Provides:
- Unique error ID generation (format: ERR-YYYYMMDD-HHMMSS-<random>)
- Error context tracking
- Integration with logging
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels for tracking."""
    CRITICAL = "critical"  # System failure, requires immediate attention
    HIGH = "high"          # Feature broken, but system functional
    MEDIUM = "medium"      # Degraded performance or non-critical feature
    LOW = "low"            # Minor issue, does not affect functionality


class ErrorTracker:
    """
    Track errors with unique IDs for Sentry integration.

    Features:
    - Unique error ID generation
    - Error context preservation
    - Automatic Sentry tagging (when available)
    - Error history tracking
    """

    def __init__(self):
        """Initialize error tracker."""
        self.error_history: list[Dict[str, Any]] = []
        self._sentry_available = False

        # Try to import Sentry SDK (optional)
        try:
            import sentry_sdk
            self._sentry = sentry_sdk
            self._sentry_available = True
            logger.debug("Sentry SDK available for error tracking")
        except ImportError:
            self._sentry = None
            logger.debug("Sentry SDK not available (install with: pip install sentry-sdk)")

    def generate_error_id(self) -> str:
        """
        Generate unique error ID.

        Format: ERR-YYYYMMDD-HHMMSS-<short-uuid>

        Example: ERR-20251111-143025-a3b2c1

        Returns:
            Unique error ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_uuid = str(uuid.uuid4())[:6]
        return f"ERR-{timestamp}-{short_uuid}"

    def track_error(
        self,
        error: Exception,
        error_id: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> str:
        """
        Track error with unique ID and context.

        Args:
            error: Exception that occurred
            error_id: Optional pre-generated error ID
            severity: Error severity level
            context: Additional context dict
            agent_name: Name of agent where error occurred
            tool_name: Name of tool where error occurred

        Returns:
            Error ID string

        Example:
            >>> tracker = ErrorTracker()
            >>> error_id = tracker.track_error(
            ...     error=ValueError("Invalid query"),
            ...     severity=ErrorSeverity.HIGH,
            ...     context={"query": "test"},
            ...     agent_name="extractor"
            ... )
            >>> logger.error(f"[{error_id}] Extraction failed")
        """
        if error_id is None:
            error_id = self.generate_error_id()

        # Build error context
        error_context = {
            "error_id": error_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "tool_name": tool_name,
            "custom_context": context or {},
        }

        # Store in history
        self.error_history.append(error_context)

        # Send to Sentry if available
        if self._sentry_available and self._sentry:
            self._send_to_sentry(error, error_context)

        logger.debug(f"Tracked error {error_id}: {type(error).__name__}")

        return error_id

    def _send_to_sentry(self, error: Exception, error_context: Dict[str, Any]) -> None:
        """
        Send error to Sentry with context.

        Args:
            error: Exception to send
            error_context: Error context dict
        """
        try:
            with self._sentry.push_scope() as scope:
                # Add error ID as tag
                scope.set_tag("error_id", error_context["error_id"])
                scope.set_tag("severity", error_context["severity"])

                # Add agent/tool context
                if error_context["agent_name"]:
                    scope.set_tag("agent_name", error_context["agent_name"])
                if error_context["tool_name"]:
                    scope.set_tag("tool_name", error_context["tool_name"])

                # Add custom context
                if error_context["custom_context"]:
                    scope.set_context("custom", error_context["custom_context"])

                # Capture exception
                self._sentry.capture_exception(error)

        except Exception as e:
            logger.warning(f"Failed to send error to Sentry: {e}")

    def get_error_history(self, limit: int = 100) -> list[Dict[str, Any]]:
        """
        Get recent error history.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error context dicts (most recent first)
        """
        return self.error_history[-limit:][::-1]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get error tracking statistics.

        Returns:
            Dict with error counts by severity and type
        """
        if not self.error_history:
            return {
                "total_errors": 0,
                "by_severity": {},
                "by_type": {},
                "by_agent": {},
                "by_tool": {},
            }

        # Count by severity
        by_severity = {}
        for error in self.error_history:
            severity = error["severity"]
            by_severity[severity] = by_severity.get(severity, 0) + 1

        # Count by error type
        by_type = {}
        for error in self.error_history:
            error_type = error["error_type"]
            by_type[error_type] = by_type.get(error_type, 0) + 1

        # Count by agent
        by_agent = {}
        for error in self.error_history:
            agent_name = error.get("agent_name")
            if agent_name:
                by_agent[agent_name] = by_agent.get(agent_name, 0) + 1

        # Count by tool
        by_tool = {}
        for error in self.error_history:
            tool_name = error.get("tool_name")
            if tool_name:
                by_tool[tool_name] = by_tool.get(tool_name, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_severity": by_severity,
            "by_type": by_type,
            "by_agent": by_agent,
            "by_tool": by_tool,
            "sentry_available": self._sentry_available,
        }

    def clear_history(self) -> None:
        """Clear error history (for testing)."""
        self.error_history.clear()


# Global error tracker instance
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """
    Get global error tracker instance.

    Returns:
        ErrorTracker instance
    """
    global _error_tracker

    if _error_tracker is None:
        _error_tracker = ErrorTracker()

    return _error_tracker


def track_error(
    error: Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **kwargs
) -> str:
    """
    Convenience function to track error with global tracker.

    Args:
        error: Exception to track
        severity: Error severity level
        **kwargs: Additional context (agent_name, tool_name, context, etc.)

    Returns:
        Error ID string

    Example:
        >>> from src.multi_agent.core.error_tracker import track_error, ErrorSeverity
        >>> error_id = track_error(
        ...     ValueError("Invalid input"),
        ...     severity=ErrorSeverity.HIGH,
        ...     agent_name="extractor",
        ...     context={"query": "test"}
        ... )
        >>> logger.error(f"[{error_id}] Extraction failed")
    """
    tracker = get_error_tracker()
    return tracker.track_error(error, severity=severity, **kwargs)
