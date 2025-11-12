"""
EventBus - Centralized event management for multi-agent system.

Provides decoupled event emission and consumption pattern using asyncio.Queue.
Supports real-time progress streaming without tight coupling between components.

Architecture:
- Agent emits events via event_bus.emit()
- Runner consumes events via event_bus.get_pending_events()
- No direct state dict manipulation
- Thread-safe async operations
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Event Schemas
# ============================================================================


class EventType(str, Enum):
    """
    Event types emitted by multi-agent system.

    Used for type-safe event categorization and filtering.
    """

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"

    # Agent events
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"

    # Workflow events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"

    # Error events
    ERROR = "error"


class Event(BaseModel):
    """
    Base event schema with validation.

    All events emitted through EventBus follow this schema.
    Pydantic validation ensures type safety and data integrity.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any]

    # Optional fields for context
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# EventBus Implementation
# ============================================================================


class EventBus:
    """
    Thread-safe event bus for multi-agent system.

    Provides centralized event management with:
    - Async-first design (asyncio.Queue)
    - Bounded queue to prevent memory leaks
    - Event validation via Pydantic
    - Subscriber pattern for logging/debugging
    - Non-blocking batch retrieval

    Example:
        ```python
        event_bus = EventBus(max_queue_size=1000)

        # Emit event
        await event_bus.emit(
            EventType.TOOL_CALL_START,
            {"agent": "extractor", "tool": "search"}
        )

        # Consume events
        events = await event_bus.get_pending_events()
        for event in events:
            print(f"{event.event_type}: {event.data}")
        ```
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize EventBus with bounded queue.

        Args:
            max_queue_size: Maximum queue size to prevent memory leaks.
                           Defaults to 1000 events (~1MB memory).
        """
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._subscribers: List[Callable[[Event], Awaitable[None]]] = []
        self._event_count: int = 0
        self._lock = asyncio.Lock()

        logger.debug(f"EventBus initialized with max_queue_size={max_queue_size}")

    async def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        agent_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Emit event to queue and notify subscribers.

        Args:
            event_type: Type of event to emit
            data: Event data (must be JSON-serializable)
            agent_name: Optional agent name for context
            tool_name: Optional tool name for context
            error: Optional error message

        Raises:
            No exceptions raised - failures are logged and gracefully handled
        """
        try:
            # Sanitize sensitive data
            sanitized_data = self._sanitize_event_data(data)

            # Create and validate event
            event = Event(
                event_type=event_type,
                data=sanitized_data,
                agent_name=agent_name,
                tool_name=tool_name,
                error=error,
            )

            # Add to queue (non-blocking)
            try:
                self._event_queue.put_nowait(event)

                async with self._lock:
                    self._event_count += 1

                logger.debug(
                    f"Event emitted: {event_type.value} "
                    f"(agent={agent_name}, tool={tool_name}, queue_size={self._event_queue.qsize()})"
                )

                # Notify subscribers asynchronously
                await self._notify_subscribers(event)

            except asyncio.QueueFull:
                logger.warning(
                    f"Event queue full (max={self._event_queue.maxsize}) - dropping event: {event_type.value}"
                )
                # Gracefully drop event - don't block agent execution

        except Exception as e:
            # NEVER crash agent due to event emission failure
            logger.error(f"Failed to emit event {event_type}: {e}", exc_info=True)

    async def get_pending_events(self, timeout: float = 0.0) -> List[Event]:
        """
        Get all pending events from queue (non-blocking batch retrieval).

        Args:
            timeout: Maximum time to wait for first event (default: 0.0 = non-blocking).
                    Used for blocking wait if no events immediately available.

        Returns:
            List of events (may be empty if queue is empty)
        """
        events = []

        try:
            # Try to get first event (with optional timeout)
            if timeout > 0:
                try:
                    first_event = await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
                    events.append(first_event)
                except asyncio.TimeoutError:
                    return events  # No events within timeout
            else:
                # Non-blocking get
                try:
                    first_event = self._event_queue.get_nowait()
                    events.append(first_event)
                except asyncio.QueueEmpty:
                    return events  # No events available

            # Batch retrieve remaining events (non-blocking)
            while True:
                try:
                    event = self._event_queue.get_nowait()
                    events.append(event)
                except asyncio.QueueEmpty:
                    break

        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}", exc_info=True)

        return events

    def subscribe(self, callback: Callable[[Event], Awaitable[None]]) -> None:
        """
        Subscribe to all events.

        Args:
            callback: Async callback function invoked on every event emission.
                     Signature: async def callback(event: Event) -> None

        Example:
            ```python
            async def log_event(event: Event):
                print(f"Event: {event.event_type} - {event.data}")

            event_bus.subscribe(log_event)
            ```
        """
        self._subscribers.append(callback)
        logger.debug(f"Subscriber added (total: {len(self._subscribers)})")

    async def _notify_subscribers(self, event: Event) -> None:
        """
        Notify all subscribers asynchronously.

        Args:
            event: Event to broadcast to subscribers
        """
        for callback in self._subscribers:
            try:
                await callback(event)
            except Exception as e:
                # Don't let subscriber errors break event emission
                logger.error(f"Subscriber callback failed: {e}", exc_info=True)

    def clear_events(self) -> None:
        """
        Clear all pending events from queue.

        Used for testing or reset scenarios.
        """
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.debug("Event queue cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get EventBus statistics for monitoring.

        Returns:
            Dict with queue size, total events emitted, subscriber count
        """
        return {
            "queue_size": self._event_queue.qsize(),
            "queue_max_size": self._event_queue.maxsize,
            "total_events_emitted": self._event_count,
            "subscriber_count": len(self._subscribers),
            "queue_utilization": (
                self._event_queue.qsize() / self._event_queue.maxsize
                if self._event_queue.maxsize > 0
                else 0.0
            ),
        }

    def _sanitize_event_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive fields from event data.

        Args:
            data: Raw event data

        Returns:
            Sanitized event data (sensitive keys removed)
        """
        sensitive_keys = {"api_key", "password", "token", "secret", "key"}

        # Recursively sanitize nested dicts
        def sanitize_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: sanitize_recursive(v) if k.lower() not in sensitive_keys else "***REDACTED***"
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [sanitize_recursive(item) for item in obj]
            else:
                return obj

        return sanitize_recursive(data)
