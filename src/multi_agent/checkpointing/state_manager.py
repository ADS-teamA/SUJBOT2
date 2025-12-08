"""
State Manager - Coordinates state operations and recovery.

Provides high-level interface for:
1. State snapshot coordination
2. Recovery from checkpoints
3. State validation
4. Periodic cleanup
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from .postgres_checkpointer import PostgresCheckpointer

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages workflow state operations and recovery.

    Coordinates between LangGraph checkpointing and custom snapshot system.
    """

    def __init__(self, checkpointer: Optional[PostgresCheckpointer] = None):
        """
        Initialize state manager.

        Args:
            checkpointer: Optional PostgreSQL checkpointer
        """
        self.checkpointer = checkpointer
        self.query_counter = 0

        logger.info("StateManager initialized")

    def should_snapshot(self) -> bool:
        """
        Determine if snapshot should be taken based on interval.

        Returns:
            True if snapshot should be taken
        """
        if not self.checkpointer or not self.checkpointer.enable_snapshots:
            return False

        self.query_counter += 1

        # Snapshot every N queries
        if self.query_counter >= self.checkpointer.snapshot_interval:
            self.query_counter = 0
            return True

        return False

    def create_thread_id(self) -> str:
        """
        Create unique thread ID for workflow.

        Returns:
            UUID-based thread ID
        """
        thread_id = f"workflow_{uuid.uuid4().hex[:16]}_{int(datetime.now().timestamp())}"

        logger.debug(f"Created thread ID: {thread_id}")

        return thread_id

    def save_state_snapshot(
        self, thread_id: str, checkpoint_id: str, query: str, state: Dict[str, Any]
    ) -> None:
        """
        Save state snapshot if checkpointer available.

        Args:
            thread_id: Workflow thread ID
            checkpoint_id: Checkpoint ID
            query: Original query
            state: Current state dict
        """
        if not self.checkpointer:
            return

        try:
            self.checkpointer.save_snapshot(thread_id, checkpoint_id, query, state)

            logger.info(f"State snapshot saved for thread {thread_id}")

        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}", exc_info=True)

    def recover_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Recover latest state for thread.

        Args:
            thread_id: Workflow thread ID

        Returns:
            Latest state dict or None
        """
        if not self.checkpointer:
            logger.warning("Cannot recover state: no checkpointer available")
            return None

        try:
            snapshot = self.checkpointer.get_latest_snapshot(thread_id)

            if snapshot:
                logger.info(f"State recovered for thread {thread_id}")
                return snapshot.get("state")

            logger.info(f"No snapshot found for thread {thread_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to recover state: {e}", exc_info=True)
            return None

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate state has required fields.

        Args:
            state: State dict to validate

        Returns:
            True if valid
        """
        required_fields = ["query", "execution_phase"]

        for field in required_fields:
            if field not in state:
                logger.error(f"Invalid state: missing field '{field}'")
                return False

        return True

    def cleanup(self) -> None:
        """Perform cleanup of old snapshots."""
        if not self.checkpointer:
            return

        try:
            deleted = self.checkpointer.cleanup_old_snapshots()

            logger.info(f"Cleanup complete: {deleted} snapshots removed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get state management statistics.

        Returns:
            Dict with stats
        """
        stats = {
            "checkpointer_enabled": self.checkpointer is not None,
            "query_counter": self.query_counter,
        }

        if self.checkpointer:
            stats.update(
                {
                    "snapshots_enabled": self.checkpointer.enable_snapshots,
                    "snapshot_interval": self.checkpointer.snapshot_interval,
                    "recovery_window_hours": self.checkpointer.recovery_window_hours,
                }
            )

        return stats
