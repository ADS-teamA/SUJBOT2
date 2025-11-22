"""Checkpointing system for multi-agent framework.

Provides optional PostgreSQL-backed state persistence for:
- Long-running workflows
- Error recovery
- Workflow resume after interruption
- State snapshots for debugging

Design note:
- psycopg / libpq are heavy optional dependencies. To keep environments
  without PostgreSQL working (e.g. local eval, benchmarks), we avoid
  importing the Postgres checkpointer unless the config explicitly
  requests a PostgreSQL backend.
"""

from typing import Any, Dict, Optional
import logging

from .state_manager import StateManager

logger = logging.getLogger(__name__)


def create_checkpointer(config: Dict[str, Any]) -> Optional["PostgresCheckpointer"]:
    """
    Create PostgreSQL checkpointer from configuration, if enabled.

    Returns:
        PostgresCheckpointer instance or None if disabled or unavailable.
    """
    checkpointing_config = config.get("checkpointing", {})
    backend = checkpointing_config.get("backend", "none")

    # Shortcut: no PostgreSQL backend requested
    if backend != "postgresql":
        logger.info(f"PostgreSQL checkpointing disabled (backend={backend})")
        return None

    # Import Postgres-specific implementation lazily so environments
    # without psycopg/libpq can still run (with checkpointing disabled).
    try:
        from .postgres_checkpointer import PostgresCheckpointer  # type: ignore
    except Exception as e:
        logger.error(
            "PostgreSQL checkpointing requested but psycopg/libpq are not available: %s. "
            "Run without checkpointing or install PostgreSQL dependencies.",
            e,
        )
        return None

    postgres_config = checkpointing_config.get("postgresql", {})
    if not postgres_config:
        logger.warning("PostgreSQL checkpointing enabled but no configuration provided")
        return None

    try:
        checkpointer = PostgresCheckpointer(postgres_config)
        checkpointer.initialize()
        logger.info("PostgreSQL checkpointer created successfully")
        return checkpointer
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL checkpointer: {e}", exc_info=True)
        return None


__all__ = ["create_checkpointer", "StateManager"]
